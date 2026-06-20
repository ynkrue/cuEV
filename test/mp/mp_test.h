/**
 * @file   mp_test.h
 * @brief  Minimal self-registering test framework for the distributed (Phase 3) suite.
 *
 * GTest and MPI don't mix cleanly (see test_mp.cpp), so this reproduces the one
 * piece of GTest we actually want — TEST()-style auto-registration — without
 * pulling in the GTest runtime. A test is a bool(Context&) function; the body
 * computes a local error, MPI_Allreduces it if needed, prints PASS/FAIL on
 * rank 0, and returns the result. test_mp.cpp owns the registry + main() and
 * runs every registered case.
 *
 * Usage (in any test_*_mp.cpp translation unit):
 *
 *   MP_TEST(QdwhMp, SdcTraceDiagonal) {
 *       ... build inputs, call the primitive under test ...
 *       bool pass = ...;
 *       if (ctx.rank == 0) printf("...  [%s]\n", pass ? "OK" : "FAIL");
 *       return pass;
 *   }
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once

#include "common.h"
#include "mp/comm.h"
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

namespace mptest {

using TestFn = std::function<bool(cuev::mp::Context &)>;

struct TestCase {
    std::string suite;
    std::string name; ///< "Suite.Case", kept for diagnostics
    TestFn fn;
};

/// Global registry, one instance for the whole binary (defined in test_mp.cpp).
std::vector<TestCase> &registry();

/// Static-init-time registration, same trick GTest's TEST() macro uses.
struct Registrar {
    Registrar(const char *suite, const char *name, TestFn fn) {
        registry().push_back({suite, name, std::move(fn)});
    }
};

// =============================================================================
// GTest-style colored output. Colors are emitted only when stdout is a TTY,
// so piping to a log file stays plain text.
// =============================================================================
namespace color {
inline bool enabled() {
    static const bool tty = isatty(fileno(stdout)) != 0;
    return tty;
}
inline const char *green() {
    return enabled() ? "\033[32m" : "";
}
inline const char *red() {
    return enabled() ? "\033[31m" : "";
}
inline const char *reset() {
    return enabled() ? "\033[0m" : "";
}
} // namespace color

/// Rank-0-only diagnostic line printed inside a running test, indented to
/// line up with the test name in "[ RUN      ] Suite.Case" above it.
/// Only printed when `pass` is false, so passing tests stay quiet.
inline void detail(cuev::mp::Context &ctx, bool pass, const char *fmt, ...) {
    if (ctx.rank != 0 || pass) return;
    printf("             "); // matches strlen("[ RUN      ] ")
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    printf("\n");
}

// =============================================================================
// Shared helpers — local <-> global index mapping (2D block-cyclic, rsrc=csrc=0)
// and the post-distributed-op sync/info-check pattern used by every test.
// =============================================================================

inline int64_t l2g_row(int64_t li, int prow, int nprow, int64_t nb) {
    return (li / nb * nprow + prow) * nb + (li % nb);
}
inline int64_t l2g_col(int64_t lj, int pcol, int npcol, int64_t nb) {
    return (lj / nb * npcol + pcol) * nb + (lj % nb);
}

inline void mp_sync(cuev::mp::Context &ctx) {
    CAL_CHECK(cal_stream_sync(ctx.cal, ctx.stream));
    CAL_CHECK(cal_comm_barrier(ctx.cal, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

inline void check_info(cuev::mp::Context &ctx, int *d_info, const char *op) {
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "[rank %d] %s: info=%d\n", ctx.rank, op, h_info);
        MPI_Abort(ctx.comm, 1);
    }
}

} // namespace mptest

#define MP_TEST(suite, name)                                                                       \
    static bool mptest_##suite##_##name(cuev::mp::Context &ctx);                                   \
    static ::mptest::Registrar mptest_reg_##suite##_##name(#suite, #suite "." #name,               \
                                                           mptest_##suite##_##name);               \
    static bool mptest_##suite##_##name(cuev::mp::Context &ctx)
