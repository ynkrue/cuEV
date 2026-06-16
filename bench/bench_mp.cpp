/**
 * @file   bench_mp.cpp
 * @brief  Benchmark distributed eigensolver.
 *
 * Run:  srun -N<nodes> --tasks-per-node=<p> --gpus-per-node=<p> \
 *           build/cuBenchMp [--nprow P] [--npcol Q] [--nb NB] [--n N] [--warmup W] [--iters I]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "mp/comm.h"
#include "mp/workspace_mp.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mpi.h>
#include <type_traits>
#include <vector>

#ifdef CUEV_HAVE_ELPA
#include <elpa/elpa.h>
#endif
#ifdef CUEV_HAVE_MAGMA
#include <magma.h>
#endif

using cuev::mp::Context;

// =============================================================================
// Utilities
// =============================================================================

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t s) {
        cudaEventRecord(start, s);
    }
    float end(cudaStream_t s) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

struct WallTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    void begin() {
        t0 = clock::now();
    }
    float end() {
        return std::chrono::duration<float, std::milli>(clock::now() - t0).count();
    }
};

static void mp_sync(Context &ctx) {
    CAL_CHECK(cal_stream_sync(ctx.cal, ctx.stream));
    CAL_CHECK(cal_comm_barrier(ctx.cal, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

template <typename T> static void fill_symmetric_local(std::vector<T> &h) {
    for (size_t i = 0; i < h.size(); ++i)
        h[i] = (T)(rand() % 200 - 100) / T(100);
}

// =============================================================================
// cuev::mp
// =============================================================================

// template <typename T>
// static float bench_cuev_mp(Context &ctx, int64_t n, int warmup, int iters) {
//     // ... cuev::mp::symm_eig_solve<T>(ctx, dA, n, d_eval, d_evec) ...
//     return 0.f;
// }

// =============================================================================
// Suite
// =============================================================================

template <typename T> static void run_suite(Context &ctx, int64_t n, int warmup, int iters) {
    const char *prec = std::is_same_v<T, float> ? "fp32" : "fp64";
    const double flops = 4.0 / 3.0 * (double)n * n * n;

    if (ctx.rank == 0) {
        printf("=== solve %s  n=%lld ===\n", prec, (long long)n);
        fflush(stdout);
    }

    auto print_result = [&](float ms, const char *name) {
        if (ctx.rank == 0) {
            printf("  %-34s  %8.3f ms   %6.3f TFLOP/s\n", name, ms, flops / (ms * 1e-3) / 1e12);
            fflush(stdout);
        }
    };

    // --- cuev::mp (TODO) ---
    // print_result(bench_cuev_mp<T>(ctx, n, warmup, iters), "cuev_mp_symm_eig_solve");

    if (ctx.rank == 0) {
        printf("\n");
        fflush(stdout);
    }
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world;
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int nprow = 0, npcol = 0, nb = 256;
    int64_t n = 8192;
    int warmup = 3, iters = 10;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--nprow") && i + 1 < argc) {
            nprow = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--npcol") && i + 1 < argc) {
            npcol = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--nb") && i + 1 < argc) {
            nb = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--n") && i + 1 < argc) {
            n = (int64_t)atoll(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--warmup") && i + 1 < argc) {
            warmup = atoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = atoi(argv[++i]);
            continue;
        }
    }
    if (nprow == 0 || npcol == 0) cuev::mp::grid_factor(world, nprow, npcol);

    Context ctx;
    cuev::mp::ctx_init(ctx, nb, nprow, npcol);

    if (ctx.rank == 0)
        printf("cuBenchMp  grid=%dx%d  nb=%d  n=%lld  ranks=%d  warmup=%d  iters=%d\n\n", ctx.nprow,
               ctx.npcol, ctx.nb, (long long)n, ctx.world_size, warmup, iters);

    // run_suite<float>(ctx, n, warmup, iters);
    run_suite<double>(ctx, n, warmup, iters);

    cuev::mp::ctx_finalize(ctx);
    MPI_Finalize();
    return 0;
}
