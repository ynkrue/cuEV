/**
 * @file   test_qdwh_mp.cpp
 * @brief  Correctness tests for the distributed QDWH primitives — cuev::mp::kernels.
 *
 * Mirrors test_qdwh.cu's coverage (shift, sign-function building blocks) but
 * for the 2D block-cyclic distributed kernels, using the MP_TEST harness
 * (see mp_test.h) since GTest and MPI don't mix.
 *
 * Tests:
 *   QdwhMp.SdcTraceDiagonal — sdc_trace_mp on a known diagonal matrix
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "mp/kernels_mp.cuh"
#include "mp/workspace_mp.h"
#include "mp_test.h"
#include <algorithm>
#include <cmath>
#include <vector>

using cuev::mp::dist_describe;
using cuev::mp::dist_free;
using cuev::mp::dist_local_count;
using cuev::mp::DistMatrix;
using cuev::mp::workspace_mp_alloc;
using cuev::mp::workspace_mp_free;
using cuev::mp::WorkspaceMp;
using mptest::l2g_col;
using mptest::l2g_row;

// =============================================================================
// SdcTraceDiagonal
//
// What it checks: sdc_trace_mp correctly sums the global diagonal across
// ranks, given a matrix whose diagonal entries are local-row/col owned by
// different ranks depending on the process grid.
//
// A[gi,gj] = (gi+1) if gi==gj else 0  →  trace = n(n+1)/2.
// =============================================================================
MP_TEST(QdwhMp, SdcTraceDiagonal) {
    using namespace cuev::mp;
    const int64_t n = 512;

    int64_t count = dist_local_count<double>(ctx, n, n);
    double *dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, count * sizeof(double)));

    DistMatrix<double> A = dist_describe<double>(ctx, n, n, dA);

    {
        std::vector<double> hA(count, 0.0);
        for (int64_t lj = 0; lj < A.local_cols; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, ctx.nb);
            for (int64_t li = 0; li < A.local_rows; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, ctx.nb);
                if (gi == gj) hA[lj * A.lld + li] = (double)(gi + 1);
            }
        }
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    }

    double trace = kernels::sdc_trace_mp(ctx, A.data, A.local_rows, A.local_cols, A.lld, ctx.prow,
                                         ctx.pcol, ctx.nprow, ctx.npcol, ctx.nb);
    double expected = (double)n * (n + 1) / 2.0;
    bool pass = std::fabs(trace - expected) < 1e-6;
    mptest::detail(ctx, pass, "sdc_trace_mp n=%lld: trace=%.6f expected=%.6f", (long long)n, trace,
                   expected);

    dist_free(A);
    CUDA_CHECK(cudaFree(dA));
    return pass;
}

// =============================================================================
// NormConstantMatrix
//
// What it checks: qdwh_norm_mp correctly reduces over the *entire* local
// tile (not just the diagonal, unlike SdcTraceDiagonal) and AllReduces
// across ranks.
//
// A[gi,gj] = 1/n for all entries → sum of n² squared entries of (1/n)²
// = 1 → ‖A‖_F = 1.
// =============================================================================
MP_TEST(QdwhMp, NormConstantMatrix) {
    using namespace cuev::mp;
    const int64_t n = 512;

    int64_t count = dist_local_count<double>(ctx, n, n);
    double *dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, count * sizeof(double)));

    DistMatrix<double> A = dist_describe<double>(ctx, n, n, dA);

    {
        std::vector<double> hA(count, 1.0 / (double)n);
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    }

    double norm = kernels::qdwh_norm_mp(ctx, A.data, A.local_rows, A.local_cols, A.lld);
    double expected = 1.0;
    bool pass = std::fabs(norm - expected) < 1e-9;
    mptest::detail(ctx, pass, "qdwh_norm_mp n=%lld: norm=%.9f expected=%.9f", (long long)n, norm,
                   expected);

    dist_free(A);
    CUDA_CHECK(cudaFree(dA));
    return pass;
}

// =============================================================================
// SignPositiveScaledIdentity
//
// What it checks: qdwh_sign_mp end-to-end — sign(α·I) = I for α > 0.
// Exercises both the QR branch (large c, early iterations) and the Cholesky
// branch (small c, later iterations) since a scaled identity still goes
// through the full coefficient schedule.
// =============================================================================
MP_TEST(QdwhMp, SignPositiveScaledIdentity) {
    using namespace cuev::mp;
    const int64_t n = 256;

    int64_t count = dist_local_count<double>(ctx, n, n);
    double *dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, count * sizeof(double)));
    DistMatrix<double> B = dist_describe<double>(ctx, n, n, dB);

    {
        std::vector<double> hB(count, 0.0);
        for (int64_t lj = 0; lj < B.local_cols; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, ctx.nb);
            for (int64_t li = 0; li < B.local_rows; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, ctx.nb);
                if (gi == gj) hB[lj * B.lld + li] = 3.0;
            }
        }
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    }

    WorkspaceMp<double> ws = workspace_mp_alloc<double>(ctx, n);
    kernels::qdwh_sign_mp(ctx, B, ws);

    std::vector<double> hS(count);
    CUDA_CHECK(cudaMemcpy(hS.data(), B.data, count * sizeof(double), cudaMemcpyDeviceToHost));

    double local_err = 0.0;
    for (int64_t lj = 0; lj < B.local_cols; ++lj) {
        int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, ctx.nb);
        for (int64_t li = 0; li < B.local_rows; ++li) {
            int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, ctx.nb);
            double expected = (gi == gj) ? 1.0 : 0.0;
            local_err = std::max(local_err, std::fabs(hS[lj * B.lld + li] - expected));
        }
    }
    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);
    bool pass = global_err < 1e-6;
    mptest::detail(ctx, pass, "qdwh_sign_mp(3I) n=%lld: max|sign(B)-I|=%.2e", (long long)n,
                   global_err);

    workspace_mp_free(ws);
    dist_free(B);
    CUDA_CHECK(cudaFree(dB));
    return pass;
}
