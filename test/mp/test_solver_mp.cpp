/**
 * @file   test_solver_mp.cpp
 * @brief  End-to-end correctness tests for the distributed spectral D&C solver.
 *
 * Mirrors test_solver.cu's intent (full symm_eig_solve correctness) but
 * compares against cusolverMpSyevd run directly on the whole matrix as an
 * independent reference, rather than a residual check — both solvers are
 * distributed, but they take genuinely different algorithm paths: cuSOLVERMp
 * tridiagonalizes the *entire* n×n matrix in one shot, while our solver only
 * calls that same primitive at *leaf* nodes (n ≤ SDC_BASE_N_MP) inside the
 * spectral divide-and-conquer recursion.
 *
 * Each case feeds a different (deterministic, symmetric) matrix family chosen to
 * stress a distinct part of the algorithm — adversarial subspace alignment,
 * indefinite spectra, clustered/degenerate eigenvalues, wide dynamic range,
 * non-block-aligned n, and deep recursion. All compare eigenvalues against
 * cusolverMpSyevd to a spectrum-relative tolerance.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "cuev_mp.h"
#include "mp/kernels_mp.cuh"
#include "mp/workspace_mp.h"
#include "mp_test.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// Symmetric (sorted-pair) pseudo-random value in [-1,1], deterministic per
// (i,j) — same generator the device-side rand_fill uses, so test matrices are
// reproducible and identical on every rank.
static double hash_sym(int64_t i, int64_t j, uint64_t seed) {
    uint64_t a = (uint64_t)std::min(i, j), b = (uint64_t)std::max(i, j);
    uint64_t h = (a + 1) * 0x9E3779B97F4A7C15ULL ^ (b + 1) * 0xC2B2AE3D27D4EB4FULL ^ seed;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return 2.0 * (double)(h >> 11) * (1.0 / 9007199254740992.0) - 1.0;
}

// Build H from a global-index fill, run symm_eig_solve_mp and cusolverMpSyevd on
// the same matrix, and compare eigenvalues to a spectrum-relative tolerance.
// FillFn: double(int64_t gi, int64_t gj), must be symmetric.
template <typename FillFn>
static bool run_solver_case(cuev::mp::Context &ctx, int64_t n, double rtol, FillFn f,
                            const char *label) {
    using namespace cuev::mp;
    int64_t count = dist_local_count<double>(ctx, n, n);
    double *dH = nullptr, *dH_ref = nullptr, *devec = nullptr, *devec_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&dH, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dH_ref, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&devec, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&devec_ref, count * sizeof(double)));

    DistMatrix<double> H = dist_describe<double>(ctx, n, n, dH);
    DistMatrix<double> H_ref = dist_describe<double>(ctx, n, n, dH_ref);
    DistMatrix<double> evec = dist_describe<double>(ctx, n, n, devec);
    DistMatrix<double> evec_ref = dist_describe<double>(ctx, n, n, devec_ref);

    {
        std::vector<double> hH(count);
        for (int64_t lj = 0; lj < H.local_cols; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, ctx.nb);
            for (int64_t li = 0; li < H.local_rows; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, ctx.nb);
                hH[lj * H.lld + li] = f(gi, gj);
            }
        }
        CUDA_CHECK(cudaMemcpy(dH, hH.data(), count * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dH_ref, hH.data(), count * sizeof(double), cudaMemcpyHostToDevice));
    }

    double *deval = nullptr, *deval_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&deval, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deval_ref, n * sizeof(double)));

    WorkspaceMp<double> ws = workspace_mp_alloc<double>(ctx, n);
    symm_eig_solve_mp(ctx, H, n, deval, evec, ws);
    cusolvermp::syevd(ctx, n, H_ref.data, 1, 1, H_ref.solverDesc, deval_ref, evec_ref.data, 1, 1,
                      evec_ref.solverDesc, ws);
    workspace_mp_free(ws);

    std::vector<double> h_eval(n), h_eval_ref(n);
    CUDA_CHECK(cudaMemcpy(h_eval.data(), deval, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(h_eval_ref.data(), deval_ref, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Both solvers return ascending eigenvalues, so the comparison is elementwise.
    double max_err = 0.0, scale = 1.0;
    for (int64_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(h_eval[i] - h_eval_ref[i]));
        scale = std::max(scale, std::fabs(h_eval_ref[i]));
    }
    double tol = rtol * scale;
    bool pass = max_err <= tol;
    // Printed for every case (not only on failure) so the run visibly confirms
    // the spread of matrices, not just an aggregate PASS.
    if (ctx.rank == 0)
        fprintf(stderr, "             %-22s n=%-4lld  max|dλ|=%.3e  scale=%.3e  tol=%.3e  [%s]\n",
                label, (long long)n, max_err, scale, tol, pass ? "OK" : "FAIL");

    dist_free(H);
    dist_free(H_ref);
    dist_free(evec);
    dist_free(evec_ref);
    CUDA_CHECK(cudaFree(dH));
    CUDA_CHECK(cudaFree(dH_ref));
    CUDA_CHECK(cudaFree(devec));
    CUDA_CHECK(cudaFree(devec_ref));
    CUDA_CHECK(cudaFree(deval));
    CUDA_CHECK(cudaFree(deval_ref));
    return pass;
}

// =============================================================================
// NearDiagonalDominant — the original adversarial case.
//
// H[i,i]=n, H[i,j]=1/(1+|i-j|). Eigenvectors ≈ coordinate axes and μ=mean lands
// at the degenerate centre, so range(P) is scattered across all indices — the
// worst case for non-pivoted QR subspace extraction (the randomized range
// finder is what makes this pass).
// =============================================================================
MP_TEST(SolverMp, NearDiagonalDominant) {
    return run_solver_case(
        ctx, 1024, 1e-8,
        [](int64_t i, int64_t j) {
            return (i == j) ? 1024.0 : 1.0 / (1.0 + std::fabs((double)(i - j)));
        },
        "NearDiagonalDominant");
}

// =============================================================================
// RandomDenseIndefinite — fully generic symmetric matrix, eigenvalues straddle
// zero. Exercises the QDWH sign function on a genuinely indefinite spectrum and
// generic (non-axis-aligned) eigenvectors.
// =============================================================================
MP_TEST(SolverMp, RandomDenseIndefinite) {
    return run_solver_case(
        ctx, 1024, 1e-7,
        [](int64_t i, int64_t j) {
            return hash_sym(i, j, 0xA11CE5ULL); // diagonal included, all in [-1,1]
        },
        "RandomDenseIndefinite");
}

// =============================================================================
// ClusteredSpectrum — two tight clusters (diagonal ∈ {1, 50}) plus a small
// random perturbation. Many near-degenerate eigenvalues within each cluster;
// μ=mean falls in the gap, so the top-level split must cleanly separate the two
// clusters. Stresses degenerate-eigenvalue handling and the split point.
// =============================================================================
MP_TEST(SolverMp, ClusteredSpectrum) {
    return run_solver_case(
        ctx, 1024, 1e-7,
        [](int64_t i, int64_t j) {
            double off = 0.05 * hash_sym(i, j, 0xC1051E5ULL);
            if (i == j) return ((i % 2 == 0) ? 1.0 : 50.0) + off;
            return off;
        },
        "ClusteredSpectrum");
}

// =============================================================================
// GradedDiagonalNonAligned — graded diagonal spanning a wide, sign-changing
// range (≈ −n/2 … +n/2) with modest off-diagonal coupling, at n=900 (NOT a
// multiple of nb=256). Stresses dynamic range, an indefinite spectrum, and
// block-cyclic splits at non-block-aligned sizes throughout the recursion.
// =============================================================================
MP_TEST(SolverMp, GradedDiagonalNonAligned) {
    return run_solver_case(
        ctx, 900, 1e-8,
        [](int64_t i, int64_t j) {
            if (i == j) return (double)i - 450.0;
            return 0.5 * hash_sym(i, j, 0x69ADED11ULL);
        },
        "GradedDiagonalNonAligned");
}

// =============================================================================
// LargeDeepRecursion — the original failing size (n=2048), near-diagonal again
// so the adversarial subspace structure persists at multiple recursion levels
// and the divide tree is several deep.
// =============================================================================
MP_TEST(SolverMp, LargeDeepRecursion) {
    return run_solver_case(
        ctx, 2048, 1e-8,
        [](int64_t i, int64_t j) {
            return (i == j) ? 2048.0 : 1.0 / (1.0 + std::fabs((double)(i - j)));
        },
        "LargeDeepRecursion");
}

// =============================================================================
// HugeMatrix — opt-in stress test at large n (default 8192). A full n×n
// reference syevd at this scale runs for minutes, so it is skipped unless
// CUEV_BIG=1 is set; size is overridable via CUEV_BIG_N (e.g. 16000).
//
//   CUEV_BIG=1 CUEV_BIG_N=16000 srun ... build/cuTestMp
// =============================================================================
MP_TEST(SolverMp, HugeMatrix) {
    const char *en = std::getenv("CUEV_BIG");
    if (!en || en[0] == '0' || en[0] == '\0') {
        if (ctx.rank == 0)
            fprintf(stderr,
                    "             HugeMatrix skipped (set CUEV_BIG=1 [CUEV_BIG_N=16000])\n");
        return true;
    }
    const char *ns = std::getenv("CUEV_BIG_N");
    int64_t n = ns ? std::atoll(ns) : 8192;
    if (n <= 0) n = 8192;
    return run_solver_case(
        ctx, n, 1e-8,
        [n](int64_t i, int64_t j) {
            return (i == j) ? (double)n : 1.0 / (1.0 + std::fabs((double)(i - j)));
        },
        "HugeMatrix");
}
