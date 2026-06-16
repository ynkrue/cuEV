/**
 * @file   test_mp.cpp
 * @brief  Correctness tests for the distributed (Phase 3) solver primitives.
 *
 * Not GTest — MPI and GTest don't mix cleanly. Each test function computes a
 * local error, MPI_Allreduces the max across ranks, and rank 0 prints PASS/FAIL.
 *
 * Run:  srun -N<nodes> --tasks-per-node=<p> --gpus-per-node=<p> build/cuTestMp [nprow npcol] [nb]
 *
 * Tests:
 *   test_nccl — NCCL AllReduce sum-of-ranks sanity check
 *   test_gemm — cublasMpGemm: A=1/n, B=1 → C should be all-1
 *   test_qr   — cusolverMpGeqrf + Ormqr; verify ||Q^T Q − I||_∞ < ε
 *   test_chol — cusolverMpPotrf + cublasMpTrsm; verify L⁻¹ on diagonal SPD
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "mp/comm.h"
#include "mp/workspace_mp.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include <vector>

using cuev::mp::Context;
using cuev::mp::dist_describe;
using cuev::mp::dist_free;
using cuev::mp::dist_local_count;
using cuev::mp::DistMatrix;

// =============================================================================
// test_nccl — NCCL AllReduce sanity: sum(rank ids) == world*(world-1)/2
// =============================================================================
static bool test_nccl(Context &ctx) {
    double host = (double)ctx.rank, *d = nullptr, *dsum = nullptr, sum = 0.0;
    CUDA_CHECK(cudaMalloc(&d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dsum, sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(d, &host, sizeof(double), cudaMemcpyHostToDevice, ctx.stream));
    NCCL_CHECK(ncclAllReduce(d, dsum, 1, ncclDouble, ncclSum, ctx.nccl, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(&sum, dsum, sizeof(double), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(dsum));

    double expect = (double)ctx.world_size * (ctx.world_size - 1) / 2.0;
    bool pass = (sum == expect);
    if (ctx.rank == 0)
        printf("NCCL AllReduce: sum=%.0f expected=%.0f  [%s]\n", sum, expect, pass ? "OK" : "FAIL");
    return pass;
}

// =============================================================================
// test_gemm — cublasMpGemm: A=1/n, B=1 → C should be all-1
// =============================================================================
static bool test_gemm(Context &ctx, int64_t n) {
    int64_t count = dist_local_count<double>(ctx, n, n);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, count * sizeof(double)));

    {
        std::vector<double> hA(count, 1.0 / (double)n);
        std::vector<double> hB(count, 1.0);
        std::vector<double> hC(count, 0.0);
        CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), count * sizeof(double), cudaMemcpyHostToDevice,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(), count * sizeof(double), cudaMemcpyHostToDevice,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(dC, hC.data(), count * sizeof(double), cudaMemcpyHostToDevice,
                                   ctx.stream));
    }

    DistMatrix<double> A = dist_describe<double>(ctx, n, n, dA);
    DistMatrix<double> B = dist_describe<double>(ctx, n, n, dB);
    DistMatrix<double> C = dist_describe<double>(ctx, n, n, dC);

    const double alpha = 1.0, beta = 0.0;
    size_t wsD = 0, wsH = 0;
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(ctx.cublasmp, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                           dA, 1, 1, A.desc, dB, 1, 1, B.desc, &beta, dC, 1, 1,
                                           C.desc, CUBLAS_COMPUTE_64F, &wsD, &wsH));
    void *d_work = nullptr, *h_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, std::max(wsD, (size_t)1)));
    h_work = std::malloc(std::max(wsH, (size_t)1));

    CUBLASMP_CHECK(cublasMpGemm(ctx.cublasmp, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, 1, 1,
                                A.desc, dB, 1, 1, B.desc, &beta, dC, 1, 1, C.desc,
                                CUBLAS_COMPUTE_64F, d_work, wsD, h_work, wsH));
    CAL_CHECK(cal_stream_sync(ctx.cal, ctx.stream));
    CAL_CHECK(cal_comm_barrier(ctx.cal, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    std::vector<double> hC(count);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, count * sizeof(double), cudaMemcpyDeviceToHost));
    double local_err = 0.0;
    for (int64_t i = 0; i < count; ++i)
        local_err = std::max(local_err, std::fabs(hC[i] - 1.0));
    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);
    bool pass = global_err < 1e-10;
    if (ctx.rank == 0)
        printf("GEMM n=%lld: max|C-1| = %.2e  [%s]\n", (long long)n, global_err,
               pass ? "OK" : "FAIL");

    dist_free(A);
    dist_free(B);
    dist_free(C);
    CUDA_CHECK(cudaFree(d_work));
    std::free(h_work);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return pass;
}

// =============================================================================
// Index helpers: local ↔ global for 2D block-cyclic with rsrc=csrc=0
// =============================================================================

static inline int64_t l2g_row(int64_t li, int prow, int nprow, int64_t nb) {
    return (li / nb * nprow + prow) * nb + (li % nb);
}
static inline int64_t l2g_col(int64_t lj, int pcol, int npcol, int64_t nb) {
    return (lj / nb * npcol + pcol) * nb + (lj % nb);
}

// =============================================================================
// Helpers shared by all tests
// =============================================================================

// Drain the cuBLASMp/cuSOLVERMp pipeline after any distributed op.
static void mp_sync(Context &ctx) {
    CAL_CHECK(cal_stream_sync(ctx.cal, ctx.stream));
    CAL_CHECK(cal_comm_barrier(ctx.cal, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

// Check the device info scalar cuSOLVERMp writes on failure.
static void check_info(Context &ctx, int *d_info, const char *op) {
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "[rank %d] %s: info=%d\n", ctx.rank, op, h_info);
        MPI_Abort(ctx.comm, 1);
    }
}

// =============================================================================
// test_qr
//
// What it checks: cusolverMpGeqrf + cusolverMpOrmqr produce an orthogonal Q.
//
// Algorithm:
//   1. Fill A (m×n) with a Hilbert-like pattern (full rank).
//   2. Geqrf(A) → Householder data stored in-place; tau holds the scalars.
//   3. Fill C (m×n) with the truncated identity [I_n; 0].
//   4. Ormqr(L, N): C ← Q · C  →  C now holds the economy Q (m×n).
//   5. Q^T Q via cublasMpGemm(OP_T, OP_N, C, C) → n×n result.
//   6. max|Q^T Q − I|_∞ < 1e-10.
//
// m=2n mirrors the QDWH inner-loop shape [√c·B ; I] that this will eventually
// factor.
// =============================================================================
static bool test_qr(Context &ctx, int64_t m, int64_t n) {
    const int64_t nb = ctx.nb;
    const int64_t k = std::min(m, n); // #Householder reflectors

    // ---- local tile dimensions for m×n matrix ----
    int64_t lr_A = cublasMpNumroc(m, nb, ctx.prow, 0, ctx.nprow);
    int64_t lc_A = cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol);
    int64_t lld_A = std::max(lr_A, (int64_t)1);

    // tau follows the column distribution (standard ScaLAPACK convention)
    int64_t l_tau = std::max(cublasMpNumroc(k, nb, ctx.pcol, 0, ctx.npcol), (int64_t)1);

    // ---- local tile dimensions for n×n result Q^T Q ----
    int64_t lr_Q = cublasMpNumroc(n, nb, ctx.prow, 0, ctx.nprow);
    int64_t lc_Q = cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol);
    int64_t lld_Q = std::max(lr_Q, (int64_t)1);

    // ---- device allocations ----
    double *dA = nullptr, *dC = nullptr, *d_tau = nullptr, *dQtQ = nullptr;
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, lr_A * lc_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, lr_A * lc_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau, l_tau * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dQtQ, lr_Q * lc_Q * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // ---- fill A: 1/(gi+gj+1) (Hilbert-like, full rank) ----
    {
        std::vector<double> hA(lr_A * lc_A);
        for (int64_t lj = 0; lj < lc_A; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, nb);
            for (int64_t li = 0; li < lr_A; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, nb);
                hA[lj * lld_A + li] = 1.0 / (double)(gi + gj + 1);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), lr_A * lc_A * sizeof(double),
                                   cudaMemcpyHostToDevice, ctx.stream));
    }

    // ---- fill C: truncated identity [I_n ; 0_{(m-n)×n}] ----
    {
        std::vector<double> hC(lr_A * lc_A, 0.0);
        for (int64_t lj = 0; lj < lc_A; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, nb);
            for (int64_t li = 0; li < lr_A; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, nb);
                hC[lj * lld_A + li] = (gi == gj) ? 1.0 : 0.0;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(dC, hC.data(), lr_A * lc_A * sizeof(double),
                                   cudaMemcpyHostToDevice, ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // ---- cuSOLVERMp descriptors for A and C (both m×n) ----
    cusolverMpMatrixDescriptor_t descrA = nullptr, descrC = nullptr;
    CUSOLVER_CHECK(
        cusolverMpCreateMatrixDesc(&descrA, ctx.solvergrid, CUDA_R_64F, m, n, nb, nb, 0, 0, lld_A));
    CUSOLVER_CHECK(
        cusolverMpCreateMatrixDesc(&descrC, ctx.solvergrid, CUDA_R_64F, m, n, nb, nb, 0, 0, lld_A));

    // ---- Geqrf ----
    size_t geqrf_wsD = 0, geqrf_wsH = 0;
    CUSOLVER_CHECK(cusolverMpGeqrf_bufferSize(ctx.cusolvermp, m, n, dA, 1, 1, descrA, CUDA_R_64F,
                                              &geqrf_wsD, &geqrf_wsH));
    // Always allocate at least 1 byte — cuSOLVERMp rejects nullptr workspace
    // even when bufferSize returns 0 (observed: wsH always 0 here).
    void *geqrf_dwork = nullptr, *geqrf_hwork = nullptr;
    CUDA_CHECK(cudaMalloc(&geqrf_dwork, std::max(geqrf_wsD, (size_t)1)));
    geqrf_hwork = std::malloc(std::max(geqrf_wsH, (size_t)1));

    CUSOLVER_CHECK(cusolverMpGeqrf(ctx.cusolvermp, m, n, dA, 1, 1, descrA, d_tau, CUDA_R_64F,
                                   geqrf_dwork, geqrf_wsD, geqrf_hwork, geqrf_wsH, d_info));
    mp_sync(ctx);
    check_info(ctx, d_info, "Geqrf");

    CUDA_CHECK(cudaFree(geqrf_dwork));
    geqrf_dwork = nullptr;
    std::free(geqrf_hwork);
    geqrf_hwork = nullptr;

    // ---- Ormqr: C ← Q · C  (C = [I_n;0] → C = economy Q) ----
    size_t ormqr_wsD = 0, ormqr_wsH = 0;
    CUSOLVER_CHECK(cusolverMpOrmqr_bufferSize(ctx.cusolvermp, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n,
                                              k, dA, 1, 1, descrA, d_tau, dC, 1, 1, descrC,
                                              CUDA_R_64F, &ormqr_wsD, &ormqr_wsH));
    void *ormqr_dwork = nullptr, *ormqr_hwork = nullptr;
    CUDA_CHECK(cudaMalloc(&ormqr_dwork, std::max(ormqr_wsD, (size_t)1)));
    ormqr_hwork = std::malloc(std::max(ormqr_wsH, (size_t)1));

    CUSOLVER_CHECK(cusolverMpOrmqr(ctx.cusolvermp, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n, k, dA, 1, 1,
                                   descrA, d_tau, dC, 1, 1, descrC, CUDA_R_64F, ormqr_dwork,
                                   ormqr_wsD, ormqr_hwork, ormqr_wsH, d_info));
    mp_sync(ctx);
    check_info(ctx, d_info, "Ormqr");

    CUDA_CHECK(cudaFree(ormqr_dwork));
    std::free(ormqr_hwork);

    // ---- cuBLASMp descriptors for GEMM: Q^T Q ----
    cublasMpMatrixDescriptor_t mpDescrC = nullptr, mpDescrQtQ = nullptr;
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(m, n, nb, nb, 0, 0, lld_A, CUDA_R_64F, ctx.grid, &mpDescrC));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(n, n, nb, nb, 0, 0, lld_Q, CUDA_R_64F, ctx.grid,
                                                  &mpDescrQtQ));

    // Q^T Q = C^T · C,  (n×m) · (m×n) → n×n
    const double alpha = 1.0, beta = 0.0;
    size_t gemm_wsD = 0, gemm_wsH = 0;
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(
        ctx.cublasmp, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, dC, 1, 1, mpDescrC, dC, 1, 1,
        mpDescrC, &beta, dQtQ, 1, 1, mpDescrQtQ, CUBLAS_COMPUTE_64F, &gemm_wsD, &gemm_wsH));
    void *gemm_dwork = nullptr, *gemm_hwork = nullptr;
    if (gemm_wsD) CUDA_CHECK(cudaMalloc(&gemm_dwork, gemm_wsD));
    if (gemm_wsH) gemm_hwork = std::malloc(gemm_wsH);

    CUBLASMP_CHECK(cublasMpGemm(ctx.cublasmp, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, dC, 1, 1,
                                mpDescrC, dC, 1, 1, mpDescrC, &beta, dQtQ, 1, 1, mpDescrQtQ,
                                CUBLAS_COMPUTE_64F, gemm_dwork, gemm_wsD, gemm_hwork, gemm_wsH));
    mp_sync(ctx);

    CUDA_CHECK(cudaFree(gemm_dwork));
    std::free(gemm_hwork);

    // ---- verify: max|Q^T Q − I|_∞ ----
    std::vector<double> hQtQ(lr_Q * lc_Q);
    CUDA_CHECK(cudaMemcpy(hQtQ.data(), dQtQ, lr_Q * lc_Q * sizeof(double), cudaMemcpyDeviceToHost));

    double local_err = 0.0;
    for (int64_t lj = 0; lj < lc_Q; ++lj) {
        int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, nb);
        for (int64_t li = 0; li < lr_Q; ++li) {
            int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, nb);
            double expected = (gi == gj) ? 1.0 : 0.0;
            local_err = std::max(local_err, std::fabs(hQtQ[lj * lld_Q + li] - expected));
        }
    }
    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);
    bool pass = global_err < 1e-10;
    if (ctx.rank == 0)
        printf("QR  m=%lld n=%lld: max|Q^TQ - I| = %.2e  [%s]\n", (long long)m, (long long)n,
               global_err, pass ? "OK" : "FAIL");

    // ---- cleanup ----
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(mpDescrC));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(mpDescrQtQ));
    CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descrA));
    CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descrC));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(dQtQ));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dA));
    return pass;
}

// =============================================================================
// test_chol
//
// What it checks: cusolverMpPotrf + cublasMpTrsm (the QDWH Cholesky branch).
//
// Algorithm:
//   1. A = (n+1)·I  — trivially SPD; exact Cholesky factor is sqrt(n+1)·I.
//   2. Potrf(lower)  → A overwritten with L in lower triangle.
//   3. B = I         — RHS for the triangular solve.
//   4. Trsm(left, lower, N, non-unit): B ← L⁻¹ · B = L⁻¹.
//   5. For a diagonal L: X[i,i] = 1/sqrt(n+1), X[i,j]=0 for i≠j.
//   6. max|X − expected|_∞ < 1e-10.
//
// This exercises exactly the two ops used in QDWH's Cholesky step:
//   Z = chol(I + c·BᵀB),  then  B ← (a/c)B + (1−a/c)·(B·Z⁻ᵀ)·Z⁻¹.
// =============================================================================
static bool test_chol(Context &ctx, int64_t n) {
    const int64_t nb = ctx.nb;

    int64_t lr = cublasMpNumroc(n, nb, ctx.prow, 0, ctx.nprow);
    int64_t lc = cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol);
    int64_t lld = std::max(lr, (int64_t)1);

    double *dA = nullptr, *dB = nullptr;
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, lr * lc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, lr * lc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // A = (n+1)·I,  B = I
    const double diag_val = (double)(n + 1);
    {
        std::vector<double> hA(lr * lc, 0.0), hB(lr * lc, 0.0);
        for (int64_t lj = 0; lj < lc; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, nb);
            for (int64_t li = 0; li < lr; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, nb);
                if (gi == gj) {
                    hA[lj * lld + li] = diag_val;
                    hB[lj * lld + li] = 1.0;
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), lr * lc * sizeof(double), cudaMemcpyHostToDevice,
                                   ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(), lr * lc * sizeof(double), cudaMemcpyHostToDevice,
                                   ctx.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // ---- descriptors ----
    cusolverMpMatrixDescriptor_t solverDescrA = nullptr;
    CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(&solverDescrA, ctx.solvergrid, CUDA_R_64F, n, n, nb,
                                              nb, 0, 0, lld));
    cublasMpMatrixDescriptor_t mpDescrA = nullptr, mpDescrB = nullptr;
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(n, n, nb, nb, 0, 0, lld, CUDA_R_64F, ctx.grid, &mpDescrA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(n, n, nb, nb, 0, 0, lld, CUDA_R_64F, ctx.grid, &mpDescrB));

    // ---- Potrf(lower) ----
    size_t potrf_wsD = 0, potrf_wsH = 0;
    CUSOLVER_CHECK(cusolverMpPotrf_bufferSize(ctx.cusolvermp, CUBLAS_FILL_MODE_LOWER, n, dA, 1, 1,
                                              solverDescrA, CUDA_R_64F, &potrf_wsD, &potrf_wsH));
    void *potrf_dwork = nullptr, *potrf_hwork = nullptr;
    CUDA_CHECK(cudaMalloc(&potrf_dwork, std::max(potrf_wsD, (size_t)1)));
    potrf_hwork = std::malloc(std::max(potrf_wsH, (size_t)1));

    CUSOLVER_CHECK(cusolverMpPotrf(ctx.cusolvermp, CUBLAS_FILL_MODE_LOWER, n, dA, 1, 1,
                                   solverDescrA, CUDA_R_64F, potrf_dwork, potrf_wsD, potrf_hwork,
                                   potrf_wsH, d_info));
    mp_sync(ctx);
    check_info(ctx, d_info, "Potrf");

    CUDA_CHECK(cudaFree(potrf_dwork));
    std::free(potrf_hwork);

    // ---- Trsm: solve L·X = I  →  X = L⁻¹ ----
    const double alpha = 1.0;
    size_t trsm_wsD = 0, trsm_wsH = 0;
    CUBLASMP_CHECK(cublasMpTrsm_bufferSize(ctx.cublasmp, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                           CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &alpha, dA, 1,
                                           1, mpDescrA, dB, 1, 1, mpDescrB, CUBLAS_COMPUTE_64F,
                                           &trsm_wsD, &trsm_wsH));
    void *trsm_dwork = nullptr, *trsm_hwork = nullptr;
    CUDA_CHECK(cudaMalloc(&trsm_dwork, std::max(trsm_wsD, (size_t)1)));
    trsm_hwork = std::malloc(std::max(trsm_wsH, (size_t)1));

    CUBLASMP_CHECK(cublasMpTrsm(ctx.cublasmp, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                CUBLAS_DIAG_NON_UNIT, n, n, &alpha, dA, 1, 1, mpDescrA, dB, 1, 1,
                                mpDescrB, CUBLAS_COMPUTE_64F, trsm_dwork, trsm_wsD, trsm_hwork,
                                trsm_wsH));
    mp_sync(ctx);

    CUDA_CHECK(cudaFree(trsm_dwork));
    std::free(trsm_hwork);

    // ---- verify ----
    // L = sqrt(n+1)·I  →  X = L⁻¹ = (1/sqrt(n+1))·I
    std::vector<double> hB(lr * lc);
    CUDA_CHECK(cudaMemcpy(hB.data(), dB, lr * lc * sizeof(double), cudaMemcpyDeviceToHost));

    const double expected_diag = 1.0 / std::sqrt(diag_val);
    double local_err = 0.0;
    for (int64_t lj = 0; lj < lc; ++lj) {
        int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, nb);
        for (int64_t li = 0; li < lr; ++li) {
            int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, nb);
            double expected = (gi == gj) ? expected_diag : 0.0;
            local_err = std::max(local_err, std::fabs(hB[lj * lld + li] - expected));
        }
    }
    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);
    bool pass = global_err < 1e-10;
    if (ctx.rank == 0)
        printf("Chol n=%lld: max|X - L^-1| = %.2e  [%s]\n", (long long)n, global_err,
               pass ? "OK" : "FAIL");

    // ---- cleanup ----
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(mpDescrA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(mpDescrB));
    CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(solverDescrA));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
    return pass;
}

// =============================================================================
// main
// =============================================================================
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world;
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int nprow = 0, npcol = 0, nb = 256;
    if (argc >= 3) {
        nprow = atoi(argv[1]);
        npcol = atoi(argv[2]);
    } else {
        cuev::mp::grid_factor(world, nprow, npcol);
    }
    if (argc >= 4) nb = atoi(argv[3]);

    Context ctx;
    cuev::mp::ctx_init(ctx, nb, nprow, npcol);

    int fails = 0;
    if (!test_nccl(ctx)) ++fails;          // NCCL comm
    if (!test_gemm(ctx, 1024)) ++fails;    // cublasMpGemm
    if (!test_qr(ctx, 1024, 512)) ++fails; // tall QR (QDWH shape: m=2n)
    if (!test_qr(ctx, 512, 512)) ++fails;  // square QR
    if (!test_chol(ctx, 512)) ++fails;     // Potrf + Trsm (QDWH Cholesky branch)

    if (ctx.rank == 0) printf("\n%s\n", fails == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    cuev::mp::ctx_finalize(ctx);
    MPI_Finalize();
    return fails ? 1 : 0;
}
