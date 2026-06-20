/**
 * @file   test_primitives_mp.cpp
 * @brief  Correctness tests for the distributed (Phase 3) communication and
 *         cuBLASMp/cuSOLVERMp building blocks.
 *
 * Tests:
 *   Primitives.Nccl   — NCCL AllReduce sum-of-ranks sanity check
 *   Primitives.Gemm   — cublasMpGemm: A=1/n, B=1 → C should be all-1
 *   Primitives.QrTall — cusolverMpGeqrf + Ormqr, m=2n (QDWH inner-loop shape)
 *   Primitives.QrSquare — same, m=n
 *   Primitives.Chol   — cusolverMpPotrf + cublasMpTrsm; verify L⁻¹ on diagonal SPD
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

using cuev::mp::Context;
using cuev::mp::dist_describe;
using cuev::mp::dist_free;
using cuev::mp::dist_local_count;
using cuev::mp::DistMatrix;
using mptest::check_info;
using mptest::l2g_col;
using mptest::l2g_row;
using mptest::mp_sync;

// =============================================================================
// Nccl — NCCL AllReduce sanity: sum(rank ids) == world*(world-1)/2
// =============================================================================
MP_TEST(Primitives, Nccl) {
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
    mptest::detail(ctx, pass, "NCCL AllReduce: sum=%.0f expected=%.0f", sum, expect);
    return pass;
}

// =============================================================================
// Gemr2D — cublasmp::gemr2d copies a NON-block-aligned column block.
//
// A (n×n) with a unique value per global cell: A[gi,gj] = gi*100000 + gj.
// Extract the column block [j0, j0+w) into a fresh n×w matrix Bm at offset
// (1,1). j0 is deliberately not a multiple of nb. Verify Bm[gi,gj] equals
// A[gi, j0+gj] — i.e. the redistribution honours the unaligned source offset.
// This mirrors exactly how the solver slices Q2 = Q[:,k:n] / C[n+1:2n,:].
// =============================================================================
MP_TEST(Primitives, Gemr2D) {
    using cuev::mp::cublasmp::gemr2d;
    const int64_t n = 1024;
    const int64_t j0 = 271; // 1-indexed start column; 270 = k from the solver run, not a block edge
    const int64_t w = n - (j0 - 1);

    int64_t cntA = dist_local_count<double>(ctx, n, n);
    int64_t cntB = dist_local_count<double>(ctx, n, w);
    double *dA = nullptr, *dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, cntA * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, std::max(cntB, (int64_t)1) * sizeof(double)));

    DistMatrix<double> A = dist_describe<double>(ctx, n, n, dA);
    DistMatrix<double> Bm = dist_describe<double>(ctx, n, w, dB);

    {
        std::vector<double> hA(cntA);
        for (int64_t lj = 0; lj < A.local_cols; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, ctx.nb);
            for (int64_t li = 0; li < A.local_rows; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, ctx.nb);
                hA[lj * A.lld + li] = (double)gi * 100000.0 + (double)gj;
            }
        }
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), cntA * sizeof(double), cudaMemcpyHostToDevice));
    }

    gemr2d(ctx, n, w, A.data, 1, j0, A.desc, Bm.data, 1, 1, Bm.desc);

    std::vector<double> hB(std::max(cntB, (int64_t)1));
    CUDA_CHECK(cudaMemcpy(hB.data(), dB, std::max(cntB, (int64_t)1) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double local_err = 0.0;
    for (int64_t lj = 0; lj < Bm.local_cols; ++lj) {
        int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, ctx.nb);
        for (int64_t li = 0; li < Bm.local_rows; ++li) {
            int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, ctx.nb);
            double expected = (double)gi * 100000.0 + (double)(j0 - 1 + gj);
            local_err = std::max(local_err, std::fabs(hB[lj * Bm.lld + li] - expected));
        }
    }
    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);
    bool pass = global_err == 0.0;
    mptest::detail(ctx, pass, "GEMR2D n=%lld j0=%lld w=%lld: max|B-A[:,j0:]| = %.2e", (long long)n,
                   (long long)j0, (long long)w, global_err);

    dist_free(A);
    dist_free(Bm);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    return pass;
}

// =============================================================================
// ProjectorQR — does non-pivoted geqrf+ormqr recover range(P) for a rank-k
// orthogonal projector? P = diag(1..1,0..0) (k leading ones). QR(P)→Q, then
// Q1 = first k cols; Q1·Q1ᵀ must equal P. Leading columns of P are e_1..e_k,
// perfectly conditioned, so this isolates the cuSOLVERMp QR path from any
// leading-column ill-conditioning of a real projector.
// =============================================================================
MP_TEST(Primitives, ProjectorQR) {
    using cuev::mp::workspace_mp_alloc;
    using cuev::mp::workspace_mp_free;
    using cuev::mp::WorkspaceMp;
    using cuev::mp::cublasmp::gemm;
    using cuev::mp::cusolvermp::geqrf;
    using cuev::mp::cusolvermp::ormqr;
    using cuev::mp::kernels::qdwh_fill_C_mp;
    const int64_t n = 1024, k = 270;

    int64_t cnt = dist_local_count<double>(ctx, n, n);
    double *dP = nullptr, *dQ = nullptr, *dPc = nullptr, *dQQ = nullptr;
    CUDA_CHECK(cudaMalloc(&dP, cnt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dQ, cnt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dPc, cnt * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dQQ, cnt * sizeof(double)));
    DistMatrix<double> P = dist_describe<double>(ctx, n, n, dP);
    DistMatrix<double> Q = dist_describe<double>(ctx, n, n, dQ);
    DistMatrix<double> Pc = dist_describe<double>(ctx, n, n, dPc);
    DistMatrix<double> QQ = dist_describe<double>(ctx, n, n, dQQ);

    {
        std::vector<double> hP(cnt, 0.0);
        for (int64_t lj = 0; lj < P.local_cols; ++lj) {
            int64_t gj = l2g_col(lj, ctx.pcol, ctx.npcol, ctx.nb);
            for (int64_t li = 0; li < P.local_rows; ++li) {
                int64_t gi = l2g_row(li, ctx.prow, ctx.nprow, ctx.nb);
                hP[lj * P.lld + li] = (gi == gj && gi < k) ? 1.0 : 0.0;
            }
        }
        CUDA_CHECK(cudaMemcpy(dP, hP.data(), cnt * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPc, hP.data(), cnt * sizeof(double), cudaMemcpyHostToDevice));
    }

    WorkspaceMp<double> ws = workspace_mp_alloc<double>(ctx, n);
    geqrf(ctx, n, n, P.data, 1, 1, P.solverDesc, ws.qdwh_tau, ws);
    qdwh_fill_C_mp(Q.data, n, n, Q.lld, Q.local_cols, ctx.prow, ctx.pcol, ctx.nprow, ctx.npcol,
                   ctx.nb, ctx.stream);
    ormqr(ctx, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, n, n, n, P.data, 1, 1, P.solverDesc, ws.qdwh_tau,
          Q.data, 1, 1, Q.solverDesc, ws);

    const double one = 1.0, zero = 0.0, mone = -1.0;
    gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &one, Q.data, 1, 1, Q.desc, Q.data, 1, 1, Q.desc,
         &zero, QQ.data, 1, 1, QQ.desc); // Q1·Q1ᵀ (first k cols of Q)
    cuev::mp::cublasmp::geadd(ctx, CUBLAS_OP_N, n, n, &mone, Pc.data, 1, 1, Pc.desc, &one, QQ.data,
                              1, 1, QQ.desc); // QQ ← Q1Q1ᵀ − P
    double err =
        cuev::mp::kernels::qdwh_norm_mp(ctx, QQ.data, QQ.local_rows, QQ.local_cols, QQ.lld);
    workspace_mp_free(ws);

    bool pass = err < 1e-9;
    mptest::detail(ctx, pass, "ProjectorQR n=%lld k=%lld: ‖Q1Q1^T - P‖=%.3e", (long long)n,
                   (long long)k, err);

    dist_free(P);
    dist_free(Q);
    dist_free(Pc);
    dist_free(QQ);
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dPc));
    CUDA_CHECK(cudaFree(dQQ));
    return pass;
}

// =============================================================================
// Gemm — cublasMpGemm: A=1/n, B=1 → C should be all-1
// =============================================================================
MP_TEST(Primitives, Gemm) {
    const int64_t n = 1024;
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
    mp_sync(ctx);

    std::vector<double> hC(count);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, count * sizeof(double), cudaMemcpyDeviceToHost));
    double local_err = 0.0;
    for (int64_t i = 0; i < count; ++i)
        local_err = std::max(local_err, std::fabs(hC[i] - 1.0));
    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);
    bool pass = global_err < 1e-10;
    mptest::detail(ctx, pass, "GEMM n=%lld: max|C-1| = %.2e", (long long)n, global_err);

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
// Qr
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
static bool run_qr(Context &ctx, int64_t m, int64_t n) {
    const int64_t nb = ctx.nb;
    const int64_t k = std::min(m, n); // #Householder reflectors

    int64_t lr_A = cublasMpNumroc(m, nb, ctx.prow, 0, ctx.nprow);
    int64_t lc_A = cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol);
    int64_t lld_A = std::max(lr_A, (int64_t)1);

    int64_t l_tau = std::max(cublasMpNumroc(k, nb, ctx.pcol, 0, ctx.npcol), (int64_t)1);

    int64_t lr_Q = cublasMpNumroc(n, nb, ctx.prow, 0, ctx.nprow);
    int64_t lc_Q = cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol);
    int64_t lld_Q = std::max(lr_Q, (int64_t)1);

    double *dA = nullptr, *dC = nullptr, *d_tau = nullptr, *dQtQ = nullptr;
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, lr_A * lc_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, lr_A * lc_A * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tau, l_tau * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dQtQ, lr_Q * lc_Q * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

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

    cusolverMpMatrixDescriptor_t descrA = nullptr, descrC = nullptr;
    CUSOLVER_CHECK(
        cusolverMpCreateMatrixDesc(&descrA, ctx.solvergrid, CUDA_R_64F, m, n, nb, nb, 0, 0, lld_A));
    CUSOLVER_CHECK(
        cusolverMpCreateMatrixDesc(&descrC, ctx.solvergrid, CUDA_R_64F, m, n, nb, nb, 0, 0, lld_A));

    size_t geqrf_wsD = 0, geqrf_wsH = 0;
    CUSOLVER_CHECK(cusolverMpGeqrf_bufferSize(ctx.cusolvermp, m, n, dA, 1, 1, descrA, CUDA_R_64F,
                                              &geqrf_wsD, &geqrf_wsH));
    void *geqrf_dwork = nullptr, *geqrf_hwork = nullptr;
    CUDA_CHECK(cudaMalloc(&geqrf_dwork, std::max(geqrf_wsD, (size_t)1)));
    geqrf_hwork = std::malloc(std::max(geqrf_wsH, (size_t)1));

    CUSOLVER_CHECK(cusolverMpGeqrf(ctx.cusolvermp, m, n, dA, 1, 1, descrA, d_tau, CUDA_R_64F,
                                   geqrf_dwork, geqrf_wsD, geqrf_hwork, geqrf_wsH, d_info));
    mp_sync(ctx);
    check_info(ctx, d_info, "Geqrf");

    CUDA_CHECK(cudaFree(geqrf_dwork));
    std::free(geqrf_hwork);

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

    cublasMpMatrixDescriptor_t mpDescrC = nullptr, mpDescrQtQ = nullptr;
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(m, n, nb, nb, 0, 0, lld_A, CUDA_R_64F, ctx.grid, &mpDescrC));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(n, n, nb, nb, 0, 0, lld_Q, CUDA_R_64F, ctx.grid,
                                                  &mpDescrQtQ));

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
    mptest::detail(ctx, pass, "QR  m=%lld n=%lld: max|Q^TQ - I| = %.2e", (long long)m, (long long)n,
                   global_err);

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

MP_TEST(Primitives, QrTall) {
    return run_qr(ctx, 1024, 512);
} // QDWH inner-loop shape: m=2n
MP_TEST(Primitives, QrSquare) {
    return run_qr(ctx, 512, 512);
}

// =============================================================================
// Chol
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
MP_TEST(Primitives, Chol) {
    const int64_t n = 512;
    const int64_t nb = ctx.nb;

    int64_t lr = cublasMpNumroc(n, nb, ctx.prow, 0, ctx.nprow);
    int64_t lc = cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol);
    int64_t lld = std::max(lr, (int64_t)1);

    double *dA = nullptr, *dB = nullptr;
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, lr * lc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, lr * lc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

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

    cusolverMpMatrixDescriptor_t solverDescrA = nullptr;
    CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(&solverDescrA, ctx.solvergrid, CUDA_R_64F, n, n, nb,
                                              nb, 0, 0, lld));
    cublasMpMatrixDescriptor_t mpDescrA = nullptr, mpDescrB = nullptr;
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(n, n, nb, nb, 0, 0, lld, CUDA_R_64F, ctx.grid, &mpDescrA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(n, n, nb, nb, 0, 0, lld, CUDA_R_64F, ctx.grid, &mpDescrB));

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
    mptest::detail(ctx, pass, "Chol n=%lld: max|X - L^-1| = %.2e", (long long)n, global_err);

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(mpDescrA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(mpDescrB));
    CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(solverDescrA));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
    return pass;
}
