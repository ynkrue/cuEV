/**
 * @file   test_dbbr.cu
 * @brief  Correctness tests for DBBR kernels (dbbr_panel_qr: geqrf + extract + larft).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include "test.h"

using namespace cutest;

// Validates one panel QR: a random rows×b panel A is factorised, then we rebuild
// the block reflector Q = I − V·T·Vᵀ from the outputs (V, T) and check
//   (1) reconstruction:  (I − V·T·Vᵀ)·R  ==  A_original
//   (2) orthogonality:   ‖QᵀQ − I‖ ≈ 0
template <typename T> static void panel_qr_case(int rows, int b, double tol) {
    const int n = rows; // standalone panel: lda = ws->n = rows
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ws = cuev::handle_alloc<T>(n, b, b, stream);

    std::vector<T> A0(rows * b);
    fill_random(A0, 12345);

    T *dA = to_device(A0);
    T *dV = nullptr;
    CUDA_CHECK(cudaMalloc(&dV, (size_t)rows * b * sizeof(T)));

    cuev::kernels::dbbr_panel_qr(&ws, dA, dV, rows, b);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> Apk(rows * b), V(rows * b), Tm(b * b);
    to_host(Apk, dA);
    to_host(V, dV);
    to_host(Tm, ws.Tmat);

    // R_full (rows×b): upper triangle of the packed factor, zeros below
    std::vector<T> R(rows * b, T(0));
    for (int c = 0; c < b; ++c)
        for (int r = 0; r <= c; ++r)
            R[r + c * rows] = Apk[r + c * rows];

    // (1) reconstruction: recon = R − V·(T·(Vᵀ·R))  ==  (I − V·T·Vᵀ)·R
    std::vector<T> VtR(b * b), TVtR(b * b), corr(rows * b), recon(rows * b);
    gemm_host(VtR, V, R, b, b, rows, true, false, rows, rows, b);
    gemm_host(TVtR, Tm, VtR, b, b, b, false, false, b, b, b);
    gemm_host(corr, V, TVtR, rows, b, b, false, false, rows, b, rows);
    for (int i = 0; i < rows * b; ++i)
        recon[i] = R[i] - corr[i];
    double rel_recon = frob_diff(recon, A0) / frob(A0);

    // (2) orthogonality: Q = I − V·T·Vᵀ ;  ‖QᵀQ − I‖ / √rows
    std::vector<T> VT(rows * b), Q(rows * rows), QtQ(rows * rows), Id(rows * rows, T(0));
    gemm_host(VT, V, Tm, rows, b, b, false, false, rows, b, rows);
    gemm_host(Q, VT, V, rows, rows, b, false, true, rows, rows, rows); // Q = V·T·Vᵀ
    for (int j = 0; j < rows; ++j)
        for (int i = 0; i < rows; ++i)
            Q[i + j * rows] = (i == j ? T(1) : T(0)) - Q[i + j * rows];
    gemm_host(QtQ, Q, Q, rows, rows, rows, true, false, rows, rows, rows);
    for (int i = 0; i < rows; ++i)
        Id[i + i * rows] = T(1);
    double orth = frob_diff(QtQ, Id) / std::sqrt((double)rows);

    CHECK_LT(rel_recon, tol);
    CHECK_LT(orth, tol);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dV));
    cuev::handle_free(&ws);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(dbbr_panel_qr, fp64_square) {
    panel_qr_case<double>(256, 32, 1e-10);
}
TEST(dbbr_panel_qr, fp64_wide) {
    panel_qr_case<double>(512, 64, 1e-10);
}
TEST(dbbr_panel_qr, fp32_square) {
    panel_qr_case<float>(256, 32, 1e-3);
}

CUTEST_MAIN()
