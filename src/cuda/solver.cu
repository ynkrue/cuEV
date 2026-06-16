/**
 * @file   solver.cu
 * @brief  Spectral divide-and-conquer eigensolver orchestration — single GPU.
 *
 * Public entry point: cuev::symm_eig_solve<T>(H, n, eval, evec, stream).
 *
 * Memory strategy:
 *   workspace_alloc  queries cuSOLVER buffer sizes and issues one cudaMalloc
 *                    for the combined pool (geqrf, orgqr, syevd scratch + d_info).
 *                    The pool is sized for the root problem; all recursion levels
 *                    reuse it.
 *   Per-level data   (B, tau_P, H1, H2, eval1/2, evec1/2) use ws->push/mark/reset —
 *                    stack allocation, no device sync, zero fragmentation.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/kernels.cuh"
#include "cuda/workspace.h"
#include "cuev.h"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <type_traits>

namespace cuev {

// =============================================================================
// spectral_dc — internal recursive solver
// =============================================================================
namespace {

template <typename T>
void spectral_dc(cublasHandle_t cublas, cusolverDnHandle_t cusolver, T *H, int n, T *eval, T *evec,
                 SolverWorkspace<T> *ws, cudaStream_t stream) {
    // --- Base case: cuSOLVER syevd ---
    if (n <= SDC_BASE_N) {
        cusolver::syevd(cusolver, n, H, n, eval, ws, stream);
        CUDA_CHECK(
            cudaMemcpyAsync(evec, H, (size_t)n * n * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    size_t lvl = ws->mark();

    // --- Split point: μ ≈ mean eigenvalue ---
    T mu = kernels::sdc_trace(H, n, stream) / T(n);

    // --- B ← copy of H, then B ← sign(H − μI) via QDWH ---
    T *B = ws->push((size_t)n * n);
    CUDA_CHECK(cudaMemcpyAsync(B, H, (size_t)n * n * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    kernels::qdwh_shift(B, mu, n, stream);
    kernels::qdwh_sign(cublas, cusolver, B, n, ws, stream);

    // --- P = (I + sign(B)) / 2 — spectral projector onto eigenvalues > μ ---
    T *P = B;
    T half = T(0.5);
    T zero = T(0);
    cublas::geam(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &half, P, n, &zero, P, n, P, n);
    kernels::qdwh_shift(P, -half, n, stream);

    // --- Split size k = rank(P) ---
    // P is an orthogonal projector: eigenvalues are exactly 1 (×k) and 0 (×n−k),
    // hence trace(P) = rank(P) = k exactly.
    int k = (int)std::lround(kernels::sdc_trace(P, n, stream));

    // --- QR(P) → Q; Q1 = Q[:,0:k], Q2 = Q[:,k:n] ---
    T *tau_P = ws->push((size_t)n);
    cusolver::geqrf(cusolver, n, n, P, n, tau_P, ws, stream);
    cusolver::orgqr(cusolver, n, n, k, P, n, tau_P, ws, stream);

    T *Q1 = P;                 // n×k, cols 0..k-1
    T *Q2 = P + (size_t)n * k; // n×(n-k), cols k..n-1

    // --- Form subproblems ---
    T *H1 = ws->push((size_t)k * k);
    T *H2 = ws->push((size_t)(n - k) * (n - k));
    kernels::sdc_split(cublas, H, Q1, Q2, H1, H2, n, k, ws, stream);

    // --- Recurse ---
    T *eval1 = ws->push((size_t)k);
    T *evec1 = ws->push((size_t)k * k);
    spectral_dc(cublas, cusolver, H1, k, eval1, evec1, ws, stream);

    T *eval2 = ws->push((size_t)(n - k));
    T *evec2 = ws->push((size_t)(n - k) * (n - k));
    spectral_dc(cublas, cusolver, H2, n - k, eval2, evec2, ws, stream);

    // --- Merge eigenvalues: [eval2 | eval1] ascending ---
    int m = n - k;
    CUDA_CHECK(
        cudaMemcpyAsync(eval, eval2, (size_t)m * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(eval + m, eval1, (size_t)k * sizeof(T), cudaMemcpyDeviceToDevice, stream));

    // --- Back-transform eigenvectors: evec = [Q2·evec2 | Q1·evec1] ---
    kernels::sdc_combine(cublas, Q1, Q2, evec1, evec2, evec, n, k, stream);

    ws->reset(lvl);
}

} // namespace

// =============================================================================
// Public entry point
// =============================================================================

template <typename T> void symm_eig_solve(T *H, int n, T *eval, T *evec, cudaStream_t stream) {
    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    CUBLAS_CHECK(cublasCreate(&cublas));
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));

    SolverWorkspace<T> ws = workspace_alloc<T>(cusolver, n, stream);
    spectral_dc(cublas, cusolver, H, n, eval, evec, &ws, stream);
    workspace_free(ws);

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template void symm_eig_solve<float>(float *, int, float *, float *, cudaStream_t);
template void symm_eig_solve<double>(double *, int, double *, double *, cudaStream_t);
} // namespace cuev
