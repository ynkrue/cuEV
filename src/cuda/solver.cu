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
 *                    reuse it (only one level runs at a time — no aliasing).
 *   Per-level data   (B, tau_P, H1, H2, eval1/2, evec1/2) use ws->push/mark/reset —
 *                    stack allocation, no device sync, zero fragmentation.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuev.h"
#include "kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <type_traits>

namespace cuev {

// =============================================================================
// workspace_alloc / workspace_free
// =============================================================================

template <typename T>
SolverWorkspace<T> workspace_alloc(cusolverDnHandle_t h, int n, cudaStream_t stream) {
    SolverWorkspace<T> ws{};
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // Query cuSOLVER buffer sizes
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(h, 2 * n, n, nullptr, 2 * n, &ws.geqrf_lwork));
        CUSOLVER_CHECK(
            cusolverDnSorgqr_bufferSize(h, 2 * n, n, n, nullptr, 2 * n, nullptr, &ws.orgqr_lwork));
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(h, uplo, n, nullptr, n, &ws.potrf_lwork));
        CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(h, jobz, uplo, SDC_BASE_N, nullptr, SDC_BASE_N,
                                                   nullptr, &ws.syevd_lwork));
    } else {
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(h, 2 * n, n, nullptr, 2 * n, &ws.geqrf_lwork));
        CUSOLVER_CHECK(
            cusolverDnDorgqr_bufferSize(h, 2 * n, n, n, nullptr, 2 * n, nullptr, &ws.orgqr_lwork));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(h, uplo, n, nullptr, n, &ws.potrf_lwork));
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(h, jobz, uplo, SDC_BASE_N, nullptr, SDC_BASE_N,
                                                   nullptr, &ws.syevd_lwork));
    }

    auto align = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };

    size_t off_geqrf = 0;
    size_t off_orgqr = off_geqrf + align((size_t)ws.geqrf_lwork * sizeof(T));
    size_t off_potrf = off_orgqr + align((size_t)ws.orgqr_lwork * sizeof(T));
    size_t off_syevd = off_potrf + align((size_t)ws.potrf_lwork * sizeof(T));
    size_t off_info = off_syevd + align((size_t)ws.syevd_lwork * sizeof(T));
    size_t off_W = off_info + align(sizeof(int));
    size_t off_tau = off_W + align(2u * (size_t)n * n * sizeof(T));
    size_t off_data = off_tau + align((size_t)n * sizeof(T));

    // Data pool: 6n² elements (B, Q, H1/H2, evec1/2, eval1/2 recursion stack).
    // See workspace.h for the derivation; push() aborts if this is exceeded.
    ws.data_cap = (size_t)6 * n * n;
    size_t total = off_data + ws.data_cap * sizeof(T);

    CUDA_CHECK(cudaMalloc(&ws.pool, total));
    char *base = static_cast<char *>(ws.pool);
    ws.geqrf_buf = reinterpret_cast<T *>(base + off_geqrf);
    ws.orgqr_buf = reinterpret_cast<T *>(base + off_orgqr);
    ws.potrf_buf = reinterpret_cast<T *>(base + off_potrf);
    ws.syevd_buf = reinterpret_cast<T *>(base + off_syevd);
    ws.d_info = reinterpret_cast<int *>(base + off_info);
    ws.qdwh_W = reinterpret_cast<T *>(base + off_W);
    ws.qdwh_tau = reinterpret_cast<T *>(base + off_tau);
    ws.data = reinterpret_cast<T *>(base + off_data);
    ws.data_used = 0;

    (void)stream;
    return ws;
}

template <typename T> void workspace_free(SolverWorkspace<T> &ws) {
    CUDA_CHECK(cudaFree(ws.pool));
    ws = SolverWorkspace<T>{};
}

// =============================================================================
// spectral_dc — internal recursive solver
// =============================================================================
namespace {

template <typename T>
void spectral_dc(cublasHandle_t cublas, cusolverDnHandle_t cusolver, T *H, int n, T *eval, T *evec,
                 SolverWorkspace<T> *ws, cudaStream_t stream) {
    // --- Base case: cuSOLVER syevd ---
    if (n <= SDC_BASE_N) {
        cusolver::syevd(cusolver, n, H, n, eval, ws, stream); // overwrites H with eigenvectors
        CUDA_CHECK(
            cudaMemcpyAsync(evec, H, (size_t)n * n * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    // Save data pool position — reset() at end of this level frees everything.
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
    // so trace(P) = rank(P) = k exactly. This is robust; counting QR-diagonal
    // entries would require column-pivoted QR (geqp3), which cuSOLVER lacks.
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

    // --- Recurse --- allocate output buffers, then recurse
    T *eval1 = ws->push((size_t)k);
    T *evec1 = ws->push((size_t)k * k);
    spectral_dc(cublas, cusolver, H1, k, eval1, evec1, ws, stream);

    T *eval2 = ws->push((size_t)(n - k));
    T *evec2 = ws->push((size_t)(n - k) * (n - k));
    spectral_dc(cublas, cusolver, H2, n - k, eval2, evec2, ws, stream);

    // --- Merge eigenvalues: eval2 (< μ) first, eval1 (> μ) second → ascending ---
    int m = n - k;
    CUDA_CHECK(
        cudaMemcpyAsync(eval, eval2, (size_t)m * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(eval + m, eval1, (size_t)k * sizeof(T), cudaMemcpyDeviceToDevice, stream));

    // --- Back-transform eigenvectors: evec = [Q2·evec2 | Q1·evec1] ---
    kernels::sdc_combine(cublas, Q1, Q2, evec1, evec2, evec, n, k, stream);

    // Free everything allocated at this level in one shot.
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
template SolverWorkspace<float> workspace_alloc<float>(cusolverDnHandle_t, int, cudaStream_t);
template SolverWorkspace<double> workspace_alloc<double>(cusolverDnHandle_t, int, cudaStream_t);
template void workspace_free<float>(SolverWorkspace<float> &);
template void workspace_free<double>(SolverWorkspace<double> &);

template void symm_eig_solve<float>(float *, int, float *, float *, cudaStream_t);
template void symm_eig_solve<double>(double *, int, double *, double *, cudaStream_t);
} // namespace cuev
