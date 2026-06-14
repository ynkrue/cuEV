/**
 * @file   qdwh.cu
 * @brief  QDWH polar iteration for computing the matrix sign function.
 *
 * Implements sign(B) for real symmetric B via the QR-based QDWH algorithm
 * (Nakatsukasa & Higham, SISC 2010). Used as the core primitive of the
 * spectral divide-and-conquer eigensolver.
 *
 * Kernel sequence per QDWH iteration:
 *   qdwh_eye        fill bottom block of W with identity
 *   (cuBLAS)        scale top block: W[0:n,:] = √cₖ · B
 *   (cuSOLVER)      thin QR of 2n×n W → Q, R
 *   (cuBLAS)        update B ← (bₖ/cₖ) B + coeff · Q₁ Q₂ᵀ
 *   qdwh_symmetrize restore symmetry after floating-point drift
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cmath>
#include <cublas_v2.h>
#include <cuda.h>
#include <cusolverDn.h>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

// -----------------------------------------------------------------------------
// qdwh_shift
// -----------------------------------------------------------------------------
template <typename T> __global__ void qdwh_shift_kernel(T *A, T mu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] -= mu;
}

// -----------------------------------------------------------------------------
// qdwh_eye
// -----------------------------------------------------------------------------
template <typename T> __global__ void qdwh_eye_kernel(T *W, int n) {
    // W is 2n×n column-major (ld=2n); set W[n:2n, :] = I_n
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) W[col * (2 * n) + (n + row)] = (row == col) ? T(1) : T(0);
}

// -----------------------------------------------------------------------------
// qdwh_symmetrize
// -----------------------------------------------------------------------------
template <typename T> __global__ void qdwh_symmetrize_kernel(T *A, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < row) {
        T avg = (A[row * n + col] + A[col * n + row]) * T(0.5);
        A[row * n + col] = avg;
        A[col * n + row] = avg;
    }
}

} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev {
namespace kernels {

template <typename T> void qdwh_shift(T *A, T mu, int n, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    qdwh_shift_kernel<T><<<div_up(n, BLOCKSIZE), BLOCKSIZE, 0, stream>>>(A, mu, n);
}

template <typename T> void qdwh_eye(T *W, int n, cudaStream_t stream) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid(div_up(n, TILE), div_up(n, TILE));
    qdwh_eye_kernel<T><<<grid, block, 0, stream>>>(W, n);
}

template <typename T> void qdwh_symmetrize(T *A, int n, cudaStream_t stream) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid(div_up(n, TILE), div_up(n, TILE));
    qdwh_symmetrize_kernel<T><<<grid, block, 0, stream>>>(A, n);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void qdwh_shift<T>(T *, T, int, cudaStream_t);                                        \
    template void qdwh_eye<T>(T *, int, cudaStream_t);                                             \
    template void qdwh_symmetrize<T>(T *, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels

// =============================================================================
// QDWH coefficients  (CPU)
// =============================================================================
namespace {

// Compute QDWH iteration coefficients from current l (lower bound on σ_min,
// normalised so σ_max ≈ 1). Returns a, b, c for the update:
//   X ← (b/c)X + coeff · Q₁Q₂ᵀ,  coeff = (a − b/c)/√c
// Reference: Nakatsukasa & Higham, SISC 2010, Algorithm 4.1.
template <typename T> static void qdwh_coeffs(T &l, T &a, T &b, T &c) {
    T d = std::cbrt(T(4) * (T(1) - l * l) / std::pow(l, 4));
    a = std::sqrt(T(1) + d) +
        std::sqrt(0.5 * (T(8) - T(4) * d + T(8) * (T(2) - l * l) / (l * l * std::sqrt(T(1) + d))));
    b = 0.25 * (a - T(1)) * (a - T(1));
    c = a + b - T(1);
    l = l * (a + b * l * l) / (T(1) + c * l * l);
}

} // namespace

// =============================================================================
// qdwh_sign
// =============================================================================

template <typename T>
void qdwh_sign(cublasHandle_t cublas, cusolverDnHandle_t cusolver, T *B, int n,
               SolverWorkspace<T> *ws, cudaStream_t stream) {
    // W (2n×n) and tau (n) come from the pre-allocated workspace — no allocation here.
    T *W = ws->qdwh_W;
    T *tau = ws->qdwh_tau;

    // Normalise: B ← B / ‖B‖_F  so that σ_max ≈ 1.
    // TODO: cublas::nrm2 on B as a flat n²-vector, then cublas::scal

    // Initial lower bound on σ_min (safe underestimate).
    // TODO: estimate l₀ from ‖B‖₁ · ‖B‖_∞ or use 1/√n
    T l = T(0);

    constexpr int MAX_ITER = 8;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        T a, b, c;
        qdwh_coeffs(l, a, b, c);

        // Build W = [√c · B ; I_n]
        // TODO: cublas::copy + cublas::scal for top half, then:
        kernels::qdwh_eye(W, n, stream);

        // Thin QR of W (2n×n): W = Q · R
        cusolver::geqrf(cusolver, 2 * n, n, W, 2 * n, tau, ws, stream);
        // Materialise Q (2n×n) in W
        cusolver::orgqr(cusolver, 2 * n, n, n, W, 2 * n, tau, ws, stream);

        // Q₁ = W[0:n, :], Q₂ = W[n:2n, :]  (contiguous in column-major W)
        // Update: B ← (b/c)·B + (a − b/c)/√c · Q₁·Q₂ᵀ
        // TODO: cublas::gemm for Q₁Q₂ᵀ, then cublas::geam for linear combination

        kernels::qdwh_symmetrize(B, n, stream);

        // TODO: update l using QDWH recurrence
        (void)a;
        (void)b;
        (void)c;
        if (l >= T(1) - T(1e-14)) break;
    }

    // W and tau are owned by ws — nothing to free here.
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template void qdwh_sign<float>(cublasHandle_t, cusolverDnHandle_t, float *, int,
                               SolverWorkspace<float> *, cudaStream_t);
template void qdwh_sign<double>(cublasHandle_t, cusolverDnHandle_t, double *, int,
                                SolverWorkspace<double> *, cudaStream_t);

} // namespace cuev
