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
#include <limits>

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
// Reference: Nakatsukasa & Higham, SISC 2013, Eq. (3.4)–(3.5).
//
// NOTE the grouping of the second term: a = √(1+d) + ½·√(8 − 4d + …).
// Writing it as √(½·(…)) instead loses cubic convergence — the map's fixed
// point at 1 then has g'(1) ≈ 0.03 (linear), capping accuracy at ~1e-9 in 8
// iterations rather than reaching machine precision.
template <typename T> static void qdwh_coeffs(T &l, T &a, T &b, T &c) {
    T d = std::cbrt(T(4) * (T(1) - l * l) / std::pow(l, 4));
    a = std::sqrt(T(1) + d) +
        T(0.5) * std::sqrt(T(8) - T(4) * d + T(8) * (T(2) - l * l) / (l * l * std::sqrt(T(1) + d)));
    b = T(0.25) * (a - T(1)) * (a - T(1));
    c = a + b - T(1);
    l = l * (a + b * l * l) / (T(1) + c * l * l);
}

} // namespace

namespace kernels {

// =============================================================================
// qdwh_sign
// =============================================================================

template <typename T>
void qdwh_sign(cublasHandle_t cublas_h, cusolverDnHandle_t cusolver_h, T *B, int n,
               SolverWorkspace<T> *ws, cudaStream_t stream) {
    T *W = ws->qdwh_W;
    T *tau = ws->qdwh_tau;

    // Normalise: B ← B / ‖B‖_F.  ‖B‖_F ≥ ‖B‖_2 = σ_max, so σ_max(B) ≤ 1 after
    // scaling — the precondition for QDWH convergence.
    T h_norm;
    cuev::cublas::nrm2(cublas_h, n * n, B, 1, &h_norm);
    T inv_norm = T(1) / h_norm;
    cuev::cublas::scal(cublas_h, n * n, &inv_norm, B, 1);

    // l = lower bound on σ_min of the scaled matrix. QDWH needs l ≤ σ_min for the
    // iteration to sharpen *every* singular value to 1; cubic convergence then
    // reaches full accuracy in ≤6 iterations even from l ≈ machine epsilon.
    // (A larger guess such as 1/√n over-estimates σ_min and leaves singular values
    // near the split point μ under-resolved — modes there get misassigned.)
    T l = std::numeric_limits<T>::epsilon();

    constexpr int MAX_ITER = 6;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        T a, b, c;
        qdwh_coeffs(l, a, b, c);

        // W = [√c · B ; I_n]
        T scale = std::sqrt(c);
        T zero = T(0);
        cuev::cublas::geam(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &scale, B, n, &zero, W, 2 * n,
                           W, 2 * n);
        qdwh_eye(W, n, stream);

        // QR of W (2n×n): W = Q · R
        cuev::cusolver::geqrf(cusolver_h, 2 * n, n, W, 2 * n, tau, ws, stream);
        cuev::cusolver::orgqr(cusolver_h, 2 * n, n, n, W, 2 * n, tau, ws, stream);

        // Q₁ = W[0:n, :], Q₂ = W[n:2n, :]
        // B ← (b/c)·B + (a − b/c)/√c · Q₁·Q₂ᵀ
        T bc = b / c;
        T coeff = (a - bc) / scale;
        T one = T(1);

        T *Q1 = W;
        T *Q2 = W + n; // W[n:2n, :]
        cuev::cublas::scal(cublas_h, n * n, &bc, B, 1);
        cuev::cublas::gemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &coeff, Q1, 2 * n, Q2,
                           2 * n, &one, B, n);
        qdwh_symmetrize(B, n, stream);

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

} // namespace kernels
} // namespace cuev
