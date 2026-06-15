/**
 * @file   qdwh.cu
 * @brief  QDWH polar iteration for computing the matrix sign function.
 *
 * Implements sign(B) for real symmetric B via the QDWH algorithm
 * (Nakatsukasa & Higham, SISC 2013). Used as the core primitive of the
 * spectral divide-and-conquer eigensolver.
 *
 * Each iteration applies one of two mathematically equivalent updates, chosen by
 * the conditioning of the iterate (coefficient cₖ):
 *   qdwh_step_qr    cₖ > 100 — QR of [√cₖ·B ; I] (2n×n); always stable
 *   qdwh_step_chol  cₖ ≤ 100 — Cholesky of I + cₖ·BᵀB (n×n); ~half the flops
 * followed by qdwh_symmetrize to undo floating-point drift. With cubic
 * convergence the iteration reaches machine precision in ≤6 steps; typically
 * the first ~2 steps use QR and the rest Cholesky.
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

// -----------------------------------------------------------------------------
// One QDWH step via QR (always stable; used while the iterate is ill-conditioned).
//   [√c·X ; I] = [Q₁ ; Q₂]·R,   X ← (b/c)·X + (a − b/c)/√c · Q₁·Q₂ᵀ
// -----------------------------------------------------------------------------
template <typename T>
static void qdwh_step_qr(cublasHandle_t cublas_h, cusolverDnHandle_t cusolver_h, T *X, int n, T a,
                         T b, T c, SolverWorkspace<T> *ws, cudaStream_t stream) {
    T *W = ws->qdwh_W;
    T *tau = ws->qdwh_tau;
    T scale = std::sqrt(c);
    T zero = T(0);
    T one = T(1);

    // W = [√c · X ; I_n]
    cuev::cublas::geam(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &scale, X, n, &zero, W, 2 * n, W,
                       2 * n);
    qdwh_eye(W, n, stream);

    // QR of W (2n×n): W = Q · R
    cuev::cusolver::geqrf(cusolver_h, 2 * n, n, W, 2 * n, tau, ws, stream);
    cuev::cusolver::orgqr(cusolver_h, 2 * n, n, n, W, 2 * n, tau, ws, stream);

    // Q₁ = W[0:n, :], Q₂ = W[n:2n, :];  X ← (b/c)·X + (a − b/c)/√c · Q₁·Q₂ᵀ
    T bc = b / c;
    T coeff = (a - bc) / scale;
    T *Q1 = W;
    T *Q2 = W + n;
    cuev::cublas::scal(cublas_h, n * n, &bc, X, 1);
    cuev::cublas::gemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &coeff, Q1, 2 * n, Q2, 2 * n,
                       &one, X, n);
}

// -----------------------------------------------------------------------------
// One QDWH step via Cholesky (~half the flops, no orgqr; valid once Z is well
// conditioned, i.e. c ≤ CHOL_SWITCH so κ(Z) = 1 + c·σ_max² ≤ 1 + c).
//   Z = I + c·XᵀX = UᵀU,   X ← (b/c)·X + (a − b/c)·X·Z⁻¹  (= X·U⁻¹·U⁻ᵀ)
// Z and the X·Z⁻¹ temp are carved from the 2n×n QR work matrix (unused here).
// -----------------------------------------------------------------------------
template <typename T>
static void qdwh_step_chol(cublasHandle_t cublas_h, cusolverDnHandle_t cusolver_h, T *X, int n, T a,
                           T b, T c, SolverWorkspace<T> *ws, cudaStream_t stream) {
    T *Z = ws->qdwh_W;                   // n×n, ld n
    T *tmp = ws->qdwh_W + (size_t)n * n; // n×n, ld n
    T zero = T(0);
    T one = T(1);
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    // Z = I + c·XᵀX  (upper triangle only), then Cholesky Z ← U
    cuev::cublas::syrk(cublas_h, uplo, CUBLAS_OP_T, n, n, &c, X, n, &zero, Z, n);
    qdwh_shift(Z, -one, n, stream); // Z[i,i] += 1
    cuev::cusolver::potrf(cusolver_h, uplo, n, Z, n, ws, stream);

    // tmp ← X, then tmp ← X·U⁻¹·U⁻ᵀ = X·Z⁻¹  (two right triangular solves)
    cuev::cublas::copy(cublas_h, n * n, X, 1, tmp, 1);
    cuev::cublas::trsm(cublas_h, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n,
                       &one, Z, n, tmp, n);
    cuev::cublas::trsm(cublas_h, CUBLAS_SIDE_RIGHT, uplo, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, n,
                       &one, Z, n, tmp, n);

    // X ← (b/c)·X + (a − b/c)·tmp
    T bc = b / c;
    T coeff = a - bc;
    cuev::cublas::geam(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, &bc, X, n, &coeff, tmp, n, X, n);
}

// =============================================================================
// qdwh_sign
// =============================================================================

template <typename T>
void qdwh_sign(cublasHandle_t cublas_h, cusolverDnHandle_t cusolver_h, T *B, int n,
               SolverWorkspace<T> *ws, cudaStream_t stream) {
    // Normalise: B ← B / ‖B‖_F.  ‖B‖_F ≥ ‖B‖_2 = σ_max, so σ_max(B) ≤ 1 after
    // scaling — the precondition for QDWH convergence, and it guarantees
    // κ(I + c·BᵀB) ≤ 1 + c so the c ≤ CHOL_SWITCH test below is a safe gate.
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

    // c_k decreases monotonically (huge when l≈0, → 3 as l→1). Use the cheap
    // Cholesky update once c drops below this; the first ~2 iterations stay on QR.
    constexpr T CHOL_SWITCH = T(100);

    constexpr int MAX_ITER = 6;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        T a, b, c;
        qdwh_coeffs(l, a, b, c);

        if (c > CHOL_SWITCH)
            qdwh_step_qr(cublas_h, cusolver_h, B, n, a, b, c, ws, stream);
        else
            qdwh_step_chol(cublas_h, cusolver_h, B, n, a, b, c, ws, stream);

        qdwh_symmetrize(B, n, stream);

        if (l >= T(1) - T(1e-14)) break;
    }

    // qdwh_W and qdwh_tau are owned by ws — nothing to free here.
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
