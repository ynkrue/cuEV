/**
 * @file kernels.cuh
 * 
 * cuGEMV kernel declarations — y = alpha * A * x + beta * y
 * A is M×N row-major, incx = incy = 1.  T = float or double.
 */

#pragma once
#include <cuda_runtime.h>

// =============================================================
// Utility kernels
// =============================================================

/// fill x ← α
template <typename T>
void launch_fill(T alpha, T *x, int N, cudaStream_t stream);

/// copy B ← A
template <typename T>
void launch_copy(const T *A, T *B, int N, cudaStream_t stream);

/// transpose Aᵀ ← A
template <typename T>
void launch_transpose(const T *A, T *AT, int M, int N, cudaStream_t stream);


// =============================================================
// Householder kernels
// =============================================================

/// hh_reflect v, τ ← H[k+1:n, k] - alpha e_k (Householder reflector)
template <typename T>
void launch_hh_reflect(T *H, T *v, T *tau, T *d, T *e, int N, int k, cudaStream_t stream);

/// hh_gemv p ← Hv (general Householder application)
template <typename T>
void launch_hh_gemv(const T *v, const T *H, T *p, int N, int k, cudaStream_t stream);

/// hh_update u ← tau p - 0.5 tau² vᵀp v (Householder rank-2 update)
template <typename T>
void launch_hh_update(const T *v, const T *p, const T *tau, T *u, int N, int k, cudaStream_t stream);

/// hh_syr2 H ← H - v uᵀ - u vᵀ (Householder symmetric rank-2 update)
template <typename T>
void launch_hh_syr2(const T *v, const T *u, T *H, int N, int k, cudaStream_t stream);

/// hh_larft T̂ ← VᵀV (blocked Householder)
template <typename T>
void launch_hh_larft(const T *V, const T *tau, T *T̂, int M, int K, cudaStream_t stream);

/// hh_larfb C ← (I − V T̂ Vᵀ)C (blocked Householder application)
template <typename T>
void launch_hh_larfb(const T *V, const T *T̂, T *C, int M, int N, int K, cudaStream_t stream);

// =============================================================
// Vector operations
// =============================================================

/// dot vᵀw

/// axpy y ← αx + y

/// scal x ← αx

/// nrm2 ‖v‖₂

// =============================================================
// Matrix-vector kernels
// =============================================================

/// GEMV y ← αAx + βy
template <typename T>
void inline launch_gemv(T alpha, const T *A, const T *x,
                       T beta, T *y, int M, int N, cudaStream_t stream) {
    launch_gemv_gmem(alpha, A, x, beta, y, M, N, stream);
}

template <typename T>
void launch_gemv_gmem(T alpha, const T *A, const T *x,
                       T beta, T *y, int M, int N, cudaStream_t stream);

template <typename T>
void launch_gemv_smem(T alpha, const T *A, const T *x,
                      T beta, T *y, int M, int N, cudaStream_t stream);

/// GER A ← A + αxyᵀ

/// SYR2 A ← A + α(xyᵀ + yxᵀ)


// =============================================================
// Matrix-matrix kernels
// =============================================================

/// GEMM C ← αAB + βC
template <typename T>
void inline launch_gemm(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream) {
    launch_gemm_warptile(alpha, A, B, beta, C, M, N, K, stream);
}

template <typename T>
void launch_gemm_gmem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream);

template <typename T>
void launch_gemm_smem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream);

template <typename T>
void launch_gemm_tiled(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream);

template <typename T>
void launch_gemm_warptile(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream);

/// fp64 tensor-core GEMM (DMMA, sm_80+)
// void launch_gemm_dmma(double alpha, const double *A, const double *B, double beta, double *C, int M, int N, int K, cudaStream_t stream);

/// TRMM B ← αAB


// =============================================================
// Solver kernels
// =============================================================

/// STEDC eigenvalues of symmetric tridiagonal matrix

/// secular 1 + ρ·Σ zᵢ²/(dᵢ − λ) = 0 root-finding for STEDC
