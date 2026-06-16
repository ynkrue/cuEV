/**
 * @file   kernels.cuh
 * @brief  CUDA kernel interface and cuBLAS/cuSOLVER wrappers for cuEV.
 *
 * Three sections:
 *
 *   cuev::kernels   Custom GPU kernel launchers:
 *     qdwh_           QDWH polar iteration primitives      (src/cuda/qdwh.cu)
 *     sdc_            Spectral divide-and-conquer helpers  (src/cuda/sdc.cu)
 *
 *   cuev::cublas    Type-dispatching inline wrappers for cuBLAS:
 *                     gemm, symm, geam, scal, copy, nrm2   (src/cuda/cublas.cu)
 *
 *   cuev::cusolver  Type-dispatching inline wrappers for cuSOLVER:
 *                     geqrf, orgqr, syevd                  (src/cuda/cusolver.cu)
 *
 *
 * All matrices are column-major unless documented otherwise.
 * Template parameter @p T is float or double throughout.
 * cuBLAS and cuSOLVER are column-major
 * where needed. All workspace buffers for cuSOLVER are allocated in SolverWorkspace
 * and passed to the wrappers.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include "common.h"
#include "cuda/workspace.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace cuev {

namespace kernels {

// =============================================================================
// QDWH polar iteration
// =============================================================================

/**
 * @brief In-place diagonal shift: A ← A − μI.
 *
 * Used to form B = H − μI before computing sign(B) via QDWH.
 *
 * @tparam T      float or double
 * @param[in,out] A      n×n matrix, column-major; diagonal modified in place
 * @param[in]     mu     shift scalar μ
 * @param[in]     n      matrix dimension
 * @param[in]     stream CUDA stream
 */
template <typename T> void qdwh_shift(T *A, T mu, int n, cudaStream_t stream);

/**
 * @brief Set the bottom n×n block of a 2n×n matrix to the identity.
 *
 * Used to build the QDWH work matrix W = [√c·X ; I] before each QR
 * factorization. Top half (√c·X) is filled by the caller via cublasDscal /
 * cublasDcopy; this kernel fills the bottom half.
 *
 * @tparam T      float or double
 * @param[in,out] W      2n×n matrix, column-major; rows n..2n-1 set to I_n
 * @param[in]     n      half-dimension (W has 2n rows, n columns)
 * @param[in]     stream CUDA stream
 */
template <typename T> void qdwh_eye(T *W, int n, cudaStream_t stream);

/**
 * @brief Symmetrize in place: A ← (A + Aᵀ) / 2.
 *
 * Applied after each QDWH iteration to prevent floating-point drift
 * from breaking symmetry of the iterates.
 *
 * @tparam T      float or double
 * @param[in,out] A      n×n matrix, column-major; symmetrized in place
 * @param[in]     n      matrix dimension
 * @param[in]     stream CUDA stream
 */
template <typename T> void qdwh_symmetrize(T *A, int n, cudaStream_t stream);

// =============================================================================
// Spectral divide-and-conquer helpers
// =============================================================================

/**
 * @brief Compute the trace of a square matrix: returns Σ A[i,i].
 *
 * Used to estimate the spectral midpoint μ ≈ trace(H) / n before each
 * recursive split. Performs a parallel reduction over the diagonal and
 * synchronises the stream before returning the host scalar.
 *
 * @tparam T      float or double
 * @param[in]     A      n×n matrix, column-major
 * @param[in]     n      matrix dimension
 * @param[in]     stream CUDA stream (synchronised internally before return)
 * @return        host scalar Σ A[i,i]
 */
template <typename T> T sdc_trace(const T *A, int n, cudaStream_t stream);

/**
 * @brief Form the two subproblems: H₁ = Q₁ᵀHQ₁ and H₂ = Q₂ᵀHQ₂.
 *
 * Q₁ (n×k) and Q₂ (n×(n−k)) are the orthonormal bases for the two
 * invariant subspaces extracted from the spectral projector.
 * Each subproblem requires two GEMMs via cuBLAS.
 *
 * @tparam T      float or double
 * @param[in]  cublas  cuBLAS handle
 * @param[in]  H       n×n symmetric matrix, column-major
 * @param[in]  Q1      n×k basis matrix, column-major (columns = basis vectors)
 * @param[in]  Q2      n×(n−k) basis matrix, column-major
 * @param[out] H1      k×k subproblem matrix, column-major
 * @param[out] H2      (n−k)×(n−k) subproblem matrix, column-major
 * @param[in]  n       global dimension
 * @param[in]  k       split size (dimension of first subproblem)
 * @param[in]  stream  CUDA stream
 */
template <typename T>
void sdc_split(cublasHandle_t cublas, const T *H, const T *Q1, const T *Q2, T *H1, T *H2, int n,
               int k, SolverWorkspace<T> *ws, cudaStream_t stream);

/**
 * @brief Combine sub-eigenvectors into the full eigenvector matrix.
 *
 * Applies the basis matrices Q₁ and Q₂ to the sub-eigenvectors:
 *
 *   evec[:, 0:k]   ← Q1 (n×k)     · evec1 (k×k)
 *   evec[:, k:n]   ← Q2 (n×(n−k)) · evec2 ((n−k)×(n−k))
 *
 * Column j of evec is the j-th eigenvector (column-major convention).
 *
 * @tparam T      float or double
 * @param[in]  cublas  cuBLAS handle
 * @param[in]  Q1      n×k basis matrix, column-major
 * @param[in]  Q2      n×(n−k) basis matrix, column-major
 * @param[in]  evec1   k×k eigenvector matrix of H₁, column-major, columns = eigenvectors
 * @param[in]  evec2   (n−k)×(n−k) eigenvector matrix of H₂, column-major, columns = eigenvectors
 * @param[out] evec    n×n output eigenvector matrix, column-major, columns = eigenvectors
 * @param[in]  n       global dimension
 * @param[in]  k       split size
 * @param[in]  ws      workspace for temporary buffers (push/reset, no net allocation)
 * @param[in]  stream  CUDA stream
 */
template <typename T>
void sdc_combine(cublasHandle_t cublas, const T *Q1, const T *Q2, const T *evec1, const T *evec2,
                 T *evec, int n, int k, cudaStream_t stream);

/**
 * @brief Compute the matrix sign function of a symmetric matrix via QDWH.
 *
 * Overwrites B with sign(B) using the QR-based QDWH polar iteration
 * (Nakatsukasa & Higham 2010). Converges in ≤ 8 iterations.
 *
 * @tparam T         float or double
 * @param[in]     cublas    cuBLAS handle
 * @param[in]     cusolver  cuSOLVER handle
 * @param[in,out] B         n×n symmetric matrix, column-major; overwritten with sign(B)
 * @param[in]     n         matrix dimension
 * @param[in]     ws        pre-allocated workspace (owns cuSOLVER scratch + d_info)
 * @param[in]     stream    CUDA stream
 */
template <typename T>
void qdwh_sign(cublasHandle_t cublas, cusolverDnHandle_t cusolver, T *B, int n,
               SolverWorkspace<T> *ws, cudaStream_t stream);

} // namespace kernels

// =============================================================================
// cuBLAS type-dispatching wrappers  (cuev::cublas)
// =============================================================================

/**
 * @brief Type-generic wrappers for cuBLAS.
 *
 * Each function dispatches to the S/D cuBLAS overload based on @p T.
 * Arguments are forwarded unchanged — no row/col-major remapping.
 * Errors are wrapped with CUBLAS_CHECK.
 */
namespace cublas {

/// C ← α·op(A)·op(B) + β·C
template <typename T>
void gemm(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

/// C ← α·op(A) + β·op(B)
template <typename T>
void geam(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          const T *alpha, const T *A, int lda, const T *beta, const T *B, int ldb, T *C, int ldc);

/// C ← α·A·B + β·C  with A symmetric (side=left) — only @p uplo triangle of A read
template <typename T>
void symm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

/// C ← α·op(A)·op(A)ᵀ + β·C  (only @p uplo triangle of C is referenced/written)
template <typename T>
void syrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
          const T *alpha, const T *A, int lda, const T *beta, T *C, int ldc);

/// B ← α·B·op(A)⁻¹  (side = right, triangular A); B overwritten with the solution
template <typename T>
void trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
          cublasDiagType_t diag, int m, int n, const T *alpha, const T *A, int lda, T *B, int ldb);

/// x ← α·x
template <typename T> void scal(cublasHandle_t h, int n, const T *alpha, T *x, int incx);

/// y ← x
template <typename T> void copy(cublasHandle_t h, int n, const T *x, int incx, T *y, int incy);

/// result ← ‖x‖₂
template <typename T> void nrm2(cublasHandle_t h, int n, const T *x, int incx, T *result);

} // namespace cublas

// =============================================================================
// cuSOLVER type-dispatching wrappers  (cuev::cusolver)
// =============================================================================

/**
 * @brief Type-generic wrappers for cuSOLVER dense routines.
 *
 * All functions receive a @ref SolverWorkspace pointer and use its pre-allocated
 * buffers (geqrf_buf / orgqr_buf / potrf_buf / syevd_buf / d_info).  No allocation
 * occurs inside these wrappers; all scratch is managed by workspace_alloc / free.
 */
namespace cusolver {

/**
 * @brief QR factorisation: A ← Q·R  (Householder, in-place).
 *
 * Uses ws->geqrf_buf and ws->d_info.  ws->geqrf_lwork must be ≥ the size
 * required for this (m, n) — guaranteed when ws was allocated for the root n.
 *
 * @tparam T         float or double
 * @param[in]     h        cuSOLVER handle
 * @param[in]     m        number of rows
 * @param[in]     n        number of columns
 * @param[in,out] A        m×n matrix, column-major; overwritten with compact QR
 * @param[in]     lda      leading dimension of A
 * @param[out]    tau      Householder scalars, length min(m,n)
 * @param[in]     ws       pre-allocated workspace (geqrf_buf, d_info)
 * @param[in]     stream   CUDA stream
 */
template <typename T>
void geqrf(cusolverDnHandle_t h, int m, int n, T *A, int lda, T *tau, SolverWorkspace<T> *ws,
           cudaStream_t stream);

/**
 * @brief Materialise Q from compact QR representation.
 *
 * Uses ws->orgqr_buf and ws->d_info.
 *
 * @tparam T         float or double
 * @param[in]     h        cuSOLVER handle
 * @param[in]     m        rows of Q
 * @param[in]     n        columns to generate (≤ m)
 * @param[in]     k        Householder reflector count (= min(m,n) from geqrf)
 * @param[in,out] A        m×n, column-major; compact QR in → first n cols of Q out
 * @param[in]     lda      leading dimension
 * @param[in]     tau      Householder scalars from geqrf, length k
 * @param[in]     ws       pre-allocated workspace (orgqr_buf, d_info)
 * @param[in]     stream   CUDA stream
 */
template <typename T>
void orgqr(cusolverDnHandle_t h, int m, int n, int k, T *A, int lda, const T *tau,
           SolverWorkspace<T> *ws, cudaStream_t stream);

/**
 * @brief Cholesky factorisation of an SPD matrix: A ← U  (A = UᵀU, upper).
 *
 * Used by the Cholesky-based QDWH update once the iterate is well-conditioned.
 * Uses ws->potrf_buf and ws->d_info.
 *
 * @tparam T         float or double
 * @param[in]     h        cuSOLVER handle
 * @param[in]     uplo     which triangle holds A and receives the factor
 * @param[in]     n        matrix dimension
 * @param[in,out] A        n×n SPD matrix, column-major; overwritten with the factor
 * @param[in]     lda      leading dimension
 * @param[in]     ws       pre-allocated workspace (potrf_buf, d_info)
 * @param[in]     stream   CUDA stream
 */
template <typename T>
void potrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T *A, int lda,
           SolverWorkspace<T> *ws, cudaStream_t stream);

/**
 * @brief Symmetric dense eigensolver (D&C): A v = λ v.
 *
 * Base-case solver in spectral_dc. Uses ws->syevd_buf and ws->d_info.
 * @p A (column-major) overwritten with eigenvectors; @p W receives eigenvalues ascending.
 *
 * @tparam T         float or double
 * @param[in]     h        cuSOLVER handle
 * @param[in]     n        matrix dimension
 * @param[in,out] A        n×n symmetric, column-major; overwritten with eigenvectors
 * @param[in]     lda      leading dimension
 * @param[out]    W        eigenvalues ascending, length n
 * @param[in]     ws       pre-allocated workspace (syevd_buf, d_info)
 * @param[in]     stream   CUDA stream
 */
template <typename T>
void syevd(cusolverDnHandle_t h, int n, T *A, int lda, T *W, SolverWorkspace<T> *ws,
           cudaStream_t stream);

} // namespace cusolver

} // namespace cuev
