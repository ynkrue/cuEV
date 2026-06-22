/**
 * @file   kernels.cuh
 * @brief  CUDA kernel interface and cuBLAS/cuSOLVER wrappers for cuEV.
 *
 * TODO: add description
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include "common.h"
#include "cuda/handle.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace cuev {

// =============================================================================
// cuBLAS dispatching wrappers  (cuev::cublas)
// =============================================================================
namespace cublas {

/**
 * @brief General matrix-matrix multiplication (GEMM). C ← α·op(A)·op(B) + β·C
 *
 * op(X) = X, Xᵀ, or Xᴴ depending on the value of transX.  C, A, B are column-major.
 *
 * @param[in] h        cuBLAS handle
 * @param[in] transa   how to interpret A (op(A))
 * @param[in] transb   how to interpret B (op(B))
 * @param[in] m        rows of op(A) and C
 * @param[in] n        columns of op(B) and C
 * @param[in] k        columns of op(A) and rows of op(B)
 * @param[in] alpha    scalar multiplier for op(A)·op(B)
 * @param[in] A        matrix A, column-major, leading dimension lda
 * @param[in] lda      leading dimension of A (≥ rows of A)
 * @param[in] B        matrix B, column-major, leading dimension ldb
 * @param[in] ldb      leading dimension of B (≥ rows of B)
 * @param[in] beta     scalar multiplier for C
 * @param[in,out] C    matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc      leading dimension of C (≥ rows of C)
 */
template <typename T>
void gemm(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

/**
 * @brief General matrix addition: C ← α·op(A) + β·op(B)
 *
 * op(X) = X or Xᵀ depending on the value of transX.  C, A, B are column-major.
 *
 * @param[in] h        cuBLAS handle
 * @param[in] transa   how to interpret A (op(A))
 * @param[in] transb   how to interpret B (op(B))
 * @param[in] m        rows of op(A), op(B), and C
 * @param[in] n        columns of op(A), op(B), and C
 * @param[in] alpha    scalar multiplier for op(A)
 * @param[in] A        matrix A, column-major, leading dimension lda
 * @param[in] lda      leading dimension of A (≥ rows of A)
 * @param[in] beta     scalar multiplier for op(B)
 * @param[in] B        matrix B, column-major, leading dimension ldb
 * @param[in] ldb      leading dimension of B (≥ rows of B)
 * @param[in,out] C    matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc      leading dimension of C (≥ rows of C)
 */
template <typename T>
void geam(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          const T *alpha, const T *A, int lda, const T *beta, const T *B, int ldb, T *C, int ldc);

/**
 * @brief Symmetric matrix-matrix multiplication: C ← α·op(A)·op(B) + β·C
 *
 * op(X) = X or Xᵀ depending on the value of transX.  C and B are column-major; A is symmetric
 * with leading dimension lda.  Only the @p uplo triangle of A is referenced
 *
 * @param[in] h         cuBLAS handle
 * @param[in] side      whether A multiplies B from the left or right
 * @param[in] uplo      which triangle of A is referenced (and where the result is written)
 * @param[in] m         rows of op(B) and C
 * @param[in] n         columns of op(B) and C
 * @param[in] alpha     scalar multiplier for op(A)·op(B)
 * @param[in] A         symmetric matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] B         matrix B, column-major, leading dimension ldb
 * @param[in] ldb       leading dimension of B (≥ rows of B)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
template <typename T>
void symm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

/**
 * @brief Symmetric rank-k update: C ← α·op(A)·op(A)ᵀ + β·C
 *
 * Only the @p uplo triangle of C is referenced/written.
 *
 * @param[in] h         cuBLAS handle
 * @param[in] uplo      which triangle of C is referenced (and where the result is written)
 * @param[in] trans     how to interpret A (op(A))
 * @param[in] n         rows and columns of op(A) and C
 * @param[in] k         columns of op(A)
 * @param[in] alpha     scalar multiplier for op(A)·op(A)ᵀ
 * @param[in] A         matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
template <typename T>
void syrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
          const T *alpha, const T *A, int lda, const T *beta, T *C, int ldc);

/**
 * @brief Triangular solve: B ← α·op(A)⁻¹·B  (or B ← α·B·op(A)⁻¹ if side = right)
 *
 * @param[in] h         cuBLAS handle
 * @param[in] side      whether A multiplies B from the left or right
 * @param[in] uplo      which triangle of A is referenced (and where the result is written)
 * @param[in] trans     how to interpret A (op(A))
 * @param[in] diag      whether A has a unit diagonal
 * @param[in] m         rows of B
 * @param[in] n         columns of B
 * @param[in] alpha     scalar multiplier for the solution
 * @param[in] A         triangular matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in,out] B     matrix B, column-major, leading dimension ldb; overwritten with the solution
 * @param[in] ldb       leading dimension of B (≥ rows of B)
 */
template <typename T>
void trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
          cublasDiagType_t diag, int m, int n, const T *alpha, const T *A, int lda, T *B, int ldb);

/**
 * @brief Vector scaling: x ← α·x
 *
 * @param[in] h         cuBLAS handle
 * @param[in] n         length of x
 * @param[in] alpha     scalar multiplier for x
 * @param[in,out] x     vector x; overwritten with the result
 * @param[in] incx      stride of x (≥ 1)
 */
template <typename T> void scal(cublasHandle_t h, int n, const T *alpha, T *x, int incx);

/**
 * @brief Vector copy: y ← x
 *
 * @param[in] h         cuBLAS handle
 * @param[in] n         length of x and y
 * @param[in] x         vector x; not modified
 * @param[in] incx      stride of x (≥ 1)
 * @param[in,out] y     vector y; overwritten with the result
 * @param[in] incy      stride of y (≥ 1)
 */
template <typename T> void copy(cublasHandle_t h, int n, const T *x, int incx, T *y, int incy);

/**
 * @brief Vector 2-norm: result = ||x||₂
 *
 * @param[in] h         cuBLAS handle
 * @param[in] n         length of x
 * @param[in] x         vector x
 * @param[in] incx      stride of x (≥ 1)
 * @param[out] result   pointer to the result on the device
 */
template <typename T> void nrm2(cublasHandle_t h, int n, const T *x, int incx, T *result);

} // namespace cublas

// =============================================================================
// cuSOLVER type-dispatching wrappers  (cuev::cusolver)
// =============================================================================
namespace cusolver {

/**
 * @brief QR factorisation: A ← Q·R  (Householder, in-place).
 *
 * TODO: add buffer explenations and references to cuSOLVER docs.
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
void geqrf(cusolverDnHandle_t h, int m, int n, T *A, int lda, T *tau, SolverHandle<T> *ws,
           cudaStream_t stream);

/**
 * @brief Materialise Q from compact QR representation.
 *
 * TODO: add buffer explenations and references to cuSOLVER docs.
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
           SolverHandle<T> *ws, cudaStream_t stream);

/**
 * @brief Symmetric dense eigensolver (D&C): A v = λ v.
 *
 * TODO: add buffer explenations and references to cuSOLVER docs.
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
void syevd(cusolverDnHandle_t h, int n, T *A, int lda, T *W, SolverHandle<T> *ws,
           cudaStream_t stream);

} // namespace cusolver

} // namespace cuev
