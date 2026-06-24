/**
 * @file   kernels.cuh
 * @brief  Custom kernel launchers and cuBLAS/cuSOLVER wrappers for cuEV.
 *
 * Three sections:
 *   cuev::kernels   Custom GPU kernel launchers (dbbr_*, bc_*, bt_*)
 *   cuev::cublas    Type-dispatching cuBLAS wrappers  — all take SolverHandle<T>*
 *   cuev::cusolver  Type-dispatching cuSOLVER wrappers — all take SolverHandle<T>*
 *
 * All matrices are column-major. T is float or double throughout.
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
// cuev::kernels — custom GPU kernel launchers
// =============================================================================
namespace kernels {

// --- DBBR (double-blocking band reduction) ------------------------------------

/**
 * @brief Panel QR + block reflector for one DBBR panel.
 *
 * Runs geqrf on the panel and extracts the explicit unit-trapezoidal reflectors
 * into Y, and forms the b×b block factor T into ws->Tmat. The Householder
 * scalars τ are staged in ws->tau.
 *      H₀·H₁···Hₖ  =  I − Y·T·Yᵀ    with Y = [v₀, v₁, …, vₖ]
 *                                   and T = upper-triangular correction.
 *
 * @tparam T      float or double
 * @param[in, out] ws   solver handle (cusolver, geqrf scratch, tau, Tmat)
 * @param[in,out] A     rows × b panel into the working matrix upper triangle ← R,
 *                      strictly-lower ← packed Householder vectors
 * @param[out]    Y     rows × b reflectors (panel of ws->Y)
 * @param[in]     rows  number of rows in the panel (n − j − b)
 * @param[in]     b     panel width (bandwidth)
 */
template <typename T> void dbbr_panel_qr(SolverHandle<T> *ws, T *A, T *Y, int rows, int b);

/**
 * @brief Custom square-blocked symmetric rank-2k update: A ← A − Z·Yᵀ − Y·Zᵀ
 *
 * Replaces cuBLAS dsyr2k with a square tiling order that keeps GEMM shapes
 * more square on GPUsH100, significantly outperforming cuBLAS for large n.
 *
 * @tparam T      float or double
 * @param[in]     ws    solver handle
 * @param[in,out] A     n×n symmetric matrix, column-major; lower triangle updated
 * @param[in]     Z     n×k matrix, column-major
 * @param[in]     Y     n×k matrix, column-major
 * @param[in]     n     matrix dimension
 * @param[in]     k     number of columns in Z and Y
 */
template <typename T>
void dbbr_syr2k(SolverHandle<T> *ws, T *A, const T *Z, const T *Y, int n, int k);

// --- BC (bulge chasing) -------------------------------------------------------

/**
 * @brief Repack a band matrix into contiguous symmetric storage for L2 reuse.
 *
 * @tparam T      float or double
 * @param[in]     ws    solver handle (stream)
 * @param[in]     B     n×n band matrix, full storage, column-major
 * @param[out]    Bp    packed band buffer, (b+1)×n column-major
 * @param[in]     n     matrix dimension
 * @param[in]     b     bandwidth
 */
template <typename T> void bc_pack(SolverHandle<T> *ws, const T *B, T *Bp, int n, int b);

/**
 * @brief GPU bulge chasing: band → tridiagonal, stores Householder vectors U.
 *
 * Launches n−2 thread blocks (one per sweep). Spin-lock flag array enforces
 * the dependency: sweep i+1 waits until sweep i has completed ≥3 bulges ahead.
 *
 * @tparam T      float or double
 * @param[in]     ws    solver handle (stream)
 * @param[in,out] Bp    packed band matrix (bc_pack output); overwritten with tridiagonal
 * @param[out]    d     diagonal of tridiagonal, length n
 * @param[out]    e     sub-diagonal of tridiagonal, length n−1
 * @param[out]    U     Householder vectors for BC-Back, n×(n−2) column-major
 * @param[in]     n     matrix dimension
 * @param[in]     b     bandwidth
 */
template <typename T> void bc_chase(SolverHandle<T> *ws, T *Bp, T *d, T *e, T *U, int n, int b);

// --- BT (back-transform) -----------------------------------------------------

/**
 * @brief SBR-Back: apply accumulated WY reflectors to Q. Q ← (I − W·Yᵀ)·Q
 *
 * Recursive WY strategy: combines b-wide W blocks into larger k-wide blocks
 * (k ≫ b) to get square GEMMs, then applies via ormqr-style update.
 *
 * @tparam T      float or double
 * @param[in]     ws    solver handle
 * @param[in,out] Q     n×n orthogonal matrix, column-major; updated in place
 * @param[in]     W     SBR W-blocks, n×(n/b · b) column-major
 * @param[in]     Y     SBR Y-blocks, n×(n/b · b) column-major
 * @param[in]     n     matrix dimension
 * @param[in]     b     bandwidth used in DBBR
 * @param[in]     k     outer panel size used in DBBR
 */
template <typename T>
void bt_sbr_back(SolverHandle<T> *ws, T *Q, const T *W, const T *Y, int n, int b, int k);

/**
 * @brief BC-Back: apply BC Householder vectors U to Q. Q ← Q_b · Q
 *
 * BLAS2 kernel: applies each u-vector directly. uGroups staged in shared
 * memory; columns of Q kept in registers; transposed-u layout avoids bank conflicts.
 *
 * @tparam T      float or double
 * @param[in]     ws    solver handle (stream)
 * @param[in,out] Q     n×n matrix (from SBR-Back), column-major
 * @param[in]     U     BC Householder vectors from bc_chase, n×(n−2) column-major
 * @param[in]     n     matrix dimension
 * @param[in]     b     bandwidth
 */
template <typename T> void bt_bc_back(SolverHandle<T> *ws, T *Q, const T *U, int n, int b);

} // namespace kernels

// =============================================================================
// cuBLAS dispatching wrappers  (cuev::cublas)
// All functions take SolverHandle<T>* and use ws->cublas internally.
// =============================================================================
namespace cublas {

/**
 * @brief General matrix-matrix multiplication (GEMM). C ← α·op(A)·op(B) + β·C
 *
 * op(X) = X, Xᵀ, or Xᴴ depending on the value of transX.  C, A, B are column-major.
 *
 * @tparam T       float or double
 * @param[in] ws       solver handle
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
void gemm(SolverHandle<T> *ws, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C,
          int ldc);

/**
 * @brief General matrix addition: C ← α·op(A) + β·op(B)
 *
 * op(X) = X or Xᵀ depending on the value of transX.  C, A, B are column-major.
 *
 * @tparam T       float or double
 * @param[in] ws       solver handle
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
void geam(SolverHandle<T> *ws, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          const T *alpha, const T *A, int lda, const T *beta, const T *B, int ldb, T *C, int ldc);

/**
 * @brief Symmetric matrix-matrix multiplication: C ← α·A·B + β·C (or B·A if side=right)
 *
 * A is symmetric; only the @p uplo triangle is referenced. B and C are column-major.
 *
 * @tparam T       float or double
 * @param[in] ws        solver handle
 * @param[in] side      whether A multiplies B from the left or right
 * @param[in] uplo      which triangle of A is referenced
 * @param[in] m         rows of B and C
 * @param[in] n         columns of B and C
 * @param[in] alpha     scalar multiplier for A·B
 * @param[in] A         symmetric matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] B         matrix B, column-major, leading dimension ldb
 * @param[in] ldb       leading dimension of B (≥ rows of B)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
template <typename T>
void symm(SolverHandle<T> *ws, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

/**
 * @brief Symmetric rank-k update: C ← α·op(A)·op(A)ᵀ + β·C
 *
 * Only the @p uplo triangle of C is referenced/written.
 *
 * @tparam T       float or double
 * @param[in] ws        solver handle
 * @param[in] uplo      which triangle of C is referenced (and where the result is written)
 * @param[in] trans     how to interpret A (op(A))
 * @param[in] n         rows and columns of C
 * @param[in] k         columns of op(A)
 * @param[in] alpha     scalar multiplier for op(A)·op(A)ᵀ
 * @param[in] A         matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
template <typename T>
void syrk(SolverHandle<T> *ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
          const T *alpha, const T *A, int lda, const T *beta, T *C, int ldc);

/**
 * @brief Symmetric rank-2k update: C ← α·op(A)·op(B)ᵀ + α·op(B)·op(A)ᵀ + β·C
 *
 * Only the @p uplo triangle of C is referenced/written.
 * Used in DBBR trailing update: C = A, op(A)=Z, op(B)=Y, α=-1, β=1.
 *
 * @tparam T       float or double
 * @param[in] ws        solver handle
 * @param[in] uplo      which triangle of C is referenced (and where the result is written)
 * @param[in] trans     how to interpret A and B (op(A), op(B))
 * @param[in] n         rows and columns of C
 * @param[in] k         columns of op(A) and op(B)
 * @param[in] alpha     scalar multiplier
 * @param[in] A         matrix A, column-major, leading dimension lda
 * @param[in] lda       leading dimension of A (≥ rows of A)
 * @param[in] B         matrix B, column-major, leading dimension ldb
 * @param[in] ldb       leading dimension of B (≥ rows of B)
 * @param[in] beta      scalar multiplier for C
 * @param[in,out] C     matrix C, column-major, leading dimension ldc; overwritten with the result
 * @param[in] ldc       leading dimension of C (≥ rows of C)
 */
template <typename T>
void syr2k(SolverHandle<T> *ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
           const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc);

/**
 * @brief Triangular solve: B ← α·op(A)⁻¹·B  (or B ← α·B·op(A)⁻¹ if side = right)
 *
 * @tparam T       float or double
 * @param[in] ws        solver handle
 * @param[in] side      whether A multiplies B from the left or right
 * @param[in] uplo      which triangle of A is referenced
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
void trsm(SolverHandle<T> *ws, cublasSideMode_t side, cublasFillMode_t uplo,
          cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const T *alpha, const T *A,
          int lda, T *B, int ldb);

/**
 * @brief Vector scaling: x ← α·x
 *
 * @tparam T       float or double
 * @param[in] ws        solver handle
 * @param[in] n         length of x
 * @param[in] alpha     scalar multiplier for x
 * @param[in,out] x     vector x; overwritten with the result
 * @param[in] incx      stride of x (≥ 1)
 */
template <typename T> void scal(SolverHandle<T> *ws, int n, const T *alpha, T *x, int incx);

/**
 * @brief Vector copy: y ← x
 *
 * @tparam T       float or double
 * @param[in] ws        solver handle
 * @param[in] n         length of x and y
 * @param[in] x         vector x; not modified
 * @param[in] incx      stride of x (≥ 1)
 * @param[in,out] y     vector y; overwritten with the result
 * @param[in] incy      stride of y (≥ 1)
 */
template <typename T> void copy(SolverHandle<T> *ws, int n, const T *x, int incx, T *y, int incy);

/**
 * @brief Vector 2-norm: result = ‖x‖₂
 *
 * @tparam T       float or double
 * @param[in] ws        solver handle
 * @param[in] n         length of x
 * @param[in] x         vector x
 * @param[in] incx      stride of x (≥ 1)
 * @param[out] result   pointer to the result on the device
 */
template <typename T> void nrm2(SolverHandle<T> *ws, int n, const T *x, int incx, T *result);

} // namespace cublas

// =============================================================================
// cuSOLVER type-dispatching wrappers  (cuev::cusolver)
// All functions take SolverHandle<T>* and extract handles/scratch internally.
// =============================================================================
namespace cusolver {

/**
 * @brief QR factorisation: A ← compact(Q·R), tau ← Householder scalars.
 *
 * @tparam T         float or double
 * @param[in]     ws      solver handle (cusolver, geqrf_buf, d_info)
 * @param[in]     m       number of rows
 * @param[in]     n       number of columns
 * @param[in,out] A       m×n matrix, column-major; overwritten with compact QR
 * @param[in]     lda     leading dimension of A
 * @param[out]    tau     Householder scalars, length min(m,n)
 * @param[in]     stream  CUDA stream
 */
template <typename T>
void geqrf(SolverHandle<T> *ws, int m, int n, T *A, int lda, T *tau, cudaStream_t stream);

/**
 * @brief Materialise Q from compact QR representation: A ← first n columns of Q.
 *
 * @tparam T         float or double
 * @param[in]     ws      solver handle (cusolver, orgqr_buf, d_info)
 * @param[in]     m       rows of Q
 * @param[in]     n       columns to generate (≤ m)
 * @param[in]     k       Householder reflector count (= min(m,n) from geqrf)
 * @param[in,out] A       m×n, column-major; compact QR in → first n cols of Q out
 * @param[in]     lda     leading dimension
 * @param[in]     tau     Householder scalars from geqrf, length k
 * @param[in]     stream  CUDA stream
 */
template <typename T>
void orgqr(SolverHandle<T> *ws, int m, int n, int k, T *A, int lda, const T *tau,
           cudaStream_t stream);

/**
 * @brief Symmetric dense eigensolver (D&C): A v = λ v.
 *
 * A is overwritten with eigenvectors (columns, ascending eigenvalue order).
 * W receives eigenvalues in ascending order.
 *
 * @tparam T         float or double
 * @param[in]     ws      solver handle (cusolver, syevd_buf, d_info)
 * @param[in]     n       matrix dimension
 * @param[in,out] A       n×n symmetric, column-major; overwritten with eigenvectors
 * @param[in]     lda     leading dimension
 * @param[out]    W       eigenvalues ascending, length n
 * @param[in]     stream  CUDA stream
 */
template <typename T>
void syevd(SolverHandle<T> *ws, int n, T *A, int lda, T *W, cudaStream_t stream);

} // namespace cusolver

} // namespace cuev
