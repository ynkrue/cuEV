/**
 * @file   kernels.cuh
 * @brief  Public kernel interface for cuEV — all launcher declarations.
 *
 * Single include for every device-side launcher in the project.
 * Naming convention:
 *
 *   __global__ <op>_<variant>_kernel   device function, defined in its .cu file
 *   <op>_<variant>                     host launcher, defined in its .cu file
 *   <op>                               default dispatcher, inlined here
 *
 * Template parameter @p T is float or double throughout.
 * All matrices are row-major unless documented otherwise.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cuda_runtime.h>

namespace cuev {
namespace kernels {

// =============================================================================
// Utility
// =============================================================================

/**
 * @brief Fill a vector with a scalar: x ← α.
 *
 * @tparam T      float or double
 * @param[in]     alpha  fill value
 * @param[out]    x      device vector, length N
 * @param[in]     N      number of elements
 * @param[in]     stream CUDA stream
 */
template <typename T> void fill(T alpha, T *x, int N, cudaStream_t stream);

/**
 * @brief Copy a vector: y ← x.
 *
 * @tparam T      float or double
 * @param[in]     x      source device vector, length N
 * @param[out]    y      destination device vector, length N
 * @param[in]     N      number of elements
 * @param[in]     stream CUDA stream
 */
template <typename T> void copy(const T *x, T *y, int N, cudaStream_t stream);

/**
 * @brief Transpose a matrix: Aᵀ ← A.
 *
 * @tparam T      float or double
 * @param[in]     A      M×N input matrix, row-major
 * @param[out]    AT     N×M output matrix, row-major
 * @param[in]     M      number of rows of A
 * @param[in]     N      number of columns of A
 * @param[in]     stream CUDA stream
 */
template <typename T> void transpose(const T *A, T *AT, int M, int N, cudaStream_t stream);

// =============================================================================
// GEMV  —  y ← αAx + βy
// =============================================================================

/**
 * @brief General matrix-vector product.
 *
 * Inlined dispatcher to the best available variant.
 *
 * @tparam T      float or double
 * @param[in]     alpha  scalar α
 * @param[in]     A      M×N matrix, row-major
 * @param[in]     x      input vector, length N
 * @param[in]     beta   scalar β
 * @param[in,out] y      output vector, length M
 * @param[in]     M      number of rows
 * @param[in]     N      number of columns
 * @param[in]     stream CUDA stream
 */
template <typename T>
inline void gemv(T alpha, const T *A, const T *x, T beta, T *y, int M, int N, cudaStream_t stream) {
    gemv_gmem(alpha, A, x, beta, y, M, N, stream);
}

/// gmem variant — one thread per output element
template <typename T>
void gemv_gmem(T alpha, const T *A, const T *x, T beta, T *y, int M, int N, cudaStream_t stream);

/// smem variant — one block per row, shared-memory reduction
template <typename T>
void gemv_smem(T alpha, const T *A, const T *x, T beta, T *y, int M, int N, cudaStream_t stream);

// =============================================================================
// GEMM  —  C ← αAB + βC
// =============================================================================

/**
 * @brief General matrix-matrix product.
 *
 * Inlined dispatcher to the best available variant.
 *
 * @tparam T      float or double
 * @param[in]     alpha  scalar α
 * @param[in]     A      M×K matrix, row-major
 * @param[in]     B      K×N matrix, row-major
 * @param[in]     beta   scalar β
 * @param[in,out] C      M×N matrix, row-major
 * @param[in]     M      rows of A and C
 * @param[in]     N      columns of B and C
 * @param[in]     K      columns of A / rows of B
 * @param[in]     stream CUDA stream
 */
template <typename T>
inline void gemm(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
                 cudaStream_t stream) {
    gemm_warptile(alpha, A, B, beta, C, M, N, K, stream);
}

/// gmem variant — one thread per output element
template <typename T>
void gemm_gmem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
               cudaStream_t stream);

/// smem variant — shared-memory tiled, one block per output tile
template <typename T>
void gemm_smem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
               cudaStream_t stream);

/// register-tiled variant — each thread accumulates a TM×TN output tile
template <typename T>
void gemm_tiled(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
                cudaStream_t stream);

/// warp-tiled variant — 128-bit vectorized loads, sA transposed in smem for coalesced reads
template <typename T>
void gemm_warptile(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
                   cudaStream_t stream);

// =============================================================================
// Householder transforms
// =============================================================================

/**
 * @brief Compute Householder projection Pₖ = I − τvvᵀ for column k of H.
 *
 * Reads H[k+1:n, k], computes v and τ such that Pₖ·H[k+1:n, k] = α e₁.
 * Stores v into H[k+2:n, k] (in-place), τ into tau[k],
 * α into e[k], and H[k,k] into d[k].
 *
 * @tparam T      float or double
 * @param[in,out] H      n×n symmetric matrix, row-major, modified in place
 * @param[out]    v      Householder vector, length n−k−1
 * @param[out]    tau    Householder scalars, length n−1
 * @param[out]    d      diagonal elements of T being built, length n
 * @param[out]    e      subdiagonal elements of T being built, length n−1
 * @param[in]     N      matrix dimension n
 * @param[in]     k      current step index
 * @param[in]     stream CUDA stream
 */
template <typename T>
void hh_reflect(T *H, T *v, T *tau, T *d, T *e, int N, int k, cudaStream_t stream);

/**
 * @brief Trailing submatrix GEMV: p ← H[k+1:n, k+1:n] · v.
 *
 * @tparam T      float or double
 * @param[in]     v      Householder vector, length n−k−1
 * @param[in]     H      n×n symmetric matrix, row-major
 * @param[out]    p      result vector, length n−k−1
 * @param[in]     N      matrix dimension n
 * @param[in]     k      current step index
 * @param[in]     stream CUDA stream
 */
template <typename T>
void hh_trail_matvec(const T *v, const T *H, T *p, int N, int k, cudaStream_t stream);

/**
 * @brief Rank-2 update prefactor: u ← τp − ½τ²(vᵀp)v.
 *
 * @tparam T      float or double
 * @param[in]     v      Householder vector, length n−k−1
 * @param[in]     p      result of H·v, length n−k−1
 * @param[in]     tau    Householder scalars, length n−1
 * @param[out]    u      update vector, length n−k−1
 * @param[in]     N      matrix dimension n
 * @param[in]     k      current step index
 * @param[in]     stream CUDA stream
 */
template <typename T>
void hh_ortho(const T *v, const T *p, const T *tau, T *u, int N, int k, cudaStream_t stream);

/**
 * @brief Symmetric rank-2 update: H ← H − vuᵀ − uvᵀ.
 *
 * Applied to the trailing submatrix H[k+1:n, k+1:n], completing step k
 * of the reduction H ← PₖᵀHPₖ.
 *
 * @tparam T      float or double
 * @param[in]     v      Householder vector, length n−k−1
 * @param[in]     u      update vector from hh_ortho, length n−k−1
 * @param[in,out] H      n×n symmetric matrix, row-major, modified in place
 * @param[in]     N      matrix dimension n
 * @param[in]     k      current step index
 * @param[in]     stream CUDA stream
 */
template <typename T>
void hh_update(const T *v, const T *u, T *H, int N, int k, cudaStream_t stream);

/**
 * @brief Accumulate compact WY factor: Tf ← triangular factor from V and τ.
 *
 * Blocked Householder: builds the K×K upper-triangular Tf such that
 *   H₁H₂···H_K = I − V·Tf·Vᵀ.
 *
 * @tparam T      float or double
 * @param[in]     V      M×K Householder vector matrix
 * @param[in]     tau    Householder scalars, length K
 * @param[out]    Tf     K×K upper-triangular WY factor
 * @param[in]     M      number of rows of V
 * @param[in]     K      number of Householder vectors
 * @param[in]     stream CUDA stream
 */
template <typename T>
void hh_hh_wy_build(const T *V, const T *tau, T *Tf, int M, int K, cudaStream_t stream);

/**
 * @brief Apply block Householder reflector: C ← (I − V·Tf·Vᵀ) C.
 *
 * @tparam T      float or double
 * @param[in]     V      M×K Householder vector matrix
 * @param[in]     Tf     K×K upper-triangular WY factor (from hh_hh_wy_build)
 * @param[in,out] C      M×N matrix, modified in place
 * @param[in]     M      rows of C
 * @param[in]     N      columns of C
 * @param[in]     K      number of Householder vectors
 * @param[in]     stream CUDA stream
 */
template <typename T>
void hh_hh_wy_apply(const T *V, const T *Tf, T *C, int M, int N, int K, cudaStream_t stream);

// =============================================================================
// Eigensolver primitives
// =============================================================================

/**
 * @brief Solve a 2×2 symmetric tridiagonal eigenproblem.
 *
 * Given diagonal d[0:2] and subdiagonal e[0], computes eigenvalues
 * eval[0:2] in ascending order and the 2×2 orthogonal eigenvector matrix QT
 * (row-major, rows = eigenvectors).
 *
 * @tparam T      float or double
 * @param[in]     d      diagonal, length 2
 * @param[in]     e      subdiagonal, length 1
 * @param[out]    eval   eigenvalues, length 2, ascending order
 * @param[out]    QT     2×2 eigenvector matrix, row-major
 * @param[in]     stream CUDA stream
 */
template <typename T> void eig_leaf(const T *d, const T *e, T *eval, T *QT, cudaStream_t stream);

/**
 * @brief Split a symmetric tridiagonal into two decoupled halves.
 *
 * Produces d₁ = d[0:k] and d₂ = d[k:n] with the coupling element e[k-1]
 * absorbed into both corners, so that
 *   T = diag(T₁, T₂) + e[k-1]·vvᵀ,  v = [0…0, 1, 1, 0…0].
 *
 * @tparam T      float or double
 * @param[in]     d      diagonal of T, length n
 * @param[in]     e      subdiagonal of T, length n−1
 * @param[in]     n      dimension of T
 * @param[in]     k      split point: first half is [0,k), second is [k,n)
 * @param[out]    d1     diagonal of T₁, length k
 * @param[out]    d2     diagonal of T₂, length n−k
 * @param[in]     stream CUDA stream
 */
template <typename T>
void eig_split(const T *d, const T *e, int n, int k, T *d1, T *d2, cudaStream_t stream);

/**
 * @brief Merge two sorted eigenvalue arrays into one sorted array.
 *
 * Standard two-way merge of eval1[0:k] and eval2[0:n-k], both in ascending
 * order, into eval[0:n].  Required before eig_secular so the combined
 * diagonal d is globally sorted.
 *
 * @tparam T      float or double
 * @param[in]     eval1  eigenvalues of T₁, length k, ascending
 * @param[in]     eval2  eigenvalues of T₂, length n−k, ascending
 * @param[in]     n      total number of eigenvalues
 * @param[in]     k      split point: size of first half
 * @param[out]    eval   merged eigenvalues, length n, ascending
 * @param[in]     stream CUDA stream
 */
template <typename T>
void eig_sort(const T *eval1, const T *eval2, int n, int k, T *eval, cudaStream_t stream);

/**
 * @brief Solve the secular equation and compute eigenvectors of the rank-1 problem.
 *
 * Finds all n eigenvalues and eigenvectors of the rank-1 modified problem
 *   M = diag(d) + β·zzᵀ,   z[i] = Q1[k−1, i]  (i < k)
 *                                   Q2[0,   i−k] (i ≥ k)
 *
 * Each eigenvalue satisfies the secular equation
 *   f(λ) = 1 + β·Σᵢ zᵢ²/(dᵢ−λ) = 0
 * with exactly one root per interval.  The poles dᵢ are kept paired with zᵢ
 * via the unsorted split: dᵢ = eval1[i] for i < k, eval2[i−k] for i ≥ k.
 * The sorted merge eval_sorted (output of eig_sort) provides interval bounds.
 *
 * @tparam T           float or double
 * @param[in]  eval1   eigenvalues of T₁, length k, ascending (poles for i < k)
 * @param[in]  eval2   eigenvalues of T₂, length n−k, ascending (poles for i ≥ k)
 * @param[in]  eval_sorted  sorted merge of eval1 and eval2, length n (from eig_sort)
 * @param[in]  Q1      k×k eigenvector matrix of T₁, row-major (rows = eigenvectors)
 * @param[in]  Q2      (n−k)×(n−k) eigenvector matrix of T₂, row-major (rows = eigenvectors)
 * @param[in]  e       subdiagonal of T, length n−1 (e[k−1] = β)
 * @param[in]  n       total dimension
 * @param[in]  k       split point
 * @param[out] eval    secular eigenvalues, length n, ascending
 * @param[out] G       n×n eigenvector matrix of M, row-major (rows = eigenvectors)
 * @param[in]  stream  CUDA stream
 */
template <typename T>
void eig_secular(const T *eval1, const T *eval2, const T *eval_sorted, const T *Q1, const T *Q2,
                 const T *e, int n, int k, T *eval, T *G, cudaStream_t stream);

/**
 * @brief Back-transform eigenvectors: QT ← G · diag(Q1, Q2).
 *
 * Combines the sub-problem eigenvector matrices Q1 and Q2 with the secular
 * eigenvector matrix G to produce the full eigenvector matrix of T.
 * All matrices use rows-as-eigenvectors convention (row-major).
 *
 *   QT = G · diag(Q1, Q2)
 *
 * Implemented via the transpose identity to avoid strided column-block access:
 *   QT^T[0:k, :]   ← Q1^T · G^T[0:k, :]   (k×k     ·  k×n)
 *   QT^T[k:n, :]   ← Q2^T · G^T[k:n, :]   ((n−k)×(n−k) · (n−k)×n)
 *
 * @tparam T      float or double
 * @param[in]     Q1     k×k eigenvector matrix of T₁, row-major, rows = eigenvectors
 * @param[in]     Q2     (n−k)×(n−k) eigenvector matrix of T₂, row-major, rows = eigenvectors
 * @param[in]     G      n×n secular eigenvector matrix, row-major, rows = eigenvectors
 * @param[out]    QT     n×n output eigenvector matrix of T, row-major, rows = eigenvectors
 * @param[in]     n      total dimension
 * @param[in]     k      split point
 * @param[in]     stream CUDA stream
 */
template <typename T>
void eig_gemm(const T *Q1, const T *Q2, const T *G, T *QT, int n, int k, cudaStream_t stream);

} // namespace kernels
} // namespace cuev

// =============================================================================
// Public Solver API
// =============================================================================
namespace cuev {

/**
 * @brief Compute all eigenvalues and eigenvectors of a real symmetric matrix.
 *
 * Reduces H to tridiagonal form via Householder projections, solves the
 * tridiagonal eigenproblem with divide-and-conquer, then applies
 * the back-transformation to recover eigenvectors of H.
 *
 * @tparam T      float or double
 * @param[in,out] H      n×n symmetric matrix, row-major; overwritten during reduction
 * @param[in]     n      matrix dimension
 * @param[out]    eval   eigenvalues in ascending order, length n
 * @param[out]    evec   eigenvectors as rows, n×n row-major
 * @param[in]     stream CUDA stream
 */
template <typename T> void solve(T *H, int n, T *eval, T *evec, cudaStream_t stream);

} // namespace cuev
