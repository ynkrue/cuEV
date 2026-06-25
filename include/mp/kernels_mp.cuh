/**
 * @file   kernels_mp.cuh
 * @brief  Distributed kernel launchers and cuBLASMp/cuSOLVERMp wrappers for cuEV.
 *
 * Three sections:
 *
 *   cuev::mp::kernels     Distributed custom GPU kernel launchers (dbbr_*, bc_*, bt_*)
 *
 *   cuev::mp::cublasmp    Type-dispatching wrappers for cuBLASMp:
 *                         gemm, geadd, syrk, syr2k, trsm, gemr2d
 *
 *   cuev::mp::cusolvermp  Wrappers for cuSOLVERMp using WorkspaceMp pre-allocated scratch:
 *                         geqrf, ormqr, syevd
 *
 * All matrices are column-major, 2D block-cyclic (BLACS-compatible, rsrc=csrc=0).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#ifdef CUEV_ENABLE_MP

#include "common.h"
#include "mp/comm.h"
#include "mp/handle_mp.h"
#include <cstdint>
#include <cublas_v2.h>

namespace cuev {
namespace mp {

/// cuBLAS compute type selector for T.
template <typename T> inline cublasComputeType_t compute_type() {
    if constexpr (std::is_same_v<T, float>)
        return CUBLAS_COMPUTE_32F;
    else
        return CUBLAS_COMPUTE_64F;
}

// =============================================================================
// cuev::mp::cublasmp — cuBLASMp type-dispatching wrappers
// =============================================================================
namespace cublasmp {

/**
 * @brief C ← α·op(A)·op(B) + β·C  (distributed GEMM).
 *
 * @tparam T         float or double
 * @param[in]  ctx   distributed context
 * @param[in]  transa, transb  transpose flags for A and B
 * @param[in]  m,n,k  global matrix dimensions
 * @param[in]  alpha  scalar α
 * @param[in]  A      local device pointer for A; ia,ja are 1-indexed submatrix offsets
 * @param[in]  descA  cuBLASMp matrix descriptor for A
 * @param[in]  B      local device pointer for B; ib,jb are 1-indexed submatrix offsets
 * @param[in]  descB  cuBLASMp matrix descriptor for B
 * @param[in]  beta   scalar β
 * @param[in,out] C   local device pointer for C; ic,jc are 1-indexed submatrix offsets
 * @param[in]  descC  cuBLASMp matrix descriptor for C
 */
template <typename T>
void gemm(Context &ctx, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n,
          int64_t k, const T *alpha, const T *A, int64_t ia, int64_t ja,
          cublasMpMatrixDescriptor_t descA, const T *B, int64_t ib, int64_t jb,
          cublasMpMatrixDescriptor_t descB, const T *beta, T *C, int64_t ic, int64_t jc,
          cublasMpMatrixDescriptor_t descC);

/**
 * @brief C ← α·op(A) + β·C  (distributed Geadd, replaces single-GPU geam).
 *
 * @tparam T      float or double
 * @param[in]  ctx   distributed context
 * @param[in]  trans CUBLAS_OP_N or CUBLAS_OP_T applied to A
 * @param[in]  m,n   global dimensions of C
 * @param[in]  alpha scalar α
 * @param[in]  A     local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[in]  beta  scalar β
 * @param[in,out] C  local device pointer; ic,jc 1-indexed; descC descriptor
 */
template <typename T>
void geadd(Context &ctx, cublasOperation_t trans, int64_t m, int64_t n, const T *alpha, const T *A,
           int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const T *beta, T *C,
           int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC);

/**
 * @brief C ← α·op(A)·op(A)ᵀ + β·C  (rank-k update; only @p uplo triangle written).
 *
 * @tparam T      float or double
 * @param[in]  ctx   distributed context
 * @param[in]  uplo  which triangle of C to write
 * @param[in]  trans CUBLAS_OP_N: C+=A·Aᵀ; CUBLAS_OP_T: C+=Aᵀ·A
 * @param[in]  n,k   C is n×n; A is n×k (OP_N) or k×n (OP_T)
 * @param[in]  alpha scalar α
 * @param[in]  A     local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[in]  beta  scalar β
 * @param[in,out] C  local device pointer; ic,jc 1-indexed; descC descriptor
 */
template <typename T>
void syrk(Context &ctx, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k,
          const T *alpha, const T *A, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA,
          const T *beta, T *C, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC);

/**
 * @brief C ← α·op(A)·op(B)ᵀ + α·op(B)·op(A)ᵀ + β·C  (rank-2k; only @p uplo written).
 *
 * Used in distributed DBBR trailing update: C=A, A=Z, B=Y, α=-1, β=1.
 *
 * @tparam T      float or double
 * @param[in]  ctx   distributed context
 * @param[in]  uplo  which triangle of C to write
 * @param[in]  trans CUBLAS_OP_N or CUBLAS_OP_T applied to A and B
 * @param[in]  n,k   C is n×n; A and B are n×k (OP_N) or k×n (OP_T)
 * @param[in]  alpha scalar α
 * @param[in]  A     local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[in]  B     local device pointer; ib,jb 1-indexed; descB descriptor
 * @param[in]  beta  scalar β
 * @param[in,out] C  local device pointer; ic,jc 1-indexed; descC descriptor
 */
template <typename T>
void syr2k(Context &ctx, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k,
           const T *alpha, const T *A, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA,
           const T *B, int64_t ib, int64_t jb, cublasMpMatrixDescriptor_t descB, const T *beta,
           T *C, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC);

/**
 * @brief Triangular solve: B ← α·op(A)⁻¹·B  or  B ← α·B·op(A)⁻¹.
 *
 * @tparam T      float or double
 * @param[in]  ctx   distributed context
 * @param[in]  side  which side A is on
 * @param[in]  uplo  which triangle of A holds the factor
 * @param[in]  trans transpose flag for A
 * @param[in]  diag  unit or non-unit diagonal
 * @param[in]  m,n   global dimensions of B
 * @param[in]  alpha scalar α
 * @param[in]  A     local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[in,out] B  local device pointer; ib,jb 1-indexed; descB descriptor
 */
template <typename T>
void trsm(Context &ctx, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
          cublasDiagType_t diag, int64_t m, int64_t n, const T *alpha, const T *A, int64_t ia,
          int64_t ja, cublasMpMatrixDescriptor_t descA, T *B, int64_t ib, int64_t jb,
          cublasMpMatrixDescriptor_t descB);

/**
 * @brief Redistribute a submatrix: B[ib:ib+m, jb:jb+n] ← A[ia:ia+m, ja:ja+n]  (PXGEMR2D).
 *
 * Accepts non-block-aligned offsets. Used to scatter eigenvector blocks (Q_d)
 * back into the full distributed evec matrix after the D&C solve.
 *
 * @tparam T      float or double
 * @param[in]  ctx   distributed context (handle + CAL comm)
 * @param[in]  m,n   submatrix dimensions to copy
 * @param[in]  A     source local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[out] B     destination local device pointer; ib,jb 1-indexed; descB descriptor
 */
template <typename T>
void gemr2d(Context &ctx, int64_t m, int64_t n, const T *A, int64_t ia, int64_t ja,
            cublasMpMatrixDescriptor_t descA, T *B, int64_t ib, int64_t jb,
            cublasMpMatrixDescriptor_t descB);

} // namespace cublasmp

// =============================================================================
// cuev::mp::cusolvermp — cuSOLVERMp wrappers
// =============================================================================
namespace cusolvermp {

/**
 * @brief QR factorisation: A ← compact(Q·R), tau ← Householder scalars.
 *
 * Uses ws.geqrf_dwork / ws.geqrf_wsD. Followed by CAL barrier + info check.
 *
 * @tparam T      float or double
 * @param[in]     ctx    distributed context
 * @param[in]     m,n    global dimensions of A
 * @param[in,out] A      local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[out]    tau    Householder scalars, local portion (column-distributed)
 * @param[in,out] ws     workspace (geqrf_dwork, h_work, d_info)
 */
template <typename T>
void geqrf(Context &ctx, int64_t m, int64_t n, T *A, int64_t ia, int64_t ja,
           cusolverMpMatrixDescriptor_t descA, T *tau, WorkspaceMp<T> &ws);

/**
 * @brief Apply Q from a prior Geqrf to C: C ← op(Q)·C  (side=left).
 *
 * @tparam T      float or double
 * @param[in]     ctx    distributed context
 * @param[in]     side   must be CUBLAS_SIDE_LEFT
 * @param[in]     trans  CUBLAS_OP_N applies Q; CUBLAS_OP_T applies Qᵀ
 * @param[in]     m,n    global dimensions of C
 * @param[in]     k      number of Householder reflectors (= min(m,n) from Geqrf)
 * @param[in]     A      compact QR from Geqrf; ia,ja 1-indexed; descA descriptor
 * @param[in]     tau    Householder scalars from Geqrf
 * @param[in,out] C      local device pointer; ic,jc 1-indexed; descC descriptor
 * @param[in,out] ws     workspace (h_work, d_info)
 */
template <typename T>
void ormqr(Context &ctx, cublasSideMode_t side, cublasOperation_t trans, int64_t m, int64_t n,
           int64_t k, const T *A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
           const T *tau, T *C, int64_t ic, int64_t jc, cusolverMpMatrixDescriptor_t descC,
           WorkspaceMp<T> &ws);

/**
 * @brief Symmetric dense eigensolver: A v = λv  (tridiagonal D&C base case).
 *
 * A is overwritten with eigenvectors (columns, ascending order).
 * W receives eigenvalues ascending, replicated on all ranks.
 *
 * @tparam T      float or double
 * @param[in]     ctx    distributed context
 * @param[in]     n      global dimension
 * @param[in,out] A      local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[out]    W      eigenvalues, length n, device pointer (all ranks)
 * @param[out]    Z      eigenvectors output; iz,jz 1-indexed; descZ descriptor
 * @param[in,out] ws     workspace (h_work, d_info; syevd scratch self-allocated)
 */
template <typename T>
void syevd(Context &ctx, int64_t n, T *A, int64_t ia, int64_t ja,
           cusolverMpMatrixDescriptor_t descA, T *W, T *Z, int64_t iz, int64_t jz,
           cusolverMpMatrixDescriptor_t descZ, WorkspaceMp<T> &ws);

} // namespace cusolvermp

// =============================================================================
// cuev::mp::kernels — distributed custom GPU kernel launchers (Phase 2 stubs)
// =============================================================================
namespace kernels {

// --- DBBR_MP -----------------------------------------------------------------

/**
 * @brief Distributed DBBR panel QR on local column tiles.
 *
 * Performs the panel QR for one b-column panel of the distributed band reduction.
 * No inter-rank communication — each rank QRs its own local rows.
 *
 * @tparam T      float or double
 * @param[in]     ctx   distributed context
 * @param[in,out] A     distributed matrix; panel column j, rows j..n-1
 * @param[out]    tau   Householder scalars, local portion, length b
 * @param[in]     j     global starting column of this panel
 * @param[in]     b     panel width (bandwidth)
 * @param[in,out] ws    workspace
 */
template <typename T>
void dbbr_panel_qr_mp(Context &ctx, DistMatrix<T> &A, T *tau, int64_t j, int64_t b,
                      WorkspaceMp<T> &ws);

/**
 * @brief Distributed DBBR trailing syr2k update: A ← A − Z·Yᵀ − Y·Zᵀ
 *
 * Deferred trailing update applied every k columns. Uses cublasmp::syr2k
 * (or custom distributed kernel for large n).
 *
 * @tparam T      float or double
 * @param[in]     ctx   distributed context
 * @param[in,out] A     distributed n×n symmetric matrix
 * @param[in]     Z     distributed n×k accumulated Z-block
 * @param[in]     Y     distributed n×k accumulated Y-block
 * @param[in,out] ws    workspace
 */
template <typename T>
void dbbr_syr2k_mp(Context &ctx, DistMatrix<T> &A, const DistMatrix<T> &Z, const DistMatrix<T> &Y,
                   WorkspaceMp<T> &ws);

// --- BC_MP -------------------------------------------------------------------

/**
 * @brief Distributed GPU bulge chasing: band → tridiagonal.
 *
 * Novel Phase 2 contribution. The band matrix is O(n·b) data — gathered to a
 * grid-row-local layout for the GPU sweep kernel, then results redistributed.
 * Stores Householder vectors U for distributed BC-Back.
 *
 * @tparam T      float or double
 * @param[in]     ctx   distributed context
 * @param[in,out] B     distributed band matrix (DBBR output); tridiagonal on exit
 * @param[out]    d     diagonal of tridiagonal, length n (replicated)
 * @param[out]    e     sub-diagonal of tridiagonal, length n−1 (replicated)
 * @param[out]    U     distributed BC Householder vectors for BC-Back
 * @param[in]     b     bandwidth
 * @param[in,out] ws    workspace
 */
template <typename T>
void bc_chase_mp(Context &ctx, DistMatrix<T> &B, T *d, T *e, DistMatrix<T> &U, int64_t b,
                 WorkspaceMp<T> &ws);

// --- BT_MP -------------------------------------------------------------------

/**
 * @brief Distributed SBR-Back: Q ← (I − W·Yᵀ)·Q using recursive WY.
 *
 * @tparam T      float or double
 * @param[in]     ctx   distributed context
 * @param[in,out] Q     distributed n×n orthogonal matrix; updated in place
 * @param[in]     W     distributed SBR W-blocks
 * @param[in]     Y     distributed SBR Y-blocks
 * @param[in]     b     bandwidth used in DBBR
 * @param[in]     k     outer panel size used in DBBR
 * @param[in,out] ws    workspace
 */
template <typename T>
void bt_sbr_back_mp(Context &ctx, DistMatrix<T> &Q, const DistMatrix<T> &W, const DistMatrix<T> &Y,
                    int64_t b, int64_t k, WorkspaceMp<T> &ws);

/**
 * @brief Distributed BC-Back: Q ← Q_b · Q using BLAS2 Householder application.
 *
 * @tparam T      float or double
 * @param[in]     ctx   distributed context
 * @param[in,out] Q     distributed n×n matrix (from SBR-Back)
 * @param[in]     U     distributed BC Householder vectors from bc_chase_mp
 * @param[in]     b     bandwidth
 * @param[in,out] ws    workspace
 */
template <typename T>
void bt_bc_back_mp(Context &ctx, DistMatrix<T> &Q, const DistMatrix<T> &U, int64_t b,
                   WorkspaceMp<T> &ws);

} // namespace kernels
} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
