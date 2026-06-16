/**
 * @file   kernels_mp.cuh
 * @brief  Distributed CUDA kernel interface and cuBLASMp/cuSOLVERMp wrappers for cuEV.
 *
 * Three sections:
 *
 *   cuev::mp::kernels     Distributed custom GPU kernel launchers:
 *     qdwh_shift_mp       Diagonal shift on a distributed matrix
 *     qdwh_fill_W_mp      Build 2n×n work matrix W = [√c·X ; I] on local tiles
 *     qdwh_fill_C_mp      Fill C = [I_k ; 0] for Ormqr-based Q materialisation
 *     sdc_trace_mp        Local diagonal reduction + NCCL AllReduce → host scalar
 *     qdwh_sign_mp        Distributed QDWH polar iteration
 *     sdc_split_mp        H₁ = Q₁ᵀHQ₁, H₂ = Q₂ᵀHQ₂
 *     sdc_combine_mp      evec = [Q₂·evec₂ | Q₁·evec₁]
 *
 *   cuev::mp::cublasmp    Type-dispatching wrappers for cuBLASMp:
 *                         gemm, geadd, syrk, trsm
 *
 *   cuev::mp::cusolvermp  Wrappers for cuSOLVERMp using WorkspaceMp pre-allocated scratch:
 *                         geqrf, ormqr, potrf, syevd
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
#include "mp/workspace_mp.h"
#include <cstdint>
#include <cublas_v2.h>

namespace cuev {
namespace mp {

/// cudaDataType for T (CUDA_R_32F / CUDA_R_64F).
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
 * Set C = [I_k ; 0_{(m-k)×k}] before calling to materialise the economy Q.
 * Uses ws.ormqr_dwork / ws.ormqr_wsD.
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
 * @param[in,out] ws     workspace (ormqr_dwork, h_work, d_info)
 */
template <typename T>
void ormqr(Context &ctx, cublasSideMode_t side, cublasOperation_t trans, int64_t m, int64_t n,
           int64_t k, const T *A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
           const T *tau, T *C, int64_t ic, int64_t jc, cusolverMpMatrixDescriptor_t descC,
           WorkspaceMp<T> &ws);

/**
 * @brief Cholesky factorisation: A ← factor  (A = Lᵀ·L or L·Lᵀ).
 *
 * Uses ws.potrf_dwork / ws.potrf_wsD.
 *
 * @tparam T      float or double
 * @param[in]     ctx   distributed context
 * @param[in]     uplo  CUBLAS_FILL_MODE_LOWER or UPPER
 * @param[in]     n     global dimension of A
 * @param[in,out] A     local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[in,out] ws    workspace (potrf_dwork, h_work, d_info)
 */
template <typename T>
void potrf(Context &ctx, cublasFillMode_t uplo, int64_t n, T *A, int64_t ia, int64_t ja,
           cusolverMpMatrixDescriptor_t descA, WorkspaceMp<T> &ws);

/**
 * @brief Symmetric dense eigensolver: A v = λv  (base case).
 *
 * A is overwritten with eigenvectors (columns, ascending order).
 * W receives eigenvalues ascending, replicated on all ranks.
 * Allocates its own workspace per call (size depends on n at each leaf).
 *
 * @tparam T      float or double
 * @param[in]     ctx    distributed context
 * @param[in]     n      global dimension
 * @param[in,out] A      local device pointer; ia,ja 1-indexed; descA descriptor
 * @param[out]    W      eigenvalues, length n, device pointer (all ranks)
 * @param[out]    Z      eigenvectors output; iz,jz 1-indexed; descZ descriptor
 * @param[in,out] ws     workspace (h_work, d_info reused; syevd scratch self-allocated)
 */
template <typename T>
void syevd(Context &ctx, int64_t n, T *A, int64_t ia, int64_t ja,
           cusolverMpMatrixDescriptor_t descA, T *W, T *Z, int64_t iz, int64_t jz,
           cusolverMpMatrixDescriptor_t descZ, WorkspaceMp<T> &ws);

} // namespace cusolvermp

// =============================================================================
// cuev::mp::kernels — distributed element-wise kernels and high-level primitives
// =============================================================================
namespace kernels {

/**
 * @brief In-place diagonal shift on a distributed n×n matrix: A[gi,gi] -= mu.
 *
 * Only elements where global row == global col are touched; no communication.
 *
 * @tparam T          float or double
 * @param[in,out] A_local   local tile buffer (lld × local_cols, column-major)
 * @param[in]     mu        shift scalar
 * @param[in]     local_rows  rows this rank stores
 * @param[in]     local_cols  cols this rank stores
 * @param[in]     lld       local leading dimension
 * @param[in]     prow,pcol,nprow,npcol,nb  grid / block-size parameters
 * @param[in]     stream    CUDA stream
 */
template <typename T>
void qdwh_shift_mp(T *A_local, T mu, int64_t local_rows, int64_t local_cols, int64_t lld, int prow,
                   int pcol, int nprow, int npcol, int64_t nb, cudaStream_t stream);

/**
 * @brief Build the 2n×n QDWH work matrix W = [√c·X ; I_n] on local tiles.
 *
 * For global rows 0..n-1:   W[gi,gj] = √c · X[gi,gj].
 * For global rows n..2n-1:  W[gi,gj] = (gi−n == gj) ? 1 : 0.
 *
 * @tparam T        float or double
 * @param[out] W_local  local tile of the 2n×n work matrix (lld_W × lc)
 * @param[in]  X_local  local tile of the n×n input matrix  (lld_X × lc)
 * @param[in]  scale    √c
 * @param[in]  n        half-dimension (W has 2n global rows, n global cols)
 * @param[in]  lld_W    local leading dimension of W
 * @param[in]  lld_X    local leading dimension of X
 * @param[in]  lc       number of local columns (same for W and X)
 * @param[in]  prow,pcol,nprow,npcol,nb  grid / block-size parameters
 * @param[in]  stream   CUDA stream
 */
template <typename T>
void qdwh_fill_W_mp(T *W_local, const T *X_local, T scale, int64_t n, int64_t lld_W, int64_t lld_X,
                    int64_t lc, int prow, int pcol, int nprow, int npcol, int64_t nb,
                    cudaStream_t stream);

/**
 * @brief Fill a distributed m×k matrix with the truncated identity [I_k ; 0].
 *
 * Sets C[gi,gj] = 1 if gi == gj and gi < k, else 0.
 * Called before ormqr to materialise the economy Q.
 *
 * @tparam T        float or double
 * @param[out] C_local  local tile buffer (lld × lc, column-major)
 * @param[in]  m        global rows of C
 * @param[in]  k        global cols of C
 * @param[in]  lld      local leading dimension
 * @param[in]  lc       number of local columns
 * @param[in]  prow,pcol,nprow,npcol,nb  grid / block-size parameters
 * @param[in]  stream   CUDA stream
 */
template <typename T>
void qdwh_fill_C_mp(T *C_local, int64_t m, int64_t k, int64_t lld, int64_t lc, int prow, int pcol,
                    int nprow, int npcol, int64_t nb, cudaStream_t stream);

/**
 * @brief Distributed trace: Σ A[i,i] via local reduction + NCCL AllReduce.
 *
 * @tparam T          float or double
 * @param[in]  ctx         distributed context (NCCL comm, stream)
 * @param[in]  A_local     local tile buffer (lld × local_cols, column-major)
 * @param[in]  local_rows  rows this rank stores
 * @param[in]  local_cols  cols this rank stores
 * @param[in]  lld         local leading dimension
 * @param[in]  prow,pcol,nprow,npcol,nb  grid / block-size parameters
 * @return     host scalar Σ A[i,i] (same value on all ranks after AllReduce)
 */
template <typename T>
T sdc_trace_mp(Context &ctx, const T *A_local, int64_t local_rows, int64_t local_cols, int64_t lld,
               int prow, int pcol, int nprow, int npcol, int64_t nb);

/**
 * @brief Distributed QDWH sign function: B ← sign(B).
 *
 * Iterates until convergence (≤ 6 steps). Each step uses either a QR update
 * or a Cholesky update depending on the conditioning coefficient c.
 *
 * @tparam T     float or double
 * @param[in]     ctx  distributed context
 * @param[in,out] B    n×n matrix in/out; sign(B) on exit
 * @param[in,out] ws   workspace (qdwh_W, qdwh_tau, geqrf/ormqr/potrf scratch)
 */
template <typename T> void qdwh_sign_mp(Context &ctx, DistMatrix<T> &B, WorkspaceMp<T> &ws);

/**
 * @brief Form subproblems H₁ = Q₁ᵀHQ₁ and H₂ = Q₂ᵀHQ₂.
 *
 * @tparam T     float or double
 * @param[in]  ctx   distributed context
 * @param[in]  H     n×n symmetric input matrix (lower fill)
 * @param[in]  Q1    n×k orthonormal basis
 * @param[in]  Q2    n×m orthonormal basis  (m = n − k)
 * @param[out] H1    k×k output subproblem
 * @param[out] H2    m×m output subproblem
 * @param[in]  n,k   global dimensions
 * @param[in,out] ws workspace (data pool for temporary n×k and n×m buffers)
 */
template <typename T>
void sdc_split_mp(Context &ctx, const DistMatrix<T> &H, const DistMatrix<T> &Q1,
                  const DistMatrix<T> &Q2, DistMatrix<T> &H1, DistMatrix<T> &H2, int64_t n,
                  int64_t k, WorkspaceMp<T> &ws);

/**
 * @brief Back-transform eigenvectors: evec[:,0:m] = Q₂·evec₂, evec[:,m:n] = Q₁·evec₁.
 *
 * Writes into column subranges of evec via the cuBLASMp ic/jc offset mechanism.
 * m = n − k.
 *
 * @tparam T     float or double
 * @param[in]  ctx           distributed context
 * @param[in]  Q1,Q2         n×k and n×m basis matrices
 * @param[in]  evec1,evec2   k×k and m×m sub-eigenvector matrices
 * @param[out] evec          n×n output eigenvector matrix
 * @param[in]  n,k           global dimensions
 */
template <typename T>
void sdc_combine_mp(Context &ctx, const DistMatrix<T> &Q1, const DistMatrix<T> &Q2,
                    const DistMatrix<T> &evec1, const DistMatrix<T> &evec2, DistMatrix<T> &evec,
                    int64_t n, int64_t k);

} // namespace kernels
} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
