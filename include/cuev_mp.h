/**
 * @file   cuev_mp.h
 * @brief  Public API for cuEV — distributed multi-GPU symmetric dense eigensolver.
 *
 * Requires building with -DCUEV_ENABLE_MP=ON (links libcuev_mp).
 * For single-GPU usage see cuev.h.
 *
 * Matrix distribution: 2D block-cyclic, BLACS-compatible process grid p×q.
 * Distributed operations via cuBLASMp + cuSOLVERMp + NCCL.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#ifdef CUEV_ENABLE_MP

#include "mp/comm.h"
#include "mp/workspace_mp.h"

namespace cuev {
namespace mp {

/**
 * @brief Distributed symmetric dense eigensolver: H v = λv.
 *
 * H is overwritten with eigenvectors on exit (columns, ascending order).
 * d_eval receives eigenvalues ascending; replicated on all ranks.
 *
 * @tparam T       float or double
 * @param[in]     ctx    distributed context (grid, handles, stream)
 * @param[in,out] H      distributed n×n symmetric input; overwritten with eigenvectors
 * @param[in]     n      global matrix dimension
 * @param[out]    eval   device array of length n; eigenvalues ascending (all ranks)
 * @param[out]    evec   distributed n×n eigenvector matrix (column j = j-th eigenvector)
 * @param[in,out] ws     pre-allocated workspace from workspace_mp_alloc<T>(ctx, n)
 */
template <typename T>
void symm_eig_solve_mp(Context &ctx, DistMatrix<T> &H, int64_t n, T *eval, DistMatrix<T> &evec,
                       WorkspaceMp<T> &ws);

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
