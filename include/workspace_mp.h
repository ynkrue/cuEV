/**
 * @file   workspace_mp.h
 * @brief  Distributed matrix handle for cuEV MP, analogue of workspace.h.
 *
 * Design: the *caller* lays out the matrix 2D block-cyclic (BLACS-style) and
 * owns its local tiles. It only needs to *describe* the layout to cuBLASMp.
 * DistMatrix is the caller's local device pointer and the cuBLASMp descriptor.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#ifdef CUEV_ENABLE_MP

#include "comm.h"
#include <type_traits>

namespace cuev {
namespace mp {

/// cudaDataType for T.
template <typename T> inline cudaDataType_t cuda_type() {
    if constexpr (std::is_same_v<T, float>)
        return CUDA_R_32F;
    else
        return CUDA_R_64F;
}

/**
 * @brief A dense matrix distributed 2D block-cyclic over the Context grid.
 *
 * @c data points at this rank's local tiles (column-major, leading dim @c lld).
 * Storage is caller-owned and this struct only adds the cuBLASMp descriptor.
 */
template <typename T> struct DistMatrix {
    int64_t m, n;                    ///< global dimensions
    int64_t nb;                      ///< square block size
    int64_t local_rows;              ///< rows this rank stores
    int64_t local_cols;              ///< cols this rank stores
    int64_t lld;                     ///< local leading dimension = local_rows
    T *data;                         ///< device pointer to local tiles
    cublasMpMatrixDescriptor_t desc; ///< cuBLASMp layout descriptor
};

/// Local element count for an m×n matrix on this rank.
template <typename T> inline int64_t dist_local_count(Context &ctx, int64_t m, int64_t n) {
    int64_t lr = cublasMpNumroc(m, ctx.nb, ctx.prow, 0, ctx.nprow);
    int64_t lc = cublasMpNumroc(n, ctx.nb, ctx.pcol, 0, ctx.npcol);
    return lr * lc;
}

/**
 * @brief Build a DistMatrix over @p ctx's grid: compute local sizes (numroc)
 *        and create the cuBLASMp descriptor. Does not allocate @p data.
 *
 * @param local_data device pointer to this rank's local tiles (lld × local_cols)
 */
template <typename T>
inline DistMatrix<T> dist_describe(Context &ctx, int64_t m, int64_t n, T *local_data) {
    DistMatrix<T> A{};
    A.m = m;
    A.n = n;
    A.nb = ctx.nb;
    A.local_rows = cublasMpNumroc(m, ctx.nb, ctx.prow, 0, ctx.nprow);
    A.local_cols = cublasMpNumroc(n, ctx.nb, ctx.pcol, 0, ctx.npcol);
    A.lld = A.local_rows > 0 ? A.local_rows : 1; // lld must be >= 1
    A.data = local_data;
    // mb=nb (square tiles).
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(m, n, ctx.nb, ctx.nb, 0, 0, A.lld, cuda_type<T>(),
                                                  ctx.grid, &A.desc));
    return A;
}

/// Destroy the descriptor (does not free caller-owned @c data).
template <typename T> inline void dist_free(DistMatrix<T> &A) {
    if (A.desc) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(A.desc));
    A.desc = nullptr;
}

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
