/**
 * @file   handle_mp.h
 * @brief  Distributed matrix descriptor and workspace for cuEV MP.
 *
 * Design: the caller lays out the matrix 2D block-cyclic (BLACS-style) and
 * owns its local tiles. DistMatrix wraps the caller's device pointer with the
 * cuBLASMp/cuSOLVERMp descriptors that describe the layout to the runtime.
 *
 * Context (NCCL communicator, cuBLASMp grid, streams) is defined in mp/comm.h
 * which will be populated in Phase 2.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#ifdef CUEV_ENABLE_MP

#include "common.h"
// mp/comm.h provides Context (Phase 2)
#include "mp/comm.h"
#include <algorithm>
#include <type_traits>
#include <vector>

namespace cuev {
namespace mp {

/// cudaDataType selector for T.
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
 * Storage is caller-owned; this struct only adds the cuBLASMp/cuSOLVERMp descriptors.
 */
template <typename T> struct DistMatrix {
    int64_t m, n;                            ///< global dimensions
    int64_t nb;                              ///< square block size
    int64_t local_rows;                      ///< rows this rank stores
    int64_t local_cols;                      ///< cols this rank stores
    int64_t lld;                             ///< local leading dimension = local_rows
    T *data;                                 ///< device pointer to local tiles
    cublasMpMatrixDescriptor_t desc;         ///< cuBLASMp layout descriptor
    cusolverMpMatrixDescriptor_t solverDesc; ///< cuSOLVERMp layout descriptor
};

/// Local element count for an m×n matrix on this rank.
template <typename T> inline int64_t dist_local_count(Context &ctx, int64_t m, int64_t n) {
    int64_t lr = cublasMpNumroc(m, ctx.nb, ctx.prow, 0, ctx.nprow);
    int64_t lc = cublasMpNumroc(n, ctx.nb, ctx.pcol, 0, ctx.npcol);
    return lr * lc;
}

/**
 * @brief Build a DistMatrix over @p ctx's grid: compute local sizes and create
 *        cuBLASMp + cuSOLVERMp descriptors. Does not allocate @p local_data.
 *
 * @param local_data  device pointer to this rank's local tiles (lld × local_cols)
 */
template <typename T>
inline DistMatrix<T> dist_describe(Context &ctx, int64_t m, int64_t n, T *local_data) {
    DistMatrix<T> A{};
    A.m = m;
    A.n = n;
    A.nb = ctx.nb;
    A.local_rows = cublasMpNumroc(m, ctx.nb, ctx.prow, 0, ctx.nprow);
    A.local_cols = cublasMpNumroc(n, ctx.nb, ctx.pcol, 0, ctx.npcol);
    A.lld = A.local_rows > 0 ? A.local_rows : 1; // cuBLASMp requires lld >= 1
    A.data = local_data;
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(m, n, ctx.nb, ctx.nb, 0, 0, A.lld, cuda_type<T>(),
                                                  ctx.grid, &A.desc));
    CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(&A.solverDesc, ctx.solvergrid, cuda_type<T>(), m, n,
                                              ctx.nb, ctx.nb, 0, 0, A.lld));
    return A;
}

/// Destroy both descriptors (does not free caller-owned @c data).
template <typename T> inline void dist_free(DistMatrix<T> &A) {
    if (A.desc) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(A.desc));
    if (A.solverDesc) CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(A.solverDesc));
    A.desc = nullptr;
    A.solverDesc = nullptr;
}

// =============================================================================
// WorkspaceMp<T> — distributed solver handle.
//
// Analogue of SolverHandle<T> in cuda/handle.h.
//
// Fixed pool (device, 256-byte aligned):
//   geqrf_dwork     cuSOLVERMp geqrf scratch [bytes]
//   syevd_dwork     cuSOLVERMp syevd scratch [bytes]
//   d_info          1 × int
//
// Variable-lifetime tile buffers use push()/reset(): real cudaMalloc per call.
// 2D block-cyclic panel sizes are data-dependent (numroc), so a fixed pool
// cannot be reliably sized.
// =============================================================================

template <typename T> struct WorkspaceMp {
    // cuSOLVERMp device scratch — sizes in *bytes*
    void *geqrf_dwork;
    size_t geqrf_wsD;
    void *syevd_dwork;
    size_t syevd_wsD;

    // cuSOLVERMp host scratch — single shared buffer (ops are sequential)
    void *h_work;
    size_t h_work_size;

    // Device info scalar
    int *d_info;

    // Single backing device allocation for the fixed regions above.
    void *pool;

    // Variable-lifetime tile buffers: LIFO stack of cudaMalloc'd pointers.
    std::vector<void *> allocs;

    /// Allocate @p count T-elements (one cudaMalloc); returns device pointer.
    inline T *push(size_t count) {
        void *ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, std::max(count, (size_t)1) * sizeof(T)));
        allocs.push_back(ptr);
        return static_cast<T *>(ptr);
    }

    /// Save stack depth; pair with reset() to free everything pushed since.
    inline size_t mark() const {
        return allocs.size();
    }

    /// Free (LIFO) every buffer pushed since the matching mark().
    inline void reset(size_t sv) {
        while (allocs.size() > sv) {
            CUDA_CHECK(cudaFree(allocs.back()));
            allocs.pop_back();
        }
    }
};

/**
 * @brief Query cuSOLVERMp buffer sizes and allocate one device pool.
 *
 * @tparam T   float or double
 * @param ctx  distributed context (grid, handles, stream)
 * @param n    root problem dimension (global)
 */
template <typename T> WorkspaceMp<T> workspace_mp_alloc(Context &ctx, int64_t n);

/** @brief Free the workspace pool (one cudaFree + one free). */
template <typename T> void workspace_mp_free(WorkspaceMp<T> &ws);

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
