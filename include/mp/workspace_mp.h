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

#include "mp/comm.h"
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

// =============================================================================
// WorkspaceMp<T> — single-allocation workspace for the distributed solver.
//
// Direct analogue of SolverWorkspace<T> in workspace.h.  One cudaMalloc for
// the device pool, one malloc for the (typically tiny) host scratch.
//
// Pool layout (device, per-rank, 256-byte aligned):
//   geqrf_dwork     cuSOLVERMp Geqrf(2n, n) scratch  [size in bytes]
//   ormqr_dwork     cuSOLVERMp Ormqr(2n, n, n) scratch
//   potrf_dwork     cuSOLVERMp Potrf(n) scratch
//   d_info          1 × int
//   qdwh_W          local tiles of 2n×n  (QR work matrix, same role as SolverWorkspace::qdwh_W)
//   qdwh_tau        local portion of tau(n)
//   data[6·lr·lc]   mark/reset bump allocator  (lr = numroc(n,…), lc = numroc(n,…))
//
// The data pool stores local tile buffers for B, Q, H1/H2, evec1/2, eval1/2
// across recursion levels — same 6× factor as the single-GPU pool, but each
// n×n matrix contributes lr·lc elements (≈ n²/p) instead of n².
// =============================================================================

/// Base-case threshold for spectral_dc_mp recursion (matches single-GPU SDC_BASE_N).
constexpr int64_t SDC_BASE_N_MP = 512;

template <typename T> struct WorkspaceMp {
    // cuSOLVERMp device scratch — sizes in *bytes* (cuSOLVERMp API differs from cuSOLVER)
    void *geqrf_dwork;
    size_t geqrf_wsD;
    void *ormqr_dwork;
    size_t ormqr_wsD;
    void *potrf_dwork;
    size_t potrf_wsD;

    // cuSOLVERMp host scratch — single shared buffer (ops are sequential)
    void *h_work;
    size_t h_work_size;

    // Shared device info scalar (cuSOLVERMp writes failure codes here)
    int *d_info;

    // QDWH per-call scratch (fixed size, reused across all QDWH iterations)
    T *qdwh_W;   ///< local tiles of the 2n×n QR work matrix
    T *qdwh_tau; ///< local portion of the n-element Householder tau vector

    // Data pool — mark/reset bump allocator for variable-lifetime tile buffers
    T *data;
    size_t data_cap;  ///< capacity in T elements
    size_t data_used; ///< current stack pointer in T elements

    // Single backing device allocation owned by this struct
    void *pool;

    /// Allocate @p n T-elements from the data pool; returns device pointer.
    inline T *push(size_t n) {
        if (data_used + n > data_cap) {
            fprintf(stderr,
                    "WorkspaceMp::push overflow: need %zu, capacity %zu "
                    "(increase pool factor in workspace_mp_alloc)\n",
                    data_used + n, data_cap);
            abort();
        }
        T *ptr = data + data_used;
        data_used += n;
        return ptr;
    }

    inline size_t mark() const {
        return data_used;
    }
    inline void reset(size_t sv) {
        data_used = sv;
    }
};

/**
 * @brief Query cuSOLVERMp buffer sizes and issue one cudaMalloc for all regions.
 *
 * @tparam T     float or double
 * @param  ctx   distributed context (grid, handles, stream)
 * @param  n     root problem dimension (global)
 */
template <typename T> WorkspaceMp<T> workspace_mp_alloc(Context &ctx, int64_t n);

/** @brief Free the workspace pool (one cudaFree + one std::free). */
template <typename T> void workspace_mp_free(WorkspaceMp<T> &ws);

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
