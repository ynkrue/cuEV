/**
 * @file   workspace.cu
 * @brief  WorkspaceMp<T> allocation / deallocation.
 *
 * One cudaMalloc for the entire per-rank device pool; one std::malloc for the
 * (typically zero-sized, but must be non-null) cuSOLVERMp host scratch.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#ifdef CUEV_ENABLE_MP

#include "common.h"
#include "mp/comm.h"
#include "mp/workspace_mp.h"
#include <algorithm>
#include <cstdlib>
#include <type_traits>

namespace cuev {
namespace mp {

template <typename T> WorkspaceMp<T> workspace_mp_alloc(Context &ctx, int64_t n) {
    WorkspaceMp<T> ws{};
    const int64_t nb = ctx.nb;

    const cudaDataType_t dtype = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;

    // -------------------------------------------------------------------------
    // Local tile dimensions for n×n and 2n×n matrices.
    // -------------------------------------------------------------------------
    int64_t lr = cublasMpNumroc(n, nb, ctx.prow, 0, ctx.nprow);
    int64_t lc = cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol);
    int64_t lr2 = cublasMpNumroc(2 * n, nb, ctx.prow, 0, ctx.nprow);
    int64_t lld_nn = std::max(lr, (int64_t)1); // lld must be ≥ 1
    int64_t lld_2nn = std::max(lr2, (int64_t)1);
    // tau is distributed over columns
    int64_t l_tau = std::max(cublasMpNumroc(n, nb, ctx.pcol, 0, ctx.npcol), (int64_t)1);

    cusolverMpMatrixDescriptor_t descr_2n_n = nullptr, descr_n_n = nullptr;
    CUSOLVER_CHECK(cusolverMpCreateMatrixDesc(&descr_2n_n, ctx.solvergrid, dtype, 2 * n, n, nb, nb,
                                              0, 0, lld_2nn));
    CUSOLVER_CHECK(
        cusolverMpCreateMatrixDesc(&descr_n_n, ctx.solvergrid, dtype, n, n, nb, nb, 0, 0, lld_nn));

    T *d_dummy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(T)));

    // Query cuSOLVER buffer sizes
    size_t geqrf_wsH = 0;
    CUSOLVER_CHECK(cusolverMpGeqrf_bufferSize(ctx.cusolvermp, 2 * n, n, d_dummy, 1, 1, descr_2n_n,
                                              dtype, &ws.geqrf_wsD, &geqrf_wsH));

    size_t ormqr_wsH = 0;
    CUSOLVER_CHECK(cusolverMpOrmqr_bufferSize(ctx.cusolvermp, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, 2 * n,
                                              n, n, d_dummy, 1, 1, descr_2n_n, d_dummy, // A + tau
                                              d_dummy, 1, 1, descr_2n_n,                // C
                                              dtype, &ws.ormqr_wsD, &ormqr_wsH));

    size_t potrf_wsH = 0;
    CUSOLVER_CHECK(cusolverMpPotrf_bufferSize(ctx.cusolvermp, CUBLAS_FILL_MODE_LOWER, n, d_dummy, 1,
                                              1, descr_n_n, dtype, &ws.potrf_wsD, &potrf_wsH));

    CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descr_2n_n));
    CUSOLVER_CHECK(cusolverMpDestroyMatrixDesc(descr_n_n));
    CUDA_CHECK(cudaFree(d_dummy));

    // Host workspace — single shared buffer.
    ws.h_work_size = std::max({geqrf_wsH, ormqr_wsH, potrf_wsH, (size_t)1});
    ws.h_work = std::malloc(ws.h_work_size);

    // Device pool layout
    auto align = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };

    size_t off_geqrf = 0;
    size_t off_ormqr = off_geqrf + align(std::max(ws.geqrf_wsD, (size_t)1));
    size_t off_potrf = off_ormqr + align(std::max(ws.ormqr_wsD, (size_t)1));
    size_t off_info = off_potrf + align(std::max(ws.potrf_wsD, (size_t)1));
    size_t off_W = off_info + align(sizeof(int));
    size_t off_tau = off_W + align((size_t)lld_2nn * lc * sizeof(T));
    size_t off_data = off_tau + align((size_t)l_tau * sizeof(T));

    // Data pool: 6 × (local tiles of n×n)
    ws.data_cap = (size_t)6 * lld_nn * lc;
    size_t total = off_data + ws.data_cap * sizeof(T);

    CUDA_CHECK(cudaMalloc(&ws.pool, total));
    char *base = static_cast<char *>(ws.pool);

    ws.geqrf_dwork = base + off_geqrf;
    ws.ormqr_dwork = base + off_ormqr;
    ws.potrf_dwork = base + off_potrf;
    ws.d_info = reinterpret_cast<int *>(base + off_info);
    ws.qdwh_W = reinterpret_cast<T *>(base + off_W);
    ws.qdwh_tau = reinterpret_cast<T *>(base + off_tau);
    ws.data = reinterpret_cast<T *>(base + off_data);
    ws.data_used = 0;

    return ws;
}

template <typename T> void workspace_mp_free(WorkspaceMp<T> &ws) {
    CUDA_CHECK(cudaFree(ws.pool));
    std::free(ws.h_work);
    ws = WorkspaceMp<T>{};
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template WorkspaceMp<T> workspace_mp_alloc<T>(Context &, int64_t);                             \
    template void workspace_mp_free<T>(WorkspaceMp<T> &);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
