/**
 * @file   cusolver_mp.cu
 * @brief  cuSOLVERMp wrappers — cuev::mp::cusolvermp namespace.
 *
 * All functions except syevd reuse pre-allocated WorkspaceMp scratch
 * (geqrf_dwork / ormqr_dwork / potrf_dwork + h_work + d_info).
 * syevd allocates its own scratch per call (leaf size varies with recursion depth).
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#ifdef CUEV_ENABLE_MP

#include "mp/kernels_mp.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace cuev {
namespace mp {
namespace cusolvermp {

namespace {
void mp_sync(Context &ctx) {
    CAL_CHECK(cal_stream_sync(ctx.cal, ctx.stream));
    CAL_CHECK(cal_comm_barrier(ctx.cal, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

void check_info(Context &ctx, int *d_info, const char *name) {
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "[rank %d] cuSOLVERMp %s: info=%d\n", ctx.rank, name, h_info);
        MPI_Abort(ctx.comm, 1);
    }
}
} // namespace

template <typename T>
void geqrf(Context &ctx, int64_t m, int64_t n, T *A, int64_t ia, int64_t ja,
           cusolverMpMatrixDescriptor_t descA, T *tau, WorkspaceMp<T> &ws) {
    CUSOLVER_CHECK(cusolverMpGeqrf(ctx.cusolvermp, m, n, A, ia, ja, descA, tau, cuda_type<T>(),
                                   ws.geqrf_dwork, ws.geqrf_wsD, ws.h_work, ws.h_work_size,
                                   ws.d_info));
    mp_sync(ctx);
    check_info(ctx, ws.d_info, "Geqrf");
}

template <typename T>
void ormqr(Context &ctx, cublasSideMode_t side, cublasOperation_t trans, int64_t m, int64_t n,
           int64_t k, const T *A, int64_t ia, int64_t ja, cusolverMpMatrixDescriptor_t descA,
           const T *tau, T *C, int64_t ic, int64_t jc, cusolverMpMatrixDescriptor_t descC,
           WorkspaceMp<T> &ws) {
    CUSOLVER_CHECK(cusolverMpOrmqr(ctx.cusolvermp, side, trans, m, n, k, A, ia, ja, descA, tau, C,
                                   ic, jc, descC, cuda_type<T>(), ws.ormqr_dwork, ws.ormqr_wsD,
                                   ws.h_work, ws.h_work_size, ws.d_info));
    mp_sync(ctx);
    check_info(ctx, ws.d_info, "Ormqr");
}

template <typename T>
void potrf(Context &ctx, cublasFillMode_t uplo, int64_t n, T *A, int64_t ia, int64_t ja,
           cusolverMpMatrixDescriptor_t descA, WorkspaceMp<T> &ws) {
    CUSOLVER_CHECK(cusolverMpPotrf(ctx.cusolvermp, uplo, n, A, ia, ja, descA, cuda_type<T>(),
                                   ws.potrf_dwork, ws.potrf_wsD, ws.h_work, ws.h_work_size,
                                   ws.d_info));
    mp_sync(ctx);
    check_info(ctx, ws.d_info, "Potrf");
}

template <typename T>
void syevd(Context &ctx, int64_t n, T *A, int64_t ia, int64_t ja,
           cusolverMpMatrixDescriptor_t descA, T *W, T *Z, int64_t iz, int64_t jz,
           cusolverMpMatrixDescriptor_t descZ, WorkspaceMp<T> &ws) {
    size_t wsD = 0, wsH = 0;
    CUSOLVER_CHECK(cusolverMpSyevd_bufferSize(ctx.cusolvermp, const_cast<char *>("V"),
                                              CUBLAS_FILL_MODE_LOWER, n, A, ia, ja, descA, W, Z, iz,
                                              jz, descZ, cuda_type<T>(), &wsD, &wsH));

    void *dw = nullptr, *hw = nullptr;
    CUDA_CHECK(cudaMalloc(&dw, std::max(wsD, (size_t)1)));
    hw = std::malloc(std::max(wsH, (size_t)1));

    CUSOLVER_CHECK(cusolverMpSyevd(ctx.cusolvermp, const_cast<char *>("V"), CUBLAS_FILL_MODE_LOWER,
                                   n, A, ia, ja, descA, W, Z, iz, jz, descZ, cuda_type<T>(), dw,
                                   wsD, hw, wsH, ws.d_info));
    mp_sync(ctx);
    check_info(ctx, ws.d_info, "Syevd");

    CUDA_CHECK(cudaFree(dw));
    std::free(hw);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void geqrf<T>(Context &, int64_t, int64_t, T *, int64_t, int64_t,                     \
                           cusolverMpMatrixDescriptor_t, T *, WorkspaceMp<T> &);                   \
    template void ormqr<T>(Context &, cublasSideMode_t, cublasOperation_t, int64_t, int64_t,       \
                           int64_t, const T *, int64_t, int64_t, cusolverMpMatrixDescriptor_t,     \
                           const T *, T *, int64_t, int64_t, cusolverMpMatrixDescriptor_t,         \
                           WorkspaceMp<T> &);                                                      \
    template void potrf<T>(Context &, cublasFillMode_t, int64_t, T *, int64_t, int64_t,            \
                           cusolverMpMatrixDescriptor_t, WorkspaceMp<T> &);                        \
    template void syevd<T>(Context &, int64_t, T *, int64_t, int64_t,                              \
                           cusolverMpMatrixDescriptor_t, T *, T *, int64_t, int64_t,               \
                           cusolverMpMatrixDescriptor_t, WorkspaceMp<T> &);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cusolvermp
} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
