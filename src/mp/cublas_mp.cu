/**
 * @file   cublas_mp.cu
 * @brief  Type-dispatching cuBLASMp wrappers — cuev::mp::cublasmp namespace.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#ifdef CUEV_ENABLE_MP

#include "mp/kernels_mp.cuh"
#include <cstdlib>

namespace cuev {
namespace mp {
namespace cublasmp {

namespace {
void mp_sync(Context &ctx) {
    CAL_CHECK(cal_stream_sync(ctx.cal, ctx.stream));
    CAL_CHECK(cal_comm_barrier(ctx.cal, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}
} // namespace

template <typename T>
void gemm(Context &ctx, cublasOperation_t transa, cublasOperation_t transb, int64_t m, int64_t n,
          int64_t k, const T *alpha, const T *A, int64_t ia, int64_t ja,
          cublasMpMatrixDescriptor_t descA, const T *B, int64_t ib, int64_t jb,
          cublasMpMatrixDescriptor_t descB, const T *beta, T *C, int64_t ic, int64_t jc,
          cublasMpMatrixDescriptor_t descC) {
    size_t wsD = 0, wsH = 0;
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(ctx.cublasmp, transa, transb, m, n, k, alpha, A, ia, ja,
                                           descA, B, ib, jb, descB, beta, C, ic, jc, descC,
                                           compute_type<T>(), &wsD, &wsH));

    void *dw = nullptr, *hw = nullptr;
    CUDA_CHECK(cudaMalloc(&dw, wsD > 0 ? wsD : 1));
    hw = std::malloc(wsH > 0 ? wsH : 1);

    CUBLASMP_CHECK(cublasMpGemm(ctx.cublasmp, transa, transb, m, n, k, alpha, A, ia, ja, descA, B,
                                ib, jb, descB, beta, C, ic, jc, descC, compute_type<T>(), dw, wsD,
                                hw, wsH));
    mp_sync(ctx);

    CUDA_CHECK(cudaFree(dw));
    std::free(hw);
}

template <typename T>
void geadd(Context &ctx, cublasOperation_t trans, int64_t m, int64_t n, const T *alpha, const T *A,
           int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA, const T *beta, T *C,
           int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC) {
    size_t wsD = 0, wsH = 0;
    CUBLASMP_CHECK(cublasMpGeadd_bufferSize(ctx.cublasmp, trans, m, n, alpha, A, ia, ja, descA,
                                            beta, C, ic, jc, descC, &wsD, &wsH));

    void *dw = nullptr, *hw = nullptr;
    CUDA_CHECK(cudaMalloc(&dw, wsD > 0 ? wsD : 1));
    hw = std::malloc(wsH > 0 ? wsH : 1);

    CUBLASMP_CHECK(cublasMpGeadd(ctx.cublasmp, trans, m, n, alpha, A, ia, ja, descA, beta, C, ic,
                                 jc, descC, dw, wsD, hw, wsH));
    mp_sync(ctx);

    CUDA_CHECK(cudaFree(dw));
    std::free(hw);
}

template <typename T>
void syrk(Context &ctx, cublasFillMode_t uplo, cublasOperation_t trans, int64_t n, int64_t k,
          const T *alpha, const T *A, int64_t ia, int64_t ja, cublasMpMatrixDescriptor_t descA,
          const T *beta, T *C, int64_t ic, int64_t jc, cublasMpMatrixDescriptor_t descC) {
    size_t wsD = 0, wsH = 0;
    CUBLASMP_CHECK(cublasMpSyrk_bufferSize(ctx.cublasmp, uplo, trans, n, k, alpha, A, ia, ja, descA,
                                           beta, C, ic, jc, descC, compute_type<T>(), &wsD, &wsH));

    void *dw = nullptr, *hw = nullptr;
    CUDA_CHECK(cudaMalloc(&dw, wsD > 0 ? wsD : 1));
    hw = std::malloc(wsH > 0 ? wsH : 1);

    CUBLASMP_CHECK(cublasMpSyrk(ctx.cublasmp, uplo, trans, n, k, alpha, A, ia, ja, descA, beta, C,
                                ic, jc, descC, compute_type<T>(), dw, wsD, hw, wsH));
    mp_sync(ctx);

    CUDA_CHECK(cudaFree(dw));
    std::free(hw);
}

template <typename T>
void trsm(Context &ctx, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
          cublasDiagType_t diag, int64_t m, int64_t n, const T *alpha, const T *A, int64_t ia,
          int64_t ja, cublasMpMatrixDescriptor_t descA, T *B, int64_t ib, int64_t jb,
          cublasMpMatrixDescriptor_t descB) {
    size_t wsD = 0, wsH = 0;
    CUBLASMP_CHECK(cublasMpTrsm_bufferSize(ctx.cublasmp, side, uplo, trans, diag, m, n, alpha, A,
                                           ia, ja, descA, B, ib, jb, descB, compute_type<T>(), &wsD,
                                           &wsH));

    void *dw = nullptr, *hw = nullptr;
    CUDA_CHECK(cudaMalloc(&dw, wsD > 0 ? wsD : 1));
    hw = std::malloc(wsH > 0 ? wsH : 1);

    CUBLASMP_CHECK(cublasMpTrsm(ctx.cublasmp, side, uplo, trans, diag, m, n, alpha, A, ia, ja,
                                descA, B, ib, jb, descB, compute_type<T>(), dw, wsD, hw, wsH));
    mp_sync(ctx);

    CUDA_CHECK(cudaFree(dw));
    std::free(hw);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void gemm<T>(                                                                         \
        Context &, cublasOperation_t, cublasOperation_t, int64_t, int64_t, int64_t, const T *,     \
        const T *, int64_t, int64_t, cublasMpMatrixDescriptor_t, const T *, int64_t, int64_t,      \
        cublasMpMatrixDescriptor_t, const T *, T *, int64_t, int64_t, cublasMpMatrixDescriptor_t); \
    template void geadd<T>(Context &, cublasOperation_t, int64_t, int64_t, const T *, const T *,   \
                           int64_t, int64_t, cublasMpMatrixDescriptor_t, const T *, T *, int64_t,  \
                           int64_t, cublasMpMatrixDescriptor_t);                                   \
    template void syrk<T>(Context &, cublasFillMode_t, cublasOperation_t, int64_t, int64_t,        \
                          const T *, const T *, int64_t, int64_t, cublasMpMatrixDescriptor_t,      \
                          const T *, T *, int64_t, int64_t, cublasMpMatrixDescriptor_t);           \
    template void trsm<T>(Context &, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,        \
                          cublasDiagType_t, int64_t, int64_t, const T *, const T *, int64_t,       \
                          int64_t, cublasMpMatrixDescriptor_t, T *, int64_t, int64_t,              \
                          cublasMpMatrixDescriptor_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cublasmp
} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
