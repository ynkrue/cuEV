/**
 * @file   cusolver.cu
 * @brief  Type-dispatching cuSOLVER wrappers — cuev::cusolver namespace.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace cuev {
namespace cusolver {

namespace {
inline void check_info([[maybe_unused]] int *d_info, [[maybe_unused]] const char *name,
                       [[maybe_unused]] cudaStream_t stream) {
#ifndef NDEBUG
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr, "cuSOLVER %s failed: info = %d\n", name, h_info);
        exit(1);
    }
#endif
}
} // namespace

template <typename T>
void geqrf(SolverHandle<T> *ws, int m, int n, T *A, int lda, T *tau, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(cusolverDnSgeqrf(ws->cusolver, m, n, A, lda, tau, ws->geqrf_buf,
                                        ws->geqrf_lwork, ws->d_info));
    else
        CUSOLVER_CHECK(cusolverDnDgeqrf(ws->cusolver, m, n, A, lda, tau, ws->geqrf_buf,
                                        ws->geqrf_lwork, ws->d_info));
    check_info(ws->d_info, "geqrf", stream);
}

template <typename T>
void orgqr(SolverHandle<T> *ws, int m, int n, int k, T *A, int lda, const T *tau,
           cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(cusolverDnSorgqr(ws->cusolver, m, n, k, A, lda, tau, ws->orgqr_buf,
                                        ws->orgqr_lwork, ws->d_info));
    else
        CUSOLVER_CHECK(cusolverDnDorgqr(ws->cusolver, m, n, k, A, lda, tau, ws->orgqr_buf,
                                        ws->orgqr_lwork, ws->d_info));
    check_info(ws->d_info, "orgqr", stream);
}

template <typename T>
void syevd(SolverHandle<T> *ws, int n, T *A, int lda, T *W, cudaStream_t stream) {
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(cusolverDnSsyevd(ws->cusolver, jobz, uplo, n, A, lda, W, ws->syevd_buf,
                                        ws->syevd_lwork, ws->d_info));
    else
        CUSOLVER_CHECK(cusolverDnDsyevd(ws->cusolver, jobz, uplo, n, A, lda, W, ws->syevd_buf,
                                        ws->syevd_lwork, ws->d_info));
    check_info(ws->d_info, "syevd", stream);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void geqrf<T>(SolverHandle<T> *, int, int, T *, int, T *, cudaStream_t);              \
    template void orgqr<T>(SolverHandle<T> *, int, int, int, T *, int, const T *, cudaStream_t);   \
    template void syevd<T>(SolverHandle<T> *, int, T *, int, T *, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cusolver
} // namespace cuev
