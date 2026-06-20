/**
 * @file   cusolver.cu
 * @brief  Type-dispatching cuSOLVER wrappers — cuev::cusolver namespace.
 *
 * All functions receive a SolverWorkspace<T>* and use its pre-allocated
 * scratch buffers (geqrf_buf / orgqr_buf / syevd_buf / d_info).
 * No allocation occurs here; workspace_alloc / workspace_free own the memory.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "cuda/kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace cuev {
namespace cusolver {

namespace {
// Verify a cuSOLVER routine's info flag. This forces a stream sync + D2H copy,
// so it runs only in debug builds — in release (NDEBUG) it compiles to nothing,
// keeping the hot path free of host round-trips. A genuine factorisation failure
// (e.g. non-SPD potrf) still surfaces later via wrong results / a CUDA error.
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
void geqrf(cusolverDnHandle_t h, int m, int n, T *A, int lda, T *tau, SolverWorkspace<T> *ws,
           cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(
            cusolverDnSgeqrf(h, m, n, A, lda, tau, ws->geqrf_buf, ws->geqrf_lwork, ws->d_info));
    else
        CUSOLVER_CHECK(
            cusolverDnDgeqrf(h, m, n, A, lda, tau, ws->geqrf_buf, ws->geqrf_lwork, ws->d_info));
    check_info(ws->d_info, "geqrf", stream);
}

template <typename T>
void orgqr(cusolverDnHandle_t h, int m, int n, int k, T *A, int lda, const T *tau,
           SolverWorkspace<T> *ws, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(
            cusolverDnSorgqr(h, m, n, k, A, lda, tau, ws->orgqr_buf, ws->orgqr_lwork, ws->d_info));
    else
        CUSOLVER_CHECK(
            cusolverDnDorgqr(h, m, n, k, A, lda, tau, ws->orgqr_buf, ws->orgqr_lwork, ws->d_info));
    check_info(ws->d_info, "orgqr", stream);
}

template <typename T>
void potrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T *A, int lda,
           SolverWorkspace<T> *ws, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(
            cusolverDnSpotrf(h, uplo, n, A, lda, ws->potrf_buf, ws->potrf_lwork, ws->d_info));
    else
        CUSOLVER_CHECK(
            cusolverDnDpotrf(h, uplo, n, A, lda, ws->potrf_buf, ws->potrf_lwork, ws->d_info));
    check_info(ws->d_info, "potrf", stream);
}

template <typename T>
void syevd(cusolverDnHandle_t h, int n, T *A, int lda, T *W, SolverWorkspace<T> *ws,
           cudaStream_t stream) {
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(cusolverDnSsyevd(h, jobz, uplo, n, A, lda, W, ws->syevd_buf, ws->syevd_lwork,
                                        ws->d_info));
    else
        CUSOLVER_CHECK(cusolverDnDsyevd(h, jobz, uplo, n, A, lda, W, ws->syevd_buf, ws->syevd_lwork,
                                        ws->d_info));
    check_info(ws->d_info, "syevd", stream);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void geqrf<T>(cusolverDnHandle_t, int, int, T *, int, T *, SolverWorkspace<T> *,      \
                           cudaStream_t);                                                          \
    template void orgqr<T>(cusolverDnHandle_t, int, int, int, T *, int, const T *,                 \
                           SolverWorkspace<T> *, cudaStream_t);                                    \
    template void potrf<T>(cusolverDnHandle_t, cublasFillMode_t, int, T *, int,                    \
                           SolverWorkspace<T> *, cudaStream_t);                                    \
    template void syevd<T>(cusolverDnHandle_t, int, T *, int, T *, SolverWorkspace<T> *,           \
                           cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cusolver
} // namespace cuev
