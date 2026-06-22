/**
 * @file   cublas.cu
 * @brief  Type-dispatching cuBLAS wrappers — cuev::cublas namespace.
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "cuda/kernels.cuh"
#include <type_traits>

namespace cuev {
namespace cublas {

template <typename T>
void gemm(SolverHandle<T> *ws, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          int k, const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C,
          int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSgemm(ws->cublas_handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                                 beta, C, ldc));
    else
        CUBLAS_CHECK(cublasDgemm(ws->cublas_handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                                 beta, C, ldc));
}

template <typename T>
void geam(SolverHandle<T> *ws, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          const T *alpha, const T *A, int lda, const T *beta, const T *B, int ldb, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSgeam(ws->cublas_handle, transa, transb, m, n, alpha, A, lda, beta, B,
                                 ldb, C, ldc));
    else
        CUBLAS_CHECK(cublasDgeam(ws->cublas_handle, transa, transb, m, n, alpha, A, lda, beta, B,
                                 ldb, C, ldc));
}

template <typename T>
void symm(SolverHandle<T> *ws, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(
            cublasSsymm(ws->cublas_handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
    else
        CUBLAS_CHECK(
            cublasDsymm(ws->cublas_handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
}

template <typename T>
void syrk(SolverHandle<T> *ws, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
          const T *alpha, const T *A, int lda, const T *beta, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(
            cublasSsyrk(ws->cublas_handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    else
        CUBLAS_CHECK(
            cublasDsyrk(ws->cublas_handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
}

template <typename T>
void trsm(SolverHandle<T> *ws, cublasSideMode_t side, cublasFillMode_t uplo,
          cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const T *alpha, const T *A,
          int lda, T *B, int ldb) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(
            cublasStrsm(ws->cublas_handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    else
        CUBLAS_CHECK(
            cublasDtrsm(ws->cublas_handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <typename T> void scal(SolverHandle<T> *ws, int n, const T *alpha, T *x, int incx) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSscal(ws->cublas_handle, n, alpha, x, incx));
    else
        CUBLAS_CHECK(cublasDscal(ws->cublas_handle, n, alpha, x, incx));
}

template <typename T> void copy(SolverHandle<T> *ws, int n, const T *x, int incx, T *y, int incy) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasScopy(ws->cublas_handle, n, x, incx, y, incy));
    else
        CUBLAS_CHECK(cublasDcopy(ws->cublas_handle, n, x, incx, y, incy));
}

template <typename T> void nrm2(SolverHandle<T> *ws, int n, const T *x, int incx, T *result) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSnrm2(ws->cublas_handle, n, x, incx, result));
    else
        CUBLAS_CHECK(cublasDnrm2(ws->cublas_handle, n, x, incx, result));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void gemm<T>(SolverHandle<T> *, cublasOperation_t, cublasOperation_t, int, int, int,  \
                          const T *, const T *, int, const T *, int, const T *, T *, int);         \
    template void geam<T>(SolverHandle<T> *, cublasOperation_t, cublasOperation_t, int, int,       \
                          const T *, const T *, int, const T *, const T *, int, T *, int);         \
    template void symm<T>(SolverHandle<T> *, cublasSideMode_t, cublasFillMode_t, int, int,         \
                          const T *, const T *, int, const T *, int, const T *, T *, int);         \
    template void syrk<T>(SolverHandle<T> *, cublasFillMode_t, cublasOperation_t, int, int,        \
                          const T *, const T *, int, const T *, T *, int);                         \
    template void trsm<T>(SolverHandle<T> *, cublasSideMode_t, cublasFillMode_t,                   \
                          cublasOperation_t, cublasDiagType_t, int, int, const T *, const T *,     \
                          int, T *, int);                                                          \
    template void scal<T>(SolverHandle<T> *, int, const T *, T *, int);                            \
    template void copy<T>(SolverHandle<T> *, int, const T *, int, T *, int);                       \
    template void nrm2<T>(SolverHandle<T> *, int, const T *, int, T *);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cublas
} // namespace cuev
