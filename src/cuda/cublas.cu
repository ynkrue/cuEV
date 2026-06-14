/**
 * @file   cublas.cu
 * @brief  Type-dispatching cuBLAS wrappers — cuev::cublas namespace.
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#include "kernels.cuh"
#include <type_traits>

namespace cuev {
namespace cublas {

template <typename T>
void gemm(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
          const T *alpha, const T *A, int lda, const T *B, int ldb, const T *beta, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSgemm(h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
    else
        CUBLAS_CHECK(cublasDgemm(h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

template <typename T>
void geam(cublasHandle_t h, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
          const T *alpha, const T *A, int lda, const T *beta, const T *B, int ldb, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSgeam(h, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
    else
        CUBLAS_CHECK(cublasDgeam(h, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc));
}

template <typename T>
void syrk(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
          const T *alpha, const T *A, int lda, const T *beta, T *C, int ldc) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSsyrk(h, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
    else
        CUBLAS_CHECK(cublasDsyrk(h, uplo, trans, n, k, alpha, A, lda, beta, C, ldc));
}

template <typename T>
void trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
          cublasDiagType_t diag, int m, int n, const T *alpha, const T *A, int lda, T *B, int ldb) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasStrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
    else
        CUBLAS_CHECK(cublasDtrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <typename T> void scal(cublasHandle_t h, int n, const T *alpha, T *x, int incx) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSscal(h, n, alpha, x, incx));
    else
        CUBLAS_CHECK(cublasDscal(h, n, alpha, x, incx));
}

template <typename T> void copy(cublasHandle_t h, int n, const T *x, int incx, T *y, int incy) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasScopy(h, n, x, incx, y, incy));
    else
        CUBLAS_CHECK(cublasDcopy(h, n, x, incx, y, incy));
}

template <typename T> void nrm2(cublasHandle_t h, int n, const T *x, int incx, T *result) {
    if constexpr (std::is_same_v<T, float>)
        CUBLAS_CHECK(cublasSnrm2(h, n, x, incx, result));
    else
        CUBLAS_CHECK(cublasDnrm2(h, n, x, incx, result));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void gemm<T>(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,     \
                          const T *, const T *, int, const T *, int, const T *, T *, int);         \
    template void geam<T>(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int,          \
                          const T *, const T *, int, const T *, const T *, int, T *, int);         \
    template void syrk<T>(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int,           \
                          const T *, const T *, int, const T *, T *, int);                         \
    template void trsm<T>(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,   \
                          cublasDiagType_t, int, int, const T *, const T *, int, T *, int);        \
    template void scal<T>(cublasHandle_t, int, const T *, T *, int);                               \
    template void copy<T>(cublasHandle_t, int, const T *, int, T *, int);                          \
    template void nrm2<T>(cublasHandle_t, int, const T *, int, T *);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cublas
} // namespace cuev
