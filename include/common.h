#pragma once
#include <cstdlib>
#include <cuda_runtime.h>

// =============================================================================
// Error checking
// =============================================================================

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t _e = (err);                                                                    \
        if (_e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t _e = (err);                                                                 \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                                         \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)_e);              \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// =============================================================================
// Utilities
// =============================================================================

/// @cond INTERNAL
struct BenchArgs {
    int M, N;
    int warmup, iters;
};
/// @endcond

inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// Debug helpers  (compiled only when -DDEBUG)
// =============================================================================
#ifdef DEBUG
#include <iomanip>
#include <iostream>
#include <vector>

namespace cuev {
namespace debug {

/// Print a symmetric tridiagonal matrix T from its device diagonal d and subdiagonal e.
template <typename T>
void print_tridiag(const char *label, const T *d_d, const T *d_e, int n, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<T> d(n), e(n - 1);
    cudaMemcpy(d.data(), d_d, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(e.data(), d_e, (n - 1) * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "[DEBUG] " << label << ":\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "  [";
        for (int j = 0; j < n; ++j) {
            T val = (i == j) ? d[i] : (j == i - 1) ? e[j] : (j == i + 1) ? e[i] : T(0);
            std::cout << std::setw(9) << std::fixed << std::setprecision(4) << (double)val;
        }
        std::cout << " ]\n";
    }
}

/// Print the Householder vectors stored in the lower triangle of device matrix H.
template <typename T>
void print_hh_vecs(const char *label, const T *d_H, int n, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<T> H(n * n);
    cudaMemcpy(H.data(), d_H, n * n * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "[DEBUG] " << label << ":\n";
    for (int k = 0; k < n - 1; ++k) {
        std::cout << "  k=" << k << ":";
        for (int s = 0; s < k; ++s) {
            std::cout << std::setw(9) << "";
        }
        std::cout << " [";
        for (int i = 0; i < n - k - 1; ++i) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                      << (double)H[(k + 1 + i) * n + k];
        }
        std::cout << " ]\n";
    }
}

/// Print a host vector.
template <typename T> void print_vec(const char *label, const T *h_x, int n) {
    std::cout << "[DEBUG] " << label << ": [";
    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(9) << std::fixed << std::setprecision(4) << (double)h_x[i];
    }
    std::cout << " ]\n";
}

/// Print a host matrix (row-major).
template <typename T> void print_mat(const char *label, const T *h_A, int M, int N) {
    std::cout << "[DEBUG] " << label << ":\n";
    for (int i = 0; i < M; ++i) {
        std::cout << "  [";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                      << (double)h_A[i * N + j];
        }
        std::cout << " ]\n";
    }
}

} // namespace debug
} // namespace cuev
#endif // DEBUG
