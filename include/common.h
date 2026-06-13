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
// Device helpers
// =============================================================================
#ifdef __CUDACC__

/// Type-safe absolute value for device code (avoids int-overload ambiguity with C abs()).
template <typename T> __device__ __forceinline__ T tabs(T x) {
    return x < T(0) ? -x : x;
}

/// Block-wide sum reduction. Each thread passes its partial sum in @p val and
/// a shared buffer @p smem of length blockDim.x. Returns the total sum in
/// thread 0 (result is undefined in other threads). Caller is responsible for
/// the tid==0 write-back and the subsequent __syncthreads broadcast.
template <typename T, int BLOCKSIZE> __device__ __forceinline__ T block_reduce_sum(T val, T *smem) {
    smem[threadIdx.x] = val;
    __syncthreads();

    for (int s = BLOCKSIZE >> 1; s >= 32; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }

    T v = T(0);
    if (threadIdx.x < 32) {
        v = smem[threadIdx.x];
        v += __shfl_down_sync(0xffffffff, v, 16);
        v += __shfl_down_sync(0xffffffff, v, 8);
        v += __shfl_down_sync(0xffffffff, v, 4);
        v += __shfl_down_sync(0xffffffff, v, 2);
        v += __shfl_down_sync(0xffffffff, v, 1);
    }
    return v;
}

#endif // __CUDACC__

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
    std::cout << "  " << label << ":\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "    [";
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
    std::cout << "  " << label << ":\n";
    for (int k = 0; k < n - 1; ++k) {
        std::cout << "    k=" << k << ":";
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
    std::cout << "     " << label << ": [";
    for (int i = 0; i < n; ++i) {
        std::cout << std::setw(9) << std::fixed << std::setprecision(4) << (double)h_x[i];
    }
    std::cout << " ]\n";
}

/// Print a host matrix (row-major).
template <typename T> void print_mat(const char *label, const T *h_A, int M, int N) {
    std::cout << "     " << label << ":\n";
    for (int i = 0; i < M; ++i) {
        std::cout << "    [";
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                      << (double)h_A[i * N + j];
        }
        std::cout << " ]\n";
    }
}

/// Print eigenvalues from a device pointer.
template <typename T>
void print_eval(const char *label, const T *d_eval, int n, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<T> eval(n);
    cudaMemcpy(eval.data(), d_eval, n * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "  " << label << ":\n    [";
    for (int i = 0; i < n; ++i)
        std::cout << std::setw(9) << std::fixed << std::setprecision(4) << (double)eval[i];
    std::cout << " ]\n";
}

/// Print eigenvector matrix QT (n×n, row-major, rows = eigenvectors) from a device pointer.
template <typename T>
void print_evec(const char *label, const T *d_evec, int n, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<T> evec(n * n);
    cudaMemcpy(evec.data(), d_evec, n * n * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "  " << label << ":\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "    [";
        for (int j = 0; j < n; ++j)
            std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                      << (double)evec[i * n + j];
        std::cout << " ]\n";
    }
}

} // namespace debug
} // namespace cuev
#endif // DEBUG
