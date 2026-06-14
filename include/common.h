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

#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t _e = (err);                                                               \
        if (_e != CUSOLVER_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__, (int)_e);            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// =============================================================================
// Utilities
// =============================================================================

inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// =============================================================================
// Device helpers
// =============================================================================
#ifdef __CUDACC__

template <typename T> __device__ __forceinline__ T tabs(T x) {
    return x < T(0) ? -x : x;
}

/// Block-wide sum reduction into thread 0. Caller broadcasts and syncs after.
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
