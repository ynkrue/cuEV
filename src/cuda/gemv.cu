/**
 * @file gemv.cu
 * 
 * cuGEMV kernel — y = alpha * A * x + beta * y
 * A is M×N row-major, incx = incy = 1.  T = float or double.
 * 
 * @author Yannik Rüfenacht
 * @date 2026-06
 */

#include "kernels.cuh"
#include <cuda.h>

/// gmem — one thread per output element
template <typename T>
__global__ void gemv_gmem_kernel(T alpha, const T *A, const T *x,
                                  T beta, T *y, int M, int N) {
    // Each thread computes one dot product, then scales and accumulates
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    T dot = T(0);
    for (int j = 0; j < N; ++j) {
        dot += A[row * N + j] * x[j];
    }
    y[row] = alpha * dot + beta * y[row];
}

template <typename T>
void launch_gemv_gmem(T alpha, const T *A, const T *x,
                       T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (M + BLOCK - 1) / BLOCK;
    gemv_gmem_kernel<<<grid, BLOCK, 0, stream>>>(alpha, A, x, beta, y, M, N);
}

/// Shared memory reduction — one block per row
template <typename T>
__global__ void gemv_smem_kernel(T alpha, const T *A, const T *x,
                                 T beta, T *y, int M, int N) {
    // Tile x into smem, accumulate partial dot products, block-reduce
    __shared__ T sr[256];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (row >= M) return;
    
    // compute partial dot product
    T local_dot = T(0);
    for (int j = tid; j < N; j += stride) {
        local_dot += A[row * N + j] * x[j];
    }
    sr[tid] = local_dot;
    __syncthreads();

    // block reduction in shared memory
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) sr[tid] += sr[tid + s];
        __syncthreads();
    }

    // warp reduction
    if (tid < 32) {
        T val = sr[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        // store result
        if (tid == 0) y[row] = alpha * val + beta * y[row];
    }
}

template <typename T>
void launch_gemv_smem(T alpha, const T *A, const T *x,
                      T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    gemv_smem_kernel<<<M, BLOCK, 0, stream>>>(alpha, A, x, beta, y, M, N);
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                         \
    template void launch_gemv_gmem<T>(T, const T *, const T *, T, T *,       \
                                       int, int, cudaStream_t);                \
    template void launch_gemv_smem<T>(T, const T *, const T *, T, T *,        \
                                      int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
