/**
 * @file   gemv.cu
 * @brief  GEMV kernel implementations: y ← αAx + βy, A M×N row-major.
 *
 * Two variants:
 *   gmem  — one thread per output element, no data reuse
 *   smem  — one block per row, partial sums reduced through shared memory
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cuda.h>

// =============================================================================
// Device kernels (translation-unit private)
// =============================================================================
namespace {

template <typename T>
__global__ void gemv_gmem_kernel(T alpha, const T *A, const T *x,
                                  T beta, T *y, int M, int N) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    T dot = T(0);
    for (int j = 0; j < N; ++j) {
        dot += A[row * N + j] * x[j];
    }
    y[row] = alpha * dot + beta * y[row];
}

template <typename T, int BLOCKSIZE>
__global__ void gemv_smem_kernel(T alpha, const T *A, const T *x,
                                  T beta, T *y, int M, int N) {
    __shared__ T sr[BLOCKSIZE];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= M) return;

    T acc = T(0);
    for (int j = tid; j < N; j += BLOCKSIZE) {
        acc += A[row * N + j] * x[j];
    }
    sr[tid] = acc;
    __syncthreads();

    for (int s = BLOCKSIZE >> 1; s >= 32; s >>= 1) {
        if (tid < s) { sr[tid] += sr[tid + s]; }
        __syncthreads();
    }
    if (tid < 32) {
        T val = sr[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) { y[row] = alpha * val + beta * y[row]; }
    }
}

} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev::kernels {

template <typename T>
void gemv_gmem(T alpha, const T *A, const T *x,
               T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    gemv_gmem_kernel<<<div_up(M, BLOCKSIZE), BLOCKSIZE, 0, stream>>>
        (alpha, A, x, beta, y, M, N);
}

template <typename T>
void gemv_smem(T alpha, const T *A, const T *x,
               T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    gemv_smem_kernel<T, BLOCKSIZE><<<M, BLOCKSIZE, 0, stream>>>
        (alpha, A, x, beta, y, M, N);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                              \
    template void gemv_gmem<T>(T, const T *, const T *, T, T *, int, int, cudaStream_t); \
    template void gemv_smem<T>(T, const T *, const T *, T, T *, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cuev::kernels
