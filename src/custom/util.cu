/**
 * @file   util.cu
 * @brief  Utility kernels: fill, copy, transpose.
 *
 * Small building-block operations used by higher-level solvers and benchmarks.
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
__global__ void fill_kernel(T alpha, T *x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { x[idx] = alpha; }
}

template <typename T>
__global__ void copy_kernel(const T *x, T *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) { y[idx] = x[idx]; }
}

template <typename T, int BLOCKSIZE>
__global__ void transpose_kernel(const T *A, T *AT, int M, int N) {
    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
    if (row < M && col < N) {
        AT[col * M + row] = A[row * N + col];
    }
}

} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev::kernels {

template <typename T>
void fill(T alpha, T *x, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    fill_kernel<<<div_up(N, BLOCKSIZE), BLOCKSIZE, 0, stream>>>(alpha, x, N);
}

template <typename T>
void copy(const T *x, T *y, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    copy_kernel<<<div_up(N, BLOCKSIZE), BLOCKSIZE, 0, stream>>>(x, y, N);
}

template <typename T>
void transpose(const T *A, T *AT, int M, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    transpose_kernel<T, BLOCKSIZE>
        <<<dim3(div_up(N, BLOCKSIZE), div_up(M, BLOCKSIZE)), BLOCKSIZE * BLOCKSIZE, 0, stream>>>
        (A, AT, M, N);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                        \
    template void fill<T>     (T, T *, int, cudaStream_t);                   \
    template void copy<T>     (const T *, T *, int, cudaStream_t);           \
    template void transpose<T>(const T *, T *, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace cuev::kernels
