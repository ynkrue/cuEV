/**
 * @file util.cu
 * 
 * Utility kernels for cuEigen — fill, copy, transpose
 * These are used by higher-level algorithms (e.g. QR) and benchmarks.
 * 
 * @author Yannik Rüfenacht
 * @date 2026-06
 */

#include "kernels.cuh"
#include <cuda.h>

/// fill x ← α
template <typename T>
__global__ void fill_kernel(T alpha, const T *A, const T *x,
                                  T beta, T *y, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = alpha;
    }
}

template <typename T>
void launch_fill(T alpha, const T *A, const T *x,
                       T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    int grid = (M + BLOCKSIZE - 1) / BLOCKSIZE;
    fill_kernel<<<grid, BLOCKSIZE, 0, stream>>>(alpha, A, x, beta, y, M, N);
}

/// copy B ← A
template <typename T>
__global__ void copy_kernel(const T *A, T *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        B[idx] = A[idx];
    }
}

template <typename T>
void launch_copy(const T *A, T *B, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    int grid = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    copy_kernel<<<grid, BLOCKSIZE, 0, stream>>>(A, B, N);
}

/// transpose Aᵀ ← A
template <typename T, int BLOCKSIZE>
__global__ void transpose_kernel(const T *A, T *AT, int M, int N) {
    int row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    int col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (row < M && col < N) {
        AT[col * M + row] = A[row * N + col];
    }
}

template <typename T>
void launch_transpose(const T *A, T *AT, int M, int N, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    transpose_kernel<T, BLOCKSIZE><<<grid, block, 0, stream>>>(A, AT, M, N);
}


// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                          \
    template void launch_fill<T>(T, const T *, const T *, T, T *,               \
                                       int, int, cudaStream_t);                 \
    template void launch_copy<T>(const T *, T *, int, cudaStream_t);            \
    template void launch_transpose<T>(const T *, T *, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
