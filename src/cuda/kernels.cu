/**
 * @file kernels.cu
 * 
 * cuGEMV kernel stubs — y = alpha * A * x + beta * y
 * A is M×N row-major, incx = incy = 1.  T = float or double.
 * 
 * @author Yannik Rüfenacht
 * @date 2026-06
 */

#include "kernels.cuh"
#include <cuda_runtime.h>

/// 1. Naive — one thread per output element
template <typename T>
__global__ void gemv_naive_kernel(T alpha, const T *A, const T *x,
                                  T beta, T *y, int M, int N) {
    // TODO: implement
    // Each thread computes one dot product, then scales and accumulates:
    //   dot = sum_j A[row*N + j] * x[j]
    //   y[row] = alpha * dot + beta * y[row]
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)N;
}

template <typename T>
void launch_gemv_naive(T alpha, const T *A, const T *x,
                       T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (M + BLOCK - 1) / BLOCK;
    gemv_naive_kernel<<<grid, BLOCK, 0, stream>>>(alpha, A, x, beta, y, M, N);
}

/// 2. Shared memory reduction — one block per row
template <typename T>
__global__ void gemv_smem_kernel(T alpha, const T *A, const T *x,
                                 T beta, T *y, int M, int N) {
    // TODO: implement
    // Tile x into smem, accumulate partial dot products, block-reduce, then:
    //   y[row] = alpha * dot + beta * y[row]
    extern __shared__ char smem_raw[];
    T *smem = reinterpret_cast<T *>(smem_raw);
    int row = blockIdx.x;
    if (row >= M) return;
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)N; (void)smem;
}

template <typename T>
void launch_gemv_smem(T alpha, const T *A, const T *x,
                      T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    size_t smem_bytes = BLOCK * sizeof(T);
    gemv_smem_kernel<<<M, BLOCK, smem_bytes, stream>>>(alpha, A, x, beta, y, M, N);
}

/// 4. TMA (Hopper) — bulk async copy via Tensor Memory Accelerator
template <typename T>
__global__ void gemv_tma_kernel(T alpha, const T *A, const T *x,
                                T beta, T *y, int M, int N) {
    // TODO: implement
    // Steps:
    //   1. CUtensorMap descriptor (from host)
    //   2. Issue async builk copy and sync on barrier
    //   3. Compute dot product from smem.
    //   4. Block-reduce and write: y[row] = alpha * dot + beta * y[row]
    extern __shared__ char smem_raw[];
    T *smem = reinterpret_cast<T *>(smem_raw);
    int row = blockIdx.x;
    if (row >= M) return;
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)N; (void)smem;
}

template <typename T>
void launch_gemv_tma(T alpha, const T *A, const T *x,
                     T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCK = 128;
    // smem holds one row tile + x tile + barrier storage.
    size_t smem_bytes = 2 * BLOCK * sizeof(T) + 8 /* barrier */;
    gemv_tma_kernel<<<M, BLOCK, smem_bytes, stream>>>(alpha, A, x, beta, y, M, N);
}

/// 5. Warp group (Hopper) — 128-thread cooperative unit
template <typename T>
__global__ void gemv_warpgroup_kernel(T alpha, const T *A, const T *x,
                                      T beta, T *y, int M, int N) {
    // TODO: implement
    // Partition threads into warpgroups; each warpgroup handles one or more rows.
    // Use cooperative_groups to identify position within the warpgroup.
    // Final write: y[row] = alpha * dot + beta * y[row]
    extern __shared__ char smem_raw[];
    T *smem = reinterpret_cast<T *>(smem_raw);
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)M; (void)N;
    (void)smem;
}

template <typename T>
void launch_gemv_warpgroup(T alpha, const T *A, const T *x,
                           T beta, T *y, int M, int N, cudaStream_t stream) {
    // Block must be a multiple of 128 (one warpgroup = 4 warps).
    constexpr int BLOCK = 128;
    int grid = (M + (BLOCK / 128) - 1) / (BLOCK / 128);
    size_t smem_bytes = BLOCK * sizeof(T);
    gemv_warpgroup_kernel<<<grid, BLOCK, smem_bytes, stream>>>(
        alpha, A, x, beta, y, M, N);
}

/// 6. Thread block cluster (Hopper) — distributed shared memory
template <typename T>
__global__ void __cluster_dims__(2, 1, 1)
    gemv_cluster_kernel(T alpha, const T *A, const T *x,
                        T beta, T *y, int M, int N) {
    // TODO: implement
    // Use block cluster cooperative group to share smem across blocks
    // Final write: y[row] = alpha * dot + beta * y[row]
    extern __shared__ char smem_raw[];
    T *smem = reinterpret_cast<T *>(smem_raw);

    int row = blockIdx.x;
    if (row >= M) return;
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)N; (void)smem;
}

template <typename T>
void launch_gemv_cluster(T alpha, const T *A, const T *x,
                         T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCK   = 256;
    constexpr int CLUSTER = 2;
    // Grid must be a multiple of cluster size.
    int grid = ((M + CLUSTER - 1) / CLUSTER) * CLUSTER;
    size_t smem_bytes = BLOCK * sizeof(T);

    cudaLaunchConfig_t cfg = {};
    cfg.gridDim          = grid;
    cfg.blockDim         = BLOCK;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream           = stream;

    cudaLaunchAttribute attrs[1];
    attrs[0].id               = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs    = attrs;
    cfg.numAttrs = 1;

    cudaLaunchKernelEx(&cfg, gemv_cluster_kernel<T>, alpha, A, x, beta, y, M, N);
}

/// 7. Producer / consumer (Hopper) — async barrier pipeline
template <typename T>
__global__ void gemv_opt_kernel(T alpha, const T *A, const T *x,
                                T beta, T *y, int M, int N) {
    // TODO: implement
    // Layout:
    //   smem[0..TILE-1]       : ping buffer (A tile)
    //   smem[TILE..2*TILE-1]  : pong buffer (A tile)
    //   smem[2*TILE..3*TILE-1]: x tile (reused across rows)
    //   smem[3*TILE..]        : barrier storage (2 barriers)
    //
    // Warp 0 is the producer: issues TMA async copies
    // Warps 1-3 are consumers: arrive on barrier, compute dot product.
    // Final write: y[row] = alpha * dot + beta * y[row]
    extern __shared__ char smem_raw[];
    T *smem = reinterpret_cast<T *>(smem_raw);
    int row = blockIdx.x;
    if (row >= M) return;
    (void)alpha; (void)A; (void)x; (void)beta; (void)y; (void)N; (void)smem;
}

template <typename T>
void launch_gemv_opt(T alpha, const T *A, const T *x,
                                   T beta, T *y, int M, int N,
                                   cudaStream_t stream) {
    constexpr int BLOCK = 128; // warp 0 = producer, warps 1-3 = consumers
    constexpr int TILE  = 128;
    // 2 ping-pong A buffers + 1 x buffer + 2 barriers (size TBD when implemented).
    size_t smem_bytes = 3 * TILE * sizeof(T);
    gemv_opt_kernel<<<M, BLOCK, smem_bytes, stream>>>(
        alpha, A, x, beta, y, M, N);
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                         \
    template void launch_gemv_naive<T>(T, const T *, const T *, T, T *,       \
                                       int, int, cudaStream_t);                \
    template void launch_gemv_smem<T>(T, const T *, const T *, T, T *,        \
                                      int, int, cudaStream_t);                 \
    template void launch_gemv_tma<T>(T, const T *, const T *, T, T *,         \
                                     int, int, cudaStream_t);                  \
    template void launch_gemv_warpgroup<T>(T, const T *, const T *, T, T *,   \
                                           int, int, cudaStream_t);            \
    template void launch_gemv_cluster<T>(T, const T *, const T *, T, T *,     \
                                         int, int, cudaStream_t);              \
    template void launch_gemv_opt<T>(T, const T *, const T *,   \
                                     T, T *, int, int,           \
                                     cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
