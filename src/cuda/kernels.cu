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
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

/// 1. gmem — one thread per output element
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

/// 2. Shared memory reduction — one block per row
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

/// 4. TMA (Hopper) — bulk async copy via Tensor Memory Accelerator
template <typename T>
__global__ void gemv_tma_kernel(T alpha, const T *A, const T *x,
                                T beta, T *y, int M, int N) {
    constexpr int TILE = 1024;
    int num_tiles = (N + TILE - 1) / TILE;
    extern __shared__ char smem[];
    T *sA = reinterpret_cast<T *>(smem);
    T *sx = sA + TILE;
    T *sr = sx + TILE;
    auto *bar = reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>(sr + blockDim.x);
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if (row >= M) return;

    if (tid == 0) init(bar, blockDim.x);
    __syncthreads();

    // loop over tiles
    T local_dot = T(0);
    for (int t = 0; t < num_tiles; ++t) {
        const int j_global = t * TILE;

        if (tid == 0) {
            cuda::device::barrier_expect_tx(*bar, 2 * TILE * sizeof(T));
            cuda::device::memcpy_async_tx(sA, A + row*N + j_global, cuda::aligned_size_t<16>(TILE * sizeof(T)), *bar);
            cuda::device::memcpy_async_tx(sx, x + j_global,          cuda::aligned_size_t<16>(TILE * sizeof(T)), *bar);
        }
        bar->arrive_and_wait();

        // compute partial dot product
        for (int j = tid; j < TILE && j_global + j < N; j += stride) {
            local_dot += sA[j] * sx[j];
        }
    }
    sr[tid] = local_dot;
    __syncthreads();

    // reduction in shared memory
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
void launch_gemv_tma(T alpha, const T *A, const T *x,
                     T beta, T *y, int M, int N, cudaStream_t stream) {
    constexpr int BLOCK = 128;
    constexpr int TILE = 1024;
    // smem layout | sA (TILE) | sx (TILE) | sr (BLOCK) | barrier (8) |
    size_t smem_bytes = 2 * (TILE * sizeof(T)) + (BLOCK * sizeof(T)) + 8;
    gemv_tma_kernel<<<M, BLOCK, smem_bytes, stream>>>(alpha, A, x, beta, y, M, N);
}

/// 7. Double buffer TMA — async barrier pipeline
template <typename T>
__global__ void gemv_double_tma_kernel(T alpha, const T *A, const T *x,
                                T beta, T *y, int M, int N) {
    // TODO: implement
    // Layout:
    //   smem[0..TILE-1]        : ping buffer (A tile)
    //   smem[TILE..2*TILE-1]   : pong buffer (A tile)
    //   smem[2*TILE..3*TILE-1] : ping buffer (x tile)
    //   smem[3*TILE..4*TILE-1] : pong buffer (x tile)
    //
    // Ping-pong buffers allow overlapping the async copy of next
    // tile with the compute of current tile
    constexpr int TILE = 1024;
    int num_tiles = (N + TILE - 1) / TILE;

    // smem setup
    extern __shared__ char smem[];
    T *sA[2] = {reinterpret_cast<T *>(smem), reinterpret_cast<T *>(smem) + TILE};
    T *sx[2] = {sA[0] + 2*TILE, sA[0] + 3*TILE};
    T *sr = sA[0] + 4*TILE;
    auto *bar = reinterpret_cast<cuda::barrier<cuda::thread_scope_block>*>(sr + blockDim.x);
    
    // thread mapping
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    int buf = 0;
    int next_buf = 1 - buf;
    if (row >= M) return;

    // setup TMA pipeline
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        init(&bar[1], blockDim.x);
    }
    __syncthreads();
    if (tid == 0) {
        // prefetch first tile
        cuda::device::barrier_expect_tx(bar[buf], 2 * TILE * sizeof(T));
        cuda::device::memcpy_async_tx(sA[buf], A + row*N, cuda::aligned_size_t<16>(TILE * sizeof(T)), bar[buf]);
        cuda::device::memcpy_async_tx(sx[buf], x,          cuda::aligned_size_t<16>(TILE * sizeof(T)), bar[buf]);
    }
    __syncthreads();

    // loop over tiles
    T local_dot = T(0);
    for (int t = 0; t < num_tiles; ++t) {
        const int j_global = t * TILE;
        // TMA load of next tile
        if (tid == 0 && t + 1 < num_tiles) {
            cuda::device::barrier_expect_tx(bar[next_buf], 2 * TILE * sizeof(T));
            cuda::device::memcpy_async_tx(sA[next_buf], A + row*N + j_global + TILE, cuda::aligned_size_t<16>(TILE * sizeof(T)), bar[next_buf]);
            cuda::device::memcpy_async_tx(sx[next_buf], x + j_global + TILE, cuda::aligned_size_t<16>(TILE * sizeof(T)), bar[next_buf]);
        }
        
        bar[buf].arrive_and_wait();
        // compute current tile dot product
        for (int j = tid; j < TILE && j_global + j < N; j += stride) {
            local_dot += sA[buf][j] * sx[buf][j];
        }
        __syncthreads();

        // swap buffers
        buf = next_buf;
        next_buf = 1 - buf;
    }
    sr[tid] = local_dot;
    __syncthreads();

    // reduction in shared memory
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
void launch_gemv_double_tma(T alpha, const T *A, const T *x,
                                   T beta, T *y, int M, int N,
                                   cudaStream_t stream) {
    constexpr int BLOCK = 128;
    constexpr int TILE = 1024;
    // smem: [ sA_ping | sA_pong | sx_ping | sx_pong | sr | bar[0] | bar[1] ]
    size_t smem_bytes = 4 * (TILE * sizeof(T)) + (BLOCK * sizeof(T)) + 16;
    gemv_double_tma_kernel<<<M, BLOCK, smem_bytes, stream>>>(alpha, A, x, beta, y, M, N);
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

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                         \
    template void launch_gemv_gmem<T>(T, const T *, const T *, T, T *,       \
                                       int, int, cudaStream_t);                \
    template void launch_gemv_smem<T>(T, const T *, const T *, T, T *,        \
                                      int, int, cudaStream_t);                 \
    template void launch_gemv_tma<T>(T, const T *, const T *, T, T *,         \
                                     int, int, cudaStream_t);                  \
    template void launch_gemv_cluster<T>(T, const T *, const T *, T, T *,     \
                                         int, int, cudaStream_t);              \
    template void launch_gemv_double_tma<T>(T, const T *, const T *,   \
                                     T, T *, int, int,           \
                                     cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
