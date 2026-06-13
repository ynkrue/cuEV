/**
 * @file gemm.cu
 * 
 * cuGEMM kernel — C = alpha * A * B + beta * C
 * A is M×K row-major, B is K×N row-major, C is M×N row-major.
 * T = float or double.
 * 
 * @author Yannik Rüfenacht
 * @date 2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cuda.h>

/// gmem — one thread per output element
template <typename T, int BLOCKSIZE>
__global__ void gemm_gmem_kernel(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K) {
    // thread layout
    int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
    int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;

    // compute C[row, col]
    if (row < M && col < N) {
        T acc = T(0);
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

template <typename T>
void launch_gemm_gmem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(div_up(M, BLOCKSIZE), div_up(N, BLOCKSIZE));
    gemm_gmem_kernel<T, BLOCKSIZE><<<grid, block, 0, stream>>>(alpha, A, B, beta, C, M, N, K);
}

/// Shared memory reduction — one block per row
template <typename T, int BLOCKSIZE>
__global__ void gemm_smem_kernel(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K) {
    // smem layout
    __shared__ T sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ T sB[BLOCKSIZE * BLOCKSIZE];

    // thread layout
    int t_row = threadIdx.x / BLOCKSIZE; // row for A and C
    int t_col = threadIdx.x % BLOCKSIZE; // col for B and C
    int b_row = blockIdx.x;              // block row for A and C
    int b_col = blockIdx.y;              // block col for B and C

    if (b_row * BLOCKSIZE + t_row >= M || b_col * BLOCKSIZE + t_col >= N) return;

    A += b_row * BLOCKSIZE * K;
    B += b_col * BLOCKSIZE;

    T acc = T(0);
    for (int bk = 0; bk < K; bk += BLOCKSIZE) {
        // load A and B into smem
        sA[t_row * BLOCKSIZE + t_col] = (bk + t_col < K) ? A[t_row * K + t_col] : T(0);
        sB[t_row * BLOCKSIZE + t_col] = (bk + t_row < K) ? B[t_row * N + t_col] : T(0);
        __syncthreads();

        // move to next block
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // compute partial product for C[b_row, b_col]
        for (int k = 0; k < BLOCKSIZE; ++k) {
            acc += sA[t_row * BLOCKSIZE + k] * sB[k * BLOCKSIZE + t_col];
        }
        __syncthreads();
    }

    // store result
    C += b_row * BLOCKSIZE * N + b_col * BLOCKSIZE;
    C[t_row * N + t_col] = alpha * acc + beta * C[t_row * N + t_col];
}

template <typename T>
void launch_gemm_smem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(div_up(M, BLOCKSIZE), div_up(N, BLOCKSIZE));
    gemm_smem_kernel<T, BLOCKSIZE><<<grid, block, 0, stream>>>(alpha, A, B, beta, C, M, N, K);
}

/// Tiled memory reduction — one block per row
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled_kernel(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K) {
    // smem layout
    __shared__ T sA[BM * BK];
    __shared__ T sB[BK * BN];
    // register layout
    T rA[TM];
    T rB[TN];
    T acc[TM * TN] = {0};

    // thread layout
    const int num_threads = BM * BN / (TM * TN);
    // block level mapping
    int b_row = blockIdx.y;
    int b_col = blockIdx.x;
    // thread compute mapping
    int t_col_c = threadIdx.x % (BN / TN);
    int t_row_c = threadIdx.x / (BN / TN);
    // thread shared memory mapping
    int t_row_a = threadIdx.x / BK;
    int t_col_a = threadIdx.x % BK;
    int stride_a = num_threads / BK;
    int t_row_b = threadIdx.x / BN;
    int t_col_b = threadIdx.x % BN;
    int stride_b = num_threads / BN;

    if (b_row * BM + t_row_c * TM >= M || b_col * BN + t_col_c * TN >= N) return;

    A += b_row * BM * K;
    B += b_col * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // load A and B blocks into smem
        for (int offset = 0; offset < BM; offset += stride_a) {
            sA[(t_row_a + offset) * BK + t_col_a] =
                (b_row * BM + t_row_a + offset < M && bk + t_col_a < K) ? A[(t_row_a + offset) * K + t_col_a] : T(0);
        }
        for (int offset = 0; offset < BK; offset += stride_b) {
            sB[(t_row_b + offset) * BN + t_col_b] =
                (bk + t_row_b + offset < K && b_col * BN + t_col_b < N) ? B[(t_row_b + offset) * N + t_col_b] : T(0);
        }
        __syncthreads();

        // move to next block
        A += BK;
        B += BK * N;

        // compute partial product for C[b_row, b_col]
        for (int k = 0; k < BK; ++k) {
            // load A and B tiles into registers
            for (int tm = 0; tm < TM; ++tm) rA[tm] = sA[(t_row_c * TM + tm) * BK + k];
            for (int tn = 0; tn < TN; ++tn) rB[tn] = sB[k * BN + t_col_c * TN + tn];
            // compute tile product
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm * TN + tn] += rA[tm] * rB[tn];
                }
            }
        }
        __syncthreads();
    }

    // store result
    C += b_row * BM * N + b_col * BN;
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            C[(t_row_c * TM + tm) * N + t_col_c * TN + tn] =
                alpha * acc[tm * TN + tn] + beta * C[(t_row_c * TM + tm) * N + t_col_c * TN + tn];
        }
    }
}

template <typename T>
void launch_gemm_tiled(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 block(BM * BN / (TM * TN));
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_tiled_kernel<T, BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(alpha, A, B, beta, C, M, N, K);
}

/// Warp-tiled with 128-bit vectorized memory access — block → warp → thread tiles
/// Assumes M % BM == 0, N % BN == 0, K % BK == 0 (launcher falls back to gemm_tiled).
template <typename T>
struct alignas(16) Vec128 {
    static constexpr int width = 16 / sizeof(T);
    T v[width];
};

template <typename T, int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
gemm_warptile_kernel(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K) {
    using Vec = Vec128<T>;
    constexpr int V = Vec::width;
    constexpr int WARPSIZE = 32;

    // warp subtile decomposition: each warp owns WM×WN, iterated in WMITER×WNITER steps
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // smem layout — sA is stored transposed (k-major) for contiguous inner-loop reads
    __shared__ T sA[BK * BM];
    __shared__ T sB[BK * BN];
    // register layout
    T rA[WMITER * TM];
    T rB[WNITER * TN];
    T acc[WMITER * TM * WNITER * TN] = {0};

    // block level mapping
    int b_row = blockIdx.y;
    int b_col = blockIdx.x;
    // warp level mapping
    int w_row = (threadIdx.x / WARPSIZE) / (BN / WN);
    int w_col = (threadIdx.x / WARPSIZE) % (BN / WN);
    // thread level mapping within warp subtile
    int t_row = (threadIdx.x % WARPSIZE) / (WSUBN / TN);
    int t_col = (threadIdx.x % WARPSIZE) % (WSUBN / TN);
    // thread shared memory mapping (vectorized)
    int t_row_a = threadIdx.x / (BK / V);
    int t_col_a = (threadIdx.x % (BK / V)) * V;
    constexpr int stride_a = NUM_THREADS * V / BK;
    int t_row_b = threadIdx.x / (BN / V);
    int t_col_b = (threadIdx.x % (BN / V)) * V;
    constexpr int stride_b = NUM_THREADS * V / BN;

    A += b_row * BM * K;
    B += b_col * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // load A block into smem, transposed
        for (int offset = 0; offset < BM; offset += stride_a) {
            Vec tmp = *reinterpret_cast<const Vec *>(&A[(t_row_a + offset) * K + t_col_a]);
            for (int i = 0; i < V; ++i)
                sA[(t_col_a + i) * BM + t_row_a + offset] = tmp.v[i];
        }
        // load B block into smem
        for (int offset = 0; offset < BK; offset += stride_b) {
            *reinterpret_cast<Vec *>(&sB[(t_row_b + offset) * BN + t_col_b]) =
                *reinterpret_cast<const Vec *>(&B[(t_row_b + offset) * N + t_col_b]);
        }
        __syncthreads();

        // move to next block
        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            // load warp subtiles into registers
            for (int wm = 0; wm < WMITER; ++wm)
                for (int tm = 0; tm < TM; ++tm)
                    rA[wm * TM + tm] = sA[k * BM + w_row * WM + wm * WSUBM + t_row * TM + tm];
            for (int wn = 0; wn < WNITER; ++wn)
                for (int tn = 0; tn < TN; ++tn)
                    rB[wn * TN + tn] = sB[k * BN + w_col * WN + wn * WSUBN + t_col * TN + tn];
            // outer product over warp subtiles
            for (int wm = 0; wm < WMITER; ++wm)
                for (int wn = 0; wn < WNITER; ++wn)
                    for (int tm = 0; tm < TM; ++tm)
                        for (int tn = 0; tn < TN; ++tn)
                            acc[(wm * TM + tm) * (WNITER * TN) + wn * TN + tn] +=
                                rA[wm * TM + tm] * rB[wn * TN + tn];
        }
        __syncthreads();
    }

    // store result (vectorized)
    C += (b_row * BM + w_row * WM) * N + b_col * BN + w_col * WN;
    for (int wm = 0; wm < WMITER; ++wm) {
        for (int wn = 0; wn < WNITER; ++wn) {
            T *Cw = C + wm * WSUBM * N + wn * WSUBN;
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; tn += V) {
                    Vec out = *reinterpret_cast<Vec *>(&Cw[(t_row * TM + tm) * N + t_col * TN + tn]);
                    for (int i = 0; i < V; ++i)
                        out.v[i] = alpha * acc[(wm * TM + tm) * (WNITER * TN) + wn * TN + tn + i] +
                                   beta * out.v[i];
                    *reinterpret_cast<Vec *>(&Cw[(t_row * TM + tm) * N + t_col * TN + tn]) = out;
                }
            }
        }
    }
}

template <typename T>
void launch_gemm_warptile(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K, cudaStream_t stream) {
    if constexpr (sizeof(T) == 4) {
        constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 64, WNITER = 4, TM = 8, TN = 4, NT = 128;
        if (M % BM || N % BN || K % BK)
            return launch_gemm_tiled(alpha, A, B, beta, C, M, N, K, stream);
        gemm_warptile_kernel<T, BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
            <<<dim3(N / BN, M / BM), NT, 0, stream>>>(alpha, A, B, beta, C, M, N, K);
    } else {
        // fp64: smaller BN/TN halves register pressure (each double accumulator = 2 registers)
        constexpr int BM = 128, BN = 64, BK = 16, WM = 64, WN = 32, WNITER = 2, TM = 8, TN = 4, NT = 128;
        if (M % BM || N % BN || K % BK)
            return launch_gemm_tiled(alpha, A, B, beta, C, M, N, K, stream);
        gemm_warptile_kernel<T, BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
            <<<dim3(N / BN, M / BM), NT, 0, stream>>>(alpha, A, B, beta, C, M, N, K);
    }
}

// ---------------------------------------------------------------------------
// fp64 tensor-core kernel (wmma 8×8×4 DMMA, sm_80+)
// ---------------------------------------------------------------------------
// template <typename T>
// void gemm_dmma_kernel(double alpha, const double *A, const double *B, double beta, double *C, int M, int N, int K) {
// }

// template <typename T>
// void launch_gemm_dmma(double alpha, const double *A, const double *B, double beta, double *C, int M, int N, int K, cudaStream_t stream);


// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                         \
    template void launch_gemm_gmem<T>(T, const T *, const T *, T, T *,         \
                                       int, int, int, cudaStream_t);           \
    template void launch_gemm_smem<T>(T, const T *, const T *, T, T *,         \
                                       int, int, int, cudaStream_t);           \
    template void launch_gemm_tiled<T>(T, const T *, const T *, T, T *,        \
                                       int, int, int, cudaStream_t);           \
    template void launch_gemm_warptile<T>(T, const T *, const T *, T, T *,     \
                                       int, int, int, cudaStream_t);           \
    // template void launch_gemm_dmma<T>(T, const T *, const T *, T, T *,      
    //                                   int, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
