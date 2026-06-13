/**
 * @file   gemm.cu
 * @brief  GEMM kernel implementations: C ← αAB + βC, all matrices row-major.
 *
 * Four variants in order of increasing sophistication:
 *   gmem      — one thread per output element, no data reuse
 *   smem      — shared-memory tiled (BLOCKSIZE×BLOCKSIZE tiles of A and B)
 *   tiled     — register-tiled; each thread accumulates a TM×TN output tile
 *   warptile  — warp-tiled with 128-bit vectorized loads; sA stored transposed
 *               in shared memory for contiguous inner-loop reads
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cuda.h>
#include <type_traits>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

// -----------------------------------------------------------------------------
// gmem
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void gemm_gmem_kernel(T alpha, const T *A, const T *B, T beta, T *C, int M, int N,
                                 int K) {
    int row = blockIdx.x * BLOCKSIZE + threadIdx.x / BLOCKSIZE;
    int col = blockIdx.y * BLOCKSIZE + threadIdx.x % BLOCKSIZE;
    if (row < M && col < N) {
        T acc = T(0);
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

// -----------------------------------------------------------------------------
// smem
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void gemm_smem_kernel(T alpha, const T *A, const T *B, T beta, T *C, int M, int N,
                                 int K) {
    __shared__ T sA[BLOCKSIZE * BLOCKSIZE];
    __shared__ T sB[BLOCKSIZE * BLOCKSIZE];

    int t_row = threadIdx.x / BLOCKSIZE;
    int t_col = threadIdx.x % BLOCKSIZE;
    int b_row = blockIdx.x;
    int b_col = blockIdx.y;

    int row = b_row * BLOCKSIZE + t_row;
    int col = b_col * BLOCKSIZE + t_col;

    A += b_row * BLOCKSIZE * K;
    B += b_col * BLOCKSIZE;

    T acc = T(0);
    for (int bk = 0; bk < K; bk += BLOCKSIZE) {
        sA[t_row * BLOCKSIZE + t_col] = (row < M && bk + t_col < K) ? A[t_row * K + t_col] : T(0);
        sB[t_row * BLOCKSIZE + t_col] = (bk + t_row < K && col < N) ? B[t_row * N + t_col] : T(0);
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        for (int k = 0; k < BLOCKSIZE; ++k) {
            acc += sA[t_row * BLOCKSIZE + k] * sB[k * BLOCKSIZE + t_col];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C += b_row * BLOCKSIZE * N + b_col * BLOCKSIZE;
        C[t_row * N + t_col] = alpha * acc + beta * C[t_row * N + t_col];
    }
}

// -----------------------------------------------------------------------------
// tiled
// -----------------------------------------------------------------------------
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_tiled_kernel(T alpha, const T *A, const T *B, T beta, T *C, int M, int N,
                                  int K) {
    __shared__ T sA[BM * BK];
    __shared__ T sB[BK * BN];
    T rA[TM], rB[TN];
    T acc[TM * TN] = {0};

    constexpr int num_threads = BM * BN / (TM * TN);
    int b_row = blockIdx.y;
    int b_col = blockIdx.x;
    int t_col_c = threadIdx.x % (BN / TN);
    int t_row_c = threadIdx.x / (BN / TN);
    int t_row_a = threadIdx.x / BK;
    int t_col_a = threadIdx.x % BK;
    constexpr int stride_a = num_threads / BK;
    int t_row_b = threadIdx.x / BN;
    int t_col_b = threadIdx.x % BN;
    constexpr int stride_b = num_threads / BN;

    if (b_row * BM + t_row_c * TM >= M || b_col * BN + t_col_c * TN >= N) return;

    A += b_row * BM * K;
    B += b_col * BN;

    for (int bk = 0; bk < K; bk += BK) {
        for (int offset = 0; offset < BM; offset += stride_a) {
            sA[(t_row_a + offset) * BK + t_col_a] =
                (b_row * BM + t_row_a + offset < M && bk + t_col_a < K)
                    ? A[(t_row_a + offset) * K + t_col_a]
                    : T(0);
        }
        for (int offset = 0; offset < BK; offset += stride_b) {
            sB[(t_row_b + offset) * BN + t_col_b] =
                (bk + t_row_b + offset < K && b_col * BN + t_col_b < N)
                    ? B[(t_row_b + offset) * N + t_col_b]
                    : T(0);
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int tm = 0; tm < TM; ++tm) {
                rA[tm] = sA[(t_row_c * TM + tm) * BK + k];
            }
            for (int tn = 0; tn < TN; ++tn) {
                rB[tn] = sB[k * BN + t_col_c * TN + tn];
            }
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm * TN + tn] += rA[tm] * rB[tn];
                }
            }
        }
        __syncthreads();
    }

    C += b_row * BM * N + b_col * BN;
    for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
            C[(t_row_c * TM + tm) * N + t_col_c * TN + tn] =
                alpha * acc[tm * TN + tn] + beta * C[(t_row_c * TM + tm) * N + t_col_c * TN + tn];
        }
    }
}

// -----------------------------------------------------------------------------
// warptile
// -----------------------------------------------------------------------------

// 128-bit vectorized load/store helper
template <typename T> struct alignas(16) Vec128 {
    static constexpr int width = 16 / sizeof(T);
    T v[width];
};

template <typename T, int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS) void gemm_warptile_kernel(T alpha, const T *A, const T *B,
                                                                    T beta, T *C, int M, int N,
                                                                    int K) {
    using Vec = Vec128<T>;
    constexpr int V = Vec::width;
    constexpr int WARPSIZE = 32;

    // each warp owns a WM×WN output tile, iterated in WMITER×WNITER register subtiles
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    // sA is stored transposed (k-major) so inner-loop reads are contiguous
    __shared__ T sA[BK * BM];
    __shared__ T sB[BK * BN];
    T rA[WMITER * TM], rB[WNITER * TN];
    T acc[WMITER * TM * WNITER * TN] = {0};

    int b_row = blockIdx.y;
    int b_col = blockIdx.x;
    int w_row = (threadIdx.x / WARPSIZE) / (BN / WN);
    int w_col = (threadIdx.x / WARPSIZE) % (BN / WN);
    int t_row = (threadIdx.x % WARPSIZE) / (WSUBN / TN);
    int t_col = (threadIdx.x % WARPSIZE) % (WSUBN / TN);
    int t_row_a = threadIdx.x / (BK / V);
    int t_col_a = (threadIdx.x % (BK / V)) * V;
    constexpr int stride_a = NUM_THREADS * V / BK;
    int t_row_b = threadIdx.x / (BN / V);
    int t_col_b = (threadIdx.x % (BN / V)) * V;
    constexpr int stride_b = NUM_THREADS * V / BN;

    A += b_row * BM * K;
    B += b_col * BN;

    for (int bk = 0; bk < K; bk += BK) {
        for (int offset = 0; offset < BM; offset += stride_a) {
            Vec tmp = *reinterpret_cast<const Vec *>(&A[(t_row_a + offset) * K + t_col_a]);
            for (int i = 0; i < V; ++i) {
                sA[(t_col_a + i) * BM + t_row_a + offset] = tmp.v[i];
            }
        }
        for (int offset = 0; offset < BK; offset += stride_b) {
            *reinterpret_cast<Vec *>(&sB[(t_row_b + offset) * BN + t_col_b]) =
                *reinterpret_cast<const Vec *>(&B[(t_row_b + offset) * N + t_col_b]);
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; ++k) {
            for (int wm = 0; wm < WMITER; ++wm) {
                for (int tm = 0; tm < TM; ++tm) {
                    rA[wm * TM + tm] = sA[k * BM + w_row * WM + wm * WSUBM + t_row * TM + tm];
                }
            }
            for (int wn = 0; wn < WNITER; ++wn) {
                for (int tn = 0; tn < TN; ++tn) {
                    rB[wn * TN + tn] = sB[k * BN + w_col * WN + wn * WSUBN + t_col * TN + tn];
                }
            }
            for (int wm = 0; wm < WMITER; ++wm) {
                for (int wn = 0; wn < WNITER; ++wn) {
                    for (int tm = 0; tm < TM; ++tm) {
                        for (int tn = 0; tn < TN; ++tn) {
                            acc[(wm * TM + tm) * (WNITER * TN) + wn * TN + tn] +=
                                rA[wm * TM + tm] * rB[wn * TN + tn];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    C += (b_row * BM + w_row * WM) * N + b_col * BN + w_col * WN;
    for (int wm = 0; wm < WMITER; ++wm) {
        for (int wn = 0; wn < WNITER; ++wn) {
            T *Cw = C + wm * WSUBM * N + wn * WSUBN;
            for (int tm = 0; tm < TM; ++tm) {
                for (int tn = 0; tn < TN; tn += V) {
                    Vec out =
                        *reinterpret_cast<Vec *>(&Cw[(t_row * TM + tm) * N + t_col * TN + tn]);
                    for (int i = 0; i < V; ++i) {
                        out.v[i] = alpha * acc[(wm * TM + tm) * (WNITER * TN) + wn * TN + tn + i] +
                                   beta * out.v[i];
                    }
                    *reinterpret_cast<Vec *>(&Cw[(t_row * TM + tm) * N + t_col * TN + tn]) = out;
                }
            }
        }
    }
}

} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev {
namespace kernels {

template <typename T>
void gemm_gmem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
               cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    gemm_gmem_kernel<T, BLOCKSIZE>
        <<<dim3(div_up(M, BLOCKSIZE), div_up(N, BLOCKSIZE)), BLOCKSIZE * BLOCKSIZE, 0, stream>>>(
            alpha, A, B, beta, C, M, N, K);
}

template <typename T>
void gemm_smem(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
               cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    gemm_smem_kernel<T, BLOCKSIZE>
        <<<dim3(div_up(M, BLOCKSIZE), div_up(N, BLOCKSIZE)), BLOCKSIZE * BLOCKSIZE, 0, stream>>>(
            alpha, A, B, beta, C, M, N, K);
}

template <typename T>
void gemm_tiled(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
                cudaStream_t stream) {
    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    gemm_tiled_kernel<T, BM, BN, BK, TM, TN>
        <<<dim3(div_up(N, BN), div_up(M, BM)), BM * BN / (TM * TN), 0, stream>>>(alpha, A, B, beta,
                                                                                 C, M, N, K);
}

template <typename T>
void gemm_warptile(T alpha, const T *A, const T *B, T beta, T *C, int M, int N, int K,
                   cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>) {
        constexpr int BM = 128, BN = 128, BK = 16, WM = 64, WN = 64, WNITER = 4, TM = 8, TN = 4,
                      NT = 128;
        if (M % BM || N % BN || K % BK) {
            return gemm_tiled(alpha, A, B, beta, C, M, N, K, stream);
        }
        gemm_warptile_kernel<T, BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
            <<<dim3(N / BN, M / BM), NT, 0, stream>>>(alpha, A, B, beta, C, M, N, K);
    } else {
        // fp64: narrower BN and smaller TN halve register pressure
        // (each double accumulator occupies 2 registers vs 1 for float)
        constexpr int BM = 128, BN = 64, BK = 16, WM = 64, WN = 32, WNITER = 2, TM = 8, TN = 4,
                      NT = 128;
        if (M % BM || N % BN || K % BK) {
            return gemm_tiled(alpha, A, B, beta, C, M, N, K, stream);
        }
        gemm_warptile_kernel<T, BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
            <<<dim3(N / BN, M / BM), NT, 0, stream>>>(alpha, A, B, beta, C, M, N, K);
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void gemm_gmem<T>(T, const T *, const T *, T, T *, int, int, int, cudaStream_t);      \
    template void gemm_smem<T>(T, const T *, const T *, T, T *, int, int, int, cudaStream_t);      \
    template void gemm_tiled<T>(T, const T *, const T *, T, T *, int, int, int, cudaStream_t);     \
    template void gemm_warptile<T>(T, const T *, const T *, T, T *, int, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
