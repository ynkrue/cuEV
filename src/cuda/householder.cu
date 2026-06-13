/**
 * @file householder.cu
 * 
 * Householder transformations for cuEigen QR solver
 * 
 * @author Yannik Rüfenacht
 * @date 2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cuda.h>

/// hh_reflect v, τ ← H[k+1:n, k] - alpha e_k (Householder reflector)
template <typename T, int BLOCKSIZE>
__global__ void hh_reflect_kernel(T *H, T *v, T *tau, T *d, T *e, int M, int k) {
    // thread layout
    __shared__ T snrm[BLOCKSIZE];
    const int tid = threadIdx.x;
    T nrm2 = T(0);

    for (int i = tid; i < M - k - 1; i += blockDim.x) {
        T a = H[(k + 1 + i) * M + k];
        nrm2 += a * a;
        v[i] = a;
    }
    snrm[tid] = nrm2;
    __syncthreads();

    // reduce nrm2 in smem
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) snrm[tid] += snrm[tid + s];
        __syncthreads();
    }

    // reduce nrm2 in warp
    if (tid < 32) {
        T val = snrm[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) {
            T norm = sqrt(val);
            T x0 = H[(k + 1) * M + k];
            T alpha = -copysign(norm, x0);  // alpha = -sign(x0) * ||x||
            T vTv = norm * norm - alpha * x0;

            // store Householder vector v and tau
            v[0] = x0 - alpha; // v = x - alpha e_1
            H[(k + 1) * M + k] = x0 - alpha;
            tau[k] = (vTv == T(0)) ? T(0) : T(1) / vTv;

            // store (sub)diagonal of H
            if (k < M - 1) e[k] = alpha;
            d[k] = H[k * M + k];
        }
        __syncthreads();
    }
}

template <typename T>
void launch_hh_reflect(T *H, T *v, T *tau, T *d, T *e, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    int grid = 1;
    hh_reflect_kernel<T, BLOCKSIZE><<<grid, BLOCKSIZE, 0, stream>>>(H, v, tau, d, e, N, k);
}

/// hh_gemv p ← Hv (general Householder application)
template <typename T, int BLOCKSIZE>
__global__ void hh_gemv_kernel(const T *v, const T *H, T *p, int N, int k) {
    // compute p = Hv for the trailing submatrix A[k+1:n, k+1:n]
    __shared__ T sr[BLOCKSIZE];

    int row = blockIdx.x + k + 1;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    if (row >= N || tid >= N - k - 1) return;

    T local_dot = T(0);
    for (int j = tid; j < N - k - 1; j += stride) {
        local_dot += H[row * N + (k + 1 + j)] * v[j];
    }
    sr[tid] = local_dot;
    __syncthreads();

    // reduce in block
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) sr[tid] += sr[tid + s];
        __syncthreads();
    }

    // reduce in warp
    if (tid < 32) {
        T val = sr[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) p[row - k - 1] = val;
    }
}

template <typename T>
void launch_hh_gemv(const T *v, const T *H, T *p, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    hh_gemv_kernel<T, BLOCKSIZE><<<N, BLOCKSIZE, 0, stream>>>(v, H, p, N, k);
}

/// hh_update u ← tau p - 0.5 tau² vᵀp v (Householder rank-2 update)
template <typename T, int BLOCKSIZE>
__global__ void hh_update_kernel(const T *v, const T *p, const T *tau, T *u, int N, int k) {
    int tid = threadIdx.x;
    __shared__ T svTp[BLOCKSIZE];

    T tau_k = tau[k];
    // compute u = tau p - 0.5 tau² (vᵀp) v
    T vTp = T(0);
    for (int i = tid; i < N - k - 1; i += BLOCKSIZE) {
        vTp += v[i] * p[i];
    }
    svTp[tid] = vTp;
    __syncthreads();

    // reduce vTp in block
    for (int s = BLOCKSIZE >> 1; s >= 32; s >>= 1) {
        if (tid < s) svTp[tid] += svTp[tid + s];
        __syncthreads();
    }

    // reduce vTp in warp
    if (tid < 32) {
        T val = svTp[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) svTp[0] = val;
    }
    __syncthreads();

    vTp = svTp[0];
    for (int i = tid; i < N - k - 1; i += BLOCKSIZE)
        u[i] = tau_k * p[i] - T(0.5) * tau_k * tau_k * vTp * v[i];
}

template <typename T>
void launch_hh_update(const T *v, const T *p, const T *tau, T *u, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    hh_update_kernel<T, BLOCKSIZE><<<1, BLOCKSIZE, 0, stream>>>(v, p, tau, u, N, k);
}

/// hh_syr2 H ← H - v uᵀ - u vᵀ (Householder symmetric rank-2 update)
template <typename T, int BLOCKSIZE>
__global__ void hh_syr2_kernel(const T *v, const T *u, T *H, int N, int k) {
    int m = N - k - 1;
    int t_row = threadIdx.x / BLOCKSIZE; // local row within tile
    int t_col = threadIdx.x % BLOCKSIZE; // local col within tile
    int row = blockIdx.y * BLOCKSIZE + t_row; // row in trailing submatrix [0, m)
    int col = blockIdx.x * BLOCKSIZE + t_col; // col in trailing submatrix [0, m)

    __shared__ T sv_row[BLOCKSIZE], su_row[BLOCKSIZE]; // indexed by local row
    __shared__ T sv_col[BLOCKSIZE], su_col[BLOCKSIZE]; // indexed by local col

    // first row of threads loads col-dimension vectors
    if (t_row == 0) {
        sv_col[t_col] = (col < m) ? v[col] : T(0);
        su_col[t_col] = (col < m) ? u[col] : T(0);
    }
    // first col of threads loads row-dimension vectors
    if (t_col == 0) {
        sv_row[t_row] = (row < m) ? v[row] : T(0);
        su_row[t_row] = (row < m) ? u[row] : T(0);
    }
    __syncthreads();

    if (row < m && col < m)
        H[(k + 1 + row) * N + (k + 1 + col)] -= sv_row[t_row] * su_col[t_col]
                                               + su_row[t_row] * sv_col[t_col];
}

template <typename T>
void launch_hh_syr2(const T *v, const T *u, T *H, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    dim3 block(BLOCKSIZE * BLOCKSIZE);
    dim3 grid(div_up(N - k - 1, BLOCKSIZE), div_up(N - k - 1, BLOCKSIZE));
    hh_syr2_kernel<T, BLOCKSIZE><<<grid, block, 0, stream>>>(v, u, H, N, k);
}

/// hh_larft T̂ ← VᵀV (blocked Householder)
template <typename T>
__global__ void hh_larft_kernel(const T *V, const T *tau, T *T̂, int M, int K) {
}

template <typename T>
void launch_hh_larft(const T *V, const T *tau, T *T̂, int M, int K, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    int grid = (K + BLOCKSIZE - 1) / BLOCKSIZE;
    hh_larft_kernel<<<grid, BLOCKSIZE, 0, stream>>>(V, tau, T̂, M, K);
}

/// hh_larfb C ← (I − V T̂ Vᵀ)C (blocked Householder application)
template <typename T>
__global__ void hh_larfb_kernel(const T *V, const T *T̂, T *C, int M, int N, int K) {
}

template <typename T>
void launch_hh_larfb(const T *V, const T *T̂, T *C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    int grid = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    hh_larfb_kernel<<<grid, BLOCKSIZE, 0, stream>>>(V, T̂, C, M, N, K);
}


// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                                \
    template void launch_hh_reflect<T>(T *, T *, T *, T *, T *, int, int, cudaStream_t); \
    template void launch_hh_gemv<T>(const T *, const T *, T *, int, int, cudaStream_t); \
    template void launch_hh_update<T>(const T *, const T *, const T *, T *, int, int, cudaStream_t); \
    template void launch_hh_syr2<T>(const T *, const T *, T *, int, int, cudaStream_t); \
    template void launch_hh_larft<T>(const T *, const T *, T *, int, int, cudaStream_t); \
    template void launch_hh_larfb<T>(const T *, const T *, T *, int, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
