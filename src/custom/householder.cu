/**
 * @file   householder.cu
 * @brief  Householder kernels for symmetric tridiagonalization.
 *
 * Reduces a real symmetric matrix H to tridiagonal form T via
 * similarity transformations H ← PₖᵀHPₖ for k = 0…n−3, where each
 * Pₖ = I − τvvᵀ is a Householder projection. The accumulated product
 * Q = P₀·P₁···P_{n-3} is stored implicitly in the zeroed-out subdiagonal
 * of H; it is only materialized during eigenvector back-transformation.
 *
 * Kernel sequence per step k:
 *   hh_reflect   v, τ ← col k of trailing submatrix
 *   hh_trail_matvec      p ← H[k+1:n, k+1:n] · v
 *   hh_ortho    u ← τp − ½τ²(vᵀp)v
 *   hh_update      H ← H − vuᵀ − uvᵀ
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

// -----------------------------------------------------------------------------
// hh_reflect
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void hh_reflect_kernel(T *H, T *v, T *tau, T *d, T *e, int N, int k) {
    __shared__ T snrm[BLOCKSIZE];
    const int tid = threadIdx.x;
    T nrm2 = T(0);

    for (int i = tid; i < N - k - 1; i += blockDim.x) {
        T a = H[(k + 1 + i) * N + k];
        nrm2 += a * a;
        v[i] = a;
    }
    snrm[tid] = nrm2;
    __syncthreads();

    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            snrm[tid] += snrm[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        T val = snrm[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) {
            T norm = sqrt(val);
            T x0 = H[(k + 1) * N + k];
            T alpha = -copysign(norm, x0); // α = −sign(x₀)‖x‖, ensures numerical stability
            T vTv = norm * norm - alpha * x0;
            v[0] = x0 - alpha;
            H[(k + 1) * N + k] = x0 - alpha;
            tau[k] = (vTv == T(0)) ? T(0) : T(1) / vTv;
            if (k < N - 1) {
                e[k] = alpha;
            }
            d[k] = H[k * N + k];
        }
        // __syncthreads() omitted — tid >= 32 do not participate; kernel returns immediately after
    }
}

// -----------------------------------------------------------------------------
// hh_trail_matvec
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void hh_trail_matvec_kernel(const T *v, const T *H, T *p, int N, int k) {
    __shared__ T sr[BLOCKSIZE];
    int row = blockIdx.x + k + 1;
    int tid = threadIdx.x;
    if (row >= N) return;

    T acc = T(0);
    for (int j = tid; j < N - k - 1; j += BLOCKSIZE) {
        acc += H[row * N + (k + 1 + j)] * v[j];
    }
    sr[tid] = acc;
    __syncthreads();

    for (int s = BLOCKSIZE >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            sr[tid] += sr[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        T val = sr[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) {
            p[row - k - 1] = val;
        }
    }
}

// -----------------------------------------------------------------------------
// hh_ortho
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void hh_ortho_kernel(const T *v, const T *p, const T *tau, T *u, int N, int k) {
    __shared__ T svTp[BLOCKSIZE];
    const int tid = threadIdx.x;
    T tau_k = tau[k];

    T vTp = T(0);
    for (int i = tid; i < N - k - 1; i += BLOCKSIZE) {
        vTp += v[i] * p[i];
    }
    svTp[tid] = vTp;
    __syncthreads();

    for (int s = BLOCKSIZE >> 1; s >= 32; s >>= 1) {
        if (tid < s) {
            svTp[tid] += svTp[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        T val = svTp[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0) {
            svTp[0] = val;
        }
    }
    __syncthreads();

    vTp = svTp[0];
    for (int i = tid; i < N - k - 1; i += BLOCKSIZE) {
        u[i] = tau_k * p[i] - T(0.5) * tau_k * tau_k * vTp * v[i];
    }
}

// -----------------------------------------------------------------------------
// hh_update
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void hh_ortho_kernel(const T *v, const T *u, T *H, int N, int k) {
    int m = N - k - 1;
    int t_row = threadIdx.x / BLOCKSIZE;
    int t_col = threadIdx.x % BLOCKSIZE;
    int row = blockIdx.y * BLOCKSIZE + t_row;
    int col = blockIdx.x * BLOCKSIZE + t_col;

    // each tile loads its v/u slices into smem to avoid redundant global reads
    __shared__ T sv_row[BLOCKSIZE], su_row[BLOCKSIZE];
    __shared__ T sv_col[BLOCKSIZE], su_col[BLOCKSIZE];

    if (t_row == 0) {
        sv_col[t_col] = (col < m) ? v[col] : T(0);
        su_col[t_col] = (col < m) ? u[col] : T(0);
    }
    if (t_col == 0) {
        sv_row[t_row] = (row < m) ? v[row] : T(0);
        su_row[t_row] = (row < m) ? u[row] : T(0);
    }
    __syncthreads();

    if (row < m && col < m) {
        H[(k + 1 + row) * N + (k + 1 + col)] -=
            sv_row[t_row] * su_col[t_col] + su_row[t_row] * sv_col[t_col];
    }
}

// -----------------------------------------------------------------------------
// hh_hh_wy_build / hh_hh_wy_apply  (stubs)
// -----------------------------------------------------------------------------
template <typename T>
__global__ void hh_wy_build_kernel(const T *V, const T *tau, T *Tf, int M, int K) {
}

template <typename T>
__global__ void hh_wy_apply_kernel(const T *V, const T *Tf, T *C, int M, int N, int K) {
}

} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev {
namespace kernels {

template <typename T>
void hh_reflect(T *H, T *v, T *tau, T *d, T *e, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    hh_reflect_kernel<T, BLOCKSIZE><<<1, BLOCKSIZE, 0, stream>>>(H, v, tau, d, e, N, k);
}

template <typename T>
void hh_trail_matvec(const T *v, const T *H, T *p, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    hh_trail_matvec_kernel<T, BLOCKSIZE><<<N - k - 1, BLOCKSIZE, 0, stream>>>(v, H, p, N, k);
}

template <typename T>
void hh_ortho(const T *v, const T *p, const T *tau, T *u, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 1024;
    hh_ortho_kernel<T, BLOCKSIZE><<<1, BLOCKSIZE, 0, stream>>>(v, p, tau, u, N, k);
}

template <typename T>
void hh_update(const T *v, const T *u, T *H, int N, int k, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 32;
    hh_ortho_kernel<T, BLOCKSIZE>
        <<<dim3(div_up(N - k - 1, BLOCKSIZE), div_up(N - k - 1, BLOCKSIZE)),
           BLOCKSIZE * BLOCKSIZE,
           0,
           stream>>>(v, u, H, N, k);
}

template <typename T>
void hh_hh_wy_build(const T *V, const T *tau, T *Tf, int M, int K, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    hh_wy_build_kernel<<<div_up(K, BLOCKSIZE), BLOCKSIZE, 0, stream>>>(V, tau, Tf, M, K);
}

template <typename T>
void hh_hh_wy_apply(const T *V, const T *Tf, T *C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    hh_wy_apply_kernel<<<div_up(N, BLOCKSIZE), BLOCKSIZE, 0, stream>>>(V, Tf, C, M, N, K);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void hh_reflect<T>(T *, T *, T *, T *, T *, int, int, cudaStream_t);                  \
    template void hh_trail_matvec<T>(const T *, const T *, T *, int, int, cudaStream_t);           \
    template void hh_ortho<T>(const T *, const T *, const T *, T *, int, int, cudaStream_t);       \
    template void hh_update<T>(const T *, const T *, T *, int, int, cudaStream_t);                 \
    template void hh_hh_wy_build<T>(const T *, const T *, T *, int, int, cudaStream_t);            \
    template void hh_hh_wy_apply<T>(const T *, const T *, T *, int, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
