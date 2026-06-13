/**
 * @file   tridiag.cu
 * @brief  Tridiagonal eigensolver primitives for divide-and-conquer STEDC.
 *
 * Kernels operating on the symmetric tridiagonal matrix T produced by
 * Householder reduction. Three primitives build up the D&C algorithm:
 *
 *   eig_leaf   base case — closed-form 2×2 symmetric tridiagonal eigensolver
 *   eig_split  split T into diag(T₁,T₂) + e[k-1]·vvᵀ at index k
 *   eig_merge  rank-1 secular equation merge (planned)
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cuda.h>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

// -----------------------------------------------------------------------------
// eig_leaf — closed-form 2×2 symmetric tridiagonal eigensolver
// -----------------------------------------------------------------------------
template <typename T> __global__ void eig_leaf_kernel(const T *d, const T *e, T *eval, T *QT) {
    // 2x2 symmetric tridiagonal matrix:
    //   [ d[0]  e[0] ]
    //   [ e[0]  d[1] ]
    T d0 = d[0];
    T d1 = d[1];
    T e0 = e[0];

    if (tabs(e0) < T(1e-12) * (tabs(d0) + tabs(d1))) {
        // T is diagonal means eigenvalues are just diag entries
        if (d0 <= d1) {
            eval[0] = d0;
            eval[1] = d1;
            QT[0] = 1;
            QT[1] = 0;
            QT[2] = 0;
            QT[3] = 1;
        } else {
            eval[0] = d1;
            eval[1] = d0;
            QT[0] = 0;
            QT[1] = 1;
            QT[2] = 1;
            QT[3] = 0;
        }
        return;
    }

    T m = (d0 + d1) / 2;
    T r = (d0 - d1) / 2;
    T p = sqrt(r * r + e0 * e0);

    T lambda1 = m - p;
    T lambda2 = m + p;
    eval[0] = lambda1;
    eval[1] = lambda2;

    T c = sqrt((p + r) / (2 * p));
    T s = (e0 < 0 ? -1 : 1) * sqrt((p - r) / (2 * p));
    QT[0] = -s;
    QT[1] = c;
    QT[2] = c;
    QT[3] = s;
}

// -----------------------------------------------------------------------------
// eig_split — split T into diag(T₁,T₂) + e[k-1]·vvᵀ at index k
// -----------------------------------------------------------------------------
template <typename T>
__global__ void eig_split_kernel(const T *d, const T *e, int n, int k, T *d1, T *d2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx < k - 1) {
        d1[idx] = d[idx];
    } else if (idx == k - 1) {
        d1[idx] = d[idx] - e[k - 1];
    } else if (idx == k) {
        d2[idx - k] = d[idx] - e[k - 1];
    } else {
        d2[idx - k] = d[idx];
    }
}

// -----------------------------------------------------------------------------
// eig_sort — merge two sorted arrays of eigenvalues
// -----------------------------------------------------------------------------
template <typename T>
__global__ void eig_sort_kernel(const T *eval1, const T *eval2, int n, int k, T *eval) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int p, q;
    int low = max(0, idx - (n - k));
    int high = min(k, idx + 1);

    // binary search
    while (low < high) {
        p = (low + high) / 2;
        q = idx - p;
        if (q > 0 && p < k && eval2[q - 1] > eval1[p]) {
            low = p + 1;
        } else {
            high = p;
        }
    }

    p = low;
    q = idx - low;

    if (p < k && (q >= n - k || eval1[p] <= eval2[q])) {
        eval[idx] = eval1[p];
    } else {
        eval[idx] = eval2[q];
    }
}

// -----------------------------------------------------------------------------
// eig_secular — rank-1 secular equation merge
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void eig_secular_kernel(const T *eval1, const T *eval2, const T *eval_sorted,
                                   const T *Q1, const T *Q2, const T *e, int n, int k, T *eval,
                                   T *G) {
    int idx = blockIdx.x;
    int tid = threadIdx.x;

    T beta = e[k - 1];
    T low, high, lambda, f_local, df_local, zi, di, diff, t;
    __shared__ T sf[BLOCKSIZE];
    __shared__ T sdf[BLOCKSIZE];
    __shared__ T smem[1];

    // interval bounds from the globally sorted merged eigenvalues
    if (beta > 0) {
        low = eval_sorted[idx];
        high = (idx < n - 1) ? eval_sorted[idx + 1] : eval_sorted[n - 1] + T(2) * tabs(beta);
    } else {
        low = (idx > 0) ? eval_sorted[idx - 1] : eval_sorted[0] - T(2) * tabs(beta);
        high = eval_sorted[idx];
    }

    // Bracketed Newton: track a narrowing bracket [lo,hi] and bisect when Newton exits it.
    constexpr int MAX_ITER = 40;
    T lo = low, hi = high;
    lambda = (lo + hi) / 2;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        f_local = T(0);
        df_local = T(0);
        // f(λ) = 1 + β·Σᵢ zᵢ²/(dᵢ−λ),  df(λ) = β·Σᵢ zᵢ²/(dᵢ−λ)²
        // poles come from eval1/eval2 to preserve the zi↔di correspondence
        for (int i = tid; i < n; i += BLOCKSIZE) {
            zi = (i < k) ? Q1[(k - 1) * k + i] : Q2[i - k];
            di = (i < k) ? eval1[i] : eval2[i - k];
            diff = di - lambda;
            t = zi / diff;
            f_local += beta * t * t * diff;
            df_local += beta * t * t;
        }
        T f = block_reduce_sum<T, BLOCKSIZE>(f_local, sf);
        T df = block_reduce_sum<T, BLOCKSIZE>(df_local, sdf);
        if (tid == 0) {
            bool root_above = (beta > T(0)) ? (T(1) + f < T(0)) : (T(1) + f > T(0));
            if (root_above)
                lo = lambda;
            else
                hi = lambda;
            T lambda_new = lambda - (T(1) + f) / df;
            if (lambda_new <= lo || lambda_new >= hi) lambda_new = (lo + hi) / T(2);
            lambda = lambda_new;
            smem[0] = lambda;
        }
        __syncthreads();
        lambda = smem[0];
    }
    eval[idx] = lambda;

    // compute eigenvector gᵢ = zᵢ / (dᵢ − λ), then normalise
    T norm_local = T(0);
    T gi;
    for (int i = tid; i < n; i += BLOCKSIZE) {
        zi = (i < k) ? Q1[(k - 1) * k + i] : Q2[i - k];
        di = (i < k) ? eval1[i] : eval2[i - k];
        diff = di - lambda;
        gi = zi / diff;
        norm_local += gi * gi;
    }
    T norm = block_reduce_sum<T, BLOCKSIZE>(norm_local, sf);
    if (tid == 0) smem[0] = rsqrt(norm);
    __syncthreads();
    T inorm = smem[0];

    for (int i = tid; i < n; i += BLOCKSIZE) {
        zi = (i < k) ? Q1[(k - 1) * k + i] : Q2[i - k];
        di = (i < k) ? eval1[i] : eval2[i - k];
        G[idx * n + i] = (zi / (di - lambda)) * inorm;
    }
}

} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev {
namespace kernels {

template <typename T> void eig_leaf(const T *d, const T *e, T *eval, T *QT, cudaStream_t stream) {
    eig_leaf_kernel<T><<<1, 1, 0, stream>>>(d, e, eval, QT);
}

template <typename T>
void eig_split(const T *d, const T *e, int n, int k, T *d1, T *d2, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 512;
    eig_split_kernel<T><<<div_up(n, BLOCKSIZE), BLOCKSIZE, 0, stream>>>(d, e, n, k, d1, d2);
}

template <typename T>
void eig_sort(const T *eval1, const T *eval2, int n, int k, T *eval, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 512;
    eig_sort_kernel<T><<<div_up(n, BLOCKSIZE), BLOCKSIZE, 0, stream>>>(eval1, eval2, n, k, eval);
}

template <typename T>
void eig_secular(const T *eval1, const T *eval2, const T *eval_sorted, const T *Q1, const T *Q2,
                 const T *e, int n, int k, T *eval, T *G, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 512;
    eig_secular_kernel<T, BLOCKSIZE>
        <<<n, BLOCKSIZE, 0, stream>>>(eval1, eval2, eval_sorted, Q1, Q2, e, n, k, eval, G);
}

template <typename T>
void eig_gemm(const T *Q1, const T *Q2, const T *G, T *QT, int n, int k, cudaStream_t stream) {
    // rows = eigenvectors: QT = G · diag(Q1, Q2)
    // G's column blocks are strided in row-major, so use the identity:
    //   QT^T = diag(Q1^T, Q2^T) · G^T
    // QT^T[0:k, :]   = Q1^T · G^T[0:k, :]   (k×k     · k×n)
    // QT^T[k:n, :]   = Q2^T · G^T[k:n, :]   ((n-k)×(n-k) · (n-k)×n)
    constexpr int BM = 128, BK = 16;
    constexpr int BN = std::is_same_v<T, float> ? 128 : 64;

    T *GT, *Q1T, *Q2T, *QTT;
    cudaMalloc(&GT, n * n * sizeof(T));
    cudaMalloc(&Q1T, k * k * sizeof(T));
    cudaMalloc(&Q2T, (n - k) * (n - k) * sizeof(T));
    cudaMalloc(&QTT, n * n * sizeof(T));

    transpose(G, GT, n, n, stream);
    transpose(Q1, Q1T, k, k, stream);
    transpose(Q2, Q2T, n - k, n - k, stream);

    auto do_gemm = [&](const T *Q, const T *GTblock, T *QTTblock, int m, int kk) {
        if (m % BM == 0 && n % BN == 0 && kk % BK == 0)
            gemm_warptile(T(1), Q, GTblock, T(0), QTTblock, m, n, kk, stream);
        else
            gemm_smem(T(1), Q, GTblock, T(0), QTTblock, m, n, kk, stream);
    };

    do_gemm(Q1T, GT, QTT, k, k);
    do_gemm(Q2T, GT + k * n, QTT + k * n, n - k, n - k);

    transpose(QTT, QT, n, n, stream);

    cudaFree(GT);
    cudaFree(Q1T);
    cudaFree(Q2T);
    cudaFree(QTT);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void eig_leaf<T>(const T *, const T *, T *, T *, cudaStream_t);                       \
    template void eig_split<T>(const T *, const T *, int, int, T *, T *, cudaStream_t);            \
    template void eig_sort<T>(const T *, const T *, int, int, T *, cudaStream_t);                  \
    template void eig_secular<T>(const T *, const T *, const T *, const T *, const T *, const T *, \
                                 int, int, T *, T *, cudaStream_t);                                \
    template void eig_gemm<T>(const T *, const T *, const T *, T *, int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
