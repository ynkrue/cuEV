/**
 * @file   sdc.cu
 * @brief  Spectral divide-and-conquer helper kernels and cuBLAS wrappers.
 *
 * Provides the GPU primitives used by spectral_dc in solver.cu:
 *
 *   sdc_trace    diagonal reduction → split-point estimate μ
 *   sdc_rank     count significant R diagonal entries → split size k
 *   sdc_split    H₁ = Q₁ᵀHQ₁,  H₂ = Q₂ᵀHQ₂  (two cuBLAS GEMM pairs)
 *   sdc_combine  evec ← blkdiag(evec₁, evec₂) · [Q₁|Q₂]ᵀ  (two cuBLAS GEMMs)
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

// -----------------------------------------------------------------------------
// sdc_trace — parallel reduction over diagonal
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void sdc_trace_kernel(const T *A, T *result, int n) {
    __shared__ T smem[BLOCKSIZE];
    int tid = threadIdx.x;
    T acc = T(0);
    for (int i = tid; i < n; i += BLOCKSIZE)
        acc += A[i * n + i];
    T total = block_reduce_sum<T, BLOCKSIZE>(acc, smem);
    if (tid == 0) *result = total;
}

// -----------------------------------------------------------------------------
// sdc_rank — count |R[i,i]| > threshold (diagonal of upper-triangular R)
// -----------------------------------------------------------------------------
template <typename T, int BLOCKSIZE>
__global__ void sdc_rank_kernel(const T *R, int *result, int n, T threshold) {
    __shared__ int smem[BLOCKSIZE];
    int tid = threadIdx.x;
    int cnt = 0;
    for (int i = tid; i < n; i += BLOCKSIZE)
        cnt += (R[i * n + i] > threshold || R[i * n + i] < -threshold) ? 1 : 0;

    smem[tid] = cnt;
    __syncthreads();
    for (int s = BLOCKSIZE >> 1; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) *result = smem[0];
}

} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev {
namespace kernels {

template <typename T> T sdc_trace(const T *A, int n, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 512;
    T *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));
    sdc_trace_kernel<T, BLOCKSIZE><<<1, BLOCKSIZE, 0, stream>>>(A, d_result, n);
    T h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_result));
    return h_result;
}

template <typename T> int sdc_rank(const T *R, int n, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 512;
    // threshold: R diagonal entries near 1 → eigenvectors in this subspace
    T threshold = T(0.5);
    int *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));
    sdc_rank_kernel<T, BLOCKSIZE><<<1, BLOCKSIZE, 0, stream>>>(R, d_result, n, threshold);
    int h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_result));
    return h_result;
}

template <typename T>
void sdc_split(cublasHandle_t cublas, const T *H, const T *Q1, const T *Q2, T *H1, T *H2, int n,
               int k, cudaStream_t stream) {
    // All matrices column-major.
    // H₁ = Q₁ᵀ H Q₁:  tmp (n×k) = H (n×n) · Q₁ (n×k),  H₁ (k×k) = Q₁ᵀ (k×n) · tmp (n×k)
    // H₂ = Q₂ᵀ H Q₂:  tmp (n×m) = H (n×n) · Q₂ (n×m),  H₂ (m×m) = Q₂ᵀ (m×n) · tmp (n×m)
    //   where m = n - k; Q1ᵀ/Q2ᵀ achieved via CUBLAS_OP_T on the left factor
    // TODO: implement via two gemm calls per subproblem
    (void)cublas;
    (void)H;
    (void)Q1;
    (void)Q2;
    (void)H1;
    (void)H2;
    (void)n;
    (void)k;
    (void)stream;
}

template <typename T>
void sdc_combine(cublasHandle_t cublas, const T *Q1, const T *Q2, const T *evec1, const T *evec2,
                 T *evec, int n, int k, cudaStream_t stream) {
    // All matrices column-major; eigenvectors are columns.
    // evec[:, 0:k]   = Q1 (n×k)     · evec1 (k×k)        — first k cols of evec
    // evec[:, k:n]   = Q2 (n×(n-k)) · evec2 ((n-k)×(n-k)) — last n-k cols of evec
    // Destination pointers: evec (first k cols), evec + k*n (last n-k cols)
    // TODO: implement via two gemm calls
    (void)cublas;
    (void)Q1;
    (void)Q2;
    (void)evec1;
    (void)evec2;
    (void)evec;
    (void)n;
    (void)k;
    (void)stream;
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template T sdc_trace<T>(const T *, int, cudaStream_t);                                         \
    template int sdc_rank<T>(const T *, int, cudaStream_t);                                        \
    template void sdc_split<T>(cublasHandle_t, const T *, const T *, const T *, T *, T *, int,     \
                               int, cudaStream_t);                                                 \
    template void sdc_combine<T>(cublasHandle_t, const T *, const T *, const T *, const T *, T *,  \
                                 int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
