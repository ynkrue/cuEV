/**
 * @file   sdc.cu
 * @brief  Spectral divide-and-conquer helper kernels and cuBLAS wrappers.
 *
 * Provides the GPU primitives used by spectral_dc in solver.cu:
 *
 *   sdc_trace    diagonal reduction → split-point estimate μ and rank k = trace(P)
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

template <typename T>
void sdc_split(cublasHandle_t cublas, const T *H, const T *Q1, const T *Q2, T *H1, T *H2, int n,
               int k, SolverWorkspace<T> *ws, cudaStream_t stream) {
    // H₁ = Q₁ᵀ H Q₁:  tmp (n×k) = H (n×n) · Q₁ (n×k),  H₁ (k×k) = Q₁ᵀ (k×n) · tmp (n×k)
    // H₂ = Q₂ᵀ H Q₂:  tmp (n×m) = H (n×n) · Q₂ (n×m),  H₂ (m×m) = Q₂ᵀ (m×n) · tmp (n×m)
    T one = T(1);
    T zero = T(0);
    int m = n - k;
    size_t mark = ws->mark();
    T *tmp = ws->push((size_t)n * k);
    cublas::gemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, H, n, Q1, n, &zero, tmp, n);
    cublas::gemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n, &one, Q1, n, tmp, n, &zero, H1, k);
    ws->reset(mark);

    tmp = ws->push((size_t)n * m);
    cublas::gemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, H, n, Q2, n, &zero, tmp, n);
    cublas::gemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, Q2, n, tmp, n, &zero, H2, m);
    ws->reset(mark);
}

template <typename T>
void sdc_combine(cublasHandle_t cublas, const T *Q1, const T *Q2, const T *evec1, const T *evec2,
                 T *evec, int n, int k, cudaStream_t stream) {
    // evec[:, 0:k]   = Q1 (n×k)     · evec1 (k×k)        — first k cols of evec
    // evec[:, k:n]   = Q2 (n×m)     · evec2 (m×m)        — last n-k cols of evec
    T one = T(1);
    T zero = T(0);
    int m = n - k;

    // Q2·evec2 → first m cols (eigenvalues < μ, ascending)
    cublas::gemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one, Q2, n, evec2, m, &zero, evec, n);
    // Q1·evec1 → last k cols (eigenvalues > μ, ascending)
    cublas::gemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, k, k, &one, Q1, n, evec1, k, &zero,
                 evec + (size_t)m * n, n);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template T sdc_trace<T>(const T *, int, cudaStream_t);                                         \
    template void sdc_split<T>(cublasHandle_t, const T *, const T *, const T *, T *, T *, int,     \
                               int, SolverWorkspace<T> *, cudaStream_t);                           \
    template void sdc_combine<T>(cublasHandle_t, const T *, const T *, const T *, const T *, T *,  \
                                 int, int, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
