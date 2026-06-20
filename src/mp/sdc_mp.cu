/**
 * @file   sdc_mp.cu
 * @brief  Distributed spectral D&C helpers — cuev::mp::kernels namespace.
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#ifdef CUEV_ENABLE_MP

#include "mp/kernels_mp.cuh"

namespace cuev {
namespace mp {
namespace kernels {

template <typename T>
void sdc_split_mp(Context &ctx, const DistMatrix<T> &H, const DistMatrix<T> &Q1,
                  const DistMatrix<T> &Q2, DistMatrix<T> &H1, DistMatrix<T> &H2, int64_t n,
                  int64_t k, WorkspaceMp<T> &ws) {
    // H₁ = Q₁ᵀ H Q₁:  tmp (n×k) = H (n×n) · Q₁ (n×k),  H₁ (k×k) = Q₁ᵀ (k×n) · tmp (n×k)
    // H₂ = Q₂ᵀ H Q₂:  tmp (n×m) = H (n×n) · Q₂ (n×m),  H₂ (m×m) = Q₂ᵀ (m×n) · tmp (n×m)
    //
    // Q1 and Q2 are standalone block-aligned matrices (the caller materialised
    // them with gemr2d), so every GEMM uses offset (1,1) — no mid-block submatrix
    // slicing, which PXGEMM rejects with INVALID_VALUE.
    T one = T(1);
    T zero = T(0);
    int64_t m = n - k;
    size_t mark = ws.mark();

    T *tmp_data = ws.push(dist_local_count<T>(ctx, n, k));
    DistMatrix<T> tmp = dist_describe<T>(ctx, n, k, tmp_data);
    cublasmp::gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, H.data, 1, 1, H.desc, Q1.data, 1,
                   1, Q1.desc, &zero, tmp.data, 1, 1, tmp.desc);
    cublasmp::gemm(ctx, CUBLAS_OP_T, CUBLAS_OP_N, k, k, n, &one, Q1.data, 1, 1, Q1.desc, tmp.data,
                   1, 1, tmp.desc, &zero, H1.data, 1, 1, H1.desc);
    dist_free(tmp);
    ws.reset(mark);

    tmp_data = ws.push(dist_local_count<T>(ctx, n, m));
    tmp = dist_describe<T>(ctx, n, m, tmp_data);
    cublasmp::gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, H.data, 1, 1, H.desc, Q2.data, 1,
                   1, Q2.desc, &zero, tmp.data, 1, 1, tmp.desc);
    cublasmp::gemm(ctx, CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, &one, Q2.data, 1, 1, Q2.desc, tmp.data,
                   1, 1, tmp.desc, &zero, H2.data, 1, 1, H2.desc);
    dist_free(tmp);
    ws.reset(mark);
}

template <typename T>
void sdc_combine_mp(Context &ctx, const DistMatrix<T> &Q1, const DistMatrix<T> &Q2,
                    const DistMatrix<T> &evec1, const DistMatrix<T> &evec2, DistMatrix<T> &evec,
                    int64_t n, int64_t k, WorkspaceMp<T> &ws) {
    T one = T(1);
    T zero = T(0);
    int64_t m = n - k;

    // evec[:, 1:m] = Q2 (n×m) · evec2 (m×m)  — eigenvalues < μ, ascending.
    // Destination offset (1,1) is block-aligned, so a direct GEMM is fine.
    cublasmp::gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, &one, Q2.data, 1, 1, Q2.desc, evec2.data,
                   1, 1, evec2.desc, &zero, evec.data, 1, 1, evec.desc);

    // evec[:, m+1:n] = Q1 (n×k) · evec1 (k×k)  — eigenvalues > μ, ascending.
    // The column offset m+1 is not a multiple of nb, so PXGEMM cannot write it
    // directly. Compute into a block-aligned temp, then scatter with gemr2d.
    size_t mark = ws.mark();
    T *tmp_data = ws.push(dist_local_count<T>(ctx, n, k));
    DistMatrix<T> tmp = dist_describe<T>(ctx, n, k, tmp_data);
    cublasmp::gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, n, k, k, &one, Q1.data, 1, 1, Q1.desc, evec1.data,
                   1, 1, evec1.desc, &zero, tmp.data, 1, 1, tmp.desc);
    cublasmp::gemr2d(ctx, n, k, tmp.data, 1, 1, tmp.desc, evec.data, 1, m + 1, evec.desc);
    dist_free(tmp);
    ws.reset(mark);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void sdc_split_mp<T>(Context &, const DistMatrix<T> &, const DistMatrix<T> &,         \
                                  const DistMatrix<T> &, DistMatrix<T> &, DistMatrix<T> &,         \
                                  int64_t, int64_t, WorkspaceMp<T> &);                             \
    template void sdc_combine_mp<T>(Context &, const DistMatrix<T> &, const DistMatrix<T> &,       \
                                    const DistMatrix<T> &, const DistMatrix<T> &, DistMatrix<T> &, \
                                    int64_t, int64_t, WorkspaceMp<T> &);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
