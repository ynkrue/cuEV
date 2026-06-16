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
    // TODO: allocate tmp from ws.push, call cublasmp::symm + cublasmp::gemm for each subproblem
}

template <typename T>
void sdc_combine_mp(Context &ctx, const DistMatrix<T> &Q1, const DistMatrix<T> &Q2,
                    const DistMatrix<T> &evec1, const DistMatrix<T> &evec2, DistMatrix<T> &evec,
                    int64_t n, int64_t k) {
    // TODO: two cublasmp::gemm calls writing into column subranges of evec via ic/jc
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
                                    int64_t, int64_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
