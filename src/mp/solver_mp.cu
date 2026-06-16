/**
 * @file   solver_mp.cu
 * @brief  Distributed spectral divide-and-conquer eigensolver.
 *
 * Public entry point: cuev::mp::symm_eig_solve_mp<T>(ctx, H, n, d_eval, evec, ws).
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#ifdef CUEV_ENABLE_MP

#include "mp/comm.h"
#include "mp/kernels_mp.cuh"
#include "mp/workspace_mp.h"
#include <algorithm>
#include <cmath>

namespace cuev {
namespace mp {

namespace {

template <typename T>
void spectral_dc_mp(Context &ctx, DistMatrix<T> &H, int64_t n, T *d_eval, DistMatrix<T> &evec,
                    WorkspaceMp<T> &ws) {
    // TODO: base case (cusolvermp::syevd) + recursive split/sign/QR/combine
}

} // namespace

template <typename T>
void symm_eig_solve_mp(Context &ctx, DistMatrix<T> &H, int64_t n, T *d_eval, DistMatrix<T> &evec,
                       WorkspaceMp<T> &ws) {
    spectral_dc_mp(ctx, H, n, d_eval, evec, ws);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template void symm_eig_solve_mp<float>(Context &, DistMatrix<float> &, int64_t, float *,
                                       DistMatrix<float> &, WorkspaceMp<float> &);
template void symm_eig_solve_mp<double>(Context &, DistMatrix<double> &, int64_t, double *,
                                        DistMatrix<double> &, WorkspaceMp<double> &);

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
