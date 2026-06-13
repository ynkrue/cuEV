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
// Host launchers
// =============================================================================
namespace cuev {
namespace kernels {

template <typename T> void eig_leaf(const T *d, const T *e, T *eval, T *QT, cudaStream_t stream) {
    // stub — closed-form 2×2 symmetric tridiagonal eigensolver not yet implemented
    (void)d;
    (void)e;
    (void)eval;
    (void)QT;
    (void)stream;
}

template <typename T>
void eig_split(const T *d, const T *e, int k, T *d1, T *d2, cudaStream_t stream) {
    // stub — tridiagonal split with rank-1 correction not yet implemented
    (void)d;
    (void)e;
    (void)k;
    (void)d1;
    (void)d2;
    (void)stream;
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void eig_leaf<T>(const T *, const T *, T *, T *, cudaStream_t);                       \
    template void eig_split<T>(const T *, const T *, int, T *, T *, cudaStream_t);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
