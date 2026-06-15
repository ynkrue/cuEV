/**
 * @file   cuev.h
 * @brief  Public API for cuEV — single-GPU symmetric dense eigensolver.
 *
 * For multi-GPU usage see cuev_mp.h (requires -DCUEV_ENABLE_MP=ON).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include "workspace.h"
#include <cuda_runtime.h>

namespace cuev {

/**
 * @brief Compute all eigenvalues and eigenvectors of a real symmetric matrix.
 *
 * Solves H v = λ v for all n eigenvalue/eigenvector pairs using spectral
 * divide-and-conquer via QDWH polar iteration. H is overwritten during
 * computation and must not be reused afterwards.
 *
 * Output convention: eigenvectors are stored as **columns** of @p evec (column-major),
 * i.e. `evec[j * n + i]` is the i-th component of the j-th eigenvector.
 * Directly compatible with cuSOLVER dsyevd/ssyevd column-major output.
 *
 * @tparam T      float or double
 * @param[in,out] H      n×n real symmetric matrix, column-major; overwritten on return
 * @param[in]     n      matrix dimension
 * @param[out]    eval   eigenvalues in ascending order, length n
 * @param[out]    evec   eigenvectors as columns, n×n column-major
 * @param[in]     stream CUDA stream; all operations are submitted to this stream
 */
template <typename T> void symm_eig_solve(T *H, int n, T *eval, T *evec, cudaStream_t stream);

} // namespace cuev
