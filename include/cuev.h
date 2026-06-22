/**
 * @file   cuev.h
 * @brief  Public API for cuEV — single-GPU symmetric dense eigensolver.
 *
 * TODO: add description
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include "cuda/handle.h"
#include <cuda_runtime.h>

namespace cuev {

/**
 * @brief Compute all eigenvalues and eigenvectors of a real symmetric matrix.
 *
 * TODO: add description
 * Solves H v = λ v for all n eigenvalue/eigenvector pairs ...
 *
 * @tparam T      float or double
 * @param[in,out] H      n×n real symmetric matrix (column-major)
 * @param[in]     n      matrix dimension
 * @param[out]    eval   eigenvalues in ascending order, length n
 * @param[out]    evec   eigenvectors as columns, n×n (column-major)
 * @param[in]     stream CUDA stream
 */
template <typename T> void symm_eig_solve(T *H, int n, T *eval, T *evec, cudaStream_t stream);

} // namespace cuev
