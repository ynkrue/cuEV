/**
 * @file   handle.h
 * @brief  cuEV handle for memory management and cuBLAS/cuSOLVER handles.
 *
 * TODO: add description
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace cuev {

/// Base-case threshold for spectral_dc recursion.
constexpr int SDC_BASE_N = 512;

/**
 * @brief Single-allocation workspace for the entire eigensolver.
 *
 * TODO: add description
 *
 * @tparam T  float or double
 */
template <typename T> struct SolverHandle {
    // cuSOLVER scratch
    cusolverDnHandle_t cusolver_handle;
    T *geqrf_buf;
    int geqrf_lwork;
    T *orgqr_buf;
    int orgqr_lwork;
    T *syevd_buf;
    int syevd_lwork;

    // cuBLAS scratch
    cublasHandle_t cublas_handle;

    // cuEV scratch
    int *d_info;
    T *pool;
};

/**
 * @brief Allocate the workspace pool and initialize cuBLAS and cuSOLVER handles.
 *
 * @tparam T         float or double
 * @param[in] n      root problem dimension
 */
template <typename T> SolverHandle<T> handle_alloc(int n, cudaStream_t stream);

/**
 * @brief Free the workspace pool (single cudaFree).
 *
 * @tparam T  float or double
 * @param[in,out] ws  workspace; all pointers zeroed on return
 */
template <typename T> void handle_free(SolverHandle<T> *ws);

} // namespace cuev
