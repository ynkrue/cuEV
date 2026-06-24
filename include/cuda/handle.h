/**
 * @file   handle.h
 * @brief  SolverHandle<T> — cuBLAS/cuSOLVER handles and scratch buffers for cuEV.
 *
 * TODO: add description
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace cuev {

/**
 * @brief Per-solve context: library handles + pre-allocated scratch buffers.
 *
 * Created once in symm_eig_solve via handle_alloc, threaded through all stages,
 * destroyed via handle_free. No cudaMalloc/cudaFree in the hot path.
 *
 * @tparam T  float or double
 */
template <typename T> struct SolverHandle {
    int n;   ///< problem dimension
    int nbw; ///< bandwidth of banded matrix
    int nk;  ///< outer panel size
    cudaStream_t stream;

    cublasHandle_t cublas;
    cusolverDnHandle_t cusolver;
    int *d_info;

    // DBBR buffers
    T *Y;    ///< n*n - Householder reflectors; current block = Y[:,i:], retained for SBR-Back
    T *Z;    ///< n*k - trailing two-sided companion (syr2k factor), transient per block
    T *tau;  ///< nbw - Householder scalars
    T *Tmat; ///< nbw*nbw - block reflector triangular factor T (larft output, transient)
    T *Dwk;  ///< nk*nbw - panel scratch (Yᵀ·AY and deferral-correction GEMM temps)

    // SBR back buffer
    T *W; ///< n*n - SBR-Back companion W = Y·T, retained

    // BC buffers
    T *B; ///< (b+1)*n - packed banded matrix
    T *U; ///< n*(n-2) - BC Householder vectors

    // cuSOLVER buffers
    T *geqrf_buf;
    int geqrf_lwork;
    T *orgqr_buf;
    int orgqr_lwork;
    T *syevd_buf;
    int syevd_lwork;

    // handle allocation
    void *pool;
    size_t pool_bytes;
};

/**
 * @brief Allocate all scratch and initialise cuBLAS/cuSOLVER handles.
 *
 * @tparam T     float or double
 * @param[in] n       root problem dimension
 * @param[in] nbw     bandwidth of banded matrix
 * @param[in] nk      outer panel size
 * @param[in] stream  CUDA stream all operations will run on
 */
template <typename T> SolverHandle<T> handle_alloc(int n, int nbw, int nk, cudaStream_t stream);

/**
 * @brief Destroy handles and free scratch.
 *
 * @tparam T  float or double
 * @param[in,out] ws  handle to destroy
 */
template <typename T> void handle_free(SolverHandle<T> *ws);

} // namespace cuev
