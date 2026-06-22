/**
 * @file   handle.h
 * @brief  cuEV handle for memory management and cuBLAS/cuSOLVER handles.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "cuda/handle.h"

namespace cuev {

template <typename T> SolverHandle<T> handle_alloc(int n, cudaStream_t stream) {
    SolverHandle<T> ws;
    // ws.stream = stream;

    // // create cuBLAS handle
    // CUBLAS_CHECK(cublasCreate(&ws.cublas_handle));
    // CUBLAS_CHECK(cublasSetStream(ws.cublas_handle, stream));

    // // create cuSOLVER handle
    // CUSOLVER_CHECK(cusolverDnCreate(&ws.cusolver_handle));
    // CUSOLVER_CHECK(cusolverDnSetStream(ws.cusolver_handle, stream));

    // // allocate device memory for info
    // CUDA_CHECK(cudaMalloc(&ws.d_info, sizeof(int)));

    return ws;
}

template <typename T> void handle_free(SolverHandle<T> *ws) {
    // // destroy cuBLAS handle
    // CUBLAS_CHECK(cublasDestroy(ws->cublas_handle));
    // // destroy cuSOLVER handle
    // CUSOLVER_CHECK(cusolverDnDestroy(ws->cusolver_handle));
    // // free device memory for info
    // CUDA_CHECK(cudaFree(ws->d_info));
    (void)ws;
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template SolverHandle<float> handle_alloc<float>(int, cudaStream_t);
template SolverHandle<double> handle_alloc<double>(int, cudaStream_t);
template void handle_free<float>(SolverHandle<float> *);
template void handle_free<double>(SolverHandle<double> *);
} // namespace cuev
