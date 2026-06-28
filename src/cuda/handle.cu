/**
 * @file   handle.cu
 * @brief  SolverHandle allocation and teardown — cuBLAS/cuSOLVER init + scratch sizing.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include <algorithm>
#include <type_traits>

namespace cuev {

template <typename T> SolverHandle<T> handle_alloc(int n, int nbw, int nk, cudaStream_t stream) {
    SolverHandle<T> ws{};
    ws.n = n;
    ws.nbw = nbw;
    ws.nk = nk;
    ws.ldu = n + 512; // padded for efficient back-transform kernels
    ws.stream = stream;

    CUBLAS_CHECK(cublasCreate(&ws.cublas));
    CUBLAS_CHECK(cublasSetStream(ws.cublas, stream));
    CUSOLVER_CHECK(cusolverDnCreate(&ws.cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(ws.cusolver, stream));
    CUDA_CHECK(cudaMalloc(&ws.d_info, sizeof(int)));

    // Workspace query
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(
            cusolverDnSgeqrf_bufferSize(ws.cusolver, n, nbw, nullptr, n, &ws.geqrf_lwork));
    else
        CUSOLVER_CHECK(
            cusolverDnDgeqrf_bufferSize(ws.cusolver, n, nbw, nullptr, n, &ws.geqrf_lwork));

    // Pool layout
    auto align_up = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    const size_t s = sizeof(T);

    size_t off = 0;
    size_t off_Y = off;
    off += align_up((size_t)n * n * s);
    size_t off_Z = off;
    off += align_up((size_t)n * nk * s);
    size_t off_tau = off;
    off += align_up((size_t)nbw * s);
    size_t off_Tri = off;
    off += align_up((size_t)nbw * nbw * s);
    size_t off_Dwk = off;
    off += align_up((size_t)nk * nbw * s);
    size_t off_W = off;
    off += align_up((size_t)n * n * s);
    size_t off_M = off;
    off += align_up((size_t)ws.ldu * n * s);
    size_t off_B = off;
    off += align_up((size_t)(2 * nbw) * n * s);
    size_t off_U = off;
    off += align_up((size_t)ws.ldu * n * s);
    size_t off_d = off;
    off += align_up((size_t)n * s);
    size_t off_e = off;
    off += align_up((size_t)n * s);
    size_t off_prog = off;
    off += align_up((size_t)n * sizeof(int));
    size_t off_geqrf = off;
    off += align_up((size_t)ws.geqrf_lwork * s);
    ws.pool_bytes = off;

    CUDA_CHECK(cudaMalloc(&ws.pool, ws.pool_bytes));
    // zeroed out padding for U and M
    CUDA_CHECK(cudaMemset((uint8_t *)ws.pool + off_U, 0, align_up((size_t)ws.ldu * n * s)));

    auto base = (uint8_t *)ws.pool;
    ws.Y = (T *)(base + off_Y);
    ws.Z = (T *)(base + off_Z);
    ws.tau = (T *)(base + off_tau);
    ws.Tri = (T *)(base + off_Tri);
    ws.Dwk = (T *)(base + off_Dwk);
    ws.W = (T *)(base + off_W);
    ws.M = (T *)(base + off_M);
    ws.B = (T *)(base + off_B);
    ws.U = (T *)(base + off_U);
    ws.d = (T *)(base + off_d);
    ws.e = (T *)(base + off_e);
    ws.prog = (int *)(base + off_prog);
    ws.geqrf_buf = (T *)(base + off_geqrf);

    return ws;
}

template <typename T> void handle_free(SolverHandle<T> *ws) {
    CUBLAS_CHECK(cublasDestroy(ws->cublas));
    CUSOLVER_CHECK(cusolverDnDestroy(ws->cusolver));
    CUDA_CHECK(cudaFree(ws->d_info));
    CUDA_CHECK(cudaFree(ws->pool));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template SolverHandle<float> handle_alloc<float>(int, int, int, cudaStream_t);
template SolverHandle<double> handle_alloc<double>(int, int, int, cudaStream_t);
template void handle_free<float>(SolverHandle<float> *);
template void handle_free<double>(SolverHandle<double> *);

} // namespace cuev
