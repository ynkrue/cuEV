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
    ws.stream = stream;

    CUBLAS_CHECK(cublasCreate(&ws.cublas));
    CUBLAS_CHECK(cublasSetStream(ws.cublas, stream));
    CUSOLVER_CHECK(cusolverDnCreate(&ws.cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(ws.cusolver, stream));
    CUDA_CHECK(cudaMalloc(&ws.d_info, sizeof(int)));

    // Workspace queries
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(
            cusolverDnSgeqrf_bufferSize(ws.cusolver, n, nbw, nullptr, n, &ws.geqrf_lwork));
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(ws.cusolver, n, nbw, nbw, nullptr, n, nullptr,
                                                   &ws.orgqr_lwork));
        // dense syevd unused (D&C runs on the tridiagonal); size that workspace when tridi_dc
        // lands. Note: dense syevd needs ~2n² and its int lwork overflows for n ≥ 32768.
        ws.syevd_lwork = 0;
    } else {
        CUSOLVER_CHECK(
            cusolverDnDgeqrf_bufferSize(ws.cusolver, n, nbw, nullptr, n, &ws.geqrf_lwork));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(ws.cusolver, n, nbw, nbw, nullptr, n, nullptr,
                                                   &ws.orgqr_lwork));
        // dense syevd unused (D&C runs on the tridiagonal); size that workspace when tridi_dc
        // lands. Note: dense syevd needs ~2n² and its int lwork overflows for n ≥ 32768.
        ws.syevd_lwork = 0;
    }

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
    size_t off_Tmat = off;
    off += align_up((size_t)nbw * nbw * s);
    size_t off_Dwk = off;
    off += align_up((size_t)nk * nbw * s);
    size_t off_W = off;
    off += align_up((size_t)n * n * s);
    size_t off_B = off;
    off += align_up((size_t)(2 * nbw) * n * s);
    size_t off_U = off;
    off += align_up((size_t)n * std::max(n - 2, 1) * s);
    size_t off_d = off;
    off += align_up((size_t)n * s);
    size_t off_e = off;
    off += align_up((size_t)n * s);
    size_t off_prog = off;
    off += align_up((size_t)n * sizeof(int));
    size_t off_geqrf = off;
    off += align_up((size_t)ws.geqrf_lwork * s);
    size_t off_orgqr = off;
    off += align_up((size_t)ws.orgqr_lwork * s);
    size_t off_syevd = off;
    off += align_up((size_t)ws.syevd_lwork * s);
    ws.pool_bytes = off;

    CUDA_CHECK(cudaMalloc(&ws.pool, ws.pool_bytes));

    auto base = (uint8_t *)ws.pool;
    ws.Y = (T *)(base + off_Y);
    ws.Z = (T *)(base + off_Z);
    ws.tau = (T *)(base + off_tau);
    ws.Tmat = (T *)(base + off_Tmat);
    ws.Dwk = (T *)(base + off_Dwk);
    ws.W = (T *)(base + off_W);
    ws.B = (T *)(base + off_B);
    ws.U = (T *)(base + off_U);
    ws.d = (T *)(base + off_d);
    ws.e = (T *)(base + off_e);
    ws.prog = (int *)(base + off_prog);
    ws.geqrf_buf = (T *)(base + off_geqrf);
    ws.orgqr_buf = (T *)(base + off_orgqr);
    ws.syevd_buf = (T *)(base + off_syevd);

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
