/**
 * @file   workspace.cu
 * @brief  WorkspaceMp<T> allocation / deallocation — distributed analogue of
 *         workspace_alloc / workspace_free in src/cuda/solver.cu.
 *
 * One cudaMalloc for the entire per-rank device pool; one std::malloc for the
 * (typically zero-sized, but must be non-null) cuSOLVERMp host scratch.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/workspace.h"

namespace cuev {

template <typename T>
SolverWorkspace<T> workspace_alloc(cusolverDnHandle_t h, int n, cudaStream_t stream) {
    SolverWorkspace<T> ws{};
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // Query cuSOLVER buffer sizes
    if constexpr (std::is_same_v<T, float>) {
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(h, 2 * n, n, nullptr, 2 * n, &ws.geqrf_lwork));
        CUSOLVER_CHECK(
            cusolverDnSorgqr_bufferSize(h, 2 * n, n, n, nullptr, 2 * n, nullptr, &ws.orgqr_lwork));
        CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(h, uplo, n, nullptr, n, &ws.potrf_lwork));
        CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(h, jobz, uplo, SDC_BASE_N, nullptr, SDC_BASE_N,
                                                   nullptr, &ws.syevd_lwork));
    } else {
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(h, 2 * n, n, nullptr, 2 * n, &ws.geqrf_lwork));
        CUSOLVER_CHECK(
            cusolverDnDorgqr_bufferSize(h, 2 * n, n, n, nullptr, 2 * n, nullptr, &ws.orgqr_lwork));
        CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(h, uplo, n, nullptr, n, &ws.potrf_lwork));
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(h, jobz, uplo, SDC_BASE_N, nullptr, SDC_BASE_N,
                                                   nullptr, &ws.syevd_lwork));
    }

    auto align = [](size_t x) -> size_t { return (x + 255u) & ~size_t(255u); };

    size_t off_geqrf = 0;
    size_t off_orgqr = off_geqrf + align((size_t)ws.geqrf_lwork * sizeof(T));
    size_t off_potrf = off_orgqr + align((size_t)ws.orgqr_lwork * sizeof(T));
    size_t off_syevd = off_potrf + align((size_t)ws.potrf_lwork * sizeof(T));
    size_t off_info = off_syevd + align((size_t)ws.syevd_lwork * sizeof(T));
    size_t off_W = off_info + align(sizeof(int));
    size_t off_tau = off_W + align(2u * (size_t)n * n * sizeof(T));
    size_t off_data = off_tau + align((size_t)n * sizeof(T));

    // Data pool: 6n² elements see workspace.h for the derivation.
    ws.data_cap = (size_t)6 * n * n;
    size_t total = off_data + ws.data_cap * sizeof(T);

    CUDA_CHECK(cudaMalloc(&ws.pool, total));
    char *base = static_cast<char *>(ws.pool);
    ws.geqrf_buf = reinterpret_cast<T *>(base + off_geqrf);
    ws.orgqr_buf = reinterpret_cast<T *>(base + off_orgqr);
    ws.potrf_buf = reinterpret_cast<T *>(base + off_potrf);
    ws.syevd_buf = reinterpret_cast<T *>(base + off_syevd);
    ws.d_info = reinterpret_cast<int *>(base + off_info);
    ws.qdwh_W = reinterpret_cast<T *>(base + off_W);
    ws.qdwh_tau = reinterpret_cast<T *>(base + off_tau);
    ws.data = reinterpret_cast<T *>(base + off_data);
    ws.data_used = 0;

    (void)stream;
    return ws;
}

template <typename T> void workspace_free(SolverWorkspace<T> &ws) {
    CUDA_CHECK(cudaFree(ws.pool));
    ws = SolverWorkspace<T>{};
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template SolverWorkspace<float> workspace_alloc<float>(cusolverDnHandle_t, int, cudaStream_t);
template SolverWorkspace<double> workspace_alloc<double>(cusolverDnHandle_t, int, cudaStream_t);
template void workspace_free<float>(SolverWorkspace<float> &);
template void workspace_free<double>(SolverWorkspace<double> &);

} // namespace cuev
