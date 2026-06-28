/**
 * @file   tridi.cu
 * @brief  Tridiagonalization eigensolver orchestration — single GPU.
 *
 * TODO
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <algorithm>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

extern "C" {
void dstedc_(const char *compz, const int *n, double *d, double *e, double *z, const int *ldz,
             double *work, const int *lwork, int *iwork, const int *liwork, int *info);

void sstedc_(const char *compz, const int *n, float *d, float *e, float *z, const int *ldz,
             float *work, const int *lwork, int *iwork, const int *liwork, int *info);
}

} // namespace

namespace cuev {
namespace kernels {

template <typename T> void tridi_dc(SolverHandle<T> *ws, T *d, T *e, T *eval, T *evec) {
    int n = ws->n;
    // Stubbed CPU solver
    T *h_eval = new T[ws->n];
    T *h_e = new T[ws->n];
    T *h_evec = new T[ws->n * ws->n];
    CUDA_CHECK(cudaMemcpy(h_eval, d, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e, e, n * sizeof(T), cudaMemcpyDeviceToHost));

    int info;
    int lwork = 1 + 4 * n + n * n;
    int liwork = 6 + 6 * n;
    T *work = (T *)malloc(lwork * sizeof(T));
    int *iwork = (int *)malloc(liwork * sizeof(int));
    int ldz = n;

    if constexpr (std::is_same_v<T, float>) {
        sstedc_("I", &n, h_eval, h_e, h_evec, &ldz, work, &lwork, iwork, &liwork, &info);
    } else {
        dstedc_("I", &n, h_eval, h_e, h_evec, &ldz, work, &lwork, iwork, &liwork, &info);
    }

    free(work);
    free(iwork);
    if (info != 0) {
        fprintf(stderr, "Error in stedc: info = %d\n", info);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(eval, h_eval, n * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(evec, h_evec, n * n * sizeof(T), cudaMemcpyHostToDevice));
    delete[] h_eval;
    delete[] h_e;
    delete[] h_evec;
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T) template void tridi_dc<T>(SolverHandle<T> *, T *, T *, T *, T *);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
