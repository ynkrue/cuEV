/**
 * @file   test_tridi.cu
 * @brief  Correctness tests for the tridiagonal divide-and-conquer eigensolver.
 *
 * Validates eigenpairs (eval, evec) of a random symmetric tridiagonal (d, e) directly:
 *   residual       ‖T·V − V·Λ‖_F / ‖T‖_F
 *   orthogonality  ‖Vᵀ·V − I‖_F
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include "test.h"
#include <cmath>
#include <vector>

using namespace cutest;

template <typename T> static void tridi_case(int n, double res_tol, double orth_tol) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto ws = cuev::handle_alloc<T>(n, 32, 512, stream);

    std::vector<T> d(n), e(n - 1);
    fill_random(d, 11);
    fill_random(e, 23);

    T *dd = to_device(d), *de = to_device(e), *deval, *devec;
    CUDA_CHECK(cudaMalloc(&deval, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&devec, (size_t)n * n * sizeof(T)));

    cuev::kernels::tridi_dc(&ws, dd, de, deval, devec);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> w(n), V((size_t)n * n);
    to_host(w, deval);
    to_host(V, devec);

    std::vector<T> Tm((size_t)n * n, T(0));
    for (int i = 0; i < n; ++i)
        Tm[i + (size_t)i * n] = d[i];
    for (int i = 0; i < n - 1; ++i)
        Tm[(i + 1) + (size_t)i * n] = Tm[i + (size_t)(i + 1) * n] = e[i];

    std::vector<T> TV((size_t)n * n);
    gemm_host(TV, Tm, V, n, n, n, false, false, n, n, n);
    double rnum = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double r = (double)TV[i + (size_t)j * n] - (double)w[j] * (double)V[i + (size_t)j * n];
            rnum += r * r;
        }
    CHECK_LT(std::sqrt(rnum) / frob(Tm), res_tol);

    std::vector<T> VtV((size_t)n * n);
    gemm_host(VtV, V, V, n, n, n, true, false, n, n, n);
    double onum = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double t = (double)VtV[i + (size_t)j * n] - (i == j ? 1.0 : 0.0);
            onum += t * t;
        }
    CHECK_LT(std::sqrt(onum) / std::sqrt((double)n), orth_tol);

    CUDA_CHECK(cudaFree(dd));
    CUDA_CHECK(cudaFree(de));
    CUDA_CHECK(cudaFree(deval));
    CUDA_CHECK(cudaFree(devec));
    cuev::handle_free(&ws);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(tridi_dc, fp64_leaf) {
    tridi_case<double>(48, 1e-10, 1e-10);
}
TEST(tridi_dc, fp64_onemerge) {
    tridi_case<double>(100, 1e-10, 1e-10);
}
TEST(tridi_dc, fp64_multilevel) {
    tridi_case<double>(500, 1e-9, 1e-9);
}
TEST(tridi_dc, fp32_multilevel) {
    tridi_case<float>(500, 1e-3, 1e-3);
}
