/**
 * @file   bench_evd.cpp
 * @brief  Eigenvalues-only EVD benchmark: cuEV (DBBR + BC + tridiagonal QL) vs cuSOLVER.
 *
 * Times our tridiagonalization pipeline (DBBR full→band, then GPU bulge chasing
 * band→tridiagonal) plus a CPU symmetric-tridiagonal QL solve for the eigenvalues,
 * and checks the spectrum against cuSOLVER's dsyevd/ssyevd (NOVECTOR).
 *
 * NOTE: bulge chasing is the step-b wavefront (Algorithm 2, O(n²b)); it is still
 * the pipeline bottleneck vs cuSOLVER at moderate n but feasible at large n.
 *
 * Usage: cuBenchEvd [--n N]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <type_traits>
#include <vector>

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer() {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }
    ~GpuTimer() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    void begin(cudaStream_t s) {
        cudaEventRecord(a, s);
    }
    float end(cudaStream_t s) {
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

template <typename T> static void fill_symmetric(std::vector<T> &h, int n) {
    srand(1234);
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i)
            h[i + j * n] = h[j + i * n] = (T)(rand() % 200 - 100) / T(100);
}

// Eigenvalues of a symmetric tridiagonal (implicit-shift QL, no vectors). d[0..n-1] is the
// diagonal (overwritten with ascending eigenvalues); e[0..n-1] the sub-diagonal, e[n-1]=0.
static int tridiag_eigvals(std::vector<double> &d, std::vector<double> &e, int n) {
    auto sgn = [](double x, double y) { return y >= 0 ? std::fabs(x) : -std::fabs(x); };
    auto pyth = [](double x, double y) { return std::hypot(x, y); };
    for (int l = 0; l < n; ++l) {
        int iter = 0, m;
        do {
            for (m = l; m < n - 1; ++m)
                if (std::fabs(e[m]) <= DBL_EPSILON * (std::fabs(d[m]) + std::fabs(d[m + 1]))) break;
            if (m != l) {
                if (iter++ == 60) return -1;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = pyth(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + sgn(r, g));
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = m - 1; i >= l; --i) {
                    double f = s * e[i], bb = c * e[i];
                    r = pyth(f, g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * bb;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - bb;
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
    std::sort(d.begin(), d.begin() + n);
    return 0;
}

// Reference eigenvalues via the 64-bit cusolverDnXsyevd (size_t workspace, so it does not
// overflow like the legacy int-lwork Dsyevd, which fails for n ≳ 26.8k).
template <typename T>
static std::vector<double> cusolver_eigvals(cusolverDnHandle_t h, const std::vector<T> &hA, int n,
                                            cudaStream_t stream, float &ms) {
    const cudaDataType dt = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
    const auto job = CUSOLVER_EIG_MODE_NOVECTOR;
    const auto uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    T *dA, *dW;
    int *info;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dW, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));

    size_t devBytes = 0, hostBytes = 0;
    cusolverStatus_t st = cusolverDnXsyevd_bufferSize(
        h, params, job, uplo, (int64_t)n, dt, dA, (int64_t)n, dt, dW, dt, &devBytes, &hostBytes);
    if (st != CUSOLVER_STATUS_SUCCESS) { // dense EVD unsupported at this n (int32 limit ~46340)
        cudaFree(dA);
        cudaFree(dW);
        cudaFree(info);
        cusolverDnDestroyParams(params);
        ms = -1.0f;
        return {};
    }
    void *devWork = nullptr, *hostWork = nullptr;
    CUDA_CHECK(cudaMalloc(&devWork, devBytes));
    if (hostBytes) hostWork = malloc(hostBytes);

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)n * n * sizeof(T), cudaMemcpyHostToDevice));
    GpuTimer t;
    t.begin(stream);
    CUSOLVER_CHECK(cusolverDnXsyevd(h, params, job, uplo, (int64_t)n, dt, dA, (int64_t)n, dt, dW,
                                    dt, devWork, devBytes, hostWork, hostBytes, info));
    ms = t.end(stream);

    std::vector<T> w(n);
    CUDA_CHECK(cudaMemcpy(w.data(), dW, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(info));
    CUDA_CHECK(cudaFree(devWork));
    free(hostWork);
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    return std::vector<double>(w.begin(), w.end());
}

// Time cuSOLVER's direct tridiagonalization (sytrd) — the apples-to-apples comparison for our
// DBBR + BC (it produces D/E, not eigenvalues). Small workspace, so no large-n overflow.
template <typename T>
static float cusolver_sytrd_ms(cusolverDnHandle_t h, const std::vector<T> &hA, int n,
                               cudaStream_t stream) {
    const auto uplo = CUBLAS_FILL_MODE_LOWER;
    T *dA, *dD, *dE, *dTau, *work;
    int *info, lwork = 0;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dD, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dE, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dTau, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    cusolverStatus_t st;
    if constexpr (std::is_same_v<T, float>)
        st = cusolverDnSsytrd_bufferSize(h, uplo, n, dA, n, dD, dE, dTau, &lwork);
    else
        st = cusolverDnDsytrd_bufferSize(h, uplo, n, dA, n, dD, dE, dTau, &lwork);
    if (st != CUSOLVER_STATUS_SUCCESS) { // unsupported at this n
        cudaFree(dA);
        cudaFree(dD);
        cudaFree(dE);
        cudaFree(dTau);
        cudaFree(info);
        return -1.0f;
    }
    CUDA_CHECK(cudaMalloc(&work, (size_t)lwork * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)n * n * sizeof(T), cudaMemcpyHostToDevice));
    GpuTimer t;
    t.begin(stream);
    if constexpr (std::is_same_v<T, float>)
        CUSOLVER_CHECK(cusolverDnSsytrd(h, uplo, n, dA, n, dD, dE, dTau, work, lwork, info));
    else
        CUSOLVER_CHECK(cusolverDnDsytrd(h, uplo, n, dA, n, dD, dE, dTau, work, lwork, info));
    float ms = t.end(stream);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaFree(dE));
    CUDA_CHECK(cudaFree(dTau));
    CUDA_CHECK(cudaFree(work));
    CUDA_CHECK(cudaFree(info));
    return ms;
}

template <typename T>
static void run_suite(cusolverDnHandle_t cusolver, int n, cudaStream_t stream) {
    const char *prec = std::is_same_v<T, float> ? "fp32" : "fp64";
    const int nbw = 64, nk = 512;

    std::vector<T> hA((size_t)n * n);
    fill_symmetric(hA, n);

    // reference spectrum (full EVD) + direct tridiagonalization timing
    float ms_ref = 0;
    auto ev_ref = cusolver_eigvals<T>(cusolver, hA, n, stream, ms_ref);
    float ms_sytrd = cusolver_sytrd_ms<T>(cusolver, hA, n, stream);

    // cuev tridiagonalization: DBBR (full→band) then BC (band→tridiagonal)
    auto ws = cuev::handle_alloc<T>(n, nbw, nk, stream);
    T *dA;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)n * n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), (size_t)n * n * sizeof(T), cudaMemcpyHostToDevice));

    GpuTimer t;
    t.begin(stream);
    cuev::kernels::dbbr_reduce(&ws, dA);
    float ms_dbbr = t.end(stream);

    t.begin(stream);
    cuev::kernels::bc_pack(&ws, dA, ws.B, n, nbw);
    cuev::kernels::bc_chase(&ws, ws.B, ws.d, ws.e, ws.U, n, nbw);
    float ms_bc = t.end(stream);

    // tridiagonal eigenvalues on CPU (QL)
    std::vector<T> hd(n), he(n);
    CUDA_CHECK(cudaMemcpy(hd.data(), ws.d, n * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(he.data(), ws.e, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<double> d(hd.begin(), hd.end()), e(he.begin(), he.end());
    e[n - 1] = 0.0;
    auto cpu0 = std::chrono::high_resolution_clock::now();
    int qlst = tridiag_eigvals(d, e, n);
    double ms_ql =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - cpu0)
            .count();

    const bool have_ref = !ev_ref.empty();         // cuSOLVER dense EVD supported at this n?
    const float ms_tridi = ms_dbbr + ms_bc;        // our tridiagonalization (full → tridiagonal)
    const float ms_cuev = ms_tridi + (float)ms_ql; // + tridiagonal eigenvalues
    const char c = std::is_same_v<T, float> ? 's' : 'd';
    const char *unsup = "  unsupported (n > cuSOLVER int32 limit ~46340)";

    printf("=== EVD (eigenvalues) %s  n=%d  b=%d ===\n", prec, n, nbw);
    printf("  -- tridiagonalization (full -> tridiagonal) --\n");
    if (ms_sytrd >= 0) {
        printf("  cusolver_%csytrd (direct)   %10.2f ms\n", c, ms_sytrd);
        printf("  cuev DBBR + BC             %10.2f ms   (%.2fx %csytrd)\n", ms_tridi,
               ms_tridi / ms_sytrd, c);
    } else {
        printf("  cusolver_%csytrd (direct) %s\n", c, unsup);
        printf("  cuev DBBR + BC             %10.2f ms\n", ms_tridi);
    }
    printf("    DBBR (full->band)        %10.2f ms\n", ms_dbbr);
    printf("    BC   (band->tridi)       %10.2f ms\n", ms_bc);

    printf("  -- full eigenvalues --\n");
    if (have_ref) {
        printf("  cusolver_%csyevd            %10.2f ms\n", c, ms_ref);
        printf("  cuev DBBR+BC + CPU QL      %10.2f ms   (%.2fx %csyevd)\n", ms_cuev,
               ms_cuev / ms_ref, c);
    } else {
        printf("  cusolver_%csyevd          %s\n", c, unsup);
        printf("  cuev DBBR+BC + CPU QL      %10.2f ms\n", ms_cuev);
    }
    printf("    QL (tridi->evals)        %10.2f ms   [CPU%s]\n", ms_ql,
           qlst ? " — NOT CONVERGED" : "");

    if (have_ref) {
        double range = ev_ref[n - 1] - ev_ref[0], maxdiff = 0;
        for (int i = 0; i < n; ++i)
            maxdiff = std::max(maxdiff, std::fabs(d[i] - ev_ref[i]));
        printf("  max |eig - cusolver|         %9.2e   (spectrum range %.3e)\n", maxdiff, range);
    } else {
        printf("  (no cuSOLVER reference — spectrum not checked)\n");
    }
    printf("\n");

    CUDA_CHECK(cudaFree(dA));
    cuev::handle_free(&ws);
}

int main(int argc, char **argv) {
    int n = 16000;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = atoi(argv[++i]);

    printf("cuBenchEvd  n=%d  (eigenvalues only)\n\n", n);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cusolverDnHandle_t cusolver;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));

    run_suite<double>(cusolver, n, stream);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolver));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
