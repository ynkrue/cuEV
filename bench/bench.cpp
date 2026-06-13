/**
 * @file   bench.cpp
 * @brief  Benchmark GEMV, GEMM, and transpose against cuBLAS.
 *
 * Build:  make bench
 * Run:    ./build/cuBench [--M N] [--N N] [--K N] [--warmup N] [--iters N]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

// =============================================================================
// Shared utilities
// =============================================================================

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin(cudaStream_t s) { cudaEventRecord(start, s); }
    float end(cudaStream_t s) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

// fp32 accumulation can diverge from cuBLAS's reduction order, so rtol is loose for float.
template <typename T>
static bool check(const T *ref, const T *got, int n,
                  double rtol = std::is_same_v<T, float> ? 1e-2 : 1e-5,
                  double atol = std::is_same_v<T, float> ? 1e-4 : 1e-9) {
    for (int i = 0; i < n; ++i) {
        double r    = (double)ref[i];
        double g    = (double)got[i];
        double diff = fabs(r - g);
        if (diff > atol + rtol * fabs(r)) {
            printf("  MISMATCH at [%d]: ref=%.8f  got=%.8f  abserr=%.2e\n", i, r, g, diff);
            return false;
        }
    }
    return true;
}

// =============================================================================
// GEMV  (y = alpha*A*x + beta*y,  A row-major M×N)
// =============================================================================

// Row-major A is passed as col-major Aᵀ (N×M) with CUBLAS_OP_T.
template <typename T>
static void cublas_gemv(cublasHandle_t handle,
                        T alpha, const T *dA, const T *dx,
                        T beta,  T *dy,
                        int M, int N, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, N, M,
                                 (const float *)&alpha, (const float *)dA, N,
                                 (const float *)dx, 1,
                                 (const float *)&beta,  (float *)dy, 1));
    } else {
        CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_T, N, M,
                                 (const double *)&alpha, (const double *)dA, N,
                                 (const double *)dx, 1,
                                 (const double *)&beta,  (double *)dy, 1));
    }
}

template <typename T>
using GemvLauncher = void (*)(T, const T *, const T *, T, T *, int, int, cudaStream_t);

template <typename T>
static void bench_gemv(const char *name, GemvLauncher<T> fn,
                       T alpha, const T *dA, const T *dx,
                       T beta,  T *dy_tmp,
                       const T *ref_host,
                       int M, int N, int warmup, int iters,
                       cudaStream_t stream) {
    GpuTimer timer;

    for (int i = 0; i < warmup; ++i) { fn(alpha, dA, dx, beta, dy_tmp, M, N, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Zero y so beta*y doesn't carry stale values into the correctness run.
    CUDA_CHECK(cudaMemsetAsync(dy_tmp, 0, M * sizeof(T), stream));
    fn(alpha, dA, dx, T(0), dy_tmp, M, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> got(M);
    CUDA_CHECK(cudaMemcpy(got.data(), dy_tmp, M * sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = check(ref_host, got.data(), M);

    timer.begin(stream);
    for (int i = 0; i < iters; ++i) { fn(alpha, dA, dx, beta, dy_tmp, M, N, stream); }
    float ms = timer.end(stream);

    double bytes = ((double)M * N + N + M) * sizeof(T);
    double gbps  = bytes * iters / (ms * 1e-3) / 1e9;
    printf("  %-30s  %7.3f ms/iter   %7.2f GB/s   %s\n", name, ms / iters, gbps, ok ? "OK" : "WRONG");
}

template <typename T>
static void run_gemv_suite(cublasHandle_t cublas,
                           int M, int N, int warmup, int iters,
                           cudaStream_t stream) {
    printf("=== gemv %s  M=%d  N=%d ===\n",
           std::is_same_v<T, float> ? "fp32" : "fp64", M, N);

    std::vector<T> hA(M * N), hx(N);
    for (auto &v : hA) { v = (T)(rand() % 100 - 50) / T(50); }
    for (auto &v : hx) { v = (T)(rand() % 100 - 50) / T(50); }

    T *dA, *dx, *dy_ref, *dy_tmp;
    CUDA_CHECK(cudaMalloc(&dA,     M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dx,         N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dy_ref,     M * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dy_tmp,     M * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(),     N * sizeof(T), cudaMemcpyHostToDevice));

    T alpha = T(1), beta = T(0);

    CUDA_CHECK(cudaMemsetAsync(dy_ref, 0, M * sizeof(T), stream));
    cublas_gemv(cublas, alpha, dA, dx, beta, dy_ref, M, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> hy_ref(M);
    CUDA_CHECK(cudaMemcpy(hy_ref.data(), dy_ref, M * sizeof(T), cudaMemcpyDeviceToHost));

    GpuTimer timer;
    timer.begin(stream);
    for (int i = 0; i < iters; ++i) { cublas_gemv(cublas, alpha, dA, dx, beta, dy_ref, M, N, stream); }
    float ms_ref = timer.end(stream);

    double bytes_ref = ((double)M * N + N + M) * sizeof(T);
    printf("  %-30s  %7.3f ms/iter   %7.2f GB/s   (cuBLAS reference)\n\n",
           std::is_same_v<T, float> ? "cublas_sgemv" : "cublas_dgemv",
           ms_ref / iters, bytes_ref * iters / (ms_ref * 1e-3) / 1e9);

    struct Entry { const char *name; GemvLauncher<T> fn; };
    Entry kernels[] = {
        {"gemv_gmem", cuev::kernels::gemv_gmem<T>},
        {"gemv_smem", cuev::kernels::gemv_smem<T>},
    };
    for (auto &e : kernels) {
        bench_gemv<T>(e.name, e.fn, alpha, dA, dx, beta, dy_tmp,
                      hy_ref.data(), M, N, warmup, iters, stream);
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy_ref));
    CUDA_CHECK(cudaFree(dy_tmp));
    printf("\n");
}

// =============================================================================
// GEMM  (C = alpha*A*B + beta*C,  all row-major, A M×K, B K×N, C M×N)
// =============================================================================

// Row-major C=A*B ↔ col-major Cᵀ = Bᵀ·Aᵀ, so swap A/B and swap M/N.
template <typename T>
static void cublas_gemm(cublasHandle_t handle,
                        T alpha, const T *dA, const T *dB,
                        T beta,  T *dC,
                        int M, int N, int K, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 (const float *)&alpha,
                                 (const float *)dB, N,
                                 (const float *)dA, K,
                                 (const float *)&beta, (float *)dC, N));
    } else {
        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 (const double *)&alpha,
                                 (const double *)dB, N,
                                 (const double *)dA, K,
                                 (const double *)&beta, (double *)dC, N));
    }
}

template <typename T>
using GemmLauncher = void (*)(T, const T *, const T *, T, T *, int, int, int, cudaStream_t);

template <typename T>
static void bench_gemm(const char *name, GemmLauncher<T> fn,
                       T alpha, const T *dA, const T *dB,
                       T beta,  T *dC_tmp,
                       const T *ref_host,
                       int M, int N, int K, int warmup, int iters,
                       cudaStream_t stream) {
    GpuTimer timer;

    for (int i = 0; i < warmup; ++i) { fn(alpha, dA, dB, beta, dC_tmp, M, N, K, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaMemsetAsync(dC_tmp, 0, M * N * sizeof(T), stream));
    fn(alpha, dA, dB, T(0), dC_tmp, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> got(M * N);
    CUDA_CHECK(cudaMemcpy(got.data(), dC_tmp, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = check(ref_host, got.data(), M * N);

    timer.begin(stream);
    for (int i = 0; i < iters; ++i) { fn(alpha, dA, dB, beta, dC_tmp, M, N, K, stream); }
    float ms = timer.end(stream);

    double tflops = 2.0 * M * N * K * iters / (ms * 1e-3) / 1e12;
    printf("  %-30s  %7.3f ms/iter   %7.3f TFLOP/s   %s\n", name, ms / iters, tflops, ok ? "OK" : "WRONG");
}

template <typename T>
static void run_gemm_suite(cublasHandle_t cublas,
                           int M, int N, int K, int warmup, int iters,
                           cudaStream_t stream) {
    printf("=== gemm %s  M=%d  N=%d  K=%d ===\n",
           std::is_same_v<T, float> ? "fp32" : "fp64", M, N, K);

    std::vector<T> hA(M * K), hB(K * N);
    for (auto &v : hA) { v = (T)(rand() % 100 - 50) / T(50); }
    for (auto &v : hB) { v = (T)(rand() % 100 - 50) / T(50); }

    T *dA, *dB, *dC_ref, *dC_tmp;
    CUDA_CHECK(cudaMalloc(&dA,     M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dB,     K * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dC_ref, M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dC_tmp, M * N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(T), cudaMemcpyHostToDevice));

    T alpha = T(1), beta = T(0);

    CUDA_CHECK(cudaMemsetAsync(dC_ref, 0, M * N * sizeof(T), stream));
    cublas_gemm(cublas, alpha, dA, dB, beta, dC_ref, M, N, K, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> hC_ref(M * N);
    CUDA_CHECK(cudaMemcpy(hC_ref.data(), dC_ref, M * N * sizeof(T), cudaMemcpyDeviceToHost));

    GpuTimer timer;
    timer.begin(stream);
    for (int i = 0; i < iters; ++i) { cublas_gemm(cublas, alpha, dA, dB, beta, dC_ref, M, N, K, stream); }
    float ms_ref = timer.end(stream);

    double tflops_ref = 2.0 * M * N * K * iters / (ms_ref * 1e-3) / 1e12;
    printf("  %-30s  %7.3f ms/iter   %7.3f TFLOP/s   (cuBLAS reference)\n\n",
           std::is_same_v<T, float> ? "cublas_sgemm" : "cublas_dgemm",
           ms_ref / iters, tflops_ref);

    struct Entry { const char *name; GemmLauncher<T> fn; };
    Entry kernels[] = {
        {"gemm_gmem",     cuev::kernels::gemm_gmem<T>},
        {"gemm_smem",     cuev::kernels::gemm_smem<T>},
        {"gemm_tiled",    cuev::kernels::gemm_tiled<T>},
        {"gemm_warptile", cuev::kernels::gemm_warptile<T>},
    };
    for (auto &e : kernels) {
        bench_gemm<T>(e.name, e.fn, alpha, dA, dB, beta, dC_tmp,
                      hC_ref.data(), M, N, K, warmup, iters, stream);
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_ref));
    CUDA_CHECK(cudaFree(dC_tmp));
    printf("\n");
}

// =============================================================================
// Transpose  (AT = Aᵀ,  A is M×N row-major)
// =============================================================================

// cublasSgeam/cublasDgeam: treat row-major M×N A as col-major N×M A, request
// CUBLAS_OP_T to get col-major M×N output = row-major N×M AT.
template <typename T>
static void cublas_transpose(cublasHandle_t handle,
                             const T *dA, T *dAT,
                             int M, int N, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    T alpha = T(1), beta = T(0);
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
                                 (const float *)&alpha, (const float *)dA, N,
                                 (const float *)&beta,  (const float *)dA, M,
                                 (float *)dAT, M));
    } else {
        CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
                                 (const double *)&alpha, (const double *)dA, N,
                                 (const double *)&beta,  (const double *)dA, M,
                                 (double *)dAT, M));
    }
}

template <typename T>
static void run_transpose_suite(cublasHandle_t cublas,
                                int M, int N, int warmup, int iters,
                                cudaStream_t stream) {
    printf("=== transpose %s  M=%d  N=%d ===\n",
           std::is_same_v<T, float> ? "fp32" : "fp64", M, N);

    std::vector<T> hA(M * N);
    for (auto &v : hA) { v = (T)(rand() % 100 - 50) / T(50); }

    T *dA, *dAT_ref, *dAT_tmp;
    CUDA_CHECK(cudaMalloc(&dA,      M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dAT_ref, N * M * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dAT_tmp, N * M * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));

    cublas_transpose(cublas, dA, dAT_ref, M, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> hAT_ref(N * M);
    CUDA_CHECK(cudaMemcpy(hAT_ref.data(), dAT_ref, N * M * sizeof(T), cudaMemcpyDeviceToHost));

    GpuTimer timer;
    timer.begin(stream);
    for (int i = 0; i < iters; ++i) { cublas_transpose(cublas, dA, dAT_ref, M, N, stream); }
    float ms_ref = timer.end(stream);

    double bytes_ref = 2.0 * M * N * sizeof(T);
    printf("  %-30s  %7.3f ms/iter   %7.2f GB/s   (cuBLAS reference)\n\n",
           std::is_same_v<T, float> ? "cublas_sgeam" : "cublas_dgeam",
           ms_ref / iters, bytes_ref * iters / (ms_ref * 1e-3) / 1e9);

    for (int i = 0; i < warmup; ++i) { cuev::kernels::transpose(dA, dAT_tmp, M, N, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> hAT_got(N * M);
    CUDA_CHECK(cudaMemcpy(hAT_got.data(), dAT_tmp, N * M * sizeof(T), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < M && ok; ++i) {
        for (int j = 0; j < N && ok; ++j) {
            if (hAT_got[j * M + i] != hA[i * N + j]) { ok = false; }
        }
    }

    timer.begin(stream);
    for (int i = 0; i < iters; ++i) { cuev::kernels::transpose(dA, dAT_tmp, M, N, stream); }
    float ms = timer.end(stream);

    double bytes = 2.0 * M * N * sizeof(T);
    printf("  %-30s  %7.3f ms/iter   %7.2f GB/s   %s\n",
           "transpose", ms / iters, bytes * iters / (ms * 1e-3) / 1e9, ok ? "OK" : "WRONG");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dAT_ref));
    CUDA_CHECK(cudaFree(dAT_tmp));
    printf("\n");
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char **argv) {
    BenchArgs args = {4096, 4096, 5, 25};
    int K = 4096;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--M")      && i+1 < argc) { args.M      = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--N")      && i+1 < argc) { args.N      = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--K")      && i+1 < argc) { K           = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) { args.warmup = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--iters")  && i+1 < argc) { args.iters  = atoi(argv[++i]); }
    }

    printf("cuBench  M=%d  N=%d  K=%d  warmup=%d  iters=%d\n\n",
           args.M, args.N, K, args.warmup, args.iters);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    run_gemv_suite<float> (cublas, args.M, args.N,    args.warmup, args.iters, stream);
    run_gemv_suite<double>(cublas, args.M, args.N,    args.warmup, args.iters, stream);
    run_gemm_suite<float> (cublas, args.M, args.N, K, args.warmup, args.iters, stream);
    run_gemm_suite<double>(cublas, args.M, args.N, K, args.warmup, args.iters, stream);
    run_transpose_suite<float> (cublas, args.M, args.N, args.warmup, args.iters, stream);
    run_transpose_suite<double>(cublas, args.M, args.N, args.warmup, args.iters, stream);

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
