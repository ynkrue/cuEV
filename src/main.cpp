#include "common.h"
#include "cuda/kernels.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

/// Timing helpers
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

/// cuBLAS reference: y = alpha*A*x + beta*y  (row-major)
static void run_cublas_ref(cublasHandle_t handle,
                           float alpha, const float *dA, const float *dx,
                           float beta, float *dy,
                           int M, int N, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                             N, M, &alpha, dA, N, dx, 1, &beta, dy, 1));
}

static void run_cublas_ref(cublasHandle_t handle,
                           double alpha, const double *dA, const double *dx,
                           double beta, double *dy,
                           int M, int N, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_T,
                             N, M, &alpha, dA, N, dx, 1, &beta, dy, 1));
}

/// Correctness check
// fp32 accumulation across N terms can diverge from cuBLAS's reduction order
// especially near zero (cancellation), so we use a loose rtol for float.
template <typename T>
static bool check(const T *ref, const T *got, int n,
                  double rtol = sizeof(T) == 4 ? 1e-2 : 1e-5,
                  double atol = sizeof(T) == 4 ? 1e-4 : 1e-9) {
    for (int i = 0; i < n; ++i) {
        double r    = (double)ref[i];
        double g    = (double)got[i];
        double diff = fabs(r - g);
        if (diff > atol + rtol * fabs(r)) {
            printf("  MISMATCH at [%d]: ref=%.8f  got=%.8f  abserr=%.2e\n",
                   i, r, g, diff);
            return false;
        }
    }
    return true;
}

/// Benchmark one precision
template <typename T>
using Launcher = void (*)(T, const T *, const T *, T, T *, int, int,
                          cudaStream_t);

template <typename T>
static void bench(const char *name, Launcher<T> fn,
                  T alpha, const T *dA, const T *dx,
                  T beta, T *dy_tmp,
                  const T *ref_host,
                  int M, int N, int warmup, int iters,
                  cudaStream_t stream) {
    GpuTimer timer;

    for (int i = 0; i < warmup; ++i)
        fn(alpha, dA, dx, beta, dy_tmp, M, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Correctness: zero y first so beta*y term doesn't carry stale values.
    CUDA_CHECK(cudaMemsetAsync(dy_tmp, 0, M * sizeof(T), stream));
    fn(alpha, dA, dx, (T)0, dy_tmp, M, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> got(M);
    CUDA_CHECK(cudaMemcpy(got.data(), dy_tmp, M * sizeof(T),
                          cudaMemcpyDeviceToHost));
    bool ok = check(ref_host, got.data(), M);

    timer.begin(stream);
    for (int i = 0; i < iters; ++i)
        fn(alpha, dA, dx, beta, dy_tmp, M, N, stream);
    float ms = timer.end(stream);

    double bytes = ((double)M * N + N + M) * sizeof(T);
    double gbps  = bytes * iters / (ms * 1e-3) / 1e9;

    printf("  %-30s  %7.3f ms/iter   %7.2f GB/s   %s\n",
           name, ms / iters, gbps, ok ? "OK" : "WRONG");
}

/// Run full suite
template <typename T>
static void run_suite(cublasHandle_t cublas, int M, int N,
                      int warmup, int iters, cudaStream_t stream) {
    const char *tag = sizeof(T) == 4 ? "fp32" : "fp64";
    printf("=== %s  M=%d  N=%d ===\n", tag, M, N);

    // host data
    std::vector<T> hA(M * N), hx(N);
    for (int i = 0; i < M * N; ++i) hA[i] = (T)(rand() % 100 - 50) / (T)50;
    for (int i = 0; i < N;     ++i) hx[i] = (T)(rand() % 100 - 50) / (T)50;

    // device allocations
    T *dA, *dx, *dy, *dy_tmp;
    CUDA_CHECK(cudaMalloc(&dA,     M * N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dx,         N * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dy,         M * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dy_tmp,     M * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(),     N * sizeof(T), cudaMemcpyHostToDevice));

    T alpha = (T)1, beta = (T)0;

    // cuBLAS reference
    CUDA_CHECK(cudaMemsetAsync(dy, 0, M * sizeof(T), stream));
    run_cublas_ref(cublas, alpha, dA, dx, beta, dy, M, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<T> hy_ref(M);
    CUDA_CHECK(cudaMemcpy(hy_ref.data(), dy, M * sizeof(T), cudaMemcpyDeviceToHost));

    GpuTimer timer;
    timer.begin(stream);
    for (int i = 0; i < iters; ++i)
        run_cublas_ref(cublas, alpha, dA, dx, beta, dy, M, N, stream);
    float ms_ref = timer.end(stream);

    double bytes_ref = ((double)M * N + N + M) * sizeof(T);
    double gbps_ref  = bytes_ref * iters / (ms_ref * 1e-3) / 1e9;
    printf("  %-30s  %7.3f ms/iter   %7.2f GB/s   (reference)\n\n",
           sizeof(T) == 4 ? "cublas_sgemv" : "cublas_dgemv",
           ms_ref / iters, gbps_ref);

    struct Entry { const char *name; Launcher<T> fn; };
    Entry kernels[] = {
        {"gemv_gmem",             launch_gemv_gmem<T>},
        {"gemv_smem",              launch_gemv_smem<T>},
        {"gemv_tma",               launch_gemv_tma<T>},
        {"gemv_double_tma",        launch_gemv_double_tma<T>},
        {"gemv_cluster",           launch_gemv_cluster<T>},
    };

    for (auto &e : kernels)
        bench<T>(e.name, e.fn, alpha, dA, dx, beta, dy_tmp,
                 hy_ref.data(), M, N, warmup, iters, stream);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dy_tmp));
    printf("\n");
}


// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    GemvArgs args = {8192, 16384, 5, 25};

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--M")      && i+1 < argc) args.M      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N")      && i+1 < argc) args.N      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) args.warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")  && i+1 < argc) args.iters  = atoi(argv[++i]);
    }

    printf("cuGEMV benchmark  M=%d  N=%d  warmup=%d  iters=%d\n\n",
           args.M, args.N, args.warmup, args.iters);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    run_suite<float> (cublas, args.M, args.N, args.warmup, args.iters, stream);
    run_suite<double>(cublas, args.M, args.N, args.warmup, args.iters, stream);

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
