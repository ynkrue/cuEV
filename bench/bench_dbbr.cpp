/**
 * @file   bench_dbbr.cpp
 * @brief  Minimal timing benchmark for the DBBR band reduction (kernels::dbbr_reduce).
 *
 * Usage: cuBenchDbbr [--iters I]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

template <typename T>
static double bench_dbbr(int n, int nbw, int nk, cudaStream_t stream, int iters) {
    std::vector<T> hA((size_t)n * n);
    std::mt19937 rng(1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto &x : hA)
        x = (T)dist(rng);

    T *dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)n * n * sizeof(T)));
    auto ws = cuev::handle_alloc<T>(n, nbw, nk, stream);

    auto reset = [&] {
        CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), (size_t)n * n * sizeof(T), cudaMemcpyHostToDevice,
                                   stream));
    };

    // warmup
    reset();
    cuev::kernels::dbbr_reduce(&ws, dA);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    double total_ms = 0;
    for (int it = 0; it < iters; ++it) {
        reset(); // dbbr_reduce overwrites A; restore (untimed) each iteration
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaEventRecord(s, stream);
        cuev::kernels::dbbr_reduce(&ws, dA);
        cudaEventRecord(e, stream);
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        total_ms += ms;
    }
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cuev::handle_free(&ws);
    CUDA_CHECK(cudaFree(dA));
    return total_ms / iters;
}

int main(int argc, char **argv) {
    int iters = 3;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int nbw = 64, nk = 512;
    printf("DBBR band reduction  (fp64, b=%d, k=%d)  iters=%d\n\n", nbw, nk, iters);
    printf("  %8s   %12s   %10s\n", "n", "time (ms)", "TFLOP/s");
    for (int n : {8192, 16384, 32768, 65536}) {
        double ms = bench_dbbr<double>(n, nbw, nk, stream, iters);
        double flops = 4.0 / 3.0 * (double)n * n * n; // ~ tridiagonalization leading order
        printf("  %8d   %12.2f   %10.2f\n", n, ms, flops / (ms * 1e-3) / 1e12);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
