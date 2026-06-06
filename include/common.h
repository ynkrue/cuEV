#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _e = (err);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(_e));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t _e = (err);                                             \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,   \
                    (int)_e);                                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

struct GemvArgs {
    int M;          // rows
    int N;          // cols
    int warmup;
    int iters;
};
