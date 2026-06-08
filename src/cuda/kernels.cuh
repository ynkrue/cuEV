/**
 * @file kernels.cuh
 * 
 * cuGEMV kernel declarations — y = alpha * A * x + beta * y
 * A is M×N row-major, incx = incy = 1.  T = float or double.
 */

#pragma once
#include <cuda_runtime.h>

template <typename T>
void launch_gemv_gmem(T alpha, const T *A, const T *x,
                       T beta, T *y, int M, int N, cudaStream_t stream);

template <typename T>
void launch_gemv_smem(T alpha, const T *A, const T *x,
                      T beta, T *y, int M, int N, cudaStream_t stream);

// Hopper: TMA bulk-copy from global to shared
template <typename T>
void launch_gemv_tma(T alpha, const T *A, const T *x,
                     T beta, T *y, int M, int N, cudaStream_t stream);
                     
// Hopper: persistent producer/consumer kernel using barrier pipelines.
template <typename T>
void launch_gemv_double_tma(T alpha, const T *A, const T *x,
                            T beta, T *y, int M, int N,
                            cudaStream_t stream);

// Hopper: thread block cluster, where neighbouring CTAs share smem via cluster bars.
template <typename T>
void launch_gemv_cluster(T alpha, const T *A, const T *x,
                         T beta, T *y, int M, int N, cudaStream_t stream);

