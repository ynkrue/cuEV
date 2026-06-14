/**
 * @file   debug.cpp
 * @brief  Eigensolver smoke test — run solve<double> on a small known matrix.
 *
 * Build:  make debug
 * Run:    ./build/cuDebug
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuev.h"
#include <iomanip>
#include <iostream>
#include <vector>

static void print_matrix(const char *label, const double *h, int rows, int cols) {
    std::cout << label << ":\n";
    for (int i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(9) << std::fixed << std::setprecision(4) << h[i * cols + j];
        }
        std::cout << " ]\n";
    }
    std::cout << "\n";
}

int main() {

    constexpr int N = 4;
    double hA[] = {
        4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1,
    };
    print_matrix("A (input)", hA, N, N);

    double *dA, *d_eval, *d_evec;
    CUDA_CHECK(cudaMalloc(&dA, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eval, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_evec, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, hA, N * N * sizeof(double), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cuev::symm_eig_solve<double>(dA, N, d_eval, d_evec, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> h_eval(N), h_evec(N * N);
    CUDA_CHECK(cudaMemcpy(h_eval.data(), d_eval, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "eigenvalues:\n  [";
    for (int i = 0; i < N; ++i) {
        std::cout << std::setw(9) << std::fixed << std::setprecision(4) << h_eval[i];
    }
    std::cout << " ]\n\n";

    print_matrix("eigenvectors (cols, col-major)", h_evec.data(), N, N);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_eval));
    CUDA_CHECK(cudaFree(d_evec));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
