#include "common.h"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

// forward-declare the solver (defined in solver.cu)
template <typename T>
void solve(T *H, int n, T *eval, T *evec, cudaStream_t stream);

static void print_matrix(const char *name, const double *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
        printf("  [");
        for (int j = 0; j < cols; ++j)
            printf(" %8.4f", M[i * cols + j]);
        printf(" ]\n");
    }
    printf("\n");
}

int main() {
    // A is symmetric — eigenvalues are real
    // Row-major, 4×4
    const int N = 4;
    double hA[] = {
         4,  1, -2,  2,
         1,  2,  0,  1,
        -2,  0,  3, -2,
         2,  1, -2, -1,
    };
    print_matrix("A (input)", hA, N, N);

    double *dA, *d_eval, *d_evec;
    CUDA_CHECK(cudaMalloc(&dA,     N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eval, N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_evec, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, hA, N * N * sizeof(double), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    solve<double>(dA, N, d_eval, d_evec, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> h_eval(N), h_evec(N * N), h_A_out(N * N);
    CUDA_CHECK(cudaMemcpy(h_eval.data(), d_eval, N     * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_evec.data(), d_evec, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    printf("eigenvalues:\n  [");
    for (int i = 0; i < N; ++i) printf(" %8.4f", h_eval[i]);
    printf(" ]\n\n");

    print_matrix("eigenvectors (rows)", h_evec.data(), N, N);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_eval));
    CUDA_CHECK(cudaFree(d_evec));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
