/**
 * @file   test_solver.cu
 * @brief  End-to-end integration tests for cuev::symm_eig_solve.
 *
 * Build:  cmake --build build --target cuTest
 * Run:    ./build/cuTest --gtest_filter="Solver.*"
 *
 * Each test checks three properties:
 *   1. Eigenvalues match a direct cuSOLVER syevd reference (double-precision tol)
 *   2. Eigenvector matrix is orthonormal: ||VᵀV − I||_F < tol
 *   3. Eigenvalue equation residual: ||HV − V·diag(λ)||_F / ||H||_F < tol
 *
 * Sizes:
 *   n ≤ SDC_BASE_N (256): hits the cuSOLVER base-case only
 *   n > SDC_BASE_N      : exercises the spectral D&C recursion
 */

#include "common.h"
#include "cuev.h"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

// =============================================================================
// Helpers
// =============================================================================

// Generate a random n×n symmetric matrix in column-major order.
static std::vector<double> random_symmetric(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> A(n * n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i)
            A[j * n + i] = A[i * n + j] = dist(rng);
    return A;
}

// Direct cuSOLVER syevd — independent of cuev wrappers. Returns eigenvalues ascending.
static std::vector<double> reference_eigenvalues(std::vector<double> H, int n) {
    double *dA, *dW, *dWork;
    int *dInfo, lwork;
    cusolverDnHandle_t h;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dW, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, H.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUSOLVER_CHECK(cusolverDnCreate(&h));
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                               n, dA, n, dW, &lwork));
    CUDA_CHECK(cudaMalloc(&dWork, (size_t)lwork * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDsyevd(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, n,
                                    dW, dWork, lwork, dInfo));
    std::vector<double> evals(n);
    CUDA_CHECK(cudaMemcpy(evals.data(), dW, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(dA);
    cudaFree(dW);
    cudaFree(dWork);
    cudaFree(dInfo);
    CUSOLVER_CHECK(cusolverDnDestroy(h));
    return evals;
}

// =============================================================================
// Fixture
// =============================================================================

class Solver : public ::testing::Test {
  protected:
    cublasHandle_t cublas{};
    cudaStream_t stream{};

    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
    }
    void TearDown() override {
        cublasDestroy(cublas);
        cudaStreamDestroy(stream);
    }

    double *upload(const std::vector<double> &h) {
        double *d;
        CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
        return d;
    }

    std::vector<double> download(const double *d, size_t elems) {
        std::vector<double> h(elems);
        CUDA_CHECK(cudaMemcpy(h.data(), d, elems * sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }

    // ||VᵀV − I||_F (computed on GPU via cuBLAS GEMM, then checked on host)
    double ortho_error(const double *V, int n) {
        double *dG;
        CUDA_CHECK(cudaMalloc(&dG, (size_t)n * n * sizeof(double)));
        double one = 1.0, zero = 0.0;
        CUBLAS_CHECK(
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &one, V, n, V, n, &zero, dG, n));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto G = download(dG, (size_t)n * n);
        cudaFree(dG);
        double err = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                double r = G[j * n + i] - (i == j ? 1.0 : 0.0);
                err += r * r;
            }
        return std::sqrt(err);
    }

    // ||H·V − V·diag(λ)||_F / ||H||_F  (all on GPU)
    double residual(const double *H_orig, const double *V, const double *eval, int n) {
        // R = H·V  (reuse H_orig — it is a separate copy)
        double *dR, *dHcopy;
        CUDA_CHECK(cudaMalloc(&dR, (size_t)n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dHcopy, (size_t)n * n * sizeof(double)));
        CUDA_CHECK(
            cudaMemcpy(dHcopy, H_orig, (size_t)n * n * sizeof(double), cudaMemcpyDeviceToDevice));
        double one = 1.0, zero = 0.0;
        CUBLAS_CHECK(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, dHcopy, n, V, n,
                                 &zero, dR, n));

        // R ← R − V·diag(λ): scale each column j of V by eval[j] and subtract
        // Download eval to host, then scale columns with cublasDscal / loop
        auto heval = download(eval, n);
        for (int j = 0; j < n; ++j) {
            double neg_lam = -heval[j];
            CUBLAS_CHECK(
                cublasDaxpy(cublas, n, &neg_lam, V + (size_t)j * n, 1, dR + (size_t)j * n, 1));
        }

        // ||R||_F
        double res_norm;
        CUBLAS_CHECK(cublasDnrm2(cublas, n * n, dR, 1, &res_norm));

        // ||H||_F
        double h_norm;
        CUBLAS_CHECK(cublasDnrm2(cublas, n * n, dHcopy, 1, &h_norm));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(dR);
        cudaFree(dHcopy);
        return res_norm / h_norm;
    }

    // Run symm_eig_solve and verify all three properties.
    void check(const std::vector<double> &hH, int n, double eval_tol, double ortho_tol,
               double res_tol) {
        // Keep original H on device for residual check (symm_eig_solve overwrites H).
        double *dH_orig = upload(hH);
        double *dH = upload(hH); // working copy — will be overwritten
        double *dEval, *dEvec;
        CUDA_CHECK(cudaMalloc(&dEval, (size_t)n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dEvec, (size_t)n * n * sizeof(double)));

        cuev::symm_eig_solve<double>(dH, n, dEval, dEvec, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto hEval = download(dEval, n);
        auto hRef = reference_eigenvalues(hH, n);

        // 1. Eigenvalue comparison
        double max_abs = *std::max_element(
            hRef.begin(), hRef.end(), [](double a, double b) { return std::abs(a) < std::abs(b); });
        max_abs = std::max(std::abs(max_abs), 1.0); // guard against near-zero spectrum
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hEval[i], hRef[i], eval_tol * max_abs)
                << "eigenvalue mismatch at index " << i;

        // 2. Orthonormality
        double oe = ortho_error(dEvec, n);
        EXPECT_LT(oe, ortho_tol) << "||VᵀV − I||_F = " << oe;

        // 3. Residual
        double re = residual(dH_orig, dEvec, dEval, n);
        EXPECT_LT(re, res_tol) << "||HV − Vλ||_F / ||H||_F = " << re;

        cudaFree(dH_orig);
        cudaFree(dH);
        cudaFree(dEval);
        cudaFree(dEvec);
    }
};

// =============================================================================
// Base-case tests (n ≤ SDC_BASE_N = 256 → single cuSOLVER syevd call)
// =============================================================================

TEST_F(Solver, DiagonalN4) {
    // Diagonal matrix — trivial eigenvalue check.
    int n = 4;
    std::vector<double> hH(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        hH[i * n + i] = double(i + 1);
    check(hH, n, /*eval*/ 1e-12, /*ortho*/ 1e-12, /*res*/ 1e-12);
}

TEST_F(Solver, DenseN64) {
    int n = 64;
    check(random_symmetric(n, 1), n, 1e-10, 1e-10, 1e-10);
}

TEST_F(Solver, DenseN128) {
    int n = 128;
    check(random_symmetric(n, 2), n, 1e-10, 1e-10, 1e-10);
}

// =============================================================================
// Large tests (n > SDC_BASE_N → spectral D&C recursion)
// =============================================================================

// Recursive path: with cubic QDWH convergence the sign function reaches machine
// precision each level, so accuracy stays near 1e-12 even at the largest sizes.
TEST_F(Solver, DenseN512) {
    int n = 512;
    check(random_symmetric(n, 3), n, /*eval*/ 1e-11, /*ortho*/ 1e-11, /*res*/ 1e-11);
}

TEST_F(Solver, DenseN768) {
    int n = 768;
    check(random_symmetric(n, 4), n, /*eval*/ 1e-11, /*ortho*/ 1e-11, /*res*/ 1e-11);
}

TEST_F(Solver, DenseN1024) {
    int n = 1024;
    check(random_symmetric(n, 5), n, /*eval*/ 1e-11, /*ortho*/ 1e-11, /*res*/ 1e-11);
}

TEST_F(Solver, DenseN2048) {
    int n = 2048;
    check(random_symmetric(n, 5), n, /*eval*/ 1e-11, /*ortho*/ 1e-11, /*res*/ 1e-11);
}

TEST_F(Solver, DenseN4096) {
    int n = 4096;
    check(random_symmetric(n, 5), n, /*eval*/ 1e-11, /*ortho*/ 1e-11, /*res*/ 1e-11);
}

TEST_F(Solver, DenseN8192) {
    int n = 8192;
    check(random_symmetric(n, 5), n, /*eval*/ 1e-11, /*ortho*/ 1e-11, /*res*/ 1e-11);
}

TEST_F(Solver, DenseN16384) {
    int n = 16384;
    check(random_symmetric(n, 5), n, /*eval*/ 1e-11, /*ortho*/ 1e-11, /*res*/ 1e-11);
}
