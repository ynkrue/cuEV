/**
 * @file   test_sdc.cu
 * @brief  Unit tests for spectral D&C helpers: sdc_split and sdc_combine.
 *
 * Build:  cmake --build build --target cuTest
 * Run:    ./build/cuTest --gtest_filter="SdcGPU.*"
 */

#include "common.h"
#include "cuev.h"
#include "kernels.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// =============================================================================
// GPU fixture
// =============================================================================

class SdcGPU : public ::testing::Test {
  protected:
    cublasHandle_t cublas{};
    cusolverDnHandle_t cusolver{};
    cudaStream_t stream{};

    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUBLAS_CHECK(cublasSetStream(cublas, stream));
        CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
        CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    }
    void TearDown() override {
        cublasDestroy(cublas);
        cusolverDnDestroy(cusolver);
        cudaStreamDestroy(stream);
    }

    // Create a workspace sized for dimension n — caller must free it.
    cuev::SolverWorkspace<double> make_ws(int n) {
        return cuev::workspace_alloc<double>(cusolver, n, stream);
    }

    // Upload host vector to a freshly allocated device buffer.
    double *upload(const std::vector<double> &h) {
        double *d;
        CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
        return d;
    }

    // Download device buffer of given element count into a host vector.
    std::vector<double> download(const double *d, int elems) {
        std::vector<double> h(elems);
        CUDA_CHECK(cudaMemcpy(h.data(), d, elems * sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }
};

// =============================================================================
// sdc_split
// =============================================================================

// Simplest case: Q1 = first k cols of I_n, Q2 = last (n-k) cols.
// Then H1 = Q1ᵀ H Q1 = top-left k×k block of H,
//      H2 = Q2ᵀ H Q2 = bottom-right (n-k)×(n-k) block of H.
TEST_F(SdcGPU, SplitIdentityBasis) {
    constexpr int n = 4, k = 2;

    // H = col-major 4×4, diagonal 1..4
    std::vector<double> hH(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        hH[i * n + i] = double(i + 1);

    // Q1 = first k=2 cols of I_4 (col-major 4×2)
    std::vector<double> hQ1(n * k, 0.0);
    for (int i = 0; i < k; ++i)
        hQ1[i * n + i] = 1.0;

    // Q2 = last (n-k)=2 cols of I_4 (col-major 4×2)
    std::vector<double> hQ2(n * (n - k), 0.0);
    for (int i = 0; i < n - k; ++i)
        hQ2[i * n + (k + i)] = 1.0;

    double *dH = upload(hH);
    double *dQ1 = upload(hQ1);
    double *dQ2 = upload(hQ2);
    double *dH1, *dH2;
    CUDA_CHECK(cudaMalloc(&dH1, k * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dH2, (n - k) * (n - k) * sizeof(double)));

    auto ws = make_ws(n);
    cuev::kernels::sdc_split(cublas, dH, dQ1, dQ2, dH1, dH2, n, k, &ws, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cuev::workspace_free(ws);

    auto hH1 = download(dH1, k * k);
    auto hH2 = download(dH2, (n - k) * (n - k));

    // H1 should be diag(1, 2)
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i)
            EXPECT_NEAR(hH1[j * k + i], (i == j) ? double(i + 1) : 0.0, 1e-10)
                << "H1 mismatch at (" << i << "," << j << ")";

    // H2 should be diag(3, 4)
    for (int j = 0; j < n - k; ++j)
        for (int i = 0; i < n - k; ++i)
            EXPECT_NEAR(hH2[j * (n - k) + i], (i == j) ? double(k + i + 1) : 0.0, 1e-10)
                << "H2 mismatch at (" << i << "," << j << ")";

    cudaFree(dH);
    cudaFree(dQ1);
    cudaFree(dQ2);
    cudaFree(dH1);
    cudaFree(dH2);
}

// H1 and H2 must be symmetric when H is symmetric and Q1, Q2 are orthonormal.
TEST_F(SdcGPU, SplitPreservesSymmetry) {
    constexpr int n = 4, k = 2;

    // Symmetric H (col-major)
    std::vector<double> hH = {
        4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1,
    };
    // Q1 = first k cols of I_4
    std::vector<double> hQ1(n * k, 0.0);
    for (int i = 0; i < k; ++i)
        hQ1[i * n + i] = 1.0;

    // Q2 = last (n-k) cols of I_4
    std::vector<double> hQ2(n * (n - k), 0.0);
    for (int i = 0; i < n - k; ++i)
        hQ2[i * n + (k + i)] = 1.0;

    double *dH = upload(hH);
    double *dQ1 = upload(hQ1);
    double *dQ2 = upload(hQ2);
    double *dH1, *dH2;
    CUDA_CHECK(cudaMalloc(&dH1, k * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dH2, (n - k) * (n - k) * sizeof(double)));

    auto ws = make_ws(n);
    cuev::kernels::sdc_split(cublas, dH, dQ1, dQ2, dH1, dH2, n, k, &ws, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cuev::workspace_free(ws);

    auto hH1 = download(dH1, k * k);
    auto hH2 = download(dH2, (n - k) * (n - k));

    for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i)
            EXPECT_NEAR(hH1[j * k + i], hH1[i * k + j], 1e-10)
                << "H1 not symmetric at (" << i << "," << j << ")";

    for (int j = 0; j < n - k; ++j)
        for (int i = 0; i < n - k; ++i)
            EXPECT_NEAR(hH2[j * (n - k) + i], hH2[i * (n - k) + j], 1e-10)
                << "H2 not symmetric at (" << i << "," << j << ")";

    cudaFree(dH);
    cudaFree(dQ1);
    cudaFree(dQ2);
    cudaFree(dH1);
    cudaFree(dH2);
}

// =============================================================================
// sdc_combine
// =============================================================================

// sdc_combine writes Q2 cols first, Q1 cols second (ascending eigenvalue order).
// Q2 = first m cols of I_n, Q1 = last k cols, evec1 = I_k, evec2 = I_m → result = I_n.
TEST_F(SdcGPU, CombineIdentityBasis) {
    constexpr int n = 4, k = 2;

    // Q1 = last k cols of I_n  (eigenvectors with eigenvalues > μ)
    std::vector<double> hQ1(n * k, 0.0);
    for (int i = 0; i < k; ++i)
        hQ1[i * n + (n - k + i)] = 1.0;

    // Q2 = first m=(n-k) cols of I_n  (eigenvectors with eigenvalues < μ)
    std::vector<double> hQ2(n * (n - k), 0.0);
    for (int i = 0; i < n - k; ++i)
        hQ2[i * n + i] = 1.0;

    std::vector<double> hE1(k * k, 0.0);
    for (int i = 0; i < k; ++i)
        hE1[i * k + i] = 1.0;

    std::vector<double> hE2((n - k) * (n - k), 0.0);
    for (int i = 0; i < n - k; ++i)
        hE2[i * (n - k) + i] = 1.0;

    double *dQ1 = upload(hQ1);
    double *dQ2 = upload(hQ2);
    double *dE1 = upload(hE1);
    double *dE2 = upload(hE2);
    double *dEvec;
    CUDA_CHECK(cudaMalloc(&dEvec, n * n * sizeof(double)));

    cuev::kernels::sdc_combine(cublas, dQ1, dQ2, dE1, dE2, dEvec, n, k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto hEvec = download(dEvec, n * n);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hEvec[j * n + i], (i == j) ? 1.0 : 0.0, 1e-10)
                << "evec mismatch at (" << i << "," << j << ")";

    cudaFree(dQ1);
    cudaFree(dQ2);
    cudaFree(dE1);
    cudaFree(dE2);
    cudaFree(dEvec);
}

// Columns of the combined evec must be orthonormal when inputs are orthonormal.
TEST_F(SdcGPU, CombinePreservesOrthonormality) {
    constexpr int n = 4, k = 2;

    // Use non-trivial Q1, Q2: first/last cols of a Hadamard-like rotation.
    // Simple choice: Q1 = (1/√2)*[[1,1],[1,-1],[0,0],[0,0]]ᵀ etc.
    // Easier: just use identity basis with a 45° rotation for evec1/evec2.
    std::vector<double> hQ1(n * k, 0.0);
    for (int i = 0; i < k; ++i)
        hQ1[i * n + i] = 1.0;

    std::vector<double> hQ2(n * (n - k), 0.0);
    for (int i = 0; i < n - k; ++i)
        hQ2[i * n + (k + i)] = 1.0;

    // evec1 = 45° rotation (2×2)
    double c = std::cos(M_PI / 4), s = std::sin(M_PI / 4);
    std::vector<double> hE1 = {c, s, -s, c}; // col-major

    // evec2 = 45° rotation (2×2)
    std::vector<double> hE2 = {c, s, -s, c};

    double *dQ1 = upload(hQ1), *dQ2 = upload(hQ2);
    double *dE1 = upload(hE1), *dE2 = upload(hE2);
    double *dEvec;
    CUDA_CHECK(cudaMalloc(&dEvec, n * n * sizeof(double)));

    cuev::kernels::sdc_combine(cublas, dQ1, dQ2, dE1, dE2, dEvec, n, k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto hEvec = download(dEvec, n * n);

    // Check VᵀV = I  (columns orthonormal)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double dot = 0.0;
            for (int r = 0; r < n; ++r)
                dot += hEvec[i * n + r] * hEvec[j * n + r];
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-10) << "VᵀV ≠ I at (" << i << "," << j << ")";
        }
    }

    cudaFree(dQ1);
    cudaFree(dQ2);
    cudaFree(dE1);
    cudaFree(dE2);
    cudaFree(dEvec);
}
