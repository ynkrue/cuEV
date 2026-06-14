/**
 * @file   test_qdwh.cu
 * @brief  Unit tests for the QDWH sign-function implementation.
 *
 * Layers:
 *   CPU  — qdwh_coeffs properties (no GPU needed)
 *   GPU  — individual kernels (qdwh_shift, qdwh_eye, qdwh_symmetrize)
 *   GPU  — end-to-end qdwh_sign on known matrices
 *
 * Build:  cmake --build build --target cuTest
 * Run:    ./build/cuTest
 *         ./build/cuTest --gtest_filter="QdwhCoeffs.*"
 */

#include "common.h"
#include "cuev.h"
#include "kernels.cuh"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// =============================================================================
// Helpers
// =============================================================================

// Replicate the qdwh_coeffs formula (the function is in an anonymous namespace
// in qdwh.cu, so we can't call it directly — we test the math here).
template <typename T> static void host_coeffs(T &l, T &a, T &b, T &c) {
    T d = std::cbrt(T(4) * (T(1) - l * l) / (l * l * l * l));
    a = std::sqrt(T(1) + d) +
        std::sqrt(T(0.5) *
                  (T(8) - T(4) * d + T(8) * (T(2) - l * l) / (l * l * std::sqrt(T(1) + d))));
    b = T(0.25) * (a - T(1)) * (a - T(1));
    c = a + b - T(1);
    l = l * (a + b * l * l) / (T(1) + c * l * l);
}

// Frobenius norm of a host matrix (col-major).
static double frob(const std::vector<double> &M, int n) {
    double s = 0;
    for (double v : M)
        s += v * v;
    return std::sqrt(s);
}

// Matrix multiply C = A*B on host (col-major, all square n×n).
static std::vector<double> matmul(const std::vector<double> &A, const std::vector<double> &B,
                                  int n) {
    std::vector<double> C(n * n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                C[j * n + i] += A[k * n + i] * B[j * n + k];
    return C;
}

// =============================================================================
// CPU tests — qdwh_coeffs
// =============================================================================

// r(x) = x(a + bx²)/(1 + cx²) has fixed point at x=1 iff a+b = 1+c.
TEST(QdwhCoeffs, FixedPointCondition) {
    for (double l0 : {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        double l = l0, a, b, c;
        host_coeffs(l, a, b, c);
        EXPECT_NEAR(a + b, 1.0 + c, 1e-10) << "fixed-point condition failed at l₀=" << l0;
    }
}

// All coefficients must be strictly positive.
TEST(QdwhCoeffs, Positivity) {
    for (double l0 : {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99}) {
        double l = l0, a, b, c;
        host_coeffs(l, a, b, c);
        EXPECT_GT(a, 0.0) << "a ≤ 0 at l₀=" << l0;
        EXPECT_GT(b, 0.0) << "b ≤ 0 at l₀=" << l0;
        EXPECT_GT(c, 0.0) << "c ≤ 0 at l₀=" << l0;
    }
}

// l must strictly increase toward 1 and converge within 8 iterations.
TEST(QdwhCoeffs, LConvergesInEightIterations) {
    for (double l0 : {0.05, 0.1, 0.3, 0.5}) {
        double l = l0;
        double l_prev = l;
        for (int iter = 0; iter < 8; ++iter) {
            double a, b, c;
            host_coeffs(l, a, b, c);
            EXPECT_GT(l, l_prev) << "l did not increase at iter=" << iter << " l₀=" << l0;
            EXPECT_LE(l, 1.0 + 1e-10) << "l exceeded 1 at iter=" << iter;
            l_prev = l;
            if (l >= 1.0 - 1e-12) break;
        }
        EXPECT_NEAR(l, 1.0, 1e-10) << "l did not converge to 1 within 8 iters, l₀=" << l0;
    }
}

// Verify r maps l into itself correctly (the update formula is self-consistent).
TEST(QdwhCoeffs, ScalarUpdateConsistency) {
    // r(l₀) should equal the new l returned by host_coeffs.
    for (double l0 : {0.2, 0.5, 0.8}) {
        double l = l0, a, b, c;
        host_coeffs(l, a, b, c); // l is now l'
        // Recompute r(l₀) directly from the returned a,b,c.
        double r_l0 = l0 * (a + b * l0 * l0) / (1.0 + c * l0 * l0);
        EXPECT_NEAR(l, r_l0, 1e-10) << "l update inconsistency at l₀=" << l0;
    }
}

// =============================================================================
// GPU fixture
// =============================================================================

class QdwhGPU : public ::testing::Test {
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

    // Allocate device copy of host matrix, run qdwh_sign, copy result back.
    std::vector<double> run_sign(std::vector<double> hB, int n) {
        double *dB;
        CUDA_CHECK(cudaMalloc(&dB, (size_t)n * n * sizeof(double)));
        CUDA_CHECK(
            cudaMemcpy(dB, hB.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
        auto ws = cuev::workspace_alloc<double>(cusolver, n, stream);
        cuev::kernels::qdwh_sign(cublas, cusolver, dB, n, &ws, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> hS(n * n);
        CUDA_CHECK(
            cudaMemcpy(hS.data(), dB, (size_t)n * n * sizeof(double), cudaMemcpyDeviceToHost));
        cuev::workspace_free(ws);
        CUDA_CHECK(cudaFree(dB));
        return hS;
    }
};

// =============================================================================
// GPU tests — individual kernels
// =============================================================================

TEST_F(QdwhGPU, ShiftDiagonal) {
    // qdwh_shift subtracts mu from diagonal; off-diagonal unchanged.
    constexpr int n = 4;
    // col-major identity × 3 → after shift by 1 → 2·I
    std::vector<double> hA(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        hA[i * n + i] = 3.0;

    double *dA;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    cuev::kernels::qdwh_shift(dA, 1.0, n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> hR(n * n);
    CUDA_CHECK(cudaMemcpy(hR.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hR[j * n + i], (i == j) ? 2.0 : 0.0, 1e-14);
}

TEST_F(QdwhGPU, EyeBottomBlock) {
    // qdwh_eye sets rows [n:2n,:] of a 2n×n matrix to identity.
    constexpr int n = 3;
    std::vector<double> hW(2 * n * n, 0.0);
    double *dW;
    CUDA_CHECK(cudaMalloc(&dW, 2 * n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), 2 * n * n * sizeof(double), cudaMemcpyHostToDevice));
    cuev::kernels::qdwh_eye(dW, n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(hW.data(), dW, 2 * n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dW));

    // col-major: column j, row i is hW[j*(2n)+i]
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hW[j * (2 * n) + (n + i)], (i == j) ? 1.0 : 0.0, 1e-14)
                << "eye mismatch at row=" << (n + i) << " col=" << j;
}

TEST_F(QdwhGPU, SymmetrizeRestoresSymmetry) {
    // An almost-symmetric matrix should become exactly symmetric after qdwh_symmetrize.
    constexpr int n = 3;
    // col-major
    std::vector<double> hA = {1, 2, 3, 2.001, 4, 5, 3.002, 5.003, 6};
    double *dA;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    cuev::kernels::qdwh_symmetrize(dA, n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<double> hR(n * n);
    CUDA_CHECK(cudaMemcpy(hR.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hR[j * n + i], hR[i * n + j], 1e-14)
                << "not symmetric at (" << i << "," << j << ")";
}

// =============================================================================
// GPU tests — qdwh_sign end-to-end
// =============================================================================

// sign(α·I) = sign(α)·I
TEST_F(QdwhGPU, PositiveScaledIdentity) {
    constexpr int n = 4;
    std::vector<double> hB(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        hB[i * n + i] = 3.0;

    auto hS = run_sign(hB, n);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hS[j * n + i], (i == j) ? 1.0 : 0.0, 1e-8)
                << "sign(3I) ≠ I at (" << i << "," << j << ")";
}

TEST_F(QdwhGPU, NegativeScaledIdentity) {
    constexpr int n = 4;
    std::vector<double> hB(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        hB[i * n + i] = -2.0;

    auto hS = run_sign(hB, n);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hS[j * n + i], (i == j) ? -1.0 : 0.0, 1e-8)
                << "sign(-2I) ≠ -I at (" << i << "," << j << ")";
}

// sign(diag(+a, -b)) = diag(+1, -1)
TEST_F(QdwhGPU, DiagonalMixedSign) {
    constexpr int n = 2;
    // col-major 2×2
    std::vector<double> hB = {3.0, 0.0, 0.0, -2.0};
    auto hS = run_sign(hB, n);
    EXPECT_NEAR(hS[0], +1.0, 1e-8); // (0,0)
    EXPECT_NEAR(hS[1], 0.0, 1e-12); // (1,0)
    EXPECT_NEAR(hS[2], 0.0, 1e-12); // (0,1)
    EXPECT_NEAR(hS[3], -1.0, 1e-8); // (1,1)
}

// sign(B)² = I  (eigenvalues of sign(B) are ±1, so squaring gives I)
TEST_F(QdwhGPU, SignSquaredIsIdentity) {
    constexpr int n = 4;
    // Symmetric matrix with mixed-sign eigenvalues (col-major).
    // A = [[4,1,-2,2],[1,2,0,1],[-2,0,3,-2],[2,1,-2,-1]]
    std::vector<double> hB = {
        4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1,
    };
    auto hS = run_sign(hB, n);
    auto hS2 = matmul(hS, hS, n);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hS2[j * n + i], (i == j) ? 1.0 : 0.0, 1e-6)
                << "sign(B)² ≠ I at (" << i << "," << j << ")";
}

// sign(B) must itself be symmetric.
TEST_F(QdwhGPU, SignIsSymmetric) {
    constexpr int n = 4;
    std::vector<double> hB = {
        4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2, 1, -2, -1,
    };
    auto hS = run_sign(hB, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(hS[j * n + i], hS[i * n + j], 1e-8)
                << "sign(B) not symmetric at (" << i << "," << j << ")";
}
