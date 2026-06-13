/**
 * @file   solver.cu
 * @brief  Top-level eigensolver orchestration.
 *
 * Public entry point: cuev::solve<T>(H, n, eval, evec, stream).
 *
 * Algorithm:
 *   tridiag_hh        H → T     (Householder tridiagonalization, n−1 steps)
 *   tridiag_eig       T → Λ, QT (divide-and-conquer eigensolver for eigenvalues and eigenvectors of
 * T) tridiag_hh_back   Λ → Q     (back-transform Householder vectors × eigenvectors of T)
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "kernels.cuh"
#include <cuda.h>

// =============================================================================
// Solver components
// =============================================================================
namespace {

template <typename T> void tridiag_hh(T *H, int n, T *d, T *e, T *tau, cudaStream_t stream) {
    T *v, *p, *u;
    cudaMalloc(&v, n * sizeof(T));
    cudaMalloc(&p, n * sizeof(T));
    cudaMalloc(&u, n * sizeof(T));

    for (int k = 0; k < n - 1; ++k) {
        cuev::kernels::hh_reflect(H, v, tau, d, e, n, k, stream);
        cuev::kernels::hh_trail_matvec(v, H, p, n, k, stream);
        cuev::kernels::hh_ortho(v, p, tau, u, n, k, stream);
        cuev::kernels::hh_update(v, u, H, n, k, stream);
    }
    cudaMemcpyAsync(d + (n - 1), H + (n - 1) * n + (n - 1), sizeof(T), cudaMemcpyDeviceToDevice,
                    stream);

    cudaFree(v);
    cudaFree(p);
    cudaFree(u);
}

template <typename T> void tridiag_eig(T *d, T *e, int n, T *eval, T *QT, cudaStream_t stream) {
    if (n == 2) {
        cuev::kernels::eig_leaf(d, e, eval, QT, stream);
        return;
    }

    int k = n / 2;
    T *d1, *d2, *eval1, *eval2, *eval_sorted, *Q1, *Q2, *G;
    cudaMalloc(&d1, k * sizeof(T));
    cudaMalloc(&d2, (n - k) * sizeof(T));
    cudaMalloc(&eval1, k * sizeof(T));
    cudaMalloc(&eval2, (n - k) * sizeof(T));
    cudaMalloc(&eval_sorted, n * sizeof(T));
    cudaMalloc(&Q1, k * k * sizeof(T));
    cudaMalloc(&Q2, (n - k) * (n - k) * sizeof(T));
    cudaMalloc(&G, n * n * sizeof(T));

    // T = diag(T₁,T₂) + e[k-1]·vvᵀ, v = [0…0 1 1 0…0]
    cuev::kernels::eig_split(d, e, n, k, d1, d2, stream);

    tridiag_eig(d1, e, k, eval1, Q1, stream);
    tridiag_eig(d2, e + k, n - k, eval2, Q2, stream);

    // merge: sort gives interval bounds; secular uses eval1/eval2 as poles
    cuev::kernels::eig_sort(eval1, eval2, n, k, eval_sorted, stream);
    cuev::kernels::eig_secular(eval1, eval2, eval_sorted, Q1, Q2, e, n, k, eval, G, stream);
    cuev::kernels::eig_gemm(Q1, Q2, G, QT, n, k, stream);

    cudaFree(d1);
    cudaFree(d2);
    cudaFree(eval1);
    cudaFree(eval2);
    cudaFree(eval_sorted);
    cudaFree(Q1);
    cudaFree(Q2);
    cudaFree(G);
}

template <typename T>
void tridiag_hh_back(T *H, int n, T *QT, T *tau, T *evec, cudaStream_t stream) {
    // stub — blocked back-transform via hh_wy_build / hh_wy_apply not yet implemented
    (void)H;
    (void)n;
    (void)QT;
    (void)tau;
    (void)evec;
    (void)stream;
}

} // namespace

// =============================================================================
// Public API
// =============================================================================
namespace cuev {

template <typename T> void solve(T *H, int n, T *eval, T *evec, cudaStream_t stream) {
    T *d, *e, *tau;
    cudaMalloc(&d, n * sizeof(T));
    cudaMalloc(&e, (n - 1) * sizeof(T));
    cudaMalloc(&tau, (n - 1) * sizeof(T));

    // --- Tridiagonalization ---
    tridiag_hh(H, n, d, e, tau, stream);
#ifdef DEBUG
    std::cout << "[DEBUG] Tridiagonalization --------------------------------\n";
    cuev::debug::print_tridiag("tridiag_hh — T", d, e, n, stream);
    cuev::debug::print_hh_vecs("tridiag_hh — Householder vectors", H, n, stream);
    std::cout << std::endl;
#endif

    // --- Eigenvalue decomposition ---
    T *QT;
    cudaMalloc(&QT, n * n * sizeof(T));

    tridiag_eig(d, e, n, eval, QT, stream);
#ifdef DEBUG
    std::cout << "[DEBUG] Eigenvalue decomposition --------------------------\n";
    cuev::debug::print_eval("tridiag_eig — eigenvalues", eval, n, stream);
    cuev::debug::print_evec("tridiag_eig — eigenvectors", QT, n, stream);
    std::cout << std::endl;
#endif

    // --- Back-transform eigenvectors ---
    tridiag_hh_back(H, n, QT, tau, evec, stream);
#ifdef DEBUG
    std::cout << "[DEBUG] Backtransformation --------------------------------\n";
    std::cout << "\n\n";
#endif

    cudaFree(d);
    cudaFree(e);
    cudaFree(tau);
    cudaFree(QT);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template void solve<float>(float *, int, float *, float *, cudaStream_t);
template void solve<double>(double *, int, double *, double *, cudaStream_t);

} // namespace cuev
