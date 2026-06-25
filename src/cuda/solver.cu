/**
 * @file   solver.cu
 * @brief  2-stage tridiagonalization eigensolver orchestration — single GPU.
 *
 * Pipeline: DBBR → bulge chasing → D&C (tridiagonal) → back-transform.
 * Public entry point: cuev::symm_eig_solve<T>(A, n, eval, evec, stream).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include "cuev.h"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <type_traits>

namespace cuev {

// dbbr_reduce lives in cuev::kernels (declared in kernels.cuh) so tests can call it directly.
namespace kernels {

template <typename T> void dbbr_reduce(SolverHandle<T> *ws, T *A) {
    int n = ws->n, b = ws->nbw, k = ws->nk;
    int lda = ws->n;
    T zero = T(0);
    T one = T(1);
    T neg1 = T(-1);
    T neg_half = T(-0.5);

    // outer (block) loop
    for (int i = 0; i < n; i += k) {
        int kc = std::min(k, n - i); // current block width
        int block_cols = 0;          // reflector columns Y/Z accumulated so far in this block

        // inner (panel) loop
        for (int j = i; j < i + kc && j + b < n; j += b) {
            int pc = std::min(b, n - j); // current panel width
            int rows = n - (j + pc);
            int off = j * lda + (j + pc);
            int tr = (j + pc) * lda + (j + pc);
            int zc = (j - i) * lda + (j + pc);

            /// Panel QR factorization red panel in [Algorithm 1, Wang et al. 2025]
            kernels::dbbr_panel_qr(ws, A + off, ws->Y + off, rows, pc);

            /// Update trailing green panel in [Algorithm 1, Wang et al. 2025]
            // W = Y·T
            cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, pc, &one, ws->Y + off, lda,
                         ws->Tmat, pc, &zero, ws->W + off, lda);

            // P = A[off]·W
            cublas::symm(ws, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, rows, pc, &one, A + tr, lda,
                         ws->W + off, lda, &zero, ws->Z + zc, lda);

            // Subtract previous panel contribution from P (w = block_cols):
            //   P -= Y[j+b:n, i:i+w]·(Z[j+b:n, 0:w]ᵀ·W) + Z[j+b:n, 0:w]·(Y[j+b:n, i:i+w]ᵀ·W)
            if (block_cols > 0) {
                // D = Z[j+b:n, 0:w]ᵀ·W
                cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, block_cols, pc, rows, &one,
                             ws->Z + (j + pc), lda, ws->W + off, lda, &zero, ws->Dwk, block_cols);
                // P -= Y[j+b:n, i:i+w]·D
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, block_cols, &neg1,
                             ws->Y + i * lda + (j + pc), lda, ws->Dwk, block_cols, &one, ws->Z + zc,
                             lda);
                // D = Y[j+b:n, i:i+w]ᵀ·W
                cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, block_cols, pc, rows, &one,
                             ws->Y + i * lda + (j + pc), lda, ws->W + off, lda, &zero, ws->Dwk,
                             block_cols);
                // P -= Z[j+b:n, 0:w]·D
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, block_cols, &neg1,
                             ws->Z + (j + pc), lda, ws->Dwk, block_cols, &one, ws->Z + zc, lda);
            }

            // C = Wᵀ·P
            cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, pc, pc, rows, &one, ws->W + off, lda,
                         ws->Z + zc, lda, &zero, ws->Dwk, pc);

            // Z = P - 0.5·W·C
            cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, pc, pc, &neg_half, ws->Y + off, lda,
                         ws->Dwk, pc, &one, ws->Z + zc, lda);

            block_cols += pc;

            // green update if next panel is within this block
            if (j + pc < i + kc) {
                // A[j+b:n, j+b:j+2b] -= Z[j+b:n, 0:w]·Y[j+b:j+2b, i:i+w]ᵀ
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_T, rows, pc, block_cols, &neg1,
                             ws->Z + (j + pc), lda, ws->Y + i * lda + (j + pc), lda, &one, A + tr,
                             lda);
                // A[j+b:n, j+b:j+2b] -= Y[j+b:n, i:i+w]·Z[j+b:j+2b, 0:w]ᵀ
                cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_T, rows, pc, block_cols, &neg1,
                             ws->Y + i * lda + (j + pc), lda, ws->Z + (j + pc), lda, &one, A + tr,
                             lda);
            }
        }

        // Update trailing green block in [Algorithm 1, Wang et al. 2025]:
        //   A[i+k:n, i+k:n] -= Y[i+k:n, i:i+w]·Z[i+k:n, 0:w]ᵀ + Z[i+k:n, 0:w]·Y[i+k:n, i:i+w]ᵀ
        cublas::syr2k(ws, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n - (i + kc), block_cols, &neg1,
                      ws->Y + i * lda + (i + kc), lda, ws->Z + (i + kc), lda, &one,
                      A + (i + kc) * lda + (i + kc), lda);

        // stash Z, Y into V, W at column i for back-transform
    }
}

template <typename T> void bc_tridi(SolverHandle<T> *ws, T *A) {
    int n = ws->n, b = ws->nbw;

    // band (A lower triangle) → packed (b+1)×n
    kernels::bc_pack(ws, A, ws->B, n, b);

    // packed band → tridiagonal (d, e) + Householder vectors U for BC-Back
    kernels::bc_chase(ws, ws->B, ws->d, ws->e, ws->U, n, b);
}

template <typename T> void tridi_dc(SolverHandle<T> *ws) {
    // TODO: implement solver
    (void)ws;
}

template <typename T> void back_transform(SolverHandle<T> *ws) {
    // TODO: implement back-transform of eigenvectors
    (void)ws;
}

} // namespace kernels

// =============================================================================
// Public entry point
// =============================================================================

template <typename T> void symm_eig_solve(T *A, int n, T *eval, T *evec, cudaStream_t stream) {
    // Create cuEV handles
    // nbw/nk: b=64 inner bandwidth, k=512 outer panel
    SolverHandle<T> ws = handle_alloc<T>(n, 64, 512, stream);

    // double-blocking band reduction
    kernels::dbbr_reduce(&ws, A);

    // bc_tridi
    kernels::bc_tridi(&ws, A);

    // divide-and-conquer solve
    kernels::tridi_dc(&ws);

    // back-transform eigenvectors
    kernels::back_transform(&ws);

    // cleanup
    handle_free(&ws);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template void symm_eig_solve<float>(float *, int, float *, float *, cudaStream_t);
template void symm_eig_solve<double>(double *, int, double *, double *, cudaStream_t);
template void kernels::dbbr_reduce<float>(SolverHandle<float> *, float *);
template void kernels::dbbr_reduce<double>(SolverHandle<double> *, double *);
} // namespace cuev
