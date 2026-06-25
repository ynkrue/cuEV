/**
 * @file   dbbr.cu
 * @brief  Double-blocking band reduction kernels — panel QR, larft, custom syr2k.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <type_traits>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

/**
 * @brief Masked copy of the packed QR panel A into the reflector buffer Y.
 *
 * Column-major, leading dimension lda for both A and Y.
 *   r <  c  →  Y[r,c] = 0          (above diagonal — R's upper part)
 *   r == c  →  Y[r,c] = 1          (unit diagonal, implicit in geqrf output)
 *   r >  c  →  Y[r,c] = A[r,c]     (essential Householder entries)
 *
 * @param[in]  A     packed panel (column-major)
 * @param[out] Y     explicit unit lower-trapezoidal reflectors (column-major)
 * @param[in]  rows  rows in the panel
 * @param[in]  cols  columns in the panel (bandwidth b)
 * @param[in]  lda   leading dimension of A and Y (= ws->n)
 */
template <typename T>
__launch_bounds__(256) __global__
    void dbbr_extract_reflectors(const T *A, T *Y, int rows, int cols, int lda) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;

    T v;
    if (r < c)
        v = T(0);
    else if (r == c)
        v = T(1);
    else
        v = A[c * lda + r];

    Y[c * lda + r] = v;
}

/**
 * @brief Form the b×b block reflector factor T.
 *
 * Given the Gram matrix G = Vᵀ·V (upper triangle) and the Householder scalars τ,
 * builds upper-triangular T with H₀···H_{b-1} = I − V·T·Vᵀ via the recurrence
 *   T[i,i]    =  τᵢ
 *   T[0:i, i] = −τᵢ · (T[0:i, 0:i] · G[0:i, i]).
 *
 * @param[in]  G    Vᵀ·V, b×b column-major, ldg = b (upper triangle referenced)
 * @param[in]  tau  Householder scalars, length b
 * @param[out] Tmat b×b block factor, column-major, ldt = b
 * @param[in]  b    panel width (≤ 64)
 */
template <typename T> __global__ void dbbr_larft(const T *G, const T *tau, T *Tmat, int b) {
    __shared__ T sT[64 * 64];
    const int tid = threadIdx.x;

    for (int c = 0; c < b; ++c)
        sT[tid + c * b] = T(0);
    __syncthreads();

    // Columns of T are built left to right; column i needs columns 0..i-1.
    for (int i = 0; i < b; ++i) {
        if (tid < i) {
            T acc = T(0);
            for (int q = tid; q < i; ++q)
                acc += sT[tid + q * b] * G[q + i * b]; // T[p,q]·G[q,i]
            sT[tid + i * b] = -tau[i] * acc;
        } else if (tid == i) {
            sT[tid + i * b] = tau[i];
        }
        __syncthreads();
    }

    for (int c = 0; c < b; ++c)
        Tmat[tid + c * b] = sT[tid + c * b];
}

} // namespace

namespace cuev {
namespace kernels {

template <typename T> void dbbr_panel_qr(SolverHandle<T> *ws, T *A, T *Y, int rows, int b) {
    const int lda = ws->n;

    // QR factorization: R in upper triangle of A, Householder vectors
    // below the diagonal, scalars into ws->tau.
    cusolver::geqrf(ws, rows, b, A, lda, ws->tau, ws->stream);

    // extract lower-trapezoidal reflectors  A → Y
    constexpr int BX = 64, BY = 4;
    dim3 block(BX, BY);
    dim3 grid(div_up(rows, BX), div_up(b, BY));
    dbbr_extract_reflectors<<<grid, block, 0, ws->stream>>>(A, Y, rows, b, lda);

    // larft: build b×b block factor T from Y and ws->tau into ws->Tmat
    //   Gram matrix G = Yᵀ·Y with syrk → ws->Dwk
    const T one = T(1), zero = T(0);
    cublas::syrk(ws, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, b, rows, &one, Y, lda, &zero, ws->Dwk, b);
    //   triangular recurrence → ws->Tmat
    dbbr_larft<<<1, b, 0, ws->stream>>>(ws->Dwk, ws->tau, ws->Tmat, b);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T) template void dbbr_panel_qr<T>(SolverHandle<T> *, T *, T *, int, int);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
