/**
 * @file   backtransform.cu
 * @brief  Eigenvector back-transform — evec ← Q_s·Q_b·Q_d.
 *
 * Two-stage tridiagonalization gives A = Q_s (Q_b T Q_bᵀ) Q_sᵀ, so the eigenvectors of A are
 * evec = Q_s·Q_b·Q_d, with Q_d the tridiagonal eigenvectors from the D&C solve. The transform
 * runs in the padded buffer ws->M (ldu rows × n): load Q_d, apply Q_b then Q_s in place, copy
 * back. Applying the reflectors to Q_d directly.
 *
 *   BC-Back  (Q_b): bulge-chasing reflectors in U (column = sweep, padded ldu, unit w).
 *   SBR-Back (Q_s): DBBR WY block reflectors (I − W·Yᵀ) per panel, from W and Y.
 *
 * The fast BC-Back kernel is adapted from Wang et al. SC'25 artifact
 * (BC_kernel_computerQ_1Col_V8_10_noBandU): one warp holds a 256-row window of an M column in
 * registers and *slides* it (register shift) past the staircased reflectors staged in shared.
 * Requires b=32 and the padded U/M layout.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <algorithm>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

constexpr int BT_PER_THREAD = 8;           // M elements per thread (window = 32·8 = 256 rows)
constexpr int BT_WIN = BT_PER_THREAD * 32; // 256 — register-resident window height
constexpr int BT_WARPS = 8;                // warps per block
constexpr int BT_RED = 32 / BT_PER_THREAD; // 4 — a b=32 reflector spans 4 threads
constexpr int BT_UTILE = 32;               // reflectors staged in shared per pass

/// Partial reduction over the BT_RED threads that span one reflector.
template <typename T> __device__ __forceinline__ T bt_reduce(T v) {
#pragma unroll
    for (int mask = BT_RED / 2; mask > 0; mask /= 2)
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    return v;
}

/**
 * @brief BC-Back: M ← Q_b · M, one warp per column with a register-resident sliding window.
 *
 * The reduction's reflectors compose to Q_bᵀ in forward (top→bottom) order; since each Householder
 * is symmetric, Q_b is the same reflectors applied in *reverse* order. So this kernel walks sweeps
 * high→low and, within each pass, applies the BT_UTILE staircased reflectors from h=BT_UTILE−1 down
 * to 0 while sliding the window *up*: each step the bottom row leaves (→ sDone, written back) and a
 * fresh row enters at the top from sHead (register shuffle, no shared round-trip). Columns are
 * partitioned across blocks (first `extra` own one more); per pass the span
 * [baseRow, baseRow+BT_WIN+BT_UTILE) of each column is read and written.
 *
 * @param[in]     n        matrix dimension
 * @param[in]     cols     columns owned by this block (before the +1 for large blocks)
 * @param[in]     extra    number of leading blocks that own one extra column
 * @param[in]     nsweeps  number of window sweeps (div_up(n-2, BT_WIN))
 * @param[in]     lastU    reflector columns reaching the deepest hop-band
 * @param[in]     U        BC reflectors, ldu×n column-major
 * @param[in]     ldu      leading dim of U
 * @param[in,out] M        padded working buffer, ldm×n column-major
 * @param[in]     ldm      leading dim of M
 */
template <typename T>
__global__ void bc_back_kernel(int n, int cols, int extra, int nsweeps, int lastU, const T *U,
                               long ldu, T *M, long ldm) {
    extern __shared__ __align__(16) unsigned char smem[];
    T *sU = reinterpret_cast<T *>(smem); // [BT_UTILE * BT_WIN] reflector tile

    __shared__ T sHead[BT_WARPS * BT_UTILE]; // rows entering the window top this pass
    __shared__ T sDone[BT_WARPS * BT_UTILE]; // rows that left the window bottom (write back)
    T rM[BT_PER_THREAD];                     // this thread's contiguous slice of the window

    const int blk = blockIdx.x;
    if (blk < extra) {
        cols += 1;
        M += (long)blk * cols * ldm;
    } else {
        M += ((long)blk * cols + extra) * ldm;
    }
    const int lane = threadIdx.x, warp = threadIdx.y;

    // Reverse of the forward order: sweeps high→low, passes within a sweep last→first.
    for (int sw = nsweeps - 1; sw >= 0; sw--) {
        const long baseRow0 = (long)(nsweeps - 1 - sw) * BT_WIN;
        const int remU0 = lastU + sw * BT_WIN;
        const int npass = (remU0 + BT_UTILE - 1) / BT_UTILE;
        for (int p = npass - 1; p >= 0; p--) {
            const long baseRow = baseRow0 + (long)p * BT_UTILE;
            const long uOff = (long)p * BT_UTILE;
            const int remU = remU0 - p * BT_UTILE; // valid reflectors in this tile (h ≥ remU ⇒ 0)
            __syncthreads();
            // stage up to BT_UTILE reflectors of this pass into shared (zero-filled beyond remU)
            for (int k = warp; k < BT_UTILE; k += BT_WARPS)
                for (int t = 0; t < BT_PER_THREAD; t++) {
                    sU[k * BT_WIN + lane + t * 32] = T(0);
                    if (k < remU)
                        sU[k * BT_WIN + lane + t * 32] =
                            U[(uOff + k) * ldu + baseRow + 1 + k + lane * BT_PER_THREAD + t];
                }
            __syncthreads();

            for (int col = warp; col < cols; col += BT_WARPS) {
                T *Mc = M + (long)col * ldm; // this column
                // window = bottom BT_WIN rows of the span: [baseRow+BT_UTILE, +BT_WIN)
#pragma unroll
                for (int t = 0; t < BT_PER_THREAD; t++)
                    rM[t] = Mc[baseRow + BT_UTILE + lane * BT_PER_THREAD + t];
                // sHead = the BT_UTILE rows above the window: [baseRow, baseRow+BT_UTILE)
#pragma unroll
                for (int t = lane; t < BT_UTILE; t += 32)
                    sHead[warp * BT_UTILE + t] = Mc[baseRow + t];
                __syncwarp();

                for (int h = BT_UTILE - 1; h >= 0; h--) {
                    // apply reflector h:  m ← m − 2·(wᵀm)·w  (unit-w convention, H = I − 2wwᵀ)
                    T proj = T(0);
#pragma unroll
                    for (int t = 0; t < BT_PER_THREAD; t++)
                        proj += sU[h * BT_WIN + lane + t * 32] * rM[t];
                    proj = bt_reduce(proj);
#pragma unroll
                    for (int t = 0; t < BT_PER_THREAD; t++)
                        rM[t] -= T(2) * proj * sU[h * BT_WIN + lane + t * 32];

                    // slide window up one row: bottom row leaves, top enters from neighbor/sHead.
                    // register shuffle (no shared round-trip): lane gets lane−1's old bottom row.
                    const T bottom = rM[BT_PER_THREAD - 1];
                    if (lane == 31) sDone[warp * BT_UTILE + h] = bottom; // bottom row leaves
#pragma unroll
                    for (int t = BT_PER_THREAD - 1; t > 0; t--)
                        rM[t] = rM[t - 1];
                    const T fromAbove = __shfl_up_sync(0xffffffffu, bottom, 1);
                    rM[0] = (lane != 0) ? fromAbove : sHead[warp * BT_UTILE + h];
                }
                __syncwarp(); // sDone (written by lane 31) visible before the read-out below

                // write back the slid-up window [baseRow, +BT_WIN) and the BT_UTILE rows that left
#pragma unroll
                for (int t = 0; t < BT_PER_THREAD; t++)
                    Mc[baseRow + lane * BT_PER_THREAD + t] = rM[t];
#pragma unroll
                for (int t = lane; t < BT_UTILE; t += 32)
                    Mc[baseRow + BT_WIN + t] = sDone[warp * BT_UTILE + t];
            }
        }
    }
}

} // namespace

namespace cuev {
namespace kernels {

/// BC-Back: M ← Q_b · M, in place on the padded buffer M (ldu×n, padding rows zeroed).
template <typename T> void bc_back(SolverHandle<T> *ws, const T *U, T *M) {
    const int n = ws->n;
    const long ldu = ws->ldu, ldm = ws->ldu;

    const int nsweeps = div_up(n - 2, BT_WIN);
    // reflector columns reaching the deepest hop-band: s ≤ n-2-(nsweeps-1)·BT_WIN, i.e. count
    // below.
    const int lastU = n - 1 - (nsweeps - 1) * BT_WIN;

    const size_t shmem = (size_t)BT_UTILE * BT_WIN * sizeof(T);
    CUDA_CHECK(cudaFuncSetAttribute(bc_back_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)shmem));

    int blocksPerSM = 0, numSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, bc_back_kernel<T>,
                                                             32 * BT_WARPS, shmem));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
    const int grid = std::max(1, std::min(blocksPerSM * numSM, n));
    const int cols = n / grid;
    const int extra = n % grid;

    dim3 block(32, BT_WARPS);
    bc_back_kernel<<<grid, block, shmem, ws->stream>>>(n, cols, extra, nsweeps, lastU, U, ldu, M,
                                                       ldm);
}

/// SBR-Back: M ← Q_s · M, in place. M is n×n (ld=ldm); WY panels applied in reverse order.
template <typename T> void sbr_back(SolverHandle<T> *ws, const T *Y, const T *W, T *M) {
    const int n = ws->n, b = ws->nbw;
    const int lda = ws->n, ldm = ws->ldu;
    const T one = T(1), zero = T(0), neg1 = T(-1);

    int jmax = 0;
    for (int j = 0; j + b < n; j += b)
        jmax = j;
    for (int j = jmax; j >= 0; j -= b) {
        const int rows = n - (j + b);
        const T *Yp = Y + (size_t)j * lda + (j + b);
        const T *Wp = W + (size_t)j * lda + (j + b);
        T *Mb = M + (j + b); // bottom row-block of M, all n columns

        // Mb ← (I − W·Yᵀ)·Mb = Mb − W·(Yᵀ·Mb)
        cublas::gemm(ws, CUBLAS_OP_T, CUBLAS_OP_N, b, n, rows, &one, Yp, lda, Mb, ldm, &zero, ws->Z,
                     b);
        cublas::gemm(ws, CUBLAS_OP_N, CUBLAS_OP_N, rows, n, b, &neg1, Wp, lda, ws->Z, b, &one, Mb,
                     ldm);
    }
}

template <typename T>
void back_transform(SolverHandle<T> *ws, const T *Y, const T *W, const T *U, T *evec,
                    SolveTimer *timer) {
    const int n = ws->n;
    const int lda = ws->n, ldm = ws->ldu;
    T *M = ws->M;

    // Optional per-phase events (copyin / Qb / Qs / copyout)
    cudaEvent_t e0{}, e_in{}, e_qb{}, e_qs{}, e_out{};
    if (timer) {
        for (cudaEvent_t *e : {&e0, &e_in, &e_qb, &e_qs, &e_out})
            CUDA_CHECK(cudaEventCreate(e));
        CUDA_CHECK(cudaEventRecord(e0, ws->stream));
    }

    // 1. M ← Q_d, padding rows below n zeroed for the sliding-window kernel.
    CUDA_CHECK(cudaMemsetAsync(M, 0, (size_t)ldm * n * sizeof(T), ws->stream));
    CUDA_CHECK(cudaMemcpy2DAsync(M, ldm * sizeof(T), evec, lda * sizeof(T), n * sizeof(T), n,
                                 cudaMemcpyDeviceToDevice, ws->stream));
    if (timer) CUDA_CHECK(cudaEventRecord(e_in, ws->stream));

    // 2. BC-Back: M ← Q_b · M
    bc_back(ws, U, M);
    if (timer) CUDA_CHECK(cudaEventRecord(e_qb, ws->stream));

    // 3. SBR-Back: M ← Q_s · M
    sbr_back(ws, Y, W, M);
    if (timer) CUDA_CHECK(cudaEventRecord(e_qs, ws->stream));

    // 4. evec ← M[:n,:]
    CUDA_CHECK(cudaMemcpy2DAsync(evec, lda * sizeof(T), M, ldm * sizeof(T), n * sizeof(T), n,
                                 cudaMemcpyDeviceToDevice, ws->stream));

    if (timer) {
        CUDA_CHECK(cudaEventRecord(e_out, ws->stream));
        CUDA_CHECK(cudaEventSynchronize(e_out));
        CUDA_CHECK(cudaEventElapsedTime(&timer->bt_copyin_ms, e0, e_in));
        CUDA_CHECK(cudaEventElapsedTime(&timer->bt_qb_ms, e_in, e_qb));
        CUDA_CHECK(cudaEventElapsedTime(&timer->bt_qs_ms, e_qb, e_qs));
        CUDA_CHECK(cudaEventElapsedTime(&timer->bt_copyout_ms, e_qs, e_out));
        for (cudaEvent_t e : {e0, e_in, e_qb, e_qs, e_out})
            CUDA_CHECK(cudaEventDestroy(e));
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void bc_back<T>(SolverHandle<T> *, const T *, T *);                                   \
    template void sbr_back<T>(SolverHandle<T> *, const T *, const T *, T *);                       \
    template void back_transform<T>(SolverHandle<T> *, const T *, const T *, const T *, T *,       \
                                    SolveTimer *);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
