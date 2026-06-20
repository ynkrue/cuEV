/**
 * @file   qdwh_mp.cu
 * @brief  Distributed QDWH polar iteration — cuev::mp::kernels namespace.
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#ifdef CUEV_ENABLE_MP

#include "mp/kernels_mp.cuh"
#include <cmath>
#include <limits>
#include <type_traits>

// =============================================================================
// Device kernels
// =============================================================================
namespace {

// Local ↔ global index helpers (2D block-cyclic, rsrc=csrc=0).
__device__ inline int64_t l2g_row(int64_t li, int prow, int nprow, int64_t nb) {
    return (li / nb * nprow + prow) * nb + (li % nb);
}
__device__ inline int64_t l2g_col(int64_t lj, int pcol, int npcol, int64_t nb) {
    return (lj / nb * npcol + pcol) * nb + (lj % nb);
}

template <typename T> ncclDataType_t nccl_type() {
    if constexpr (std::is_same_v<T, float>)
        return ncclFloat32;
    else
        return ncclFloat64;
}

template <typename T, int BLOCKSIZE>
__global__ void sdc_trace_mp_kernel(const T *A, T *result, int64_t local_cols, int64_t lld,
                                    int prow, int pcol, int nprow, int npcol, int64_t nb) {
    __shared__ T smem[BLOCKSIZE];
    int tid = threadIdx.x;
    T acc = T(0);
    for (int64_t lj = tid; lj < local_cols; lj += BLOCKSIZE) {
        int64_t gj = l2g_col(lj, pcol, npcol, nb);
        int64_t rb = gj / nb;
        int owner_prow = (int)(rb % nprow);
        if (owner_prow == prow) {
            int64_t li = (rb / nprow) * nb + (gj % nb);
            acc += A[lj * lld + li];
        }
    }
    T total = block_reduce_sum<T, BLOCKSIZE>(acc, smem);
    if (tid == 0) *result = total;
}

// Multi-block reduction (unlike sdc_trace_mp_kernel's single block): the norm
// scans every local element, not just the diagonal, so there's real
// parallelism to exploit. Each block reduces its grid-stride share into
// shared memory, then block 0's lane atomically adds into the global sum.
template <typename T, int BLOCKSIZE>
__global__ void qdwh_norm2_mp_kernel(const T *A, T *result, int64_t local_rows, int64_t local_cols,
                                     int64_t lld) {
    __shared__ T smem[BLOCKSIZE];
    int tid = threadIdx.x;
    int64_t total = local_rows * local_cols;
    T acc = T(0);
    for (int64_t idx = (int64_t)blockIdx.x * BLOCKSIZE + tid; idx < total;
         idx += (int64_t)BLOCKSIZE * gridDim.x) {
        int64_t li = idx % local_rows;
        int64_t lj = idx / local_rows;
        T v = A[lj * lld + li];
        acc += v * v;
    }
    T block_sum = block_reduce_sum<T, BLOCKSIZE>(acc, smem);
    if (tid == 0) atomicAdd(result, block_sum);
}

template <typename T>
__global__ void qdwh_shift_mp_kernel(T *A, T mu, int64_t local_rows, int64_t local_cols,
                                     int64_t lld, int prow, int pcol, int nprow, int npcol,
                                     int64_t nb) {
    (void)local_rows;
    for (int64_t lj = blockIdx.x * blockDim.x + threadIdx.x; lj < local_cols;
         lj += (int64_t)blockDim.x * gridDim.x) {
        int64_t gj = l2g_col(lj, pcol, npcol, nb);
        int64_t rb = gj / nb;
        if ((int)(rb % nprow) == prow) {
            int64_t li = (rb / nprow) * nb + (gj % nb);
            A[lj * lld + li] -= mu;
        }
    }
}

// Purely local element-wise scale — every local element is touched once,
// no diagonal-ownership check needed (unlike qdwh_shift_mp_kernel).
template <typename T>
__global__ void qdwh_scal_mp_kernel(T *A, T alpha, int64_t local_rows, int64_t local_cols,
                                    int64_t lld) {
    int64_t total = local_rows * local_cols;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t li = idx % local_rows;
        int64_t lj = idx / local_rows;
        A[lj * lld + li] *= alpha;
    }
}

template <typename T>
__global__ void qdwh_fill_W_mp_kernel(T *W, const T *X, T scale, int64_t n, int64_t lld_W,
                                      int64_t lld_X, int64_t local_rows_W, int64_t lc, int prow,
                                      int pcol, int nprow, int npcol, int64_t nb) {
    int64_t total = local_rows_W * lc;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t li = idx % local_rows_W;
        int64_t lj = idx / local_rows_W;
        int64_t gi = l2g_row(li, prow, nprow, nb);
        if (gi < n) {
            W[lj * lld_W + li] = scale * X[lj * lld_X + li];
        } else {
            int64_t gj = l2g_col(lj, pcol, npcol, nb);
            W[lj * lld_W + li] = ((gi - n) == gj) ? T(1) : T(0);
        }
    }
}

// Deterministic per-(global-index) pseudo-random fill in [-1,1]. Used to build
// the Gaussian-like sketch Ω for the randomized range finder. Hashing the
// *global* (gi,gj) makes the sketch independent of the process grid and
// reproducible across ranks (no communication, no RNG state).
template <typename T>
__global__ void rand_fill_mp_kernel(T *A, int64_t local_rows, int64_t local_cols, int64_t lld,
                                    int prow, int pcol, int nprow, int npcol, int64_t nb,
                                    unsigned long long seed) {
    int64_t total = local_rows * local_cols;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t li = idx % local_rows;
        int64_t lj = idx / local_rows;
        int64_t gi = l2g_row(li, prow, nprow, nb);
        int64_t gj = l2g_col(lj, pcol, npcol, nb);
        unsigned long long h = (unsigned long long)(gi + 1) * 0x9E3779B97F4A7C15ULL ^
                               (unsigned long long)(gj + 1) * 0xC2B2AE3D27D4EB4FULL ^ seed;
        h ^= h >> 33;
        h *= 0xFF51AFD7ED558CCDULL;
        h ^= h >> 33;
        h *= 0xC4CEB9FE1A85EC53ULL;
        h ^= h >> 33;
        double r = (double)(h >> 11) * (1.0 / 9007199254740992.0); // [0,1)
        A[lj * lld + li] = (T)(2.0 * r - 1.0);
    }
}

template <typename T>
__global__ void qdwh_fill_C_mp_kernel(T *C, int64_t m, int64_t k, int64_t lld, int64_t local_rows,
                                      int64_t lc, int prow, int pcol, int nprow, int npcol,
                                      int64_t nb) {
    int64_t total = local_rows * lc;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)blockDim.x * gridDim.x) {
        int64_t li = idx % local_rows;
        int64_t lj = idx / local_rows;
        int64_t gi = l2g_row(li, prow, nprow, nb);
        int64_t gj = l2g_col(lj, pcol, npcol, nb);
        C[lj * lld + li] = (gi == gj && gi < k) ? T(1) : T(0);
    }
}

} // namespace

// =============================================================================
// QDWH coefficients
// =============================================================================
namespace {
template <typename T> static void qdwh_coeffs(T &l, T &a, T &b, T &c) {
    T d = std::cbrt(T(4) * (T(1) - l * l) / std::pow(l, 4));
    a = std::sqrt(T(1) + d) +
        T(0.5) * std::sqrt(T(8) - T(4) * d + T(8) * (T(2) - l * l) / (l * l * std::sqrt(T(1) + d)));
    b = T(0.25) * (a - T(1)) * (a - T(1));
    c = a + b - T(1);
    l = l * (a + b * l * l) / (T(1) + c * l * l);
}
} // namespace

// =============================================================================
// Host launchers
// =============================================================================
namespace cuev {
namespace mp {
namespace kernels {

template <typename T>
void qdwh_shift_mp(T *A_local, T mu, int64_t local_rows, int64_t local_cols, int64_t lld, int prow,
                   int pcol, int nprow, int npcol, int64_t nb, cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    int64_t grid = (local_cols + BLOCKSIZE - 1) / BLOCKSIZE;
    if (grid < 1) grid = 1;
    qdwh_shift_mp_kernel<T><<<(unsigned)grid, BLOCKSIZE, 0, stream>>>(
        A_local, mu, local_rows, local_cols, lld, prow, pcol, nprow, npcol, nb);
}

template <typename T>
void qdwh_scal_mp(T *A_local, T alpha, int64_t local_rows, int64_t local_cols, int64_t lld,
                  cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    int64_t total = local_rows * local_cols;
    int64_t grid = (total + BLOCKSIZE - 1) / BLOCKSIZE;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    qdwh_scal_mp_kernel<T>
        <<<(unsigned)grid, BLOCKSIZE, 0, stream>>>(A_local, alpha, local_rows, local_cols, lld);
}

template <typename T>
void qdwh_fill_W_mp(T *W_local, const T *X_local, T scale, int64_t n, int64_t lld_W, int64_t lld_X,
                    int64_t lc, int prow, int pcol, int nprow, int npcol, int64_t nb,
                    cudaStream_t stream) {
    int64_t local_rows_W = cublasMpNumroc(2 * n, nb, prow, 0, nprow);
    constexpr int BLOCKSIZE = 256;
    int64_t total = local_rows_W * lc;
    int64_t grid = (total + BLOCKSIZE - 1) / BLOCKSIZE;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    qdwh_fill_W_mp_kernel<T><<<(unsigned)grid, BLOCKSIZE, 0, stream>>>(
        W_local, X_local, scale, n, lld_W, lld_X, local_rows_W, lc, prow, pcol, nprow, npcol, nb);
}

template <typename T>
void rand_fill_mp(T *A_local, int64_t local_rows, int64_t local_cols, int64_t lld, int prow,
                  int pcol, int nprow, int npcol, int64_t nb, unsigned long long seed,
                  cudaStream_t stream) {
    constexpr int BLOCKSIZE = 256;
    int64_t total = local_rows * local_cols;
    int64_t grid = (total + BLOCKSIZE - 1) / BLOCKSIZE;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    rand_fill_mp_kernel<T><<<(unsigned)grid, BLOCKSIZE, 0, stream>>>(
        A_local, local_rows, local_cols, lld, prow, pcol, nprow, npcol, nb, seed);
}

template <typename T>
void qdwh_fill_C_mp(T *C_local, int64_t m, int64_t k, int64_t lld, int64_t lc, int prow, int pcol,
                    int nprow, int npcol, int64_t nb, cudaStream_t stream) {
    int64_t local_rows = cublasMpNumroc(m, nb, prow, 0, nprow);
    constexpr int BLOCKSIZE = 256;
    int64_t total = local_rows * lc;
    int64_t grid = (total + BLOCKSIZE - 1) / BLOCKSIZE;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;
    qdwh_fill_C_mp_kernel<T><<<(unsigned)grid, BLOCKSIZE, 0, stream>>>(
        C_local, m, k, lld, local_rows, lc, prow, pcol, nprow, npcol, nb);
}

template <typename T>
T sdc_trace_mp(Context &ctx, const T *A_local, int64_t local_rows, int64_t local_cols, int64_t lld,
               int prow, int pcol, int nprow, int npcol, int64_t nb) {
    (void)local_rows; // diagonal ownership is derived from the column's row-block owner
    constexpr int BLOCKSIZE = 256;
    T *trace;
    CUDA_CHECK(cudaMalloc(&trace, sizeof(T)));
    sdc_trace_mp_kernel<T, BLOCKSIZE><<<1, BLOCKSIZE, 0, ctx.stream>>>(
        A_local, trace, local_cols, lld, prow, pcol, nprow, npcol, nb);
    NCCL_CHECK(ncclAllReduce(trace, trace, 1, nccl_type<T>(), ncclSum, ctx.nccl, ctx.stream));
    T h_result;
    CUDA_CHECK(cudaMemcpyAsync(&h_result, trace, sizeof(T), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaFree(trace));
    return h_result;
}

template <typename T>
T qdwh_norm_mp(Context &ctx, const T *A_local, int64_t local_rows, int64_t local_cols,
               int64_t lld) {
    constexpr int BLOCKSIZE = 256;
    int64_t total = local_rows * local_cols;
    int64_t grid = (total + BLOCKSIZE - 1) / BLOCKSIZE;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535; // grid-stride loop covers any remainder

    T *d_sum;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(T)));
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(T), ctx.stream));
    qdwh_norm2_mp_kernel<T, BLOCKSIZE>
        <<<(unsigned)grid, BLOCKSIZE, 0, ctx.stream>>>(A_local, d_sum, local_rows, local_cols, lld);
    NCCL_CHECK(ncclAllReduce(d_sum, d_sum, 1, nccl_type<T>(), ncclSum, ctx.nccl, ctx.stream));
    T h_sum;
    CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaFree(d_sum));
    return std::sqrt(h_sum);
}

// One QDWH step via QR (always stable; used while the iterate is
// ill-conditioned). B ← coeff·Q1·Q2ᵀ + bc·B, where [Q1;Q2] = QR([√c·B; I]).
template <typename T>
static void qdwh_step_qr_mp(Context &ctx, DistMatrix<T> &B, T a, T b, T c, WorkspaceMp<T> &ws) {
    T scale = std::sqrt(c);
    DistMatrix<T> W = dist_describe<T>(ctx, 2 * B.n, B.n, ws.qdwh_W);
    qdwh_fill_W_mp(W.data, B.data, scale, B.n, W.lld, B.lld, B.local_cols, ctx.prow, ctx.pcol,
                   ctx.nprow, ctx.npcol, ctx.nb, ctx.stream);
    cusolvermp::geqrf(ctx, 2 * B.n, B.n, W.data, 1, 1, W.solverDesc, ws.qdwh_tau, ws);

    // cuSOLVERMp has no orgqr — materialise Q by applying it to a truncated identity.
    T *C_data = ws.push(dist_local_count<T>(ctx, 2 * B.n, B.n));
    DistMatrix<T> C = dist_describe<T>(ctx, 2 * B.n, B.n, C_data);
    qdwh_fill_C_mp(C.data, 2 * B.n, B.n, C.lld, B.local_cols, ctx.prow, ctx.pcol, ctx.nprow,
                   ctx.npcol, ctx.nb, ctx.stream);
    cusolvermp::ormqr(ctx, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, 2 * B.n, B.n, B.n, W.data, 1, 1,
                      W.solverDesc, ws.qdwh_tau, C.data, 1, 1, C.solverDesc, ws);

    // B ← coeff·Q1·Q2ᵀ + bc·B, where Q1 = C[1:n,:], Q2 = C[n+1:2n,:].
    // Q1 sits at offset (1,1) — block-aligned, usable directly. Q2 starts at row
    // n+1, which is not a multiple of nb whenever n isn't (true at most recursion
    // levels), and PXGEMM rejects non-block-aligned submatrix offsets. Extract Q2
    // into its own block-aligned n×n buffer with gemr2d (PXGEMR2D tolerates the
    // offset), then GEMM at (1,1).
    T *Q2_data = ws.push(dist_local_count<T>(ctx, B.n, B.n));
    DistMatrix<T> Q2 = dist_describe<T>(ctx, B.n, B.n, Q2_data);
    cublasmp::gemr2d(ctx, B.n, B.n, C.data, B.n + 1, 1, C.desc, Q2.data, 1, 1, Q2.desc);

    T bc = b / c;
    T coeff = (a - bc) / scale;
    cublasmp::gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_T, B.n, B.n, B.n, &coeff, C.data, 1, 1, C.desc,
                   Q2.data, 1, 1, Q2.desc, &bc, B.data, 1, 1, B.desc);

    dist_free(Q2);
    dist_free(C);
    dist_free(W);
}

// One QDWH step via Cholesky (~half the flops; valid once c ≤ CHOL_SWITCH so
// κ(Z) = 1 + c·σ_max² ≤ 1 + c). B ← coeff·B·Z⁻¹ + bc·B, where Z = I + c·BᵀB.
template <typename T>
static void qdwh_step_chol_mp(Context &ctx, DistMatrix<T> &B, T a, T b, T c, WorkspaceMp<T> &ws) {
    T zero = T(0);
    T one = T(1);

    T *Z_data = ws.push(dist_local_count<T>(ctx, B.n, B.n));
    DistMatrix<T> Z = dist_describe<T>(ctx, B.n, B.n, Z_data);
    cublasmp::syrk(ctx, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, B.n, B.n, &c, B.data, 1, 1, B.desc,
                   &zero, Z.data, 1, 1, Z.desc);
    qdwh_shift_mp(Z.data, T(-1), Z.local_rows, Z.local_cols, Z.lld, ctx.prow, ctx.pcol, ctx.nprow,
                  ctx.npcol, ctx.nb, ctx.stream);
    cusolvermp::potrf(ctx, CUBLAS_FILL_MODE_UPPER, B.n, Z.data, 1, 1, Z.solverDesc, ws);

    T *tmp_data = ws.push(dist_local_count<T>(ctx, B.n, B.n));
    DistMatrix<T> tmp = dist_describe<T>(ctx, B.n, B.n, tmp_data);
    CUDA_CHECK(cudaMemcpyAsync(tmp.data, B.data, (size_t)B.local_rows * B.local_cols * sizeof(T),
                               cudaMemcpyDeviceToDevice, ctx.stream));
    cublasmp::trsm(ctx, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                   CUBLAS_DIAG_NON_UNIT, B.n, B.n, &one, Z.data, 1, 1, Z.desc, tmp.data, 1, 1,
                   tmp.desc);
    cublasmp::trsm(ctx, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                   CUBLAS_DIAG_NON_UNIT, B.n, B.n, &one, Z.data, 1, 1, Z.desc, tmp.data, 1, 1,
                   tmp.desc);

    T bc = b / c;
    T coeff = a - bc;
    cublasmp::geadd(ctx, CUBLAS_OP_N, B.n, B.n, &coeff, tmp.data, 1, 1, tmp.desc, &bc, B.data, 1, 1,
                    B.desc);

    dist_free(tmp);
    dist_free(Z);
}

template <typename T> void qdwh_sign_mp(Context &ctx, DistMatrix<T> &B, WorkspaceMp<T> &ws) {
    // Normalise: B ← B / ‖B‖_F such that σ_max(B) ≤ 1. This guarantees
    // κ(I + c·BᵀB) ≤ 1 + c so the c ≤ CHOL_SWITCH test below is a safe gate.
    T zero = T(0);
    T one = T(1);
    T norm = qdwh_norm_mp(ctx, B.data, B.local_rows, B.local_cols, B.lld);
    T scale_b = one / norm;
    qdwh_scal_mp(B.data, scale_b, B.local_rows, B.local_cols, B.lld, ctx.stream);

    // c_k decreases monotonically, so use the cheaper Cholesky update once it's safe.
    T l = std::numeric_limits<T>::epsilon();
    constexpr T CHOL_SWITCH = T(100);
    constexpr int MAX_ITER = 6;
    size_t mark = ws.mark();
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        T a, b, c;
        qdwh_coeffs(l, a, b, c);

        if (c > CHOL_SWITCH)
            // QR for update: X ← (b/c)·X + (a − b/c)/√c · Q₁·Q₂ᵀ
            qdwh_step_qr_mp(ctx, B, a, b, c, ws);
        else
            // Cholesky for update: X ← (b/c)·X + (a − b/c)·X·Z⁻¹
            qdwh_step_chol_mp(ctx, B, a, b, c, ws);

        // symmetrize
        T *sym_data = ws.push(dist_local_count<T>(ctx, B.n, B.n));
        DistMatrix<T> sym = dist_describe<T>(ctx, B.n, B.n, sym_data);

        T half = T(0.5);
        cublasmp::geadd(ctx, CUBLAS_OP_T, B.n, B.n, &half, B.data, 1, 1, B.desc, &zero, sym.data, 1,
                        1, sym.desc);
        cublasmp::geadd(ctx, CUBLAS_OP_N, B.n, B.n, &half, B.data, 1, 1, B.desc, &one, sym.data, 1,
                        1, sym.desc);
        CUDA_CHECK(cudaMemcpyAsync(B.data, sym.data,
                                   (size_t)B.local_rows * B.local_cols * sizeof(T),
                                   cudaMemcpyDeviceToDevice, ctx.stream));
        dist_free(sym);
        ws.reset(mark);

        if (l >= one - T(1e-14)) break;
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T)                                                                             \
    template void qdwh_shift_mp<T>(T *, T, int64_t, int64_t, int64_t, int, int, int, int, int64_t, \
                                   cudaStream_t);                                                  \
    template void qdwh_scal_mp<T>(T *, T, int64_t, int64_t, int64_t, cudaStream_t);                \
    template void qdwh_fill_W_mp<T>(T *, const T *, T, int64_t, int64_t, int64_t, int64_t, int,    \
                                    int, int, int, int64_t, cudaStream_t);                         \
    template void qdwh_fill_C_mp<T>(T *, int64_t, int64_t, int64_t, int64_t, int, int, int, int,   \
                                    int64_t, cudaStream_t);                                        \
    template void rand_fill_mp<T>(T *, int64_t, int64_t, int64_t, int, int, int, int, int64_t,     \
                                  unsigned long long, cudaStream_t);                               \
    template T sdc_trace_mp<T>(Context &, const T *, int64_t, int64_t, int64_t, int, int, int,     \
                               int, int64_t);                                                      \
    template T qdwh_norm_mp<T>(Context &, const T *, int64_t, int64_t, int64_t);                   \
    template void qdwh_sign_mp<T>(Context &, DistMatrix<T> &, WorkspaceMp<T> &);

INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
