/**
 * @file   solver_mp.cu
 * @brief  Distributed spectral divide-and-conquer eigensolver.
 *
 * Public entry point: cuev::mp::symm_eig_solve_mp<T>(ctx, H, n, d_eval, evec, ws).
 *
 * @author Yannik Rüfenacht
 * @date   2026-06
 */

#ifdef CUEV_ENABLE_MP

#include "mp/comm.h"
#include "mp/kernels_mp.cuh"
#include "mp/workspace_mp.h"
#include <algorithm>
#include <cmath>

namespace cuev {
namespace mp {

namespace {

template <typename T>
void spectral_dc_mp(Context &ctx, DistMatrix<T> &H, int64_t n, T *eval, DistMatrix<T> &evec,
                    WorkspaceMp<T> &ws) {
    // --- Base case: cuSOLVERMp syevd ---
    if (n <= SDC_BASE_N_MP) {
        cusolvermp::syevd(ctx, n, H.data, 1, 1, H.solverDesc, eval, evec.data, 1, 1,
                          evec.solverDesc, ws);
        return;
    }

    size_t lvl = ws.mark();

    // --- Split point: μ ≈ mean eigenvalue ---
    T mu = kernels::sdc_trace_mp(ctx, H.data, H.local_rows, H.local_cols, H.lld, ctx.prow, ctx.pcol,
                                 ctx.nprow, ctx.npcol, ctx.nb) /
           T(n);

    // --- B ← copy of H, then B ← sign(H − μI) via QDWH ---
    T *B_data = ws.push((size_t)H.local_rows * H.local_cols);
    DistMatrix<T> B = dist_describe<T>(ctx, n, n, B_data);
    CUDA_CHECK(cudaMemcpyAsync(B.data, H.data, (size_t)H.local_rows * H.local_cols * sizeof(T),
                               cudaMemcpyDeviceToDevice, ctx.stream));
    kernels::qdwh_shift_mp(B.data, mu, B.local_rows, B.local_cols, B.lld, ctx.prow, ctx.pcol,
                           ctx.nprow, ctx.npcol, ctx.nb, ctx.stream);
    kernels::qdwh_sign_mp(ctx, B, ws);

    // --- P = (I + sign(B)) / 2 — spectral projector onto eigenvalues > μ ---
    T half = T(0.5);
    kernels::qdwh_scal_mp(B.data, half, B.local_rows, B.local_cols, B.lld, ctx.stream);
    kernels::qdwh_shift_mp(B.data, -half, B.local_rows, B.local_cols, B.lld, ctx.prow, ctx.pcol,
                           ctx.nprow, ctx.npcol, ctx.nb, ctx.stream);
    DistMatrix<T> &P = B;

    // --- Split size k = rank(P) ---
    // P as orthogonal projector implies trace(P) = rank(P) = k.
    T trace_p = kernels::sdc_trace_mp(ctx, P.data, P.local_rows, P.local_cols, P.lld, ctx.prow,
                                      ctx.pcol, ctx.nprow, ctx.npcol, ctx.nb);
    int k = (int)std::lround(trace_p);
    int64_t m = n - k;

    // --- Orthonormal bases Q1 (n×k, range P) and Q2 (n×m, null P) via a
    // randomized range finder.
    //
    // The obvious method — QR of P's own columns, taking the first k — fails
    // when those leading columns are rank-deficient: column j of P is P·eⱼ, and
    // if axis j lies (near) the complementary subspace its column is ~0. That
    // happens whenever range(P) is not aligned with the leading coordinates
    // (e.g. a near-diagonal H), and unpivoted geqrf cannot skip the dead
    // columns. cuSOLVERMp has no column-pivoted QR, so we sketch instead:
    //
    //   Q1 = orth(P · Ω1),       Ω1 random n×k   → spans range(P)
    //   Q2 = orth((I − P) · Ω2), Ω2 random n×m   → spans null(P) = range(I−P)
    //
    // Random Ω makes every sketched column a generic full-rank mix, so plain
    // geqrf suffices; range(P) ⟂ null(P) gives Q1 ⟂ Q2 automatically. ---
    T one = T(1), zero = T(0), mone = T(-1);

    T *Q1_data = ws.push(dist_local_count<T>(ctx, n, k));
    DistMatrix<T> Q1 = dist_describe<T>(ctx, n, k, Q1_data);
    T *Q2_data = ws.push(dist_local_count<T>(ctx, n, m));
    DistMatrix<T> Q2 = dist_describe<T>(ctx, n, m, Q2_data);

    { // Q1 ← orth(P · Ω1)
        size_t qm = ws.mark();
        T *om = ws.push(dist_local_count<T>(ctx, n, k));
        DistMatrix<T> Om = dist_describe<T>(ctx, n, k, om);
        T *y = ws.push(dist_local_count<T>(ctx, n, k));
        DistMatrix<T> Y = dist_describe<T>(ctx, n, k, y);
        kernels::rand_fill_mp(Om.data, Om.local_rows, Om.local_cols, Om.lld, ctx.prow, ctx.pcol,
                              ctx.nprow, ctx.npcol, ctx.nb, 0x51D3C0DEULL, ctx.stream);
        cublasmp::gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, n, k, n, &one, P.data, 1, 1, P.desc, Om.data,
                       1, 1, Om.desc, &zero, Y.data, 1, 1, Y.desc); // Y = P·Ω1
        cusolvermp::geqrf(ctx, n, k, Y.data, 1, 1, Y.solverDesc, ws.qdwh_tau, ws);
        kernels::qdwh_fill_C_mp(Q1.data, n, k, Q1.lld, Q1.local_cols, ctx.prow, ctx.pcol, ctx.nprow,
                                ctx.npcol, ctx.nb, ctx.stream); // [I_k; 0]
        cusolvermp::ormqr(ctx, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, n, k, k, Y.data, 1, 1, Y.solverDesc,
                          ws.qdwh_tau, Q1.data, 1, 1, Q1.solverDesc, ws);
        ws.reset(qm);
    }
    { // Q2 ← orth((I − P) · Ω2)
        size_t qm = ws.mark();
        T *om = ws.push(dist_local_count<T>(ctx, n, m));
        DistMatrix<T> Om = dist_describe<T>(ctx, n, m, om);
        T *y = ws.push(dist_local_count<T>(ctx, n, m));
        DistMatrix<T> Y = dist_describe<T>(ctx, n, m, y);
        kernels::rand_fill_mp(Om.data, Om.local_rows, Om.local_cols, Om.lld, ctx.prow, ctx.pcol,
                              ctx.nprow, ctx.npcol, ctx.nb, 0xC0FFEE11ULL, ctx.stream);
        cublasmp::gemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &one, P.data, 1, 1, P.desc, Om.data,
                       1, 1, Om.desc, &zero, Y.data, 1, 1, Y.desc); // Y = P·Ω2
        cublasmp::geadd(ctx, CUBLAS_OP_N, n, m, &mone, Y.data, 1, 1, Y.desc, &one, Om.data, 1, 1,
                        Om.desc); // Om ← Ω2 − P·Ω2 = (I−P)·Ω2
        cusolvermp::geqrf(ctx, n, m, Om.data, 1, 1, Om.solverDesc, ws.qdwh_tau, ws);
        kernels::qdwh_fill_C_mp(Q2.data, n, m, Q2.lld, Q2.local_cols, ctx.prow, ctx.pcol, ctx.nprow,
                                ctx.npcol, ctx.nb, ctx.stream); // [I_m; 0]
        cusolvermp::ormqr(ctx, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, n, m, m, Om.data, 1, 1, Om.solverDesc,
                          ws.qdwh_tau, Q2.data, 1, 1, Q2.solverDesc, ws);
        ws.reset(qm);
    }

    // --- Form subproblems ---
    T *H1_data = ws.push(dist_local_count<T>(ctx, k, k));
    DistMatrix<T> H1 = dist_describe<T>(ctx, k, k, H1_data);
    T *H2_data = ws.push(dist_local_count<T>(ctx, m, m));
    DistMatrix<T> H2 = dist_describe<T>(ctx, m, m, H2_data);
    kernels::sdc_split_mp(ctx, H, Q1, Q2, H1, H2, n, k, ws);

    // --- Recurse ---
    T *eval1 = ws.push((size_t)k);
    T *evec1_data = ws.push(dist_local_count<T>(ctx, k, k));
    DistMatrix<T> evec1 = dist_describe<T>(ctx, k, k, evec1_data);
    spectral_dc_mp(ctx, H1, k, eval1, evec1, ws);

    T *eval2 = ws.push((size_t)m);
    T *evec2_data = ws.push(dist_local_count<T>(ctx, m, m));
    DistMatrix<T> evec2 = dist_describe<T>(ctx, m, m, evec2_data);
    spectral_dc_mp(ctx, H2, m, eval2, evec2, ws);

    // --- Merge eigenvalues: [eval2 | eval1] ascending (eval arrays are
    // plain replicated device vectors, not block-cyclic — a flat memcpy) ---
    CUDA_CHECK(
        cudaMemcpyAsync(eval, eval2, (size_t)m * sizeof(T), cudaMemcpyDeviceToDevice, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(eval + m, eval1, (size_t)k * sizeof(T), cudaMemcpyDeviceToDevice,
                               ctx.stream));

    // --- Back-transform eigenvectors: evec = [Q2·evec2 | Q1·evec1] ---
    kernels::sdc_combine_mp(ctx, Q1, Q2, evec1, evec2, evec, n, k, ws);

    dist_free(evec2);
    dist_free(evec1);
    dist_free(H2);
    dist_free(H1);
    dist_free(Q2);
    dist_free(Q1);
    dist_free(B);
    ws.reset(lvl);
}

} // namespace

template <typename T>
void symm_eig_solve_mp(Context &ctx, DistMatrix<T> &H, int64_t n, T *eval, DistMatrix<T> &evec,
                       WorkspaceMp<T> &ws) {
    spectral_dc_mp(ctx, H, n, eval, evec, ws);
}

// =============================================================================
// Explicit instantiations
// =============================================================================
template void symm_eig_solve_mp<float>(Context &, DistMatrix<float> &, int64_t, float *,
                                       DistMatrix<float> &, WorkspaceMp<float> &);
template void symm_eig_solve_mp<double>(Context &, DistMatrix<double> &, int64_t, double *,
                                        DistMatrix<double> &, WorkspaceMp<double> &);

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
