/**
 * @file   tridi.cu
 * @brief  Tridiagonal divide-and-conquer eigensolver — single GPU (hybrid).
 *
 * Phase-A port of MAGMA's hybrid D&C (dstedx → dlaex0 → dlaex1 → dlaex3):
 * the algorithm runs on the host (LAPACK/MKL for the leaf solve `*steqr`,
 * deflation `*laed2`, secular roots `*laed4`, and merge sort `*lamrg`), but the
 * dominant O(n³) eigenvector-update GEMMs in laex3 are executed on the GPU via
 * cublas::gemm. Eigenvalues/vectors are uploaded once at the end; `evec` is the
 * tridiagonal Q_d consumed in place by the back-transform.
 *
 * Only the MagmaRangeAll case (all eigenpairs) is implemented — the solver
 * always wants the full spectrum.
 *
 * Device scratch (no extra pool allocation): laex3 stages the per-merge GEMM
 * operands in ws->Sdc (Q2 block) and ws->M (S matrix + output); ws->M is
 * re-initialised by back_transform before stage 4, so clobbering it here is safe.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuda/handle.h"
#include "cuda/kernels.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

// =============================================================================
// LAPACK (MKL) f77 entry points — only the routines reused as-is on the host.
// All are void / pointer-arg (no return-by-value, ABI-safe).
// =============================================================================
namespace {
extern "C" {
void dsteqr_(const char *, const int *, double *, double *, double *, const int *, double *, int *);
void ssteqr_(const char *, const int *, float *, float *, float *, const int *, float *, int *);
void dlaed2_(int *, const int *, const int *, double *, double *, const int *, int *, double *,
             double *, double *, double *, double *, int *, int *, int *, int *, int *);
void slaed2_(int *, const int *, const int *, float *, float *, const int *, int *, float *,
             float *, float *, float *, float *, int *, int *, int *, int *, int *);
void dlaed4_(const int *, const int *, const double *, const double *, double *, const double *,
             double *, int *);
void slaed4_(const int *, const int *, const float *, const float *, float *, const float *,
             float *, int *);
void dlamrg_(const int *, const int *, const double *, const int *, const int *, int *);
void slamrg_(const int *, const int *, const float *, const int *, const int *, int *);
}
} // namespace

namespace cuev {
namespace kernels {

namespace {

/// Minimum sub-problem size handed to the dense base solver `*steqr`.
/// Sub-problems of this size or smaller are solved directly (no further divide).
constexpr int SMLSIZ = 64;

// --- thin templated dispatch over the f77 LAPACK routines --------------------
template <typename T>
void lap_steqr(const char *compz, int n, T *d, T *e, T *Z, int ldz, T *work, int *info) {
    if constexpr (std::is_same_v<T, float>)
        ssteqr_(compz, &n, d, e, Z, &ldz, work, info);
    else
        dsteqr_(compz, &n, d, e, Z, &ldz, work, info);
}

template <typename T>
void lap_laed2(int *k, int n, int n1, T *d, T *Q, int ldq, int *indxq, T *rho, T *z, T *dlamda,
               T *w, T *Q2, int *indx, int *indxc, int *indxp, int *coltyp, int *info) {
    if constexpr (std::is_same_v<T, float>)
        slaed2_(k, &n, &n1, d, Q, &ldq, indxq, rho, z, dlamda, w, Q2, indx, indxc, indxp, coltyp,
                info);
    else
        dlaed2_(k, &n, &n1, d, Q, &ldq, indxq, rho, z, dlamda, w, Q2, indx, indxc, indxp, coltyp,
                info);
}

template <typename T>
void lap_laed4(int k, int i, const T *d, const T *z, T *delta, T rho, T *dlam, int *info) {
    if constexpr (std::is_same_v<T, float>)
        slaed4_(&k, &i, d, z, delta, &rho, dlam, info);
    else
        dlaed4_(&k, &i, d, z, delta, &rho, dlam, info);
}

template <typename T> void lap_lamrg(int n1, int n2, const T *a, int d1, int d2, int *index) {
    if constexpr (std::is_same_v<T, float>)
        slamrg_(&n1, &n2, a, &d1, &d2, index);
    else
        dlamrg_(&n1, &n2, a, &d1, &d2, index);
}

[[noreturn]] static void fail(const char *what, int info) {
    fprintf(stderr, "tridi_dc: %s failed, info = %d\n", what, info);
    exit(EXIT_FAILURE);
}

// --- small host helpers (replace lanst / nrm2 / column ops) ------------------
template <typename T> T tridi_maxnorm(int n, const T *d, const T *e) {
    T m = 0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::abs(d[i]));
    for (int i = 0; i < n - 1; ++i)
        m = std::max(m, std::abs(e[i]));
    return m;
}

template <typename T> T nrm2(int n, const T *x) {
    T s = 0;
    for (int i = 0; i < n; ++i)
        s += x[i] * x[i];
    return std::sqrt(s);
}

// =============================================================================
// laex3 — secular equation + GPU eigenvector update (MagmaRangeAll)
// =============================================================================
// Solves D + rho z zᵀ for the k non-deflated roots, forms the rank-1 modification
// eigenvectors in the first k columns of Q, then multiplies the split eigenvectors
// Q2 by that k×k system (the two block GEMMs) on the GPU to update Q in place.
template <typename T>
void laex3(SolverHandle<T> *ws, int k, int n, int n1, T *d, T *Q, int ldq, T rho, T *dlamda, T *Q2,
           const int *indx, const int *ctot, T *w, T *s, int *indxq) {
    const T one = 1, zero = 0;
    if (k == 0) return;

    int info = 0;
    // Secular roots — each call fills column j of Q with its delta vector.
    // Independent across roots → parallelise (each laed4 is a pure function).
#pragma omp parallel for schedule(dynamic, 32)
    for (int j = 0; j < k; ++j) {
        int jinfo = 0;
        lap_laed4<T>(k, j + 1, dlamda, w, &Q[(size_t)j * ldq], rho, &d[j], &jinfo);
        if (jinfo != 0) info = jinfo;
    }
    if (info != 0) fail("laed4", info);

    // Reintegration permutation: d[0..k) ascending, d[k..n) descending.
    lap_lamrg<T>(k, n - k, d, 1, -1, indxq);

    if (k == 2) {
        for (int j = 0; j < k; ++j) {
            T tmp2[2] = {Q[0 + (size_t)j * ldq], Q[1 + (size_t)j * ldq]};
            Q[0 + (size_t)j * ldq] = tmp2[indx[0] - 1];
            Q[1 + (size_t)j * ldq] = tmp2[indx[1] - 1];
        }
    } else if (k != 1) {
        // Updated weights (Löwner / Gu-Eisenstat). Each w[i] is an independent
        // product over columns, so parallelise over i. s[] keeps the old signs.
        for (int i = 0; i < k; ++i)
            s[i] = w[i];
#pragma omp parallel for schedule(static)
        for (int i = 0; i < k; ++i) {
            T wi = Q[i + (size_t)i * ldq]; // diag of delta
            for (int j = 0; j < i; ++j)
                wi *= Q[i + (size_t)j * ldq] / (dlamda[i] - dlamda[j]);
            for (int j = i + 1; j < k; ++j)
                wi *= Q[i + (size_t)j * ldq] / (dlamda[i] - dlamda[j]);
            w[i] = std::copysign(std::sqrt(-wi), s[i]);
        }

        // Eigenvectors of the rank-1 modification, scattered by indx.
        // Columns are independent; each thread uses private scratch.
#pragma omp parallel
        {
            std::vector<T> sl(k);
#pragma omp for schedule(dynamic, 32)
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < k; ++i)
                    sl[i] = w[i] / Q[i + (size_t)j * ldq];
                T t = nrm2(k, sl.data());
                for (int i = 0; i < k; ++i) {
                    int ii = indx[i] - 1;
                    Q[i + (size_t)j * ldq] = sl[ii] / t;
                }
            }
        }
    }

    // --- GPU eigenvector update: Q <- Q2 · S, two blocks --------------------
    int n2 = n - n1;
    int n12 = ctot[0] + ctot[1];
    int n23 = ctot[1] + ctot[2];
    int iq2 = n1 * n12; // offset of block-2 inside Q2

    // device scratch carved from the pool (no extra allocation)
    T *dQ2 = ws->Sdc;                                    // Q2 block
    T *dS = ws->M;                                       // S (copy of Q rows)
    T *dQ = ws->M + ((size_t)ws->n * ws->n / 2 + ws->n); // output, disjoint from dS
    cudaStream_t st = ws->stream;
    const size_t es = sizeof(T);

    auto up = [&](const T *hsrc, int ld_src, T *ddst, int rows, int cols) {
        CUDA_CHECK(cudaMemcpy2DAsync(ddst, rows * es, hsrc, ld_src * es, rows * es, cols,
                                     cudaMemcpyHostToDevice, st));
    };
    auto down = [&](T *hdst, int ld_dst, const T *dsrc, int rows, int cols) {
        CUDA_CHECK(cudaMemcpy2DAsync(hdst, ld_dst * es, dsrc, rows * es, rows * es, cols,
                                     cudaMemcpyDeviceToHost, st));
    };

    // Block 2: rows [n1, n) <- Q2[iq2] (n2×n23) · Q(ctot0.., 0..k) (n23×k)
    if (n23 != 0) {
        up(&Q2[iq2], n2, dQ2, n2, n23);
        up(&Q[ctot[0]], ldq, dS, n23, k);
        cublas::gemm<T>(ws, CUBLAS_OP_N, CUBLAS_OP_N, n2, k, n23, &one, dQ2, n2, dS, n23, &zero, dQ,
                        n2);
        down(&Q[n1], ldq, dQ, n2, k);
    } else {
        for (int j = 0; j < k; ++j)
            for (int i = n1; i < n; ++i)
                Q[i + (size_t)j * ldq] = 0;
    }

    // Block 1: rows [0, n1) <- Q2[0] (n1×n12) · Q(0.., 0..k) (n12×k)
    if (n12 != 0) {
        up(&Q2[0], n1, dQ2, n1, n12);
        up(&Q[0], ldq, dS, n12, k);
        cublas::gemm<T>(ws, CUBLAS_OP_N, CUBLAS_OP_N, n1, k, n12, &one, dQ2, n1, dS, n12, &zero, dQ,
                        n1);
        down(&Q[0], ldq, dQ, n1, k);
    } else {
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < n1; ++i)
                Q[i + (size_t)j * ldq] = 0;
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
}

// =============================================================================
// laex1 — one rank-1 merge: form z, deflate (laed2), then laex3
// =============================================================================
// work holds three length-n vectors: z @ [0,n), dlamda @ [n,2n), w @ [2n,3n).
template <typename T>
void laex1(SolverHandle<T> *ws, int n, T *d, T *Q, int ldq, int *indxq, T rho, int cutpnt, T *work,
           T *Q2, T *s, int *indx, int *indxc, int *indxp, int *coltyp) {
    T *z = work, *dlamda = work + n, *w = work + 2 * n;

    // z = [ last row of Q1 | first row of Q2 ]
    for (int i = 0; i < cutpnt; ++i)
        z[i] = Q[(cutpnt - 1) + (size_t)i * ldq];
    for (int i = cutpnt; i < n; ++i)
        z[i] = Q[cutpnt + (size_t)i * ldq];

    int k = 0, info = 0;
    lap_laed2<T>(&k, n, cutpnt, d, Q, ldq, indxq, &rho, z, dlamda, w, Q2, indx, indxc, indxp,
                 coltyp, &info);
    if (info != 0) fail("laed2", info);

    if (k != 0)
        // coltyp[0..3] hold the column-type counts (ctot) on exit from laed2.
        laex3<T>(ws, k, n, cutpnt, d, Q, ldq, rho, dlamda, Q2, indxc, coltyp, w, s, indxq);
    else
        for (int i = 0; i < n; ++i)
            indxq[i] = i + 1;
}

// =============================================================================
// laex0 — divide-and-conquer driver over one (irreducible) block
// =============================================================================
template <typename T>
void laex0(SolverHandle<T> *ws, int n, T *d, T *e, T *Q, int ldq, T *work, T *Q2, T *s, int *indxq,
           int *indx, int *indxc, int *indxp, int *coltyp, int *part) {
    // Sizes and placement of the bottom-level sub-problems (block boundaries).
    part[0] = n;
    int subpbs = 1, tlvls = 0;
    while (part[subpbs - 1] > SMLSIZ) {
        for (int j = subpbs; j > 0; --j) {
            part[2 * j - 1] = (part[j - 1] + 1) / 2;
            part[2 * j - 2] = part[j - 1] / 2;
        }
        ++tlvls;
        subpbs *= 2;
    }
    for (int j = 1; j < subpbs; ++j)
        part[j] += part[j - 1];

    // Split into sub-problems via rank-1 modifications (cuts on the diagonal).
    for (int i = 0; i < subpbs - 1; ++i) {
        int submat = part[i];
        d[submat - 1] -= std::abs(e[submat - 1]);
        d[submat] -= std::abs(e[submat - 1]);
    }

    // Base solve at the leaves.
    for (int i = 0; i < subpbs; ++i) {
        int submat = (i == 0) ? 0 : part[i - 1];
        int matsiz = (i == 0) ? part[0] : part[i] - part[i - 1];
        int info = 0;
        lap_steqr<T>("I", matsiz, &d[submat], &e[submat], &Q[submat + (size_t)submat * ldq], ldq,
                     work, &info);
        if (info != 0) fail("steqr (leaf)", info);
        for (int j = submat, kk = 1; j < part[i]; ++j, ++kk)
            indxq[j] = kk;
    }

    // Merge adjacent eigensystems bottom-up.
    while (subpbs > 1) {
        for (int i = 0; i < subpbs - 1; i += 2) {
            int submat, matsiz, msd2;
            if (i == 0) {
                submat = 0;
                matsiz = part[1];
                msd2 = part[0];
            } else {
                submat = part[i - 1];
                matsiz = part[i + 1] - part[i - 1];
                msd2 = matsiz / 2;
            }
            laex1<T>(ws, matsiz, &d[submat], &Q[submat + (size_t)submat * ldq], ldq, &indxq[submat],
                     e[submat + msd2 - 1], msd2, work, Q2, s, indx, indxc, indxp, coltyp);
            part[i / 2] = part[i + 1];
        }
        subpbs /= 2;
    }

    // Re-merge deflated eigenpairs into ascending order (column permutation).
    // Uses Q2 (n×n) as the reordered destination and the first n of `work` for d.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int j = indxq[i] - 1;
        work[i] = d[j];
        for (int r = 0; r < n; ++r)
            Q2[r + (size_t)i * ldq] = Q[r + (size_t)j * ldq];
    }
    for (int i = 0; i < n; ++i)
        d[i] = work[i];
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < n; ++r)
            Q[r + (size_t)j * ldq] = Q2[r + (size_t)j * ldq];
}

// =============================================================================
// stedx — top level: scale, split at tiny off-diagonals, drive laex0
// (MagmaRangeAll). Q must be n×n with ld = ldq.
// =============================================================================
template <typename T>
void stedx(SolverHandle<T> *ws, int n, T *d, T *e, T *Q, int ldq, T *work, T *Q2, T *s, int *indxq,
           int *indx, int *indxc, int *indxp, int *coltyp, int *part) {
    // Q <- I
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Q[i + (size_t)j * ldq] = (i == j) ? T(1) : T(0);

    if (n < SMLSIZ) {
        int info = 0;
        lap_steqr<T>("I", n, d, e, Q, ldq, work, &info);
        if (info != 0) fail("steqr (small-n)", info);
        return;
    }

    T orgnrm = tridi_maxnorm(n, d, e);
    if (orgnrm == T(0)) return;
    const T eps = std::numeric_limits<T>::epsilon();

    int last_m = n;
    int start = 0;
    while (start < n) {
        // Next independent sub-problem: [start, end).
        int end = start + 1;
        for (; end < n; ++end) {
            T tiny = eps * std::sqrt(std::abs(d[end - 1] * d[end]));
            if (std::abs(e[end - 1]) <= tiny) break;
        }
        int m = end - start;
        last_m = m;
        if (m == 1) {
            start = end;
            continue;
        }
        if (m > SMLSIZ) {
            // Scale block to unit norm, solve, scale eigenvalues back.
            T bn = tridi_maxnorm(m, &d[start], &e[start]);
            for (int i = 0; i < m; ++i)
                d[start + i] /= bn;
            for (int i = 0; i < m - 1; ++i)
                e[start + i] /= bn;
            laex0<T>(ws, m, &d[start], &e[start], &Q[start + (size_t)start * ldq], ldq, work, Q2, s,
                     &indxq[start], indx, indxc, indxp, coltyp, part);
            for (int i = 0; i < m; ++i)
                d[start + i] *= bn;
        } else {
            int info = 0;
            lap_steqr<T>("I", m, &d[start], &e[start], &Q[start + (size_t)start * ldq], ldq, work,
                         &info);
            if (info != 0) fail("steqr (block)", info);
        }
        start = end;
    }

    // If the matrix split, eigenvalues across blocks are unordered — sort.
    if (last_m < n) {
        for (int i = 1; i < n; ++i) {
            int kmin = i - 1;
            T p = d[i - 1];
            for (int j = i; j < n; ++j)
                if (d[j] < p) {
                    kmin = j;
                    p = d[j];
                }
            if (kmin != i - 1) {
                d[kmin] = d[i - 1];
                d[i - 1] = p;
                for (int r = 0; r < n; ++r)
                    std::swap(Q[r + (size_t)(i - 1) * ldq], Q[r + (size_t)kmin * ldq]);
            }
        }
    }
}

} // namespace

// =============================================================================
// Public entry: tridiagonal D&C, GPU-accelerated eigenvector updates.
// =============================================================================
template <typename T> void tridi_dc(SolverHandle<T> *ws, T *d, T *e, T *eval, T *evec) {
    const int n = ws->n;
    const size_t nn = (size_t)n * n;

    // Pull the tridiagonal to the host. e holds n-1 sub-diagonal entries.
    std::vector<T> hd(n), he(n, T(0));
    CUDA_CHECK(cudaMemcpyAsync(hd.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost, ws->stream));
    if (n > 1)
        CUDA_CHECK(
            cudaMemcpyAsync(he.data(), e, (n - 1) * sizeof(T), cudaMemcpyDeviceToHost, ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));

    // Host workspace. Q/Q2 are fully overwritten before use, so skip zero-init.
    std::unique_ptr<T[]> Qbuf(new T[nn]);  // eigenvectors (column-major, ld = n)
    std::unique_ptr<T[]> Q2buf(new T[nn]); // deflation eigenvectors / reorder scratch
    T *Q = Qbuf.get(), *Q2 = Q2buf.get();
    std::vector<T> s(n);                    // laex3 sign scratch (length k <= n)
    std::vector<T> work(3 * (size_t)n + 2); // z, dlamda, w (>= 2n for steqr too)
    std::vector<int> indxq(n), indx(n), indxc(n), indxp(n), coltyp(n), part(2 * n + 1);

    stedx<T>(ws, n, hd.data(), he.data(), Q, n, work.data(), Q2, s.data(), indxq.data(),
             indx.data(), indxc.data(), indxp.data(), coltyp.data(), part.data());

    // Upload results.
    CUDA_CHECK(cudaMemcpyAsync(eval, hd.data(), n * sizeof(T), cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaMemcpyAsync(evec, Q, nn * sizeof(T), cudaMemcpyHostToDevice, ws->stream));
    CUDA_CHECK(cudaStreamSynchronize(ws->stream));
}

// =============================================================================
// Explicit instantiations
// =============================================================================
#define INSTANTIATE(T) template void tridi_dc<T>(SolverHandle<T> *, T *, T *, T *, T *);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE

} // namespace kernels
} // namespace cuev
