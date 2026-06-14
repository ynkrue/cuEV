/**
 * @file   workspace.h
 * @brief  SolverWorkspace<T> — single-allocation bump-allocator for cuEV.
 *
 * Included by kernels.cuh and any translation unit that calls
 * workspace_alloc / workspace_free directly (e.g. tests).
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace cuev {

/// Base-case threshold for spectral_dc recursion.
constexpr int SDC_BASE_N = 512;

/**
 * @brief Single-allocation workspace for the entire spectral D&C eigensolver.
 *
 * Created once in symm_eig_solve, threaded through all recursion levels.
 * workspace_alloc<T> issues one cudaMalloc for the combined pool; workspace_free<T>
 * issues the matching single cudaFree.  Zero dynamic allocation in the hot path.
 *
 * Memory layout (contiguous, 256-byte aligned regions):
 *
 *   cuSOLVER scratch:
 *     geqrf_buf    geqrf(2n, n)     — reused for geqrf(n, n), which is smaller
 *     orgqr_buf    orgqr(2n, n)     — reused for orgqr(n, n)
 *     potrf_buf    potrf(n)         — Cholesky-based QDWH update
 *     syevd_buf    syevd(SDC_BASE_N)
 *     d_info       1 × int
 *
 *   QDWH per-call scratch (fixed size, reused across all QDWH calls):
 *     qdwh_W       2n × n  (QR work matrix)
 *     qdwh_tau     n       (Householder scalars)
 *
 *   Data pool (mark/reset bump allocator for variable-lifetime data):
 *     6n² elements of T   — covers B, Q, H1/H2, evec1/2, eval1/2 across
 *                           all recursion levels; see workspace_alloc for derivation.
 *
 *   Derivation: at the deepest point of recursion the live data is the sum,
 *   over the active path, of each frame's footprint
 *       B(m²) + H1(k²) + H2((m−k)²) + evec1(k²) + evec2((m−k)²) ≈ m² + 2k² + 2(m−k)²
 *   which is ≤ 3m².  For near-balanced splits (k ≈ m/2, the regime for matrices
 *   with spectra symmetric about the mean) this sums to ≈ 8/3·n² ≈ 2.67n²; 6n²
 *   gives ~2× headroom.  Strongly unbalanced spectra (split point = mean, not
 *   median) can need more — push() aborts loudly if the pool is exceeded.
 *
 * @tparam T  float or double
 */
template <typename T> struct SolverWorkspace {
    // cuSOLVER scratch
    T *geqrf_buf;
    int geqrf_lwork;
    T *orgqr_buf;
    int orgqr_lwork;
    T *potrf_buf;
    int potrf_lwork;
    T *syevd_buf;
    int syevd_lwork;
    int *d_info;

    // QDWH scratch (2n×n and n)
    T *qdwh_W;
    T *qdwh_tau;

    // Data pool — mark/reset bump allocator
    T *data;
    size_t data_cap;  ///< total capacity in elements of T
    size_t data_used; ///< current stack pointer in elements of T

    // Single backing allocation owned by this struct
    void *pool;

    /// Allocate @p n elements from the data pool; returns device pointer.
    /// Aborts loudly on overflow — a silent overrun hands out a pointer past the
    /// pool and corrupts the next cudaMalloc region / faults later.
    inline T *push(size_t n) {
        if (data_used + n > data_cap) {
            fprintf(stderr,
                    "SolverWorkspace::push overflow: need %zu elements, capacity %zu "
                    "(increase pool factor in workspace_alloc)\n",
                    data_used + n, data_cap);
            abort();
        }
        T *ptr = data + data_used;
        data_used += n;
        return ptr;
    }

    /// Save the current stack position for a later reset().
    inline size_t mark() const {
        return data_used;
    }

    /// Restore the stack to a position saved by mark(), freeing everything above.
    inline void reset(size_t saved) {
        data_used = saved;
    }
};

/**
 * @brief Query cuSOLVER buffer sizes and issue one cudaMalloc for all regions.
 *
 * @tparam T         float or double
 * @param[in] h      cuSOLVER handle (buffer-size queries only; no GPU work)
 * @param[in] n      root problem dimension
 * @param[in] stream CUDA stream (unused; reserved for future async alloc)
 */
template <typename T>
SolverWorkspace<T> workspace_alloc(cusolverDnHandle_t h, int n, cudaStream_t stream);

/**
 * @brief Free the workspace pool (single cudaFree).
 *
 * @tparam T  float or double
 * @param[in,out] ws  workspace; all pointers zeroed on return
 */
template <typename T> void workspace_free(SolverWorkspace<T> &ws);

} // namespace cuev
