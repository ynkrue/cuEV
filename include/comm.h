/**
 * @file   comm.h
 * @brief  Distributed execution context for cuEV — process grid, GPU binding,
 *         NCCL communicator, and cuBLASMp / cuSOLVERMp handles.
 *
 * Internal harness layer, analogous to workspace.h on the single-GPU
 * side: pulls in the heavy MPI / NCCL / cuBLASMp / cuSOLVERMp headers.
 * Included by src/mp/*.cu and the MP harness driver.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#pragma once
#ifdef CUEV_ENABLE_MP

#include <cal.h>
#include <cstdio>
#include <cstdlib>
#include <cublasmp.h>
#include <cuda_runtime.h>
#include <cusolverMp.h>
#include <mpi.h>
#include <nccl.h>

// =============================================================================
// Error checking (MP-only). cuSOLVERMp returns cusolverStatus_t — use the
// existing CUSOLVER_CHECK from common.h for it. NCCL, MPI, cuBLASMp, and CAL
// each need their own.
// =============================================================================

#define NCCL_CHECK(err)                                                                            \
    do {                                                                                           \
        ncclResult_t _e = (err);                                                                   \
        if (_e != ncclSuccess) {                                                                   \
            fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(_e)); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define MPI_CHECK(err)                                                                             \
    do {                                                                                           \
        int _e = (err);                                                                            \
        if (_e != MPI_SUCCESS) {                                                                   \
            char _s[MPI_MAX_ERROR_STRING];                                                         \
            int _l;                                                                                \
            MPI_Error_string(_e, _s, &_l);                                                         \
            fprintf(stderr, "MPI error %s:%d: %s\n", __FILE__, __LINE__, _s);                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CUBLASMP_CHECK(err)                                                                        \
    do {                                                                                           \
        cublasMpStatus_t _e = (err);                                                               \
        if (_e != CUBLASMP_STATUS_SUCCESS) {                                                       \
            fprintf(stderr, "cuBLASMp error %s:%d: %d\n", __FILE__, __LINE__, (int)_e);            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CAL_CHECK(err)                                                                             \
    do {                                                                                           \
        calError_t _e = (err);                                                                     \
        if (_e != CAL_OK) {                                                                        \
            fprintf(stderr, "CAL error %s:%d: %d\n", __FILE__, __LINE__, (int)_e);                 \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

namespace cuev {
namespace mp {

/**
 * @brief Per-rank execution context for the distributed solver.
 *
 * One Context per MPI rank. Owns this rank's GPU binding, CUDA stream, the NCCL
 * communicator, and the cuBLASMp / cuSOLVERMp handles.
 *
 * Process grid: the `world` ranks are arranged as a 2D grid nprow × npcol. Rank
 * r maps to cell (prow, pcol) with column-major ordering:
 * prow = r % nprow, pcol = r / nprow.
 * A distributed matrix is then 2D block-cyclic with block size @ref nb.
 *
 * Lifetime: bootstrap with @ref ctx_init (collective over MPI_COMM_WORLD), tear
 * down with @ref ctx_finalize.
 */
struct Context {
    // --- MPI world ---
    MPI_Comm comm;  ///< duplicated MPI communicator
    int rank;       ///< this rank's id in `comm`
    int world_size; ///< number of ranks = nprow * npcol

    // --- process grid ---
    int nprow; ///< grid rows
    int npcol; ///< grid cols
    int prow;  ///< this rank's grid row  = rank % nprow
    int pcol;  ///< this rank's grid col  = rank / nprow
    int nb;    ///< block-cyclic block size

    // --- device ---
    int device;          ///< CUDA device bound to this rank
    cudaStream_t stream; ///< stream all library handles run on

    // --- communication / compute handles ---
    ncclComm_t nccl;               ///< NCCL comm over `comm` - custom kernels
    cal_comm_t cal;                ///< CAL comm — cuBLASMp/cuSOLVERMp grids
    cublasMpHandle_t cublasmp;     ///< distributed BLAS-3 engine
    cublasMpGrid_t grid;           ///< cuBLASMp view of the nprow×npcol grid
    cusolverMpHandle_t cusolvermp; ///< distributed solver engine
    cusolverMpGrid_t solvergrid;   ///< cuSOLVERMp view of the same grid
};

/**
 * @brief Collective bootstrap of a Context.
 *
 * Performs, in order:
 *   1. MPI_Comm_dup(MPI_COMM_WORLD) → ctx.comm; read rank / world_size.
 *   2. Validate nprow * npcol == world_size and compute (prow, pcol).
 *   3. Bind to a GPU: device = local_rank % cudaGetDeviceCount.
 *   4. NCCL: rank 0 makes an ncclUniqueId, MPI_Bcast it, then every rank inits.
 *   5. cuBLASMp: cublasMpCreate on this device / stream.
 *   6. cuSOLVERMp: cusolverMpCreate on this device / stream.
 *
 * The caller owns the grid shape nprow * npcol must equal the number of MPI
 * ranks. For a convenience near-square factorization, see @ref grid_factor.
 *
 * @param[out] ctx    context to populate
 * @param[in]  nb     block-cyclic block size
 * @param[in]  nprow  grid rows
 * @param[in]  npcol  grid cols
 */
void ctx_init(Context &ctx, int nb, int nprow, int npcol);

/**
 * @brief Convenience Helper: factor @p world_size into a near-square nprow×npcol
 *        grid (nprow ≤ npcol, the largest divisor ≤ √world_size).
 */
void grid_factor(int world_size, int &nprow, int &npcol);

/**
 * @brief Collective teardown: destroy handles, NCCL comm, stream, free comm.
 *        Mirror of ctx_init in reverse order. Does not call MPI_Finalize.
 */
void ctx_finalize(Context &ctx);

} // namespace mp
} // namespace cuev

#endif // CUEV_ENABLE_MP
