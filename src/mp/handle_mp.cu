/**
 * @file   handle.cu
 * @brief  Distributed execution context bootstrap for cuEV Mp.
 *
 * Implements ctx_init / ctx_finalize / grid_factor from comm.h.
 *
 * Communicator story on library version (cuBLASMp 0.4 / CAL 0.4):
 *   - NCCL  : our own collectives. Bootstrapped with the canonical
 *             "broadcast a ncclUniqueId over MPI" handshake.
 *   - CAL   : required by cublasMpGridCreate / cusolverMpCreateDeviceGrid. CAL
 *             bootstraps from a user-provided MPI all-gather callback.
 *   Both are independent handles.
 *
 * TODO Portability: newer cuBLASMp/cuSOLVERMp take an ncclComm_t directly and drop
 * CAL.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "mp/comm.h"

#include <mpi.h>

namespace {

// -----------------------------------------------------------------------------
// CAL bootstrap callbacks.
//
// `data` carries the MPI_Comm (set to &ctx.comm in params.data).
// Each in-flight all-gather is tracked by a heap MPI_Request, freed in req_free.
// -----------------------------------------------------------------------------

calError_t mpi_allgather(void *src, void *recv, size_t size, void *data, void **request) {
    MPI_Comm comm = *static_cast<MPI_Comm *>(data);
    MPI_Request *req = static_cast<MPI_Request *>(std::malloc(sizeof(MPI_Request)));
    if (!req) return CAL_ERROR;
    int err = MPI_Iallgather(src, (int)size, MPI_BYTE, recv, (int)size, MPI_BYTE, comm, req);
    if (err != MPI_SUCCESS) {
        std::free(req);
        return CAL_ERROR;
    }
    *request = req;
    return CAL_OK;
}

calError_t mpi_req_test(void *request) {
    MPI_Request *req = static_cast<MPI_Request *>(request);
    int done = 0;
    if (MPI_Test(req, &done, MPI_STATUS_IGNORE) != MPI_SUCCESS) return CAL_ERROR;
    return done ? CAL_OK : CAL_ERROR_INPROGRESS;
}

calError_t mpi_req_free(void *request) {
    std::free(request);
    return CAL_OK;
}

} // namespace

namespace cuev {
namespace mp {

// =============================================================================
// grid_factor — convenience near-square factorization
// =============================================================================
void grid_factor(int world_size, int &nprow, int &npcol) {
    // Largest divisor of world_size that is <= sqrt(world_size); pair it with
    // the cofactor so nprow <= npcol and nprow*npcol == world_size.
    int p = 1;
    for (int d = 1; d * d <= world_size; ++d)
        if (world_size % d == 0) p = d;
    nprow = p;
    npcol = world_size / p;
}

// =============================================================================
// ctx_init — collective bootstrap
// =============================================================================
void ctx_init(Context &ctx, int nb, int nprow, int npcol) {
    // MPI world
    MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &ctx.comm));
    MPI_CHECK(MPI_Comm_rank(ctx.comm, &ctx.rank));
    MPI_CHECK(MPI_Comm_size(ctx.comm, &ctx.world_size));

    // map the process grid
    if (nprow * npcol != ctx.world_size) {
        if (ctx.rank == 0)
            fprintf(stderr, "ctx_init: grid %dx%d = %d != world_size %d\n", nprow, npcol,
                    nprow * npcol, ctx.world_size);
        MPI_Abort(ctx.comm, 1);
    }
    ctx.nprow = nprow;
    ctx.npcol = npcol;
    ctx.nb = nb;
    ctx.prow = ctx.rank % nprow;
    ctx.pcol = ctx.rank / nprow;

    // Bind rank to a GPU via its NODE-LOCAL rank
    MPI_Comm node;
    MPI_CHECK(MPI_Comm_split_type(ctx.comm, MPI_COMM_TYPE_SHARED, ctx.rank, MPI_INFO_NULL, &node));
    int local_rank;
    MPI_CHECK(MPI_Comm_rank(node, &local_rank));
    MPI_CHECK(MPI_Comm_free(&node));
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    ctx.device = local_rank % ndev;
    CUDA_CHECK(cudaSetDevice(ctx.device));
    CUDA_CHECK(cudaStreamCreate(&ctx.stream));

    // NCCL broadcast a uniqueId over MPI, then init all ranks
    ncclUniqueId id;
    if (ctx.rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, ctx.comm));
    NCCL_CHECK(ncclCommInitRank(&ctx.nccl, ctx.world_size, id, ctx.rank));

    // CAL: bootstrap over MPI
    cal_comm_create_params_t p{};
    p.allgather = mpi_allgather;
    p.req_test = mpi_req_test;
    p.req_free = mpi_req_free;
    p.data = &ctx.comm;
    p.nranks = ctx.world_size;
    p.rank = ctx.rank;
    p.local_device = ctx.device;
    CAL_CHECK(cal_comm_create(p, &ctx.cal));

    // cuBLASMp + cuSOLVERMp handles and their grid views
    CUBLASMP_CHECK(cublasMpCreate(&ctx.cublasmp, ctx.stream));
    CUBLASMP_CHECK(
        cublasMpGridCreate(nprow, npcol, CUBLASMP_GRID_LAYOUT_COL_MAJOR, ctx.cal, &ctx.grid));

    CUSOLVER_CHECK(cusolverMpCreate(&ctx.cusolvermp, ctx.device, ctx.stream));
    CUSOLVER_CHECK(cusolverMpCreateDeviceGrid(ctx.cusolvermp, &ctx.solvergrid, ctx.cal, nprow,
                                              npcol, CUSOLVERMP_GRID_MAPPING_COL_MAJOR));
}

// =============================================================================
// ctx_finalize — teardown in reverse order
// =============================================================================
void ctx_finalize(Context &ctx) {
    CUSOLVER_CHECK(cusolverMpDestroyGrid(ctx.solvergrid));
    CUSOLVER_CHECK(cusolverMpDestroy(ctx.cusolvermp));
    CUBLASMP_CHECK(cublasMpGridDestroy(ctx.grid));
    CUBLASMP_CHECK(cublasMpDestroy(ctx.cublasmp));
    CAL_CHECK(cal_comm_destroy(ctx.cal));
    NCCL_CHECK(ncclCommDestroy(ctx.nccl));
    CUDA_CHECK(cudaStreamDestroy(ctx.stream));
    MPI_CHECK(MPI_Comm_free(&ctx.comm));
}

} // namespace mp
} // namespace cuev
