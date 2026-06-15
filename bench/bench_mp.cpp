/**
 * @file   bench_mp.cpp
 * @brief  Harness driver for the distributed solver (Phase 3).
 *
 * Rung 1: bootstrap a Context and print the rank -> (grid cell, GPU) mapping.
 * Rung 2: an NCCL AllReduce sanity check — sum each rank's id and verify it
 *         equals world_size*(world_size-1)/2, proving the comm moves bytes.
 * Rung 3: first distributed GEMM.  A[i,j]=1/n, B[i,j]=1 → C should be all-1.
 *         Each rank checks its local tiles; max error is AllReduced to rank 0.
 *
 * Run:  srun -n <nprow*npcol> ./cuBenchMp [nprow npcol] [nb] [n]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "comm.h"
#include "common.h"
#include "workspace_mp.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <vector>

using cuev::mp::Context;
using cuev::mp::dist_describe;
using cuev::mp::dist_free;
using cuev::mp::dist_local_count;
using cuev::mp::DistMatrix;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// Fill a device buffer with a constant value (host → device broadcast).
static void fill_device(double *d_buf, int64_t count, double val, cudaStream_t s) {
    std::vector<double> h(count, val);
    CUDA_CHECK(cudaMemcpyAsync(d_buf, h.data(), count * sizeof(double), cudaMemcpyHostToDevice, s));
}

// ---------------------------------------------------------------------------
// rung 3: distributed GEMM
// ---------------------------------------------------------------------------
static void rung3_gemm(Context &ctx, int64_t n) {
    // Local tile counts on this rank.
    int64_t countA = dist_local_count<double>(ctx, n, n);
    int64_t countB = dist_local_count<double>(ctx, n, n);
    int64_t countC = dist_local_count<double>(ctx, n, n);

    // Allocate local tiles on device.
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, countA * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dB, countB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dC, countC * sizeof(double)));

    // Fill:  A = 1/n,  B = 1,  C = 0 (beta=0 makes this optional but good practice).
    fill_device(dA, countA, 1.0 / (double)n, ctx.stream);
    fill_device(dB, countB, 1.0, ctx.stream);
    fill_device(dC, countC, 0.0, ctx.stream);

    // Build cuBLASMp descriptors (wraps pointers + numroc layout).
    DistMatrix<double> A = dist_describe<double>(ctx, n, n, dA);
    DistMatrix<double> B = dist_describe<double>(ctx, n, n, dB);
    DistMatrix<double> C = dist_describe<double>(ctx, n, n, dC);

    // --- query workspace ---
    const double alpha = 1.0, beta = 0.0;
    size_t wsD = 0, wsH = 0;
    CUBLASMP_CHECK(cublasMpGemm_bufferSize(ctx.cublasmp, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                           dA, 1, 1, A.desc, dB, 1, 1, B.desc, &beta, dC, 1, 1,
                                           C.desc, CUBLAS_COMPUTE_64F, &wsD, &wsH));

    void *d_work = nullptr, *h_work = nullptr;
    if (wsD) CUDA_CHECK(cudaMalloc(&d_work, wsD));
    if (wsH) h_work = std::malloc(wsH);

    // --- execute ---
    CUBLASMP_CHECK(cublasMpGemm(ctx.cublasmp, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, 1, 1,
                                A.desc, dB, 1, 1, B.desc, &beta, dC, 1, 1, C.desc,
                                CUBLAS_COMPUTE_64F, d_work, wsD, h_work, wsH));

    // cuBLASMp uses CAL for cross-rank data movement; must drain the pipeline.
    CAL_CHECK(cal_stream_sync(ctx.cal, ctx.stream));
    CAL_CHECK(cal_comm_barrier(ctx.cal, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    // --- verify ---
    // Copy C tiles back and check max |C[i,j] - 1|.
    std::vector<double> hC(countC);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, countC * sizeof(double), cudaMemcpyDeviceToHost));

    double local_err = 0.0;
    for (int64_t i = 0; i < countC; ++i)
        local_err = std::max(local_err, std::fabs(hC[i] - 1.0));

    double global_err = 0.0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);

    if (ctx.rank == 0)
        printf("GEMM n=%lld: max|C-1| = %.2e  [%s]\n", (long long)n, global_err,
               global_err < 1e-10 ? "OK" : "FAIL");

    // --- cleanup ---
    dist_free(A);
    dist_free(B);
    dist_free(C);
    if (d_work) CUDA_CHECK(cudaFree(d_work));
    if (h_work) std::free(h_work);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world;
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // args: optional grid (nprow npcol), optional block size, optional n
    int nprow = 0, npcol = 0, nb = 256;
    int64_t n = 2048;
    if (argc >= 3) {
        nprow = atoi(argv[1]);
        npcol = atoi(argv[2]);
    } else {
        cuev::mp::grid_factor(world, nprow, npcol);
    }
    if (argc >= 4) nb = atoi(argv[3]);
    if (argc >= 5) n = (int64_t)atoll(argv[4]);

    // --- bootstrap ---
    Context ctx;
    cuev::mp::ctx_init(ctx, nb, nprow, npcol);

    // --- rung 1: ordered print of the grid mapping ---
    if (ctx.rank == 0)
        printf("grid %dx%d, nb=%d, %d ranks\n  rank -> (prow,pcol)  device\n", ctx.nprow, ctx.npcol,
               ctx.nb, ctx.world_size);
    for (int r = 0; r < ctx.world_size; ++r) {
        if (r == ctx.rank)
            printf("  %4d -> (%d,%d)        %d\n", ctx.rank, ctx.prow, ctx.pcol, ctx.device);
        fflush(stdout);
        MPI_Barrier(ctx.comm);
    }

    // --- rung 2: NCCL AllReduce sanity ---
    double host = (double)ctx.rank, *d = nullptr, *dsum = nullptr, sum = 0.0;
    CUDA_CHECK(cudaMalloc(&d, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dsum, sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(d, &host, sizeof(double), cudaMemcpyHostToDevice, ctx.stream));
    NCCL_CHECK(ncclAllReduce(d, dsum, 1, ncclDouble, ncclSum, ctx.nccl, ctx.stream));
    CUDA_CHECK(cudaMemcpyAsync(&sum, dsum, sizeof(double), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(dsum));

    if (ctx.rank == 0) {
        double expect = (double)ctx.world_size * (ctx.world_size - 1) / 2.0;
        printf("NCCL AllReduce: sum(rank) = %.0f, expected %.0f  [%s]\n", sum, expect,
               sum == expect ? "OK" : "FAIL");
    }

    // --- rung 3: distributed GEMM ---
    rung3_gemm(ctx, n);

    cuev::mp::ctx_finalize(ctx);
    MPI_Finalize();
    return 0;
}
