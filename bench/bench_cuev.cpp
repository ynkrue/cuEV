/**
 * @file   bench_cuev.cpp
 * @brief  cuEV-only runtime sweep — distributed spectral D&C eigensolver timed
 *         over a range of matrix sizes (no cross-solver comparison).
 *
 * Run:  srun -N<nodes> --tasks-per-node=<p> --gpus-per-node=<p> \
 *           build/cuBenchCuev [--nprow P] [--npcol Q] [--nb NB] \
 *                             [--sizes 1024,2048,4096,8192] [--warmup W] [--iters I]
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */

#include "common.h"
#include "cuev_mp.h"
#include "mp/comm.h"
#include "mp/workspace_mp.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using cuev::mp::Context;
using cuev::mp::dist_describe;
using cuev::mp::dist_free;
using cuev::mp::dist_local_count;
using cuev::mp::DistMatrix;

// Local→global index (2D block-cyclic, rsrc=csrc=0) — matches cuEV's layout.
static int64_t l2g(int64_t li, int p, int np, int64_t nb) {
    return (li / nb * np + p) * nb + (li % nb);
}

// Non-trivial test matrix: dense, indefinite, symmetric (eigenvalues straddle 0).
// Deterministic per sorted (i,j) so it is reproducible and identical on all ranks.
static double matrix_entry(int64_t i, int64_t j) {
    uint64_t a = (uint64_t)std::min(i, j), b = (uint64_t)std::max(i, j);
    uint64_t h = (a + 1) * 0x9E3779B97F4A7C15ULL ^ (b + 1) * 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    h *= 0xC4CEB9FE1A85EC53ULL;
    h ^= h >> 33;
    return 2.0 * (double)(h >> 11) * (1.0 / 9007199254740992.0) - 1.0;
}

static double wall() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world;
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int nprow = 0, npcol = 0, nb = 256, warmup = 1, iters = 3;
    std::vector<int64_t> sizes = {1024, 2048, 4096, 8192};

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--nprow") && i + 1 < argc)
            nprow = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--npcol") && i + 1 < argc)
            npcol = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nb") && i + 1 < argc)
            nb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc) {
            sizes.clear();
            std::string s = argv[++i];
            for (size_t p = 0; p < s.size();) {
                size_t c = s.find(',', p);
                if (c == std::string::npos) c = s.size();
                sizes.push_back(atoll(s.substr(p, c - p).c_str()));
                p = c + 1;
            }
        }
    }
    if (nprow == 0 || npcol == 0) cuev::mp::grid_factor(world, nprow, npcol);

    Context ctx;
    cuev::mp::ctx_init(ctx, nb, nprow, npcol);

    if (ctx.rank == 0) {
        printf("cuBenchCuev  grid=%dx%d  nb=%d  ranks=%d  warmup=%d  iters=%d\n", ctx.nprow,
               ctx.npcol, ctx.nb, ctx.world_size, warmup, iters);
        printf("%8s  %10s  %12s\n", "n", "time[ms]", "TFLOP/s");
        fflush(stdout);
    }

    for (int64_t n : sizes) {
        int64_t cnt = dist_local_count<double>(ctx, n, n);
        double *dH = nullptr, *dH0 = nullptr, *dEvec = nullptr, *dEval = nullptr;
        CUDA_CHECK(cudaMalloc(&dH, cnt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dH0, cnt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dEvec, cnt * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dEval, n * sizeof(double)));
        DistMatrix<double> H = dist_describe<double>(ctx, n, n, dH);
        DistMatrix<double> Evec = dist_describe<double>(ctx, n, n, dEvec);

        // Pristine copy dH0 of this rank's local tile (the solver overwrites H).
        {
            std::vector<double> hH((size_t)H.local_rows * H.local_cols);
            for (int64_t lj = 0; lj < H.local_cols; ++lj) {
                int64_t gj = l2g(lj, ctx.pcol, ctx.npcol, nb);
                for (int64_t li = 0; li < H.local_rows; ++li)
                    hH[(size_t)lj * H.lld + li] =
                        matrix_entry(l2g(li, ctx.prow, ctx.nprow, nb), gj);
            }
            CUDA_CHECK(
                cudaMemcpy(dH0, hH.data(), hH.size() * sizeof(double), cudaMemcpyHostToDevice));
        }

        cuev::mp::WorkspaceMp<double> ws = cuev::mp::workspace_mp_alloc<double>(ctx, n);

        double best = 1e300;
        for (int it = 0; it < warmup + iters; ++it) {
            CUDA_CHECK(cudaMemcpy(dH, dH0, cnt * sizeof(double), cudaMemcpyDeviceToDevice));
            MPI_Barrier(ctx.comm);
            double t0 = wall();
            cuev::mp::symm_eig_solve_mp(ctx, H, n, dEval, Evec, ws);
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            MPI_Barrier(ctx.comm);
            double dt = wall() - t0;
            if (it >= warmup) best = std::min(best, dt);
        }
        double tmax = 0.0;
        MPI_Allreduce(&best, &tmax, 1, MPI_DOUBLE, MPI_MAX, ctx.comm);

        if (ctx.rank == 0) {
            double tflops = (4.0 / 3.0 * (double)n * n * n) / tmax / 1e12;
            printf("%8lld  %10.3f  %12.3f\n", (long long)n, tmax * 1e3, tflops);
            fflush(stdout);
        }

        cuev::mp::workspace_mp_free(ws);
        dist_free(H);
        dist_free(Evec);
        CUDA_CHECK(cudaFree(dH));
        CUDA_CHECK(cudaFree(dH0));
        CUDA_CHECK(cudaFree(dEvec));
        CUDA_CHECK(cudaFree(dEval));
    }

    cuev::mp::ctx_finalize(ctx);
    MPI_Finalize();
    return 0;
}
