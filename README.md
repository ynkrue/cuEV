# cuEV — CUDA Eigensolver

Real symmetric dense eigensolver on NVIDIA GPUs.

```cpp
cuev::symm_eig_solve<T>(T *H, int n, T *eval, T *evec, cudaStream_t stream);
```

`H` — n×n real symmetric, row-major, device pointer, overwritten on return.
`eval` — eigenvalues ascending, length n.
`evec` — eigenvectors as rows, n×n row-major.

## Algorithm

Spectral divide-and-conquer via QDWH polar iteration (Nakatsukasa, Bai, Gygi 2013). Entirely BLAS-3 — no tridiagonalization, no sequential bottleneck.

```
symm_eig_solve
└── spectral_dc (recursive)
    ├── n ≤ 256   →  cuSOLVER dsyevd / ssyevd
    └── n  > 256
        ├── μ ← trace(H) / n                   split point
        ├── sign(H − μI)                        QDWH polar iteration (≤ 8 iters)
        │     each iter: scale + QR + GEMM
        ├── P = (I + sign) / 2                  spectral projector
        ├── QR(P)  →  Q₁ (n×k),  Q₂ (n×(n−k)) orthonormal bases
        ├── H₁ = Q₁ᵀHQ₁,  H₂ = Q₂ᵀHQ₂        subproblems
        ├── spectral_dc(H₁),  spectral_dc(H₂)  recurse
        └── evec = blkdiag(evec₁, evec₂) · [Q₁|Q₂]ᵀ
```

All GEMMs and QR factorizations go through cuBLAS / cuSOLVER. Custom CUDA kernels handle the small fused operations: diagonal shift, identity fill, symmetrize, trace reduction, projector rank estimation.

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build
```

Override architecture:

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a
```

Multi-GPU (Phase 3, requires MPI + cuBLASMp):

```bash
cmake -B build -DCUEV_ENABLE_MP=ON
```

## Binaries

| Binary | Source | Purpose |
|---|---|---|
| `cuBench` | `bench/bench.cpp` | `symm_eig_solve` vs cuSOLVER `dsyevd` / `ssyevd` |
| `cuTest` | `test/debug.cpp` | Smoke test on a 4×4 matrix |

## File Structure

```
cuEV/
├── CMakeLists.txt
├── include/
│   ├── cuev.h          public API  (symm_eig_solve)
│   ├── cuev_mp.h       multi-GPU API placeholder
│   ├── kernels.cuh     kernel + wrapper declarations, SolverWorkspace
│   └── common.h        CUDA_CHECK / CUBLAS_CHECK / CUSOLVER_CHECK / block_reduce_sum
└── src/
    ├── cuda/           single-GPU solver
    │   ├── cublas.cu   cuev::cublas  — type-dispatching cuBLAS wrappers
    │   ├── cusolver.cu cuev::cusolver — cuSOLVER wrappers (workspace-aware)
    │   ├── qdwh.cu     qdwh_sign  — QDWH polar iteration
    │   ├── sdc.cu      sdc_trace / sdc_rank / sdc_split / sdc_combine
    │   └── solver.cu   spectral_dc + symm_eig_solve + workspace_alloc/free
    └── mp/             multi-GPU (Phase 3, placeholder)
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | Custom STEDC eigensolver (learning) | done on `main` branch |
| 2 | Spectral D&C via QDWH, single GPU | **in progress** |
| 3 | Spectral D&C via QDWH, multi-GPU (cuBLASMp / NCCL) | planned |
