# cuEV — CUDA Eigensolver

Custom CUDA implementation of a real symmetric dense eigensolver: `A v = λv`.

Public entry point:
```cpp
cuev::solve<T>(T *H, int n, T *eval, T *evec, cudaStream_t stream);
```
`H` is an n×n real symmetric matrix (row-major, device pointer). On return, `eval` holds eigenvalues in ascending order and `evec` holds the corresponding eigenvectors as rows.

## Algorithm

```
solve<T>
├── tridiag_hh        H → T       Householder tridiagonalization (n−1 steps)
├── tridiag_eig       T → Λ, Qᵀ  divide-and-conquer STEDC
└── tridiag_hh_back   evec = Q·Qᵀ apply stored Householder projections
```

Each Householder step k reduces column k of the trailing submatrix and stores the reflection vector implicitly in the zeroed-out lower triangle of H. The accumulated product Q = P₀·P₁···P_{n-3} is only materialized during the back-transformation.

The D&C split: `T = diag(T₁, T₂) + β·vvᵀ`, solve halves recursively, merge via the rank-1 secular equation `1 + ρ·Σ zᵢ²/(dᵢ−λ) = 0`.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | Custom kernels — everything from scratch | **in progress** |
| 2 | cuBLAS / cuSOLVER-backed reference implementations | planned |
| 3 | Distributed (cuBLASMp, NCCL) | planned |

Phase 1 is the learning and research phase. Phase 2 adds library-backed implementations of each operation as correctness references and performance baselines. Phase 3 extends to multi-GPU.

## Build

```bash
make            # build shared library → build/libcuev.so
make bench      # build benchmark binary → build/cuBench
make debug      # build debug binary → build/cuDebug  (-DDEBUG)
make clean
```

Override architecture or CUDA path:
```bash
make ARCH=sm_90a
make CUDA_HOME=/path/to/cuda
```

Default target is `sm_80` (A100). Primary development target is `sm_90a` (H100/H200).

## Binaries

| Binary | Source | Purpose |
|---|---|---|
| `cuBench` | `bench/bench.cpp` | Benchmark GEMV / GEMM / transpose vs cuBLAS |
| `cuDebug` | `test/debug.cpp` | Run `solve<double>` on a small matrix, print eigenvalues and eigenvectors |

## File structure

```
cuEV/
├── Makefile
├── Doxyfile
├── include/
│   ├── common.h            CUDA_CHECK / CUBLAS_CHECK / div_up / BenchArgs / debug helpers
│   └── kernels.cuh         all kernel declarations — single interface file
└── src/
    └── custom/             Phase 1 — custom kernels
        ├── util.cu         fill, copy, transpose
        ├── gemv.cu         gemv_{gmem,smem}
        ├── gemm.cu         gemm_{gmem,smem,tiled,warptile}
        ├── householder.cu  hh_{reflect,trail_matvec,ortho,update,wy_build,wy_apply}
        ├── tridiag.cu      eig_{leaf,split,merge}
        └── solver.cu       solve<T> + tridiag_{hh,eig,hh_back}
```

## Documentation

```bash
doxygen Doxyfile    # output → docs/html/index.html
```

API reference: [cuev::kernels](@ref cuev::kernels) — all kernel launchers, [cuev::solve](@ref cuev::solve) — public entry point.
