# cuEV — CUDA Eigensolver

Real symmetric dense eigensolver on NVIDIA GPUs, built to scale to a 2D BLACS
block-cyclic matrix distributed over MPI + NCCL.

```cpp
cuev::symm_eig_solve<T>(T *H, int n, T *eval, T *evec);
```

- `H` — n×n real symmetric, column-major, device pointer, overwritten on return.
- `eval` — eigenvalues ascending, length n.
- `evec` — eigenvectors as columns (column j = j-th eigenvector), n×n column-major.

## Algorithm

2-stage tridiagonalization EVD:

```
DBBR            full matrix → band       (double-blocking band reduction)
bulge chasing   band → tridiagonal       (GPU bulge chasing)
D&C             tridiagonal eigensolve
back-transform  SBR-Back + BLAS2 BC-Back, reordered (Q_s·Q_b)·Q_d
```

References: Wang et al., "Improving Tridiagonalization Performance on GPU Architectures"
(PPoPP'25); Wang et al., "Rethinking Back Transformation in 2-stage EVD" (SC'25).

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80   # single GPU
cmake --build build
cmake -B build -DCUEV_ENABLE_MP=ON             # distributed (multi-GPU)
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | Single-GPU reference (DBBR + GPU-BC + back-transform) | in progress |
| 2 | Distributed 2D block-cyclic (cuBLASMp / cuSOLVERMp / NCCL) | planned |
