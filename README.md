# cuGEMV

Sandbox for developing and benchmarking GEMV kernels on Hopper (H100/H200, sm_90a), with a focus on advanced CUDA techniques: TMA, warp groups, thread block clusters, and async producer-consumer pipelines.

All kernels implement `y = α·A·x + β·y` (row-major, fp32 and fp64). cuBLAS `sgemv`/`dgemv` serves as the correctness reference and performance baseline.

## Build

Requires CUDA 12+ and a Hopper GPU (sm_90a).

```bash
make          # release build → build/cugemv
make clean
```

Override the CUDA installation path if needed:

```bash
make CUDA_HOME=/path/to/cuda
```

## Run

```bash
./build/cugemv [--M 4096] [--N 4096] [--warmup 3] [--iters 20]
```

## File structure

```
cuGEMV/
├── Makefile
├── include/
│   └── common.h              CUDA_CHECK / CUBLAS_CHECK macros, GemvArgs
└── src/
    ├── main.cpp              benchmark harness (timing, correctness)
    └── cuda/
        ├── kernels.cuh       launcher declarations
        └── kernels.cu        gemv kernels
```

## Kernels

| Name | Technique |
|---|---|
| `gemv_gmem` | one thread per row |
| `gemv_smem` | shared memory tiling + block reduce |
| `gemv_tma` | Hopper TMA bulk async copy |
| `gemv_warpgroup` | 128-thread warpgroup cooperative fetch |
| `gemv_cluster` | thread block cluster + distributed shared memory |
| `gemv_double_tma` | TMA producer / async-barrier consumer pipeline |
