# cuEV — CUDA Eigensolver

Real symmetric dense eigensolver on NVIDIA GPUs, built to scale to a 2D BLACS
block-cyclic matrix distributed over MPI + NCCL.

```cpp
cuev::symm_eig_solve<T>(T *A, int n, T *eval, T *evec, cudaStream_t stream);
```

- `A` — n×n real symmetric, column-major, device pointer, overwritten on return.
- `eval` — eigenvalues ascending, length n.
- `evec` — eigenvectors as columns (column j = j-th eigenvector), n×n column-major.

## Algorithm

2-stage tridiagonalization EVD:

### 1. Double Blocking Band Reduction (DBBR): Symm → Banded
* **Mechanism**: Accumulate Householder reflections in compact $WY$ panels to increase arithmetic intensity.
* **Strategy**: Update the next panel (1st column blocking) and defer the full trailing matrix update until the working index reaches block $k$ (2nd panel blocking).

### 2. Data Repacking
* **Mechanism**: Extract the reduced band from the hollowed-out dense matrix.
* **Strategy**: Move the band into a packed $N \times b$ contiguous array. This eliminates non-contiguous memory accesses, ensuring the entire band fits into the L2 cache for subsequent stages.

### 3. Wavefront Bulge Chasing: Banded → Tridiagonal
* **Mechanism**: Launch a persistent CUDA kernel where thread blocks are statically assigned to horizontal band tiles.
* **Strategy**: Implement a wavefront pipeline where bulges are chased across tiles and passed across thread blocks. Synchronize thread block handoffs with point-to-point atomics (`cuda::atomic_thread_fence`, `wait`/`notify`).

### 4. Divide & Conquer (D&C)
* **Mechanism**: Tridiagonal eigensolve on the **CPU** via LAPACK `dstedc` (MKL). cuSOLVER exposes no standalone tridiagonal D&C (only dense `syevd`, which re-tridiagonalizes), so it is unusable here; running on the CPU also frees the GPU for the back-transform (stage 5).

### 5. Back-Transformation
* **Mechanism**: Reordered workflow $Q = (Q_s \cdot Q_b) \cdot Q_d$.
* **Strategy**: Run CPU D&C concurrently with the GPU forming $Q_s \cdot Q_b$ (both depend only on the tridiagonal/reflectors, not each other), then one GEMM $\cdot Q_d$. Correctness first uses the simple order $Q_s \cdot (Q_b \cdot Q_d)$ synchronously; the async reorder is the performance layer.


References: Wang et al., "Improving Tridiagonalization Performance on GPU Architectures"
(PPoPP'25); Wang et al., "Rethinking Back Transformation in 2-stage EVD" (SC'25).


## Distributed Algorithm

### 1. Distributed DBBR: Symm → Banded
* **Mechanism**: Maintain a 2D Block-Cyclic grid to ensure optimal load balance for the dense trailing matrix update.
* **Strategy**: Implement Lookahead Pipeline: slice the trailing update to compute the "Lookahead Panel" first. Dispatch it via asynchronous `ncclBroadcast`. While the network handles the transfer and the next process column factorizes, execute the massive bulk `SYR2K` update on the remaining trailing matrix.

### 2. Distributed Data Repacking
* **Mechanism**: Perform a bulk synchronous `ncclAllToAllv` to extract the band from the 2D grid.
* **Strategy**: Redistribute the data into a 1D Column-wise Process Grid. This spans the narrow band across all GPUs.

### 3. Distributed Wavefront Bulge Chasing: Banded → Tridiagonal
* **Mechanism**: Launch one persistent CUDA kernel per GPU on the 1D grid.
* **Strategy**: Pipeline sweeps across local tiles using point-to-point atomics inside one GPU. Utilize asynchronous `ncclSend`/`ncclRecv` streams to pass boundary bulges directly between GPUs.

### 4. Distributed Divide & Conquer (D&C)
* **Mechanism**: Gather the tridiagonal matrix to all nodes using `ncclAllGather`.
* **Strategy**: Execute redundant local LAPACK `dstedc` (CPU) solves on every node to bypass network traffic.

### 5. Distributed Back-Transformation
* **Mechanism**: Reordered workflow $Q = (Q_s \cdot Q_b) \cdot Q_d$.
* **Strategy**: Overlap async GPU-stream D&C with the back-transformation of $Q_s$ and $Q_b$.

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

### Stage status (Phase 1)

- [x] **1 DBBR** (full → band) — complete, tested (eigenvalues vs cuSOLVER), benchmarked
- [ ] **2 Data repacking** (`bc_pack`) — stub
- [ ] **3 Bulge chasing** (`bc_chase`) — stub
- [ ] **4 D&C** (LAPACK `dstedc`, MKL) — not started
- [ ] **5 Back-transform** (`bt_sbr_back`, `bt_bc_back`) — stub

DBBR on A100 80GB (fp64, b=64, k=512): ~9.9 TFLOP/s at n=32k (~1.15× over single-blocked SBR;
the bigger DBBR win is a ≥49k phenomenon). Profile levers for later: per-block square companion
(`symm` ~34%), custom panel QR (cuSOLVER `geqrf` ~26%), custom `dbbr_syr2k` (~18%).
