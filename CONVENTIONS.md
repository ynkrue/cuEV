# cuEV — Coding Conventions

## File Headers

Every `.cu` and `.cuh` file begins with a Doxygen block:

```cpp
/**
 * @file   qdwh.cu
 * @brief  One-line description.
 *
 * Longer description: algorithm context, mathematical operation,
 * kernel sequence, anything a reader needs without reading all the code.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */
```

---

## Function Documentation

### `kernels.cuh` — full Doxygen per function

```cpp
/**
 * @brief In-place diagonal shift: A ← A − μI.
 *
 * @tparam T      float or double
 * @param[in,out] A      n×n matrix, column-major; diagonal modified in place
 * @param[in]     mu     shift scalar μ
 * @param[in]     n      matrix dimension
 * @param[in]     stream CUDA stream
 */
template <typename T>
void qdwh_shift(T *A, T mu, int n, cudaStream_t stream);
```

Rules:
- `@brief` — single sentence ending with a period
- Math in descriptions uses Unicode (`←`, `ᵀ`, `Σ`, `·`, `α`, `β`)
- `@param` direction tags: `[in]`, `[out]`, `[in,out]`
- `@tparam T` is always `float or double`

### `.cu` bodies — non-obvious WHY only

Only comment lines where the **why** is non-obvious: a numerical subtlety, a hardware constraint, a workaround. Never restate what the code does.

---

## Naming

| Thing | Convention | Example |
|---|---|---|
| Device kernel | `<prefix>_<op>_kernel` | `qdwh_shift_kernel` |
| Host launcher | `<prefix>_<op>` in `cuev::kernels` | `qdwh_shift` |
| Prefixes | `qdwh_` QDWH primitives, `sdc_` spectral D&C helpers | |
| Device pointer (host code) | `d` prefix | `dA`, `d_eval` |
| Host pointer | `h` prefix | `hA`, `h_eval` |
| Block size | `constexpr int BLOCKSIZE = N` inside launcher | |

---

## Namespaces

```
cuev            public API: symm_eig_solve, qdwh_sign, SolverWorkspace, workspace_alloc/free
cuev::kernels   custom GPU kernel launchers (qdwh_*, sdc_*)
cuev::cublas    type-dispatching cuBLAS wrappers (gemm, geam, scal, copy, nrm2)
cuev::cusolver  workspace-aware cuSOLVER wrappers (geqrf, orgqr, syevd)
cuev::mp        multi-GPU API (Phase 3)
```

`__global__` kernels live in anonymous namespaces inside their `.cu` files — never exported.

---

## Templates

- All numeric code templated on `T` (float or double)
- Use `if constexpr (std::is_same_v<T, float>)` for type dispatch
- Every `.cu` ends with explicit instantiations:

```cpp
#define INSTANTIATE(T) \
    template void qdwh_shift<T>(T *, T, int, cudaStream_t);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
```

---

## Error Handling

- `CUDA_CHECK(...)` — every CUDA API call in host code
- `CUBLAS_CHECK(...)` — every cuBLAS call
- `CUSOLVER_CHECK(...)` — every cuSOLVER call
- No error checking inside `__global__` kernels

---

## Memory

- All matrices **column-major** (matches cuBLAS/cuSOLVER natively; leading dimension = number of rows)
- No `cudaMalloc` / `cudaFree` in the hot path — use `SolverWorkspace::push/mark/reset`
- Eigenvectors stored as **columns**: `evec[j * n + i]` = i-th component of j-th eigenvector

---

## Kernel Design Patterns

- Reductions: block-reduce with `__syncthreads` → warp-reduce with `__shfl_down_sync(0xffffffff, val, N)`
- Block size: always compile-time constant (`constexpr` or template param)
- Prefer 1D `dim3 block(N)` with explicit index arithmetic over 2D blocks
- Annotate kernels with fixed thread counts: `__launch_bounds__(NUM_THREADS)`

---

## Braces

Always use `{}` except for single-line guard clauses that only `return`:

```cpp
if (row >= n) return;              // OK — guard clause
if (tid < s) { smem[tid] += ...; } // braces required
```
