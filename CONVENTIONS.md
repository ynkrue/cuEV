# cuEV — Coding Conventions

## File Headers

Every `.cu` and `.cuh` file begins with a Doxygen block:

```cpp
/**
 * @file   householder.cu
 * @brief  One-line description of what this file contains.
 *
 * Longer description: algorithm context, what mathematical operation is
 * implemented, what the kernel sequence looks like, anything a reader needs
 * to understand the file without reading all the code.
 *
 * @author  Yannik Rüfenacht
 * @date    2026-06
 */
```

Keep the longer description accurate — it is the first thing Claude and human
reviewers read. Update it when the file changes significantly.

---

## Function Documentation

### `kernels.cuh` — full Doxygen per function

All public launcher declarations use the full `@brief / @param / @tparam` form:

```cpp
/**
 * @brief Compute Householder reflector for column k.
 *
 * Reads H[k+1:n, k], computes v and τ such that
 *   (I − τvvᵀ) · H[k+1:n, k] = α e₁.
 * Stores v back into H[k+2:n, k], τ into tau[k],
 * α into e[k], H[k,k] into d[k].
 *
 * @tparam T      float or double
 * @param[in,out] H      n×n symmetric matrix, row-major, modified in place
 * @param[out]    v      Householder vector, length n−k−1
 * @param[out]    tau    Householder scalars, length n−1
 * @param[out]    d      diagonal of T being built, length n
 * @param[out]    e      subdiagonal of T being built, length n−1
 * @param[in]     N      matrix dimension
 * @param[in]     k      current step index (0-based)
 * @param[in]     stream CUDA stream
 */
template <typename T>
void hh_reflect(T *H, T *v, T *tau, T *d, T *e, int N, int k, cudaStream_t stream);
```

Rules:
- `@brief` is a single sentence ending with a period.
- Math in the longer description uses Unicode (`←`, `ᵀ`, `Σ`, `·`, `α`, `β`, `τ`).
- `@param` direction tags are `[in]`, `[out]`, or `[in,out]`. Always include them.
- `@tparam T` is always `float or double`.
- No `@return` for `void` functions.

### `.cu` bodies — short inline comments only

Only comment lines where the **why** is non-obvious: a hidden constraint, a
numerical subtlety, a workaround for a hardware limitation.

```cpp
// tau = 1/vᵀv, not 2/vᵀv — the Golub & Van Loan convention absorbs the 2
// into the rank-2 update formula
tau[k] = (vTv == T(0)) ? T(0) : T(1) / vTv;
```

Do not write comments that restate what the code already says. No `// compute
partial dot product`, no `// store result`.

---

## Naming

| Thing | Convention | Example |
|---|---|---|
| Device kernel | `<op>_<variant>_kernel` | `gemv_smem_kernel` |
| Host launcher | `<op>_<variant>` | `gemv_smem` |
| Default dispatcher (inlined in kernels.cuh) | `<op>` | `gemv` |
| Block size constant | `constexpr int BLOCKSIZE = N` inside launcher | |
| Device pointer (host code) | `d` prefix | `dA`, `d_eval` |
| Host pointer | `h` prefix | `hA`, `h_eval` |

---

## Namespaces

- `namespace cuev` — public API (`solve<T>` and anything callers use directly)
- `namespace cuev::kernels` — all kernel launchers declared in `kernels.cuh`
- Anonymous namespace inside each `.cu` file — `__global__` kernels (never exported)

```cpp
// kernels.cuh
namespace cuev::kernels {
    template <typename T> void gemv(...);
    template <typename T> void hh_reflect(...);
}

// solver.cu / public header
namespace cuev {
    template <typename T> void solve(...);
}
```

---

## Templates

- All numeric code is templated on `T` (float or double).
- Use `if constexpr (std::is_same_v<T, float>)` for type dispatch, not `sizeof(T) == 4`.
- Every `.cu` file ends with explicit instantiations via an `INSTANTIATE(T)` macro:

```cpp
#define INSTANTIATE(T)                                            \
    template void gemv_smem<T>(T, const T *, const T *, T, T *, int, int, cudaStream_t);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
```

---

## Braces

Always use `{}` for `if`, `for`, and `while` bodies — **one exception**: a guard-clause `if` that only returns may be written on one line:

```cpp
if (row >= M) return;          // OK — guard clause, return only
if (tid < s) { sr[tid] += ...; }  // braces required — not a return
for (int k = 0; k < K; ++k) {    // braces required
    acc += A[k] * x[k];
}
```

---

## Error Handling

- Wrap every CUDA API call with `CUDA_CHECK(...)` in host code.
- Wrap every cuBLAS call with `CUBLAS_CHECK(...)`.
- After kernel launches in non-hot debug paths: `CUDA_CHECK(cudaGetLastError())`.
- No error checking inside `__global__` kernels.

---

## Memory

- All matrices are **row-major** unless explicitly noted in the function doc.
- Every `cudaMalloc` in algorithm internals has a paired `cudaFree` in the same scope.

---

## Kernel Design Patterns

- **Reductions**: block-reduce with `__syncthreads`, final warp with `__shfl_down_sync(0xffffffff, val, N)`.
- Block size is always a compile-time constant (`constexpr` or template param).
- Prefer 1D `dim3 block(N)` with explicit index arithmetic over 2D blocks.
- Annotate kernels with fixed thread counts: `__launch_bounds__(NUM_THREADS)`.
- Vectorized 128-bit loads: `alignas(16) struct Vec128<T>` (defined in `gemm.cu`).

---

## Doxygen

Generate docs with:

```bash
doxygen Doxyfile
```

Output goes to `docs/html/`. Open `docs/html/index.html` in a browser.
The default theme is plain Doxygen HTML. A better theme (Doxygen Awesome) can
be wired in later.
