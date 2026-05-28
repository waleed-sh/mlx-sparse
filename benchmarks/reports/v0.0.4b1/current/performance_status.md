# v0.0.4b0 Current Performance Status

Date: 2026-05-28

Below is the native CPU baseline before v0.0.4b1 kernel optimization
work. The broad sparse-operation and native direct-solver sweeps use target
nonzeros per row rather than fixed densities, so increasing matrix dimension
keeps row-local sparse work meaningful. Explicit density sweeps remain available
through `--densities`, but the current baseline uses `target_nnz_per_row`.

The full benchmark pass was run serially such that benchmark scripts did not compete
with each other for CPU resources. The machine reported:

- CPU model: Apple M5
- MLX device: `Device(cpu, 0)` for native CPU benchmarks
- Native extension: available
- Hardware cores/threads: 10 / 10
- Configured sparse worker count: 10 default hardware-thread count
- Metal availability: true
- Accelerate availability: true

The reports use one timing iteration for the broad status pass. Treat these
numbers as a bottleneck map, not final statistically stable regression
thresholds. Every benchmark report includes SciPy as the primary CPU comparison
where SciPy exposes a comparable operation. Native Cholesky records mark SciPy
as `no_equivalent` because `scipy.sparse.linalg` does not ship a sparse
Cholesky factorization.

## Artifacts

- `native_cpu_sparse_ops_current.json`
- `native_cpu_sparse_ops_current.stdout`
- `native_cpu_direct_solvers_current.json`
- `native_cpu_direct_solvers_current.stdout`
- `accelerate_direct_solvers_current.json`
- `csr_matvec_current.txt`
- `csr_matmul_current.txt`
- `reductions_current.txt`
- `linalg_solvers_current.txt`
- `linalg_factorizations_current.txt`
- `linalg_spectral_current.txt`
- `readme_cg_cpu_current.txt`

## Native CPU Sparse Operations

Command coverage:

- Sizes: 128, 512, 2048, 8192, 32768
- Density mode: target nnz per row
- Target nnz per row: 2, 8, 32
- Uniform short-row nnz: 2, 8, 32
- Output-density sweep mode: target output nnz per row
- Target output nnz per row: 2, 8, 32
- Max density clamp: 0.25
- Sparse operand nnz cap: 1048576

Report totals:

- Records: 1710
- Timed records: 1674
- Skipped records: 36, all `fromdense` cases above the dense materialization
  cap

Median-of-medians, SciPy comparison, and slowest timed records by suite:

| suite | timed records | native median | SciPy median | native vs SciPy median | slowest native record |
|---|---:|---:|---:|---:|---|
| `fromdense` | 59 | 0.1951 ms | 0.5450 ms | 2.88x | 3.1751 ms, `imbalanced_rows_n2048_rnnz32` |
| `compressed` | 380 | 0.0931 ms | 0.0421 ms | 0.46x | 28.0694 ms, CSC `sort_indices`, `uniform_short_rows_n32768_r32` |
| `coo_conversion` | 190 | 0.1801 ms | 0.0556 ms | 0.44x | 37.2410 ms, COO-to-CSC, `uniform_short_rows_n32768_r32` |
| `spgemm` | 285 | 1.1045 ms | 0.6044 ms | 0.58x | 7187.1985 ms, COO exact-cancellation, `n32768` |
| `transpose_products` | 380 | 0.0334 ms | 0.0307 ms | 0.84x | 2.2231 ms, CSC transpose matmul, `diagonal_dominant_n32768_rnnz32` |
| `coo_csc_dense_products` | 380 | 0.0424 ms | 0.0388 ms | 0.76x | 3.5893 ms, COO matvec, `diagonal_dominant_n32768_rnnz32` |

Primary bottlenecks:

- Exact-cancellation SpGEMM remains the largest pathological case. At 32768,
  the CSR/COO/CSC products all return zero nnz but still take roughly
  6.3-7.2 s. SciPy is also slow on this case, roughly 3.2 s, but native is
  materially slower.
- Target-row and target-output SpGEMM exercise real large-row workloads.
  At 32768 and target 32 nnz per row, diagonal-dominant and output-density
  cases produce roughly 33-35 million output nnz and take roughly 575-705 ms
  for several formats. SciPy is faster on those cases, roughly 171-203 ms.
- COO-to-CSC conversion and compressed sorting became more visible under the
  corrected row-occupancy sweep. Large 32768 cases now reach tens of
  milliseconds, with SciPy generally ahead.
- Fixed-shape transpose and COO/CSC dense products remain small in absolute
  time, but SciPy is still ahead in median timing.

## Native Direct Solvers

Command coverage:

- Sizes: 64, 128, 256, 512
- Families: banded SPD, banded general, random SPD, random general
- Density mode: target nnz per row
- Target nnz per row: 4, 16
- Phases: factor-only, solve-only, factor-plus-solve

Report totals:

- Records: 144

Phase summary:

| solver | phase | records | native median | SciPy median | native vs SciPy median | slowest native record |
|---|---:|---:|---:|---:|---:|---|
| Cholesky | factor-only | 16 | 0.2558 ms | n/a | n/a | 86.5978 ms, random SPD 512, target nnz/row=16 |
| Cholesky | solve-only | 16 | 0.0445 ms | n/a | n/a | 0.3041 ms, random SPD 512, target nnz/row=16 |
| Cholesky | factor-plus-solve | 16 | 0.2753 ms | n/a | n/a | 86.8039 ms, random SPD 512, target nnz/row=16 |
| LU | factor-only | 32 | 0.6505 ms | 0.1078 ms | 0.23x | 816.4865 ms, random general 512, target nnz/row=16 |
| LU | solve-only | 32 | 0.0342 ms | 0.0083 ms | 0.18x | 0.2530 ms, random SPD 512, target nnz/row=16 |
| LU | factor-plus-solve | 32 | 0.7261 ms | 0.0909 ms | 0.20x | 830.6397 ms, random SPD 512, target nnz/row=16 |

Primary bottlenecks:

- Native LU factorization is far behind SciPy SuperLU on random 512x512
  target-row systems. The worst factor-plus-solve case is roughly 830.6 ms
  native vs 6.3 ms SciPy.
- Native Cholesky remains much cheaper than LU on the same random SPD family,
  but target 16 nnz/row at 512 still exposes an 86.8 ms factor-plus-solve
  path.
- Solve-only phases are comparatively small. Repeated-solve work should still
  focus on multi-RHS dispatch and reuse rather than rank-1 latency alone.

## Direct Solver Comparison Script

The Accelerate comparison benchmark ran on sizes 64 and 128 for CSR, CSC, and
COO inputs, pinned to CPU for MLX execution.

- Accelerate available: true
- Accelerate LU available: true
- Accelerate records: 54, median 0.0403 ms, max 5.3622 ms
- Native records: 36, median 0.0451 ms, max 0.1277 ms
- SciPy records: 36, median 0.0277 ms, max 0.5781 ms

These small matrices are not a meaningful Accelerate-vs-native conclusion, but
the comparison path is runnable and SciPy records are included.

## Fixed-Shape CSR Products

- CSR matvec, 4096x4096, nnz=16777, density=0.001:
  - Native: 0.0554 ms
  - SciPy: 0.0139 ms
  - Dense MLX: 1.0931 ms
  - Native vs SciPy: 0.25x
  - Effective sparse throughput: 303.0 million nnz/s
- CSR matmul, 2048x2048, rhs_cols=16, nnz=8389, density=0.002:
  - Native: 0.0664 ms
  - SciPy: 0.0172 ms
  - Dense MLX: 0.5069 ms
  - Native vs SciPy: 0.26x

These products are healthy relative to dense MLX baselines, but SciPy remains
substantially faster on the sampled CPU cases.

## Reductions

COO/CSC native reductions were much faster than legacy conversion paths on the
4096x4096, nnz=32768 CPU reduction benchmark. Against SciPy, the picture is
mixed:

- Native COO col sums, canonical col norms, diagonal, and trace are faster than
  SciPy, row sums are roughly tied.
- Native COO canonical row norms are slightly slower than SciPy.
- Native CSC col sums and canonical row norms are faster than SciPy, row sums,
  canonical col norms, diagonal, and trace are slower.

## Solver And Spectral Scripts

Iterative solvers at n=256, nnz=906:

- CG: 0.0380 ms, 0.84x vs SciPy CG, converged.
- GMRES: 0.1685 ms, 0.37x vs SciPy GMRES, converged.
- MINRES: 26.4807 ms, far slower than SciPy MINRES on this case, converged.

Direct factorization script at n=256, nnz=910:

- Cholesky factor-plus-solve style path: 0.9376 ms, 0.15x vs SciPy `spsolve`.
- LU: 0.6588 ms, 0.15x vs SciPy `splu(...).solve(...)`.
- `spsolve`: 17.3656 ms, 0.01x vs SciPy `spsolve`.

Spectral script at n=128:

- `eigsh`: 0.0596 ms, 8.29x faster than SciPy `eigsh`.
- `eigs`: 0.4962 ms, 2.43x faster than SciPy `eigs`.
- `svds`: 0.0520 ms, 12.36x faster than SciPy `svds`.

README CG CPU benchmark:

- 2-D Poisson, n=2304, nnz=11328
- mlx-sparse CG: 0.637 ms
- SciPy CG: 1.265 ms
- mlx-sparse was 1.99x faster
- residual: 1.01e-06


**Do not infer final regression thresholds from this first pass. Later before/after runs should use more iterations for the specific optimized family.**
