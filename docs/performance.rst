Performance guide
==================

This page explains when sparse is faster than dense, how to measure it, and
what to expect from the current kernels.

When sparse beats dense
-----------------------

Sparse-dense products are only faster than dense matmul when the sparsity is
high enough that the savings in arithmetic and memory access outweigh the
overhead of irregular memory access (indirect indexing). On a GPU, memory
latency rather than arithmetic throughput is the dominant cost.

Rough guidance for ``csr_matvec`` on Metal:

* **< 1% density**: sparse is almost always faster for matrices larger than
  approx. 512 x 512. Fewer than 10 non-zeros per row means very short inner loops,
  excellent branch prediction, and minimal memory traffic.
* **1%–5% density**: depends on matrix size, row length uniformity, and
  reuse. Benchmark before committing to sparse.
* **> 10% density**: dense MLX matmul typically wins. The CSR kernel's
  irregular memory access pattern does not amortize at high densities.

For ``csr_matmul``, the crossover density is lower when the RHS has many
columns. The implementation uses scalar output-element kernels for short rows
and vector-reduction kernels for long rows. Dense MLX can still win once the
sparse row is wide enough that irregular memory traffic dominates.

The break-even point also depends on whether the sparse structure is reused:
if you multiply the same matrix by many different vectors, the per-assembly
cost is amortized over all products.

Running the benchmarks
-----------------------

Two benchmark scripts ship in the ``benchmarks/`` directory.

``bench_csr_matvec.py``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # GPU, default shape (32768x32768), density 0.01%
   python benchmarks/bench_csr_matvec.py

   # Custom shape and density
   python benchmarks/bench_csr_matvec.py --rows 8192 --cols 8192 --density 0.001

   # CPU comparison
   python benchmarks/bench_csr_matvec.py --device cpu

   # Complex dtype
   python benchmarks/bench_csr_matvec.py --complex

Output example:

.. code-block:: text

   {'backend': 'gpu', 'shape': (32768, 32768), 'nnz': 10737,
    'density': 0.0001, 'dtype': "<class 'numpy.float32'>",
    'csr_matvec_ms': 0.412, 'dense_matvec_ms': 15.3,
    'effective_nnz_per_s': 2.6e10}

``bench_csr_matmul.py``
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 16-column dense RHS on GPU
   python benchmarks/bench_csr_matmul.py --rhs-cols 16

   # Wider RHS
   python benchmarks/bench_csr_matmul.py --rhs-cols 64 --density 0.001

   # Complex dtype
   python benchmarks/bench_csr_matmul.py --complex --rhs-cols 8

Current kernel characteristics
--------------------------------

Terminology
~~~~~~~~~~~

The kernel notes below use a few implementation terms:

* **Atomic add**: many GPU threads can contribute to the same output element,
  so the kernel uses a hardware atomic operation to make each addition safe.
  This is fast for ``float32`` transpose products and avoids a separate
  reduction pass.
* **Scatter**: each thread writes a contribution to the output position implied
  by the sparse column index. Scatter is natural for transpose products
  because CSR stores rows but ``A.T`` writes by columns.
* **Segmented reduction**: contributions are first grouped by output column,
  then each column segment is reduced independently. This avoids value atomics
  for dtypes where Metal does not provide the right atomic operation.
* **Symbolic pass**: compute the output sparsity pattern or row counts without
  writing final values. CSR x CSR uses this to know how much output storage to
  allocate.
* **Prefix sum**: convert per-row or per-column counts into an ``indptr`` array
  and stable write offsets.
* **Numeric pass**: fill the already-allocated output ``data`` and ``indices``
  using the pattern discovered by the symbolic pass.

**Metal csr_matvec**

* Short rows use a scalar row kernel: one thread reads the indptr slice for its
  row and iterates over stored values.
* Long rows use a threadgroup-per-row reduction kernel, selected from the known
  ``nnz / n_rows`` ratio without host synchronization.
* ``float16`` and ``bfloat16`` accumulate in ``float32`` and cast back to the
  storage dtype.
* Output is written once at the end. No atomic operations are needed.

**Metal csr_matmul**

* Short rows use a scalar output-element kernel.
* Long rows use a threadgroup reduction per output element.
* Better than dense when density is low enough that the sparse row fits in
  cache across multiple threads.

**Metal csr_batched_matvec / csr_batched_matmul**

* Batched dense right-hand sides use dedicated native kernels rather than
  flattening through Python.
* The public helpers accept leading batch dimensions and flatten them only for
  dispatch, the kernels still see contiguous batches, rows, and RHS columns.
* As with the rank-1/rank-2 kernels, short rows use scalar output kernels and
  long rows use threadgroup reductions.

**Metal COO dense products**

* COO products use explicit ``(row, col)`` coordinates and scatter each stored
  contribution into the dense output.

* ``float32`` matvec/matmul and batched products use ``atomic_float`` updates.
  This gives high parallelism over nonzeros and RHS columns without converting
  the structure.

* ``float16``, ``bfloat16``, and ``complex64`` stay native but use a serial
  scatter kernel on GPU because Metal does not provide compatible atomic add
  operations for those storage types.

**Metal CSC dense products**

* Forward CSC products walk compressed columns and scatter into output rows.
  ``float32`` uses atomic scatter-add, other value dtypes use native serial
  scatter for correctness.

* CSC transpose products are reductions over compressed columns, so they do
  not need output atomics. Matrix transpose products compute one
  ``(column, rhs_column)`` dot product per output element.

**Metal COO / CSC reductions**

* COO row/column sums, diagonal extraction, and trace operate directly on
  explicit coordinates. ``float32`` scatter reductions use ``atomic_float``,
  low-precision and complex sum scatters lower through native compressed
  conversion plus storage-aligned reductions where Metal lacks compatible
  storage atomics.

* COO row/column norms and CSC row norms accumulate squared magnitudes into
  ``float32`` outputs. ``float32`` inputs can use direct atomic accumulation.
  Non-``float32`` canonical norms lower through native COO/CSC-to-compressed
  conversion plus storage-aligned CSR/CSC norm reductions, avoiding a
  scatter-heavy path when Metal lacks storage-compatible atomics. Public norm
  methods canonicalize non-canonical COO/CSC inputs first so duplicates are
  summed before squaring.

* CSC column sums and column norms are storage-aligned. They use scalar
  per-column kernels for short columns and threadgroup vector reductions for
  long columns, matching the CSR row-reduction strategy but along the CSC
  compressed dimension.

* CSR/CSC diagonal extraction uses one thread per diagonal slot for short
  segments and a threadgroup-per-slot vector reduction for long compressed
  rows or columns.

* CSR, CSC, and COO trace use the fixed 128-lane reduction for small inputs.
  Large traces use a staged Metal reduction: independent threadgroups write
  accumulator-typed partials, then a final threadgroup reduces those partials
  to the scalar output. ``float16`` and ``bfloat16`` partials are stored as
  ``float32`` until the final cast.

If the native extension is unavailable, public reduction helpers fall back to
the NumPy-backed ``mlx_sparse._fallback`` implementations. That is an extension
availability fallback, not a dtype-specific path in normal wheels.

**Metal csr_transpose**

* The CPU path uses a counting transpose: count destination-row sizes, prefix
  sum to build ``out_indptr``, then fill outputs in source-row order.

* The Metal path counts destination rows in parallel, builds ``out_indptr``,
  then fills each destination row deterministically. This preserves sorted row
  indices in the transposed CSR output.

**Metal transpose matvec and transpose matmul**

* ``float32`` transpose matvec uses a parallel scatter-add kernel with
  ``atomic_float`` output updates.

* ``float32`` transpose matmul uses one thread per source row and RHS column,
  again with atomic adds into the transposed output.

* Non-``float32`` transpose products lower through native ``csr_transpose``
  followed by native ``csr_matvec`` / ``csr_matmul``. This is still entirely
  native C++/Metal and avoids NumPy. It is intentional because Metal does not
  expose a general complex or low-precision atomic add with the semantics
  needed for a direct scatter kernel.

**CSR x CSR**

* Native sparse-sparse multiplication uses a symbolic pass to determine each
  output row's column set, a prefix sum to allocate compact output buffers, and
  a numeric pass to accumulate values.

* The default implementation performs the structural assembly on the host
  because the output size is data-dependent. A staged Metal implementation is
  available behind ``ms.config.EXPERIMENTAL_METAL_SPGEMM`` for experimentation,
  but the optimized host path remains the default on the current benchmark set.

**COO x COO and CSC x CSC**

.. important::

   Sparse-sparse multiplication in this release is correctness- and
   infrastructure-focused, not yet the primary performance surface of
   ``mlx-sparse``. The native host paths are the default because they are
   currently the fastest implementation for the small and medium products in
   the test/benchmark set. Staged Metal SpGEMM paths are experimental and are
   intended to validate format-specific symbolic/numeric/prune pipelines before
   later GPU tuning work such as group-level hashing, merge-based row/column
   kernels, and lower-synchronization allocation strategies.

* COO sparse-sparse multiplication groups explicit coordinate rows, performs a
  symbolic count for each output row, allocates compact coordinate buffers, then
  fills sorted ``(row, col)`` entries and prunes exact zero cancellations.

* The default COO implementation remains the optimized native host path. An
  experimental staged Metal COO path is available behind
  ``ms.config.EXPERIMENTAL_METAL_SPGEMM``. It row-buckets the explicit COO
  coordinates as a scheduling structure, then runs COO-specific symbolic,
  numeric-fill, and prune kernels that return canonical COO output. It does not
  call CSR sparse-sparse multiplication.

* CSC sparse-sparse multiplication is column-native: each output column walks
  the right-hand compressed column and gathers matching left-hand compressed
  columns. The result is canonical CSC with sorted row indices per column.

* The default CSC implementation remains the optimized native host path. An
  experimental staged Metal CSC path is available behind
  ``ms.config.EXPERIMENTAL_METAL_SPGEMM``. It runs CSC-specific symbolic,
  numeric-fill, and prune kernels over compressed output columns. It does not
  call CSR sparse-sparse multiplication.

**svds normal-operator Lanczos**

* ``linalg.svds`` applies Lanczos to the normal operator ``A.T @ A`` without
  materializing ``A.T @ A``.

* The CSR path uses a dedicated fused normal-operator step. For each source
  row, it computes the row contribution to ``A @ v`` and immediately scatters
  that contribution into the ``A.T @ (...)`` workspace. This removes the
  previous native-host decomposition into two separate SpMVs per Lanczos step
  and avoids host materialization of the intermediate ``A @ v`` vector.

* On Metal, the Lanczos recurrence runs in a native fused kernel. The small
  tridiagonal eigensolve, Ritz back-transformation, and final singular-vector
  assembly still synchronize the Lanczos basis back to CPU.

**Accelerate direct solves**

* Accelerate-enabled Apple builds can route ``linalg.spsolve`` and
  ``linalg.factorized`` through opaque Accelerate ``float32`` direct
  factorizations for supported CPU cases. Native explicit-factor APIs remain
  available as the baseline because they return mlx-sparse CSR factors.

* ``benchmarks/bench_accelerate_direct_solvers.py`` compares native
  Cholesky/LU factor-and-solve and solve-only timings against Accelerate
  opaque-factor timings across CSR, CSC, and COO inputs. Run it once in a
  normal build and once after rebuilding with
  ``CMAKE_ARGS="-DMLX_SPARSE_ENABLE_ACCELERATE=ON"``.

**CPU backends**

The CPU backends use MLX's command encoder dispatch model. Fixed-shape
CSR sparse-dense products now have native CPU tuning in the row-owned path,
while other CPU kernels are still being optimized incrementally:

* ``csr_matvec`` uses a short-row serial fast path and fixed-worker row
  partitioning when ``MLX_SPARSE_CPU_THREADS`` resolves above one.
* ``csr_matmul`` and ``csr_batched_matmul`` specialize common RHS widths
  ``1``, ``2``, ``4``, ``8``, and ``16`` with stack accumulators to reduce
  per-row temporary traffic.
* ``csr_batched_matvec`` uses fixed-worker batch-row partitions when more than
  one CPU worker is configured.
* ``coo_batched_matmul`` and ``csc_batched_matmul`` use fixed-worker
  batch-owned partitions on CPU.  Batched matvec wrappers use the same native
  path with a single RHS column.
* Storage-aligned CSR row reductions, CSR diagonal extraction, CSR dense
  conversion, CSC column reductions, CSC diagonal extraction, and CSC dense
  conversion use fixed-worker row/column partitions when
  ``MLX_SPARSE_CPU_THREADS`` resolves above one.
* ``fromdense`` uses an immediate CPU host assembly path that scans rows once
  into canonical CSR buffers when the selected stream is CPU.  GPU streams keep
  the staged count/prefix/fill implementation.
* Compressed CSR/CSC ``sort_indices`` and ``sum_duplicates`` plus sparse-value
  VJP kernels use fixed-worker row/column/entry partitions where each worker
  owns disjoint output entries.  ``sum_duplicates`` stays on the staged
  count/prefix/fill path because measured immediate host assembly did not beat
  it on the CPU benchmark sweep.
* CSR-to-CSC, CSC-to-CSR, COO-to-CSR, COO-to-CSC, and CSR transpose CPU
  structural paths use histogram/prefix/fill style assembly with private
  per-worker histograms or private write offsets for the parallel fill.  They
  do not share mutable destination counters between workers.
* Non-batched COO/CSC forward dense products remain serial on CPU because they
  scatter into shared dense output rows.  Parallelizing those paths requires a
  measured race-free design such as output ownership or private accumulators;
  unsynchronized scatter writes are not used.
* Axis-mismatched compressed reductions and transpose products are treated as
  scatter-style kernels.  CSR transpose products and axis-mismatched COO/CSC
  reductions use private per-worker accumulators with a deterministic final
  reduction, while CSC transpose products use output-column ownership.
* Native explicit-factor solves accept matrix right-hand sides on CPU.  The
  serial path uses a row-major sparse triangular solve with multiple dense RHS
  columns so each sparse factor row is scanned once per triangular solve rather
  than once per Python-sliced RHS column.  Solver parallelism is controlled
  separately with ``MLX_SPARSE_SOLVER_PARALLEL`` and
  ``MLX_SPARSE_SOLVER_THREADS``; it is not enabled by ``SPGEMM_THREADS``.
* CSR triangular-solve structural analysis is available internally for
  benchmark and development work: diagonal positions can be precomputed, and a
  dependency-level schedule is emitted only when the factor graph has level
  width greater than one.  v0.0.4b1 keeps production explicit-factor solves on
  the row-order path because the analyzed path did not beat the serial
  regression target in the measured CPU sweep.  Cholesky solves still reuse the
  transposed upper factor after the first solve without changing the public
  factor object constructor.
* The worker count is the configured runtime value. It is not changed
  heuristically from matrix shape, density, or nnz.
* No architecture-specific SIMD intrinsics are required in the default build.
* ``float16`` and ``bfloat16`` use ``float32`` accumulators.
* ``complex64`` uses standard ``complex64`` arithmetic.

For reference: hand-tuned Sparse BLAS libraries may still be faster at the
same problem size, especially on small sparse-dense products where fixed
worker launch overhead dominates. Set ``MLX_SPARSE_CPU_THREADS=1`` when
measuring the serial native CPU path.

Measuring effective bandwidth
------------------------------

For ``csr_matvec``, the effective bandwidth is approximately:

.. math::

   \text{BW} = \frac{(\text{nnz} \times (\text{sizeof}(\text{data}) + \text{sizeof}(\text{col})) + n_{\text{rows}} \times \text{sizeof}(\text{indptr}))}{t}

For a float32/int32 matrix:

* Each non-zero reads 4 bytes of data + 4 bytes of index = 8 bytes.
* Each row reads 1 indptr entry = 4 bytes.

The benchmarks report ``effective_nnz_per_s`` as ``nnz / time_in_seconds``.
To convert to effective bandwidth, multiply by 8 bytes per non-zero.

The CSR kernels do not achieve peak bandwidth because of
random access into the dense vector ``x``. Cache utilization improves when
``nnz/n_cols`` is high.

Row imbalance effects
----------------------

CSR row imbalance is still important. Scalar row kernels are excellent for
short, balanced rows, while long-row reductions help heavy rows but add
synchronization overhead. For balanced matrices (e.g. the 2D Laplacian,
k-regular graphs), scalar kernels can still be the faster path.

To quantify imbalance, compute the coefficient of variation of row lengths:

.. code-block:: python

   import numpy as np
   import mlx.core as mx

   mx.eval(csr.indptr)
   indptr_np = np.array(csr.indptr)
   row_lengths = np.diff(indptr_np)
   cv = row_lengths.std() / (row_lengths.mean() + 1e-12)
   print(f"CV of row lengths: {cv:.2f}")
   # < 0.5: balanced, scalar kernels usually work well
   # > 1.0: imbalanced, benchmark vector-reduction behavior

Benchmarking advice
--------------------

Follow MLX's benchmark convention: warmup first, then timed iterations with
``mx.eval`` inside the loop.

.. code-block:: python

   import time
   import mlx.core as mx

   def bench(fn, warmup=5, iters=50):
       for _ in range(warmup):
           mx.eval(fn())
       t0 = time.perf_counter()
       for _ in range(iters):
           mx.eval(fn())
       return 1000 * (time.perf_counter() - t0) / iters

   ms_sparse = bench(lambda: csr @ x)
   ms_dense  = bench(lambda: dense @ x)
   print(f"sparse: {ms_sparse:.3f} ms,  dense: {ms_dense:.3f} ms")

Do not call ``mx.eval`` outside the timed loop between iterations. This would
flush the stream and exclude kernel launch overhead from the measurement. The
``mx.eval`` inside the loop is essential because MLX is lazy. Without it you
are measuring graph construction cost, not compute cost.
