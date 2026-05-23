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
  dispatch; the kernels still see contiguous batches, rows, and RHS columns.
* As with the rank-1/rank-2 kernels, short rows use scalar output kernels and
  long rows use threadgroup reductions.

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

**CPU backends**

The CPU backends use MLX's command encoder dispatch model. They are correct
and readable but not hand-tuned (**yet**):

* No SIMD intrinsics.
* No parallelism over rows (serial dispatch).
* ``float16`` and ``bfloat16`` use ``float32`` accumulators.
* ``complex64`` uses standard ``complex64`` arithmetic.

For reference: a hand-tuned MKL or BLAS SpMV on a 12-core M-series chip will
typically be faster than this CPU backend at the same problem size.
The MLX CPU backend is a correctness baseline, not a performance product. Use
the Metal path for CPU+GPU comparison.

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
