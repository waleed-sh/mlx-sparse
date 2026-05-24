Device selection and execution model
=====================================

mlx-sparse builds on MLX's device abstraction. Every MLX array belongs to a
stream on a device, and all operations that create new arrays run on the
current default device unless explicitly overridden with a stream argument.

Selecting the default device
-----------------------------

.. code-block:: python

   import mlx_sparse as ms

   ms.use_gpu()  # Apple Silicon GPU via Metal
   ms.use_cpu()  # CPU
   ms.use_device("gpu")  # same as use_gpu(). useful with argparse

These functions call ``mx.set_default_device`` and optionally probe the device
with a trivial evaluation to confirm it is available. The selected device
persists for the lifetime of the Python process or until changed by another
call.

.. note::

   Calling ``use_gpu()`` or ``use_cpu()`` after operations have already been
   dispatched does not retroactively move prior work. It only affects new
   operations.

Lazy execution
--------------

MLX uses a deferred execution model. Operations like ``A @ x`` or ``A.todense()``
do not compute anything immediately. They add nodes to a computation graph.
Computation runs when ``mx.eval()`` is called explicitly, or implicitly when a
value is read (for example via ``numpy.array(y)`` or ``print(y)``).

mlx-sparse follows this model for fixed-output numerical kernels:

* **Fixed-output operations are lazy.** ``csr_matvec``, ``csr_matmul``,
  ``todense``, ``T``, ``H``, transpose products, and autodiff primitives add
  nodes to the MLX graph and do not materialize values immediately.
* **Dynamic-output structural operations must discover output sizes.**
  ``fromdense()``, ``sum_duplicates()`` / ``canonicalize()``, and
  sparse-sparse ``matmat`` run native counting or symbolic work first, then
  synchronize compact counts or structure so final sparse buffers can be
  allocated.
* **Full validation (``validate="full"``) also reads values.** It must inspect
  ``indptr`` and ``indices`` to check bounds, so it calls ``mx.eval`` on those
  arrays. Keep this in mind when constructing from device arrays.
* **``to_numpy``** (used internally by fallback operations and full validation)
  always calls ``mx.eval``.

A graph composition example:

.. code-block:: python

   ms.use_gpu()

   y = A @ x  # lazy: one graph node
   z = mx.sin(y) + 2.0  # lazy: two more graph nodes
   mx.eval(z)  # GPU runs here. only one dispatch

This means you can build multi-step computations before triggering any GPU
work, letting MLX fuse and optimize the graph.

Which operations run on GPU
----------------------------

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Operation
     - CPU
     - Metal GPU
   * - ``csr_matvec`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``csr_matmul`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``coo_matvec`` / ``coo_matmul`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``csc_matvec`` / ``csc_matmul`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``coo_tocsr`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``csr_todense`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``csr_sort_indices`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``csr_transpose`` (all value dtypes, int32 and int64)
     - Yes
     - Yes
   * - ``csr_sum_duplicates`` / ``canonicalize``
     - Yes
     - Yes (staged count/prefix/fill, synchronizes row counts)
   * - ``fromdense``
     - Yes
     - Yes (staged count/prefix/fill, synchronizes row counts)
   * - Sparse-sparse ``CSR @ CSR``
     - Yes
     - Experimental via ``EXPERIMENTAL_METAL_SPGEMM``, host native path is
       default.
   * - Sparse-sparse ``COO @ COO`` / ``CSC @ CSC``
     - Yes
     - Not yet. Native host symbolic/numeric paths are used.
   * - Batched sparse-dense products for COO, CSR, and CSC
     - Yes
     - Yes
   * - Autodiff (JVP / VJP, sparse values and dense RHS)
     - Yes
     - Yes

When a GPU primitive encounters an unsupported configuration, it raises a
``RuntimeError`` with a clear message. Some public operations intentionally
lower to other native primitives on GPU, for example some non-``float32`` CSR
transpose products use ``csr_transpose`` followed by the ordinary product
rather than a direct complex or low-precision atomic scatter kernel. COO and
CSC scatter products keep native GPU coverage, ``float32`` uses atomic
scatter-add, while other value dtypes use native serial scatter where Metal
lacks compatible atomic adds.

Typical workflow: construct on CPU, multiply on GPU
----------------------------------------------------

The most common pattern for large-scale workloads is:

1. Assemble or canonicalize sparse structure once. Native staged constructors
   can run on CPU or GPU, but they may synchronize counts to allocate compact
   output buffers.
2. Keep the resulting CSR buffers and dense RHS arrays on the target device.
3. Run repeated COO/CSR/CSC matvec, matmul, and batched products on GPU.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   # Assembly phase: build and canonicalize once
   ms.use_cpu()
   coo = ms.coo_array((data, (row, col)), shape=(m, n))
   csr = coo.tocsr(canonical=True)
   mx.eval(csr.data, csr.indices, csr.indptr)  # materialise buffers

   # Compute phase: multiply on GPU
   ms.use_gpu()
   # Re-wrap the same buffers (already evaluated) into a new csr_array call.
   # No data is copied. MLX arrays are device-agnostic.
   csr_gpu = ms.csr_array(
       (csr.data, csr.indices, csr.indptr),
       shape=csr.shape,
       sorted_indices=csr.sorted_indices,
       canonical=csr.has_canonical_format,
       validate=False,  # buffers already validated
   )
   x = mx.array(np.random.randn(n).astype(np.float32))
   y = csr_gpu @ x  # dispatches Metal kernel
   mx.eval(y)

Stream safety
-------------

All native primitives pass MLX's ``StreamOrDevice`` parameter through to the
underlying operation wrappers and C++ primitive constructors. When the default
stream is used, MLX handles command sequencing automatically. Do not call
``mx.synchronize()`` or your own Metal synchronization inside a sparse
operation. This will deadlock with MLX's command encoder.
