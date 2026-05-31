Parallelism and performance
===========================

As of v0.0.4b1, mlx-sparse uses deliberate native CPU parallelism in the portable
backend.  The important word is deliberate: MLX's CPU stream scheduler can run
an operation, but it does not automatically split a sparse kernel's row,
column, or nonzero loop across CPU cores.  The native backend therefore uses
fixed-worker partitions only where the output ownership is clear or where a
measured private-accumulator design is used.

Use this section when you need to understand what runs in parallel, what still
synchronizes on the host, and how to time sparse arrays fairly.

Quick control examples
----------------------

The preferred public API is :mod:`mlx_sparse.runtime`.  It mirrors the
underlying configuration variables and keeps the synchronized
``MLX_SPARSE_*`` environment visible to native code.

.. code-block:: python

   import mlx_sparse as ms

   # Resolved package-wide worker count.
   print(ms.runtime.N_THREADS)

   # Force the serial native CPU path for a local comparison.
   with ms.runtime.context(n_threads=1, spgemm_parallel=False):
       serial = A @ B

   # Use four workers for package-wide CPU kernels and SpGEMM.
   with ms.runtime.context(
       n_threads=4,
       spgemm_parallel=True,
       spgemm_threads=4,
   ):
       parallel = A @ B

   print(ms.runtime.info())

The same controls can be set before Python starts:

.. code-block:: bash

   MLX_SPARSE_CPU_THREADS=4 python run_my_case.py
   MLX_SPARSE_SPGEMM_THREADS=4 MLX_SPARSE_SPGEMM_PARALLEL=1 python run_spgemm.py
   MLX_SPARSE_SPGEMM_THREADS=1 python run_serial_spgemm.py
   MLX_SPARSE_SOLVER_PARALLEL=0 python run_solvers.py

For the option table and API details, see :doc:`/api/runtime` and
:doc:`/api/configuration`.

How to benchmark sparse arrays fairly
-------------------------------------

Sparse operations are not all evaluated at the same time:

* Fixed-shape sparse-dense primitives are lazy MLX nodes.  Time the evaluated
  result, not only Python graph construction.
* Dynamic-output sparse-sparse products and constructors must discover output
  structure.  Their host assembly has real synchronization points.
* Sparse containers have several buffers.  Force every structural buffer,
  not only ``data``.

For a dense result, evaluate the array directly:

.. code-block:: python

   y = A @ x
   mx.eval(y)

For a sparse result, evaluate every buffer that defines the container:

.. code-block:: python

   C = A @ B

   if hasattr(C, "indptr"):
       mx.eval(C.data, C.indices, C.indptr)  # CSR or CSC
   else:
       mx.eval(C.data, C.row, C.col)         # COO

The v0.0.4b1 benchmark helpers use this rule so sparse dynamic work is compared
against evaluated dense work instead of against unevaluated MLX graph
construction.

Execution profiles at a glance
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 24 28 24

   * - Category
     - Evaluation shape
     - Main synchronization point
     - CPU parallel behavior as of v0.0.4b1
   * - Fixed-shape sparse-dense primitives
     - Lazy MLX primitive with known output shape
     - Evaluation of the output dense array
     - Row, column, batch-slab, nonzero, or private-accumulator partitions
       where measured and race-free.
   * - Dynamic sparse-sparse products
     - Eager host assembly for native CSR/COO/CSC SpGEMM
     - Input-buffer evaluation plus output-structure discovery
     - Same-format CSR/COO/CSC SpGEMM uses fixed-worker output-row or
       output-column ownership.
   * - Constructors and canonicalization
     - Dynamic sparse output
     - Counts, prefixes, and fills for output structure
     - CPU ``fromdense`` has immediate host assembly. Staged helpers use
       row/column/segment partitions where useful.
   * - Explicit native direct factorizations
     - Immediate host routines returning CSR factors
     - Factor construction and factor buffer materialization
     - Cholesky/LU storage was optimized, but natural-order factorization is
       still dependency-bound and not internally threaded.
   * - Repeated explicit-factor solves
     - Immediate/native solve calls
     - Triangular solve and permutation evaluation
     - Matrix RHS avoids Python column loops. Production row-order triangular
       solve stays serial unless future measured level scheduling wins.
   * - Accelerate-backed routines
     - Opaque Apple framework calls in Accelerate-enabled builds
     - Framework call boundaries
     - Controlled by build capability, not by mlx-sparse CPU worker settings.

Detailed pages
--------------

.. toctree::
   :maxdepth: 2

   execution_model
   performance_results
