Multi-threaded execution model
==============================

This page describes the native CPU execution model as of mlx-sparse v0.0.4b1.
It is intentionally specific.  Some sparse kernels are
parallel, some are faster but still serial, and some candidate parallel paths
were measured and rejected.

Core model
----------

MLX CPU primitives run through ``mx::cpu::CommandEncoder::dispatch``.  That
dispatch schedules the primitive body on MLX's stream machinery.  It does not,
by itself, shard a sparse row loop, column loop, nonzero loop, or solver
recurrence over CPU cores.  mlx-sparse adds its own native worker partitions
inside selected CPU kernels.

As of v0.0.4b1, the rules are:

* Worker tasks use already-materialized raw input and output pointers.
* Worker tasks use local scratch or private destination ranges.
* Worker tasks join before the native host operation returns.
* Worker tasks do not enqueue nested MLX operations.
* Fixed-worker kernels use the configured worker count.  They do not silently
  choose a different count from matrix size, density, or output density.
* ``CPU_THREADS=1`` remains the first-class serial regression target.

That fixed-worker policy makes behavior predictable.  It also means tiny
matrices can slow down under multi-worker settings because thread launch and
join overhead can exceed useful sparse work.

Runtime controls
----------------

Package-wide CPU workers
~~~~~~~~~~~~~~~~~~~~~~~~

``MLX_SPARSE_CPU_THREADS`` controls the package-wide native CPU worker budget.
The compatibility alias ``MLX_SPARSE_N_THREADS`` is also read at import time,
but ``MLX_SPARSE_CPU_THREADS`` is the canonical synchronized variable.

In Python:

.. code-block:: python

   import mlx_sparse as ms

   ms.runtime.N_THREADS = 4
   with ms.runtime.context(n_threads=1):
       serial_y = A @ x

``N_THREADS`` accepts a positive integer or ``"auto"``.  In ``"auto"`` mode,
runtime resolution checks, in order:

#. explicit package configuration,
#. ``OMP_NUM_THREADS``,
#. common scheduler variables such as ``SLURM_CPUS_PER_TASK``, ``PBS_NP``,
   ``LSB_DJOB_NUMPROC``, and ``NSLOTS``,
#. process affinity where the platform exposes it,
#. hardware concurrency,
#. one worker as the final fallback.

Use ``ms.runtime.resolve_n_threads()`` to inspect the resolved count and
source, or ``ms.runtime.info()`` for a full report-friendly dictionary.

SpGEMM workers
~~~~~~~~~~~~~~

Sparse-sparse products have separate controls because they are dynamic-output
host assembly kernels and often need a different budget from fixed-shape
sparse-dense primitives.

.. list-table::
   :header-rows: 1

   * - Control
     - Default
     - Meaning
   * - ``MLX_SPARSE_SPGEMM_PARALLEL``
     - ``True``
     - Enables native CPU same-format CSR/COO/CSC SpGEMM parallel paths.
   * - ``MLX_SPARSE_SPGEMM_THREADS``
     - ``"inherit"``
     - Uses ``CPU_THREADS`` by default, or a positive integer / ``"auto"``
       when set explicitly.

In Python:

.. code-block:: python

   ms.runtime.SPGEMM_PARALLEL = True
   ms.runtime.SPGEMM_THREADS = 4

   with ms.runtime.context(spgemm_parallel=False):
       serial_C = A @ B

``SPGEMM_THREADS=1`` and ``SPGEMM_PARALLEL=False`` both force the serial
Gustavson/SPA host path.

Solver workers
~~~~~~~~~~~~~~

Solver-family controls are separate from both package-wide CPU workers and
SpGEMM workers.

.. list-table::
   :header-rows: 1

   * - Control
     - Default
     - Meaning
   * - ``MLX_SPARSE_SOLVER_PARALLEL``
     - ``False``
     - Enables solver parallel routines only when a production solver path
       elects to use them.
   * - ``MLX_SPARSE_SOLVER_THREADS``
     - ``"inherit"``
     - Uses ``CPU_THREADS`` by default, or a positive integer / ``"auto"``
       when set explicitly.

The v0.0.4b1 work measured solver helper parallelism, cached triangular diagonal
positions, and triangular level scheduling.  Those experiments remain useful
for development and benchmarking, but production iterative/spectral solvers
and explicit-factor solves keep the measured-safe row-order/serial helper
paths because the parallel helper variants regressed the benchmark slice.

Experimental Metal SpGEMM
~~~~~~~~~~~~~~~~~~~~~~~~~

``MLX_SPARSE_EXPERIMENTAL_METAL_SPGEMM`` controls experimental staged Metal
same-format CSR/COO/CSC sparse-sparse products.  It is separate from the
native CPU worker controls.  ``MLX_SPARSE_FORCE_EXPERIMENTAL_METAL_SPGEMM``
can force the option from the environment.

Laziness and host synchronization
---------------------------------

Fixed-shape sparse-dense primitives are lazy MLX nodes.  For example,
``csr @ dense`` knows its dense output shape before evaluation.  Constructing
the Python result is not the same as doing the sparse work.  The sparse loop
runs when the output is evaluated with ``mx.eval`` or consumed by later MLX
work.

Dynamic-output operations are different.  Sparse-sparse products,
``fromdense``, format conversion, and canonicalization need to discover output
structure.  That requires counts, prefixes, host assembly, or structural
buffer materialization.  Those synchronization points are real and are part of
the operation being optimized.

When timing:

* Dense outputs must be evaluated with ``mx.eval(result)``.
* CSR/CSC outputs must evaluate ``data``, ``indices``, and ``indptr``.
* COO outputs must evaluate ``data``, ``row``, and ``col``.
* Explicit factor objects must evaluate the factor buffers they contain.

Primitive execution details
---------------------------

Same-format CSR/COO/CSC SpGEMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Native CPU ``CSRArray @ CSRArray``, ``COOArray @ COOArray``, and
``CSCArray @ CSCArray`` use host SpGEMM assembly when the experimental Metal
path is disabled or unavailable.

The CPU algorithm used as of v0.0.4b1 is a one-pass Gustavson/SPA
final-writer design:

* CSR and COO products are organized by output row.
* CSC products are organized by output column.
* Each worker owns a deterministic row or column range.
* Imbalanced sparse cases split ranges by cumulative scalar-product work
  estimates instead of by equal row or column count.
* Each worker reuses private marker, accumulator, and touched-index scratch.
* Rows/columns write only final nonzero entries after accumulation.
* Exact zero cancellations are pruned before final structure is stitched.
* The final CSR/COO/CSC output is deterministic and canonical.

No worker writes into another worker's row or column.  No atomic update is
needed for the output structure.

CSR sparse-dense products
~~~~~~~~~~~~~~~~~~~~~~~~~

The CSR fixed-shape sparse-dense paths are row-owned:

* ``csr_matvec`` owns one output row per partitioned row range.
* ``csr_matmul`` owns one output row and all dense RHS columns for that row.
* ``csr_batched_matvec`` and ``csr_batched_matmul`` own batch-row ranges.
* ``csr_matmul`` and ``csr_batched_matmul`` specialize common RHS widths
  ``1``, ``2``, ``4``, ``8``, and ``16`` with stack accumulators.
* Short-row serial paths remain important when row work is too small for
  worker overhead to pay off.

These kernels use ``CPU_THREADS`` rather than the SpGEMM-specific controls.
Because rows are disjoint, the output is race-free without atomics.

COO and CSC dense products
~~~~~~~~~~~~~~~~~~~~~~~~~~

COO and CSC non-batched forward dense products scatter into shared dense
output rows.  Naively parallelizing those loops would let multiple input
nonzeros update the same dense element.  As of v0.0.4b1, those non-batched
forward products remain measured serial fallbacks.

Batched COO/CSC dense products are different: each batch output slab is
disjoint.  The accepted CPU path partitions complete batch ranges across
``CPU_THREADS`` workers, so each worker owns the full output slab for its
assigned batch.

CSC storage-aligned and transpose products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSC column-aligned reductions and dense conversion use output-column
ownership.  For transpose products where the output column can be owned by one
worker, the native CPU path uses deterministic column partitions.
Scatter-style cases that cannot be expressed as simple output-column ownership
use private accumulators or remain serial.

Storage-aligned reductions and dense conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following kernels have disjoint-output partitions:

* CSR row sums,
* CSR row norms,
* CSR diagonal extraction,
* CSR dense conversion,
* CSC column sums,
* CSC column norms,
* CSC diagonal extraction,
* CSC dense conversion.

CSR row-owned work writes row-owned output entries.  CSC column-owned work
writes column-owned output entries.  Dense conversion writes disjoint dense
rows or columns depending on the storage format and target ownership.

Axis-mismatched reductions and transpose products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Axis-mismatched reductions can be scatter-style even when the sparse format is
compressed.  For example, many CSR rows can contribute to the same output
column.  As of v0.0.4b1, these are treated separately:

* CSR transpose products use deterministic private worker accumulators.
* Axis-mismatched COO/CSC reductions use private accumulators followed by a
  deterministic final reduction.
* CSC transpose products use output-column ownership where possible.
* Unsynchronized writes into shared dense output are not used.

Format conversion and compressed transpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Format conversion uses histogram, prefix, and scatter style assembly:

* CSR-to-CSC and CSC-to-CSR use per-worker histograms and private write
  offsets rather than shared mutable ``next[col]`` counters.
* COO-to-CSR and COO-to-CSC use count/prefix/fill assembly with private
  worker offsets in the parallel fill.
* CSR transpose uses the same destination-owned or histogram/scatter design.
* The remaining compressed transpose/conversion paths were audited before
  parallelization so no shared destination counter is updated concurrently.

This is the same shape as standard sparse format-conversion implementations:
first count destination segment sizes, prefix the counts to get disjoint
output ranges, then scatter into private offsets inside those ranges.

Constructors and canonicalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fromdense`` is dynamic-output because the number of nonzeros depends on the
dense input values and threshold.  On CPU, the v0.0.4b1 path has an immediate
host assembly fast path that scans dense rows directly into canonical CSR
buffers.
GPU streams keep the staged count/prefix/fill implementation.

Compressed ``sort_indices`` and ``sum_duplicates`` use segment partitions.
Each worker owns a set of compressed rows or columns.  Immediate host assembly
was measured for compressed ``sum_duplicates`` and rejected because it did not
improve over the staged count/prefix/fill path on the ``CPU_THREADS=1``
regression target.

Sparse-value VJP kernels
~~~~~~~~~~~~~~~~~~~~~~~~

The sparse-dense data-gradient VJP kernels write one output value per stored
sparse nonzero:

* ``csr_matvec_data_vjp``,
* ``csr_matmul_data_vjp``,
* ``coo_matmul_data_vjp``,
* ``csc_matmul_data_vjp``.

Those outputs are naturally independent.  As of v0.0.4b1, the native CPU path
partitions the sparse value range or the owning row/column segments so one
worker owns each output sparse value.

Scalar trace and sparse dot/vdot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CSR, COO, and CSC trace plus CSR sparse ``dot`` and ``vdot`` use thresholded
deterministic tree reductions:

* Small inputs use the serial scalar path.
* Larger inputs split row or nonzero ranges across ``CPU_THREADS`` workers.
* Each worker writes a local partial.
* The final result is reduced in a stable order.

Low-precision and complex semantics are preserved.  ``float16`` and
``bfloat16`` paths continue to use the existing ``float32`` accumulation
semantics where required, and complex ``dot``/``vdot`` keep their conjugation
and complex accumulation rules.  As of v0.0.4b1, the implementation does not
add architecture-specific SIMD intrinsics for these reductions.  It relies on
cache-friendly loop layout and compiler auto-vectorization.

Solver execution details
------------------------

Native explicit Cholesky
~~~~~~~~~~~~~~~~~~~~~~~~

``csr_cholesky`` is an immediate native CPU host routine returning explicit
CSR factors.  It stays natural-order and serial because the left-looking
factorization has row dependencies.  The v0.0.4b1 work optimizes storage and
update mechanics instead of changing the algorithm family:

* map-heavy rows were replaced by sorted sparse rows,
* reusable dense marker/work arrays reduce allocation churn,
* upper-only and lower-only symmetric input handling is preserved,
* error behavior and factor semantics stay compatible with existing tests.

No fill-reducing ordering, elimination-tree scheduling, supernodal
factorization, or public factor dataclass change is introduced.

Native explicit LU
~~~~~~~~~~~~~~~~~~

``csr_lu`` is also an immediate native CPU host routine returning explicit CSR
factors.  It keeps the existing natural-order partial pivoting behavior.  The
v0.0.4b1 storage optimization replaces per-pivot map churn with sorted sparse
row vectors and explicit lookup/update/erase helpers.

LU remains dependency-bound.  No fill-reducing ordering is applied in this
release line, and no new public solver family is added.

Repeated explicit-factor solves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repeated solves were optimized by changing the RHS shape handled by native
code:

* CSR triangular solve accepts rank-2 dense RHS matrices.
* Row order is preserved.
* Multiple RHS columns are handled inside one native call sequence.
* LU permutation accepts matrix RHS inputs instead of one Python/native call
  per RHS column.
* Cholesky solves cache the transposed upper factor after the first solve
  without changing the public factor dataclass constructor.

Cached diagonal positions and dependency-level scheduling were evaluated.
They remain as benchmark/development primitives, but production solves keep
the row-order path because the analyzed path regressed most measured cases.

Iterative and spectral solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CG, MINRES, GMRES, Arnoldi, Lanczos, ``eigs``, ``eigsh``, and ``svds`` are
secondary native CPU targets as of v0.0.4b1.  They continue to use the same
public algorithms and APIs.  The release-validation work measured host CSR SpMV reuse,
deterministic chunked dot/norm reductions, and fixed-worker helper
parallelism, but did not ship those helper changes in production because
repeated thread launches and synchronization inside every Krylov step
outweighed the available work.

No new Krylov algorithm, preconditioner, spectral API, or public solver
behavior is added as of v0.0.4b1.

Remaining serial or dependency-bound paths
------------------------------------------

The native backend is not "fully parallel."  The following paths are
intentionally serial or guarded as of v0.0.4b1:

* natural-order Cholesky numeric factorization,
* natural-order LU numeric factorization with partial pivoting,
* production single-RHS triangular solves,
* production analyzed triangular solves and level scheduling,
* non-batched COO/CSC forward dense products,
* small trace/dot/vdot inputs below the tree-reduction threshold,
* iterative/spectral recurrence helpers where fixed-worker parallelism was
  measured and rejected,
* native QR, native LDLT, and native rectangular direct solvers.

Use :doc:`performance_results` for the measured effect of the accepted and
rejected paths.
