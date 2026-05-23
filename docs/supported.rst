Currently supported
====================

This page is the authoritative record of what mlx-sparse implements, what is
planned, and what is out of scope. Status is updated with each release.

Current version: **development branch**

Sparse formats
--------------

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Feature
     - Status
     - Notes
   * - ``COOArray`` container
     - Done
     - Immutable frozen dataclass. Allows duplicates and unsorted coordinates.
   * - ``CSRArray`` container
     - Done
     - Immutable frozen dataclass. ``sorted_indices`` and
       ``has_canonical_format`` flags.
   * - CSC format
     - Planned
     - Will be added when there is a workload that specifically benefits.
   * - Block CSR (BCSR)
     - Planned
     - Internal storage format for block-structured matrices.
   * - ELLPACK / SELL-C-σ
     - Research
     - Internal format for regular row lengths. No public API commitment.
   * - Sparse tensors (rank > 2)
     - Not planned
     - MLX's lazy graph requires output shapes at graph-build time. General
       sparse tensors have dynamic shapes and are out of scope for v0.x.

Constructors
------------

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Feature
     - Status
     - Notes
   * - ``coo_array((data, (row, col)), shape)``
     - Done
     - Accepts MLX arrays, NumPy arrays, or Python lists.
   * - ``csr_array((data, indices, indptr), shape)``
     - Done
     - Same input flexibility.
   * - ``eye(n, m, k)``
     - Done
     - Sparse identity or shifted-diagonal matrix. Returns canonical CSR.
   * - ``diags(diagonals, offsets)``
     - Done
     - One or more diagonals at specified offsets. Returns canonical CSR.
   * - ``fromdense(array, threshold)``
     - Done
     - Native staged conversion with optional threshold for near-zeros.
       Counts on the active backend, synchronizes row counts to allocate
       compact output buffers, then fills CSR data natively.
   * - ``from_dense(array)`` / ``from_numpy(array)``
     - Done
     - PEP 8 and NumPy-oriented aliases for dense-to-CSR conversion.
   * - ``from_scipy(matrix)``
     - Done
     - Converts any SciPy sparse matrix/array to canonical CSR or COO.
   * - ``identity_like(x)``
     - Done
     - Extension smoke test / identity copy.
   * - ``issparse(x)``
     - Done
     - Returns ``True`` for ``COOArray`` and ``CSRArray``.
   * - ``asarray(x)``
     - Done
     - Converts existing sparse, SciPy sparse, dense MLX, NumPy, or Python
       rank-2 array-like inputs to CSR.

Conversions and structural operations
--------------------------------------

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Feature
     - Status
     - Notes
   * - ``COOArray.tocsr()``
     - Done
     - Native primitive (CPU and Metal). Sorts by row then column. Preserves
       duplicates.
   * - ``COOArray.tocsr(canonical=True)``
     - Done
     - Sorts and sums duplicates.
   * - ``CSRArray.todense()``
     - Done
     - Native primitive (CPU and Metal). Sums duplicate column entries.
   * - ``COOArray.todense()``
     - Done
     - Via ``tocsr().todense()``.
   * - ``ms.todense(array)``
     - Done
     - Module-level dispatch helper.
   * - ``CSRArray.sort_indices()``
     - Done
     - Native primitive (CPU and Metal).
   * - ``CSRArray.sum_duplicates()``
     - Done
     - Native staged primitive (CPU and Metal). Dynamic output size requires
       a row-count synchronization before compact output fill.
   * - ``CSRArray.canonicalize()``
     - Done
     - Combines ``sort_indices`` and ``sum_duplicates``.
   * - ``CSRArray.transpose()`` / ``.T``
     - Done
     - Native primitive (CPU and Metal). Returns row-sorted CSRArray.
   * - ``CSRArray.conj()`` / ``.conjugate()``
     - Done
     - ``mx.conjugate`` applied to ``data``.
   * - ``CSRArray.H``
     - Done
     - Hermitian (conjugate) transpose.

Sparse-dense arithmetic
------------------------

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Feature
     - Status
     - Notes
   * - ``csr_matvec`` (all value dtypes, int32 and int64)
     - Done
     - CPU and Metal GPU. Scalar row kernel plus vector-reduction kernel for
       long rows on Metal.
   * - ``csr_matmul`` (all value dtypes, int32 and int64)
     - Done
     - CPU and Metal GPU. Scalar element kernel plus vector-reduction kernel
       for long rows on Metal.
   * - Batched dense RHS (``CSRArray @ batch``)
     - Done
     - RHS with ``ndim > 2`` dispatches native batched sparse-dense kernels.
       Explicit helpers are ``csr_batched_matvec`` and ``csr_batched_matmul``.
   * - Sparse-sparse multiplication (``CSRArray @ CSRArray``)
     - Done
     - Native symbolic pass, prefix-sum allocation, and numeric pass returning
       canonical CSR. Dynamic output size requires host synchronization.
   * - Scalar multiply (``alpha * A``)
     - Not yet
     - Can be approximated with ``ms.csr_array((alpha * data, ...), ...)``.
   * - Sparse-sparse addition
     - Not planned
     - Dynamic output size. May be added as a host-side utility.

Sparse linear algebra
---------------------

.. list-table::
   :widths: 35 15 15 35
   :header-rows: 1

   * - Feature
     - Status
     - Backend
     - Notes
   * - ``linalg.cg``
     - Done
     - CPU + GPU
     - Full solver runs inside a single Metal kernel on GPU.
   * - ``linalg.gmres``
     - Done
     - CPU + GPU
     - Each restart's Arnoldi step dispatches the ``csr_arnoldi`` Metal
       kernel, convergence bookkeeping and the small least-squares solve
       run on CPU.
   * - ``linalg.minres``
     - Done
     - CPU + GPU
     - Lanczos tridiagonalisation dispatches the ``csr_lanczos`` Metal
       kernel, the tridiagonal least-squares solve runs on CPU.
   * - ``linalg.eigsh``
     - Done
     - CPU + GPU
     - Lanczos step dispatches ``csr_lanczos`` Metal kernel, the small
       Jacobi eigensolver runs on CPU.
   * - ``linalg.eigs``
     - Done
     - CPU + GPU
     - Arnoldi step dispatches ``csr_arnoldi`` Metal kernel, QR eigenvalues
       run on CPU.
   * - ``linalg.svds``
     - Done
     - CPU only
     - Normal-operator Lanczos (two SpMVs per step) has no dedicated Metal
       kernel and runs entirely on CPU.
   * - ``linalg.sparse_cholesky``
     - Done
     - CPU only
     - Symbolic fill-in factorisation is inherently sequential. Planned GPU
       path via supernodal Cholesky is out of scope for v0.x.
   * - ``linalg.sparse_lu`` / ``linalg.spsolve``
     - Done
     - CPU + GPU
     - LU factorisation (partial pivoting) runs on CPU. Triangular
       forward/back-substitution and permutation dispatch to Metal GPU via
       ``csr_triangular_solve`` and ``csr_permute_vector`` kernels.
   * - ``CSRArray.dot`` / ``CSRArray.vdot``
     - Done
     - CPU + GPU
     - Native CSR row-merge reductions for ``float32`` and ``complex64``.

Linalg GPU coverage notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

The table above uses a simplified "CPU + GPU" label. The precise breakdown
is:

* **CG**: the entire conjugate-gradient iteration (SpMV, dot products,
  vector updates) runs inside a single Metal threadgroup kernel.  The GPU
  path is fully independent of the CPU.

* **GMRES / MINRES / eigsh / eigs**: the expensive Krylov-subspace step
  (Arnoldi or Lanczos, which accounts for most of the wall time at large
  ``n``) runs on GPU via the ``csr_arnoldi`` or ``csr_lanczos`` Metal
  kernels.  Post-processing (a small dense eigensolve or least-squares
  solve of size ``≤ restart`` or ``≤ ncv``) runs on CPU.  An
  ``mx.eval()`` synchronisation separates the two phases, at very small
  ``n`` (≲ 1 000) the synchronisation overhead can exceed the GPU savings.

* **Cholesky / LU factorisation**: row-by-row elimination with fill-in is
  inherently sequential and runs on CPU.  The resulting triangular **solve**
  (``SparseCholesky.solve``, ``SparseLU.solve``, ``spsolve``) dispatches
  the ``csr_triangular_solve`` Metal kernel and the ``csr_permute_vector``
  Metal kernel for the LU row-permutation step.

* **svds**: uses a two-SpMV-per-step Lanczos (``A.T @ (A @ x)``).  The
  existing ``csr_lanczos`` kernel performs a single SpMV per step, so svds
  has no GPU path and runs entirely on CPU.  A dedicated two-matvec kernel
  is planned.

Automatic differentiation
--------------------------

.. list-table::
   :widths: 40 15 45
   :header-rows: 1

   * - Feature
     - Status
     - Notes
   * - VJP w.r.t. dense ``x`` in ``A @ x``
     - Done
     - Dispatches native transpose matvec. ``float32`` uses Metal atomics,
       other GPU value dtypes lower through native transpose plus matvec.
   * - JVP w.r.t. dense ``x`` in ``A @ x``
     - Done
     - Reuses forward ``csr_matvec``. CPU and Metal GPU.
   * - VJP w.r.t. dense ``X`` in ``A @ X``
     - Done
     - Dispatches native transpose matmul. ``float32`` uses Metal atomics,
       other GPU value dtypes lower through native transpose plus matmul.
   * - JVP w.r.t. dense ``X`` in ``A @ X``
     - Done
     - Reuses forward ``csr_matmul``. CPU and Metal GPU.
   * - VJP/JVP w.r.t. sparse values (``data``)
     - Done
     - Fixed-output data-gradient primitives for matvec and matmul on CPU and
       Metal GPU.
   * - Complex autodiff
     - Done
     - ``complex64`` VJP uses Hermitian adjoints and is tested against dense
       MLX matmul.
   * - VJP/JVP w.r.t. ``indices`` / ``indptr``
     - Not planned
     - Structural parameters are not differentiable variables.
   * - ``vmap`` over dense RHS
     - Done
     - Batched dense RHS uses native batched sparse-dense kernels.
   * - VJP/JVP through batched dense RHS
     - Done
     - Native batched matvec/matmul primitives support sparse-value and
       dense-RHS differentiation.
   * - ``vmap`` over sparse matrices
     - Not planned
     - Batch of sparse matrices is an unusual use case. Deferred.

Metal GPU kernel coverage
--------------------------

Most sparse primitives cover the full value and index dtype matrix. A few
linalg kernels are intentionally ``float32``-only, and dynamic-output
structural primitives synchronize counts or output structure before allocating
compact buffers.

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Kernel
     - Status
     - Notes
   * - ``csr_matvec``
     - All value and index dtypes
     - Scalar row kernel plus threadgroup vector reduction for long rows
   * - ``csr_batched_matvec``
     - All value and index dtypes
     - Native batched dense-vector RHS kernel
   * - ``csr_matvec_data_vjp``
     - All value and index dtypes
     - Fixed-output sparse-value VJP primitive
   * - ``csr_matvec_transpose``
     - All value and index dtypes
     - ``float32`` uses atomic scatter-add, other GPU value dtypes lower
       through native transpose plus matvec
   * - ``csr_matmul``
     - All value and index dtypes
     - Scalar element kernel plus threadgroup vector reduction for long rows
   * - ``csr_batched_matmul``
     - All value and index dtypes
     - Native batched dense-matrix RHS kernel
   * - ``csr_matmul_data_vjp``
     - All value and index dtypes
     - Fixed-output sparse-value VJP primitive
   * - ``csr_matmul_transpose``
     - All value and index dtypes
     - ``float32`` uses atomic scatter-add, other value dtypes lower through
       native transpose plus matmul
   * - ``csr_todense``
     - All value and index dtypes
     - Fixed-output materialization kernel
   * - ``coo_tocsr``
     - All value and index dtypes
     - Rank-based stable sort plus indptr build
   * - ``csr_transpose``
     - All value and index dtypes
     - Parallel count/prefix plus deterministic fill
   * - ``csr_sort_indices``
     - All value and index dtypes
     - Rank-based stable per-row sort
   * - ``csr_cg``
     - ``float32`` values, int32/int64 indices
     - Full CG iteration for ``linalg.cg``
   * - ``csr_lanczos``
     - ``float32`` values, int32/int64 indices
     - Krylov step for ``linalg.minres``, ``linalg.eigsh``, and the
       primitive ``linalg.lanczos``
   * - ``csr_arnoldi``
     - ``float32`` values, int32/int64 indices
     - Krylov step for ``linalg.gmres``, ``linalg.eigs``
   * - ``csr_triangular_solve``
     - ``float32`` values, int32/int64 indices
     - Forward/back-substitution for ``SparseCholesky.solve``,
       ``SparseLU.solve``, and ``linalg.spsolve``
   * - ``csr_permute_vector``
     - ``float32``, int32 permutation
     - Row permutation step in ``SparseLU.solve`` / ``linalg.spsolve``
   * - ``csr_dot`` / ``csr_vdot``
     - ``float32``/``complex64`` values, int32/int64 indices
     - Sparse Frobenius inner products with explicit complex conjugation
       semantics
   * - ``csr_sum_duplicates``
     - All value and index dtypes
     - Staged count/prefix/fill primitive, dynamic output size requires
       row-count synchronization
   * - ``csr_fromdense``
     - All value and index dtypes
     - Staged count/prefix/fill dense-to-CSR conversion
   * - ``csr_matmat``
     - All value and index dtypes
     - Optimized host path by default, experimental staged Metal path behind
       ``EXPERIMENTAL_METAL_SPGEMM``

Known limitations
-----------------------------

* GPU availability depends on the MLX and macOS Metal runtime.
* Dynamic-output helpers (``fromdense()``, ``canonicalize()``, dense/SciPy
  construction, and ``CSR @ CSR``) synchronize compact row counts or structure
  to host before allocating final output buffers.
* Sparse solver, factorization, and spectral kernels are real-valued.
  ``float16`` and ``bfloat16`` inputs are promoted to ``float32`` before
  solver dispatch. Sparse ``dot``/``vdot`` support ``complex64``.
* Full validation (``validate="full"``) may trigger host synchronization.
