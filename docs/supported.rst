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
     - Dense-to-sparse conversion with optional threshold for near-zeros.
       Synchronizes to host to determine output size.
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
     - Python/NumPy host implementation via fallback.
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
     - RHS with ``ndim > 2`` is reshaped to 2D internally.
   * - Sparse-sparse multiplication (``CSRArray @ CSRArray``)
     - Done
     - Host structural assembly returning canonical CSR. Dynamic output size
       requires host synchronization.
   * - Scalar multiply (``alpha * A``)
     - Not yet
     - Can be approximated with ``ms.csr_array((alpha * data, ...), ...)``.
   * - Sparse-sparse addition
     - Not planned
     - Dynamic output size. May be added as a host-side utility.

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
     - Dispatches ``CSRMatVecTranspose`` primitive. CPU and Metal GPU.
   * - JVP w.r.t. dense ``x`` in ``A @ x``
     - Done
     - Reuses forward ``csr_matvec``. CPU and Metal GPU.
   * - VJP w.r.t. dense ``X`` in ``A @ X``
     - Done
     - Dispatches ``CSRMatMulTranspose`` primitive. CPU and Metal GPU.
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
     - Batched dense RHS lowers to one ``csr_matmul`` over flattened batch
       columns.
   * - ``vmap`` over sparse matrices
     - Not planned
     - Batch of sparse matrices is an unusual use case. Deferred.

Metal GPU kernel coverage
--------------------------

All fixed-shape kernels cover the full value and index dtype matrix.

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Kernel
     - Status
     - Notes
   * - ``csr_matvec``
     - All value and index dtypes
     - Scalar row kernel plus threadgroup vector reduction for long rows
   * - ``csr_matvec_data_vjp``
     - All value and index dtypes
     - Fixed-output sparse-value VJP primitive
   * - ``csr_matmul``
     - All value and index dtypes
     - Scalar element kernel plus threadgroup vector reduction for long rows
   * - ``csr_matmul_data_vjp``
     - All value and index dtypes
     - Fixed-output sparse-value VJP primitive
   * - ``csr_todense``
     - All value and index dtypes
     - Fixed-output materialization kernel
   * - ``coo_tocsr``
     - All value and index dtypes
     - Rank-based stable sort plus indptr build
   * - ``csr_transpose``
     - All value and index dtypes
     - Rank-based transpose sort plus indptr build
   * - ``csr_sort_indices``
     - All value and index dtypes
     - Rank-based stable per-row sort
   * - ``csr_sum_duplicates``
     - Not implemented
     - Dynamic output size. Deferred.

Known limitations
-----------------------------

* GPU availability depends on the MLX and macOS Metal runtime.
* Dynamic-output helpers (``canonicalize()``, dense/SciPy construction, and
  ``CSR @ CSR``) synchronize to host for structural assembly.
* Sparse linear solvers, sparse eigensolvers, and general sparse tensors are
  outside the current scope.
* Full validation (``validate="full"``) may trigger host synchronization.
