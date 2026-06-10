.. _currently-supported:

Currently supported
====================

This page is the authoritative record of what mlx-sparse implements, what is
planned, and what is out of scope. Status is updated with each release.

.. warning::

   ``mlx-sparse`` supports macOS and Linux. Linux support is CPU-only in this
   release: CUDA and ROCm are not implemented, Metal is Apple-only, and Linux
   builds do not use Accelerate, BLAS, or Sparse BLAS backends.

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
   * - ``CSCArray`` container
     - Done
     - Immutable frozen dataclass. Column-compressed dual of CSR with
       ``sorted_indices`` and ``has_canonical_format`` flags.
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
   * - ``csc_array((data, indices, indptr), shape)``
     - Done
     - Explicit CSC buffers with metadata or full validation.
   * - ``eye(n, m, k)``
     - Done
     - Sparse identity or shifted-diagonal matrix. Returns canonical CSR.
   * - ``identity(n)``
     - Done
     - Square identity alias. Defaults to CSR and supports COO/CSR/CSC output.
   * - ``diags(diagonals, offsets)``
     - Done
     - One or more diagonals at specified offsets. Returns canonical CSR.
   * - ``block_array`` / ``bmat``
     - Done
     - Native COO coordinate-offset assembly for COO/CSR/CSC and dense blocks.
       ``None`` entries represent implicit zero blocks with inferred sizes.
   * - ``block_diag`` / ``vstack`` / ``hstack``
     - Done
     - Native block-offset assembly without Python loops over stored entries.
       Supports COO/CSR/CSC output and dense inputs through native
       ``fromdense``.
   * - ``tril`` / ``triu``
     - Done
     - Native staged count/fill extraction for COO, CSR, and CSC inputs.
       Dense inputs route through native ``fromdense`` first.
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
     - Converts any SciPy sparse matrix/array to canonical CSR, CSC, or COO.
   * - ``identity_like(x)``
     - Done
     - Extension smoke test / identity copy.
   * - ``issparse(x)``
     - Done
     - Returns ``True`` for ``COOArray``, ``CSRArray``, and ``CSCArray``.
   * - ``asarray(x)``
     - Done
     - Converts existing sparse, SciPy sparse, dense MLX, NumPy, or Python
       rank-2 array-like inputs. Existing CSR/CSC inputs are preserved unless
       a dtype cast is requested, dense and SciPy inputs default to CSR.
   * - ``ms.random`` namespace
     - Done
     - Public ``random_array``, ``random``, and ``rand`` support COO/CSR/CSC
       output with native CPU/Metal duplicate-free structure generation. CSR
       and CSC are generated directly in compressed form rather than through
       COO conversion. Default values use MLX uniform ``[0, 1)`` random vector
       operations, custom value samplers are called once for custom ranges or
       distributions and may explicitly provide host values.

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
   * - ``COOArray.tocsc()``
     - Done
     - Native ``coo_tocsc`` primitive (CPU and Metal). Sorts by column then
       row. Preserves duplicates.
   * - ``COOArray.tocsc(canonical=True)``
     - Done
     - Sorts row indices within columns and sums duplicates.
   * - ``CSRArray.tocsc()``
     - Done
     - Native ``csr_tocsc`` conversion with count/prefix/fill structure build.
   * - ``CSCArray.tocsr()``
     - Done
     - Native ``csc_tocsr`` conversion with count/prefix/fill structure build.
   * - ``CSRArray.tocoo()``
     - Done
     - Native row-pointer expansion to COO row coordinates. Canonical CSR
       inputs can produce canonical row-major COO metadata.
   * - ``CSCArray.tocoo()``
     - Done
     - Native column-pointer expansion to COO column coordinates. Requested
       canonical COO output routes through native CSC-to-CSR conversion and
       CSR-to-COO expansion.
   * - ``CSRArray.todense()``
     - Done
     - Native primitive (CPU and Metal). Sums duplicate column entries.
   * - ``CSCArray.todense()``
     - Done
     - Native column-wise materialization (CPU and Metal). Sums duplicate row
       entries.
   * - ``COOArray.todense()``
     - Done
     - Via ``tocsr().todense()``.
   * - ``ms.todense(array)``
     - Done
     - Module-level dispatch helper.
   * - Structural block and stack assembly
     - Done
     - CPU and Metal native coordinate-offset kernels. CSR/CSC format requests
       canonicalize through native compressed conversion.
   * - Triangular extraction
     - Done
     - CPU and Metal native count/fill kernels for COO, CSR, and CSC.
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
   * - ``CSCArray`` canonicalization
     - Done
     - Native CSC sort, duplicate-sum, and canonicalization primitives over
       compressed columns.
   * - ``CSRArray.transpose()`` / ``.T``
     - Done
     - Native primitive (CPU and Metal). Returns row-sorted CSRArray.
   * - ``CSRArray.conj()`` / ``.conjugate()``
     - Done
     - ``mx.conjugate`` applied to ``data``.
   * - ``CSRArray.H``
     - Done
     - Hermitian (conjugate) transpose.
   * - ``CSCArray.transpose()`` / ``.T`` / ``.H``
     - Done
     - ``.T`` is a zero-copy CSRArray view of the transposed structure,
       ``.H`` conjugates values first.

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
   * - ``csc_matvec`` / ``csc_matvec_transpose``
     - Done
     - Native CSC kernels. Forward matvec is column scatter-add, transpose
       matvec is segmented column reduction.
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
   * - Sparse-sparse multiplication (``COOArray @ COOArray``)
     - Done
     - Native coordinate-row symbolic/count pass, prefix allocation, sorted
       numeric fill, and zero pruning returning canonical COO.
   * - Sparse-sparse multiplication (``CSCArray @ CSCArray``)
     - Done
     - Native compressed-column symbolic/count pass, prefix allocation, sorted
       numeric fill, and zero pruning returning canonical CSC.
   * - Mixed-format sparse-sparse multiplication
     - Done
     - ``COO/CSR/CSC @ COO/CSR/CSC`` works for all format pairs. Mixed RHS
       operands are normalized through native format conversion, then the
       left-hand format's native sparse-sparse product is used. Output format
       follows the left operand: COO, CSR, or CSC.
   * - Scalar multiply (``alpha * A``)
     - Done
     - Scales stored values for COO, CSR, and CSC inputs while preserving the
       sparse format and structural metadata.
   * - Sparse-sparse addition
     - Done
     - ``A + B``, ``A - B``, :func:`mlx_sparse.add`, and
       :func:`mlx_sparse.subtract` support COO/CSR/CSC sparse operands with
       equal shape and matching value dtype. Inputs canonicalize through native
       sort/sum and conversion paths, then a native CSR CPU/Metal merge emits
       canonical output with duplicate coordinates summed and exact zero
       cancellations removed. Homogeneous CSC inputs return CSC, other
       supported combinations return CSR. Sparse+dense and nonzero scalar
       addition are rejected to avoid hidden dense outputs.
   * - Kronecker product and sum
     - Done
     - :func:`mlx_sparse.kron` accepts COO/CSR/CSC or dense rank-2 operands and
       returns COO/CSR/CSC. Dense operands are extracted with native
       ``fromdense``. CSR/CSC operands convert through native compressed-to-COO
       expansion, the native COO Kronecker CPU/Metal primitive writes product
       coordinates and values directly, and requested CSR/CSC outputs
       canonicalize through native compressed conversion. :func:`kronsum`
       composes native ``kron`` and native sparse addition for square inputs.
       The fixed-topology COO data product has sparse-value JVP/VJP support,
       duplicate-summing canonicalization remains a dynamic-topology boundary.

Sparse reductions
-----------------

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Feature
     - Status
     - Notes
   * - COO reductions
     - Done
     - Native row/column sums, row/column L2 norms, diagonal extraction, and
       trace. Sums and diagonal/trace operate directly on coordinates. Norms
       canonicalize first when duplicates may be present so the result matches
       dense semantics. Non-``float32`` canonical norm reductions on Metal use
       native COO-to-compressed conversion plus storage-aligned reductions to
       avoid scatter-heavy atomic accumulation.
   * - CSR reductions
     - Done
     - Native row/column sums, row norms, diagonal, and trace. Storage-aligned
       row reductions and long diagonal segments use threadgroup reductions on
       Metal, large traces use a staged partial-reduction path.
   * - CSC reductions
     - Done
     - Native row/column sums, row/column L2 norms, diagonal, and trace.
       Column sums and column norms are storage-aligned compressed-column
       reductions and are the fast path for CSC. Non-``float32`` row norms on
       Metal lower through native CSC-to-CSR conversion and CSR row reductions,
       long diagonal segments and large traces use staged/vector reductions.

.. _gpu-supported-linalg:

Sparse linear algebra
---------------------

For a solver-centric view of CPU, Metal GPU, and Accelerate coverage, see
:doc:`user_guide/linalg_solvers`.

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
       kernel. Diagonal/Jacobi and ILU(0) preconditioned native paths are
       available, convergence bookkeeping and the small least-squares solve
       run on CPU.
   * - ``linalg.minres``
     - Done
     - CPU + GPU
     - Shifted Paige-Saunders recurrence runs in native CPU or Metal kernels.
       Diagonal/Jacobi preconditioners are supported when SPD.
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
     - CPU + GPU
     - Dedicated normal-operator Lanczos step keeps
       ``A.T @ (A @ v)`` native. The small eigensolve and singular-vector
       assembly run on CPU.
   * - ``linalg.sparse_cholesky``
     - Done
     - CPU only
     - Symbolic fill-in factorisation is inherently sequential. Planned GPU
       path via supernodal Cholesky is out of scope for v0.x.
   * - ``linalg.sparse_lu``
     - Done
     - CPU + GPU
     - LU factorisation (partial pivoting) runs on CPU. Triangular
       forward/back-substitution and permutation dispatch to Metal GPU via
       ``csr_triangular_solve`` and ``csr_permute_vector`` kernels.
   * - ``preconditioners.ilu0``
     - Done
     - CPU + GPU
     - Natural-order ILU(0) setup runs on CPU and preserves the canonical CSR
       sparsity pattern. Application uses native CSR triangular solves for
       rank-1 or rank-2 right-hand sides on CPU or Metal.
   * - ``linalg.factorized`` / ``linalg.spsolve``
     - Optional
     - CPU only
     - Accelerate-enabled Apple builds use opaque Accelerate direct solves for
       supported real ``float32`` systems. Portable builds fall back to native
       LU for square ``spsolve``.
   * - ``CSRArray.dot`` / ``CSRArray.vdot``
     - Done
     - CPU + GPU
     - Native CSR row-merge reductions for ``float32`` and ``complex64``.

Linalg GPU coverage notes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse linalg entrypoints accept CSR, COO, and CSC inputs. CSR is the execution
format for native kernels, so COO and CSC inputs are converted once to
canonical CSR at native solver entry. This keeps the existing Metal Krylov,
triangular solve, and permutation kernels active without doing repeated CSC
scatter-add matvecs inside solver iterations. Accelerate-enabled direct solves
instead validate and normalize real ``float32`` CSR, COO, and CSC inputs into
canonical CSC storage because Apple's sparse direct solvers are CSC-native.

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

* **Cholesky / LU / ILU(0) factorisation**: row-by-row elimination with fill-in
  or no-fill incomplete updates runs on CPU.  The resulting triangular
  **solve** (``SparseCholesky.solve``, ``SparseLU.solve``, ``spsolve``, and
  ``preconditioners.ilu0`` application) dispatches the ``csr_triangular_solve``
  Metal kernel and the ``csr_permute_vector`` Metal kernel for the LU
  row-permutation step where a permutation is present.

* **svds**: uses a dedicated normal-operator Lanczos step for
  ``A.T @ (A @ x)``.  The implementation does not materialize ``A.T @ A`` and
  does not split the recurrence into Python-level sparse products.  On Metal,
  the two sparse products are fused inside the native Lanczos step, the small
  tridiagonal eigensolve, Ritz-vector back transformation, and final singular
  vector assembly remain CPU work.

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
   * - VJP/JVP through sparse-sparse ``matmat``
     - Not planned for v0.1
     - Output topology is data-dependent and returned as a sparse container.
       Fixed-output sparse-dense products are the differentiable path.
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
   * - ``coo_matvec`` / ``coo_matmul``
     - All value and index dtypes
     - Native coordinate scatter products. ``float32`` uses atomic
       scatter-add, other value dtypes use native serial scatter.
   * - ``coo_batched_matvec`` / ``coo_batched_matmul``
     - All value and index dtypes
     - Native batched coordinate scatter kernels
   * - ``coo_matmul_data_vjp``
     - All value and index dtypes
     - Fixed-output sparse-value VJP over explicit coordinates
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
   * - ``csc_matvec`` / ``csc_matvec_transpose``
     - All value and index dtypes
     - Forward ``float32`` matvec uses atomic column scatter-add, other
       forward GPU dtypes use native serial scatter. Transpose matvec uses
       scalar or threadgroup vector column reductions.
   * - ``csc_matmul`` / ``csc_matmul_transpose``
     - All value and index dtypes
     - Forward ``float32`` matmul uses atomic column scatter-add, other
       forward GPU dtypes use native serial scatter. Transpose matmul uses
       compressed-column dot products.
   * - ``csc_batched_matvec`` / ``csc_batched_matmul``
     - All value and index dtypes
     - Native batched compressed-column dense RHS kernels
   * - COO/CSC reductions
     - All value and index dtypes
     - Storage-aligned reductions use scalar or threadgroup vector kernels.
       Scatter reductions use ``atomic_float`` where possible, norm scatter
       accumulates into ``float32`` atomics, and low-precision/complex sum
       scatters lower through native compressed conversion paths.
   * - ``csc_matmul_data_vjp``
     - All value and index dtypes
     - Fixed-output sparse-value VJP over compressed columns
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
   * - ``coo_tocsc``
     - All value and index dtypes
     - Rank-based stable column-major sort plus indptr build
   * - ``csr_transpose``
     - All value and index dtypes
     - Parallel count/prefix plus deterministic fill
   * - ``csr_tocsc`` / ``csc_tocsr``
     - All value and index dtypes
     - Native count/prefix/fill conversions. GPU fill uses atomic offsets and
       does not promise sorted output, call ``canonicalize()`` when ordering
       matters.
   * - ``csc_todense``
     - All value and index dtypes
     - Parallel zero-fill plus column-wise materialization
   * - ``csr_sort_indices``
     - All value and index dtypes
     - Rank-based stable per-row sort
   * - ``csc_sort_indices``
     - All value and index dtypes
     - Rank-based stable per-column sort
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
       ``SparseLU.solve``, public ``linalg.spsolve_triangular``, and native
       ``linalg.spsolve`` fallback
   * - ``csr_permute_vector``
     - ``float32``, int32 permutation
     - Row permutation step in ``SparseLU.solve`` / native ``linalg.spsolve``
   * - ``csr_dot`` / ``csr_vdot``
     - ``float32``/``complex64`` values, int32/int64 indices
     - Sparse Frobenius inner products with explicit complex conjugation
       semantics
   * - ``csr_sum_duplicates``
     - All value and index dtypes
     - Staged count/prefix/fill primitive, dynamic output size requires
       row-count synchronization
   * - ``csc_sum_duplicates``
     - All value and index dtypes
     - Staged per-column count/prefix/fill primitive, dynamic output size
       requires column-count synchronization
   * - ``csr_fromdense``
     - All value and index dtypes
     - Staged count/prefix/fill dense-to-CSR conversion
   * - ``csr_matmat``
     - All value and index dtypes
     - Optimized host path by default, experimental staged Metal path behind
       ``EXPERIMENTAL_METAL_SPGEMM``
   * - ``coo_matmat``
     - All value and index dtypes
     - Optimized host path by default, experimental staged Metal path behind
       ``EXPERIMENTAL_METAL_SPGEMM``. The Metal path uses COO-specific
       symbolic/numeric/prune kernels and returns canonical COO.
   * - ``csc_matmat``
     - All value and index dtypes
     - Optimized host path by default, experimental staged Metal path behind
       ``EXPERIMENTAL_METAL_SPGEMM``. The Metal path uses CSC-specific
       symbolic/numeric/prune kernels and returns canonical CSC.

Known limitations
-----------------------------

* GPU availability depends on the MLX and macOS Metal runtime.
* Dynamic-output helpers (``fromdense()``, ``canonicalize()``, dense/SciPy
  construction, and sparse-sparse ``matmat``) synchronize compact counts or
  structure to host before allocating final output buffers.
* CSC currently covers construction, conversion, canonicalization, dense
  materialization, reductions, dense vector/matrix products including batched
  dense RHS, same-format sparse-sparse matmul, one-time conversion at native
  linalg solver entry, and canonical CSC normalization for Accelerate-enabled
  opaque direct solves.
* Sparse solver, factorization, and spectral kernels are real-valued.
  ``float16`` and ``bfloat16`` inputs are promoted to ``float32`` before
  solver dispatch. Sparse ``dot``/``vdot`` support ``complex64``.
* Full validation (``validate="full"``) may trigger host synchronization.
