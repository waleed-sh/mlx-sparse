Changelog
=========

mlx-sparse v0.0.4b0 (Unreleased)
----------------------------------

.. note::

    This release will focus on improving the performance of the existing sparse linear algebra solvers, specifically for the
    CPU case, by providing Accelerate based solvers when appropriate and optimised versions of the current C++ native solvers
    otherwise. See the roadmap `here <https://github.com/waleed-sh/mlx-sparse/issues/1>`_.

New features
~~~~~~~~~~~~

* Support multiplying all sparse array types by numbers.

* Added user-friendly native capability reporting via
  :data:`mlx_sparse.capabilities` and :func:`mlx_sparse.has_capability`. Users
  can check booleans such as ``ms.capabilities.METAL`` or query status strings
  for CPU, Metal, and reserved Accelerate/CUDA/ROCm backend capabilities.

* Added a CMake feature gate, ``MLX_SPARSE_ENABLE_ACCELERATE``, that detects
  and links Apple's Accelerate framework on Darwin builds for future sparse
  solver integration. No Accelerate-backed solver dispatch is enabled yet.

Improvements
~~~~~~~~~~~~

* Added an experimental staged Metal path for ``COOArray @ COOArray`` behind
  ``ms.config.EXPERIMENTAL_METAL_SPGEMM``. The path row-buckets explicit COO
  coordinates for scheduling, then uses COO-specific symbolic, numeric-fill,
  and zero-prune kernels to return canonical COO output without calling CSR
  sparse-sparse multiplication.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* None.

Documentation
~~~~~~~~~~~~~

* Updated sparse format, supported-feature, and performance documentation to
  describe the COO sparse-sparse execution paths and the experimental Metal
  gate accurately.

mlx-sparse v0.0.3b0 (25.05.2026)
----------------------------------

.. note::

    This release focuses on expanding the supported sparse formats to include COO and CSC arrays, with native C++/Metal
    kernels for basic operations.

New features
~~~~~~~~~~~~

* Added a typed runtime configuration manager exposed as
  :data:`mlx_sparse.config`, with attribute access, ``get_config`` /
  ``set_config``, context-manager overrides, environment-variable sync, and
  forced environment overrides.

* Added the ``EXPERIMENTAL_METAL_SPGEMM`` configuration flag for opting into
  the staged Metal CSR x CSR implementation while keeping the optimized native
  host implementation as the default.

* Added explicit batched sparse-dense APIs:
  :func:`mlx_sparse.csr_batched_matvec` for RHS shape ``(..., n_cols)`` and
  :func:`mlx_sparse.csr_batched_matmul` for RHS shape ``(..., n_cols, k)``.
  ``CSRArray @ dense`` with rank greater than 2 now dispatches through these
  native batched primitives.

* Added native staged ``fromdense`` and ``CSRArray.sum_duplicates`` /
  ``canonicalize`` implementations. These replace NumPy fallback behavior in
  the native path with count/prefix/fill C++ and Metal primitives.

* Added native CSR x CSR multiplication with a symbolic pass, prefix-sum
  output allocation, and numeric fill pass returning canonical CSR output.

* Added native COO x COO and CSC x CSC sparse-sparse multiplication. The new
  paths use format-specific symbolic/count passes, prefix allocation, sorted
  numeric fill, and zero pruning without routing through CSR.

* Added first-class ``CSCArray`` support with explicit constructors,
  validation, repr/metadata flags, COO/CSR conversion paths, dense
  materialization, sorting, duplicate summation, canonicalization, and native
  ``csc_matvec`` / ``csc_matvec_transpose`` entrypoints.

* Added native COO and CSC sparse-dense matrix products for dense vector,
  dense matrix, batched vector, and batched matrix right-hand sides. ``COOArray
  @ dense`` and ``CSCArray @ dense`` now dispatch through format-specific
  C++/Metal primitives instead of converting through CSR.

* Added native COO and CSC reductions: row sums, column sums, row norms,
  column norms, diagonal extraction, and trace. CSC column sums and column
  norms are storage-aligned compressed-column reductions.

* Added CSC input support to sparse linalg entrypoints. CSC matrices are
  converted once to canonical CSR at solver entry so existing CSR-native Krylov,
  direct factorization, triangular solve, spectral, and sparse inner-product
  kernels remain the execution path.

Improvements
~~~~~~~~~~~~

* Reorganized the native source tree so sparse and linalg operations live in
  operation-specific directories containing their C++, header, and Metal files.
  The previous monolithic sparse/linalg source layout has been split into
  localized implementation units.

* Improved CSR transpose. The CPU path now uses a counting transpose, and the
  Metal path performs parallel counts and prefix construction followed by a
  deterministic fill that preserves sorted row indices in the transposed CSR.

* Improved transpose-product kernels used by autodiff. ``float32`` Metal
  transpose matvec/matmul now use parallel atomic scatter-add kernels.
  Non-``float32`` GPU transpose products lower through native transpose plus
  native sparse-dense product to avoid unsupported Metal atomic semantics while
  staying out of NumPy.

* Extended JVP/VJP coverage through the new batched sparse-dense primitives,
  including sparse-value and dense-RHS gradients.

* Extended JVP/VJP coverage to COO and CSC sparse-dense products, including
  batched dense RHS gradients and fixed-output sparse-value VJP kernels.

* Added dedicated CSC native kernels instead of hidden CSR routing for the
  first CSC surface: column-major COO conversion, CSR/CSC conversion,
  dense materialization, per-column sorting, duplicate summation, forward
  matvec scatter-add, and transpose matvec segmented reductions.

* Added dedicated COO and CSC dense-RHS Metal kernels. COO uses coordinate
  scatter, CSC uses compressed-column scatter for forward products and
  compressed-column reductions for transpose products. ``float32`` scatter
  paths use ``atomic_float`` and non-``float32`` scatter paths remain native
  through serial GPU kernels where Metal lacks compatible atomic add support.

* Added reduction-specific Metal kernels for COO and CSC. COO coordinate
  scatter reductions use ``atomic_float`` where storage-compatible, COO/CSC
  norm scatters accumulate squared magnitudes into ``float32`` atomics, and
  CSC column reductions use scalar or threadgroup vector reductions over
  contiguous compressed columns.

* Broadened native correctness and regression tests against dense MLX and
  SciPy references, including GPU dtype coverage, complex gradients,
  pathological linalg cases, and performance regression checks.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* Removed several hidden NumPy fallback paths from native structural operations
  so canonicalization, dense conversion, and sparse-sparse multiplication use
  native implementations when the extension is available.

* Fixed GPU transpose correctness for solver paths by replacing the previous
  fragile transpose fill behavior with a deterministic native fill.

Documentation
~~~~~~~~~~~~~

* Added the :doc:`api/configuration` reference page.

* Updated operation, autodiff, device-execution, supported-feature, and
  performance documentation to explain COO/CSR/CSC batched sparse-dense
  dispatch, atomic scatter-add kernels, native transpose-product lowering,
  symbolic/numeric sparse-sparse assembly, and dynamic-output synchronization
  points.

* Documented COO/CSC reduction semantics, including duplicate-aware norm
  canonicalization and why CSC column reductions are the storage-aligned fast
  path.

* Added CSC container, conversion, and native matvec documentation plus a CSC
  notebook covering SciPy interop and CSR/CSC conversion semantics.


mlx-sparse v0.0.2b0 (21.05.2026)
----------------------------------

.. note::

    This release focuses on providing basic sparse linear algebra operations via native C++ and Metal kernels. The
    focus here is not performance but rather functionality first.

New features
~~~~~~~~~~~~

* Added :mod:`mlx_sparse.linalg`, a sparse linear algebra sub-package with
  three solver families, a spectral module, and a matrix-free operator interface.

* **Iterative solvers**: :func:`mlx_sparse.linalg.cg`,
  :func:`mlx_sparse.linalg.gmres`, and :func:`mlx_sparse.linalg.minres` solve
  sparse linear systems natively on CPU and Metal GPU.  All three accept
  :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, and any
  CSR-backed :class:`~mlx_sparse.linalg.LinearOperator` as the system matrix.

* **Direct factorizations**: :func:`mlx_sparse.linalg.cholesky` (SPD systems),
  :func:`mlx_sparse.linalg.splu` (general systems via sparse LU), and the
  convenience wrapper :func:`mlx_sparse.linalg.spsolve` for single right-hand
  sides.  Factor objects expose a ``.solve(b)`` method for multiple RHS
  without re-factorising.

* **Spectral methods**: :func:`mlx_sparse.linalg.eigsh` computes a few
  eigenvalues and eigenvectors of a real symmetric sparse matrix via a
  native Lanczos iteration.  :func:`mlx_sparse.linalg.eigs` handles general
  non-symmetric matrices (Arnoldi), and :func:`mlx_sparse.linalg.svds`
  computes a partial SVD via randomised bidiagonalisation.

* **LinearOperator interface**: :class:`mlx_sparse.linalg.LinearOperator`
  wraps any callable matvec (or a sparse array) into a uniform operator
  object accepted throughout the linalg sub-package.  The operator exposes
  :attr:`~mlx_sparse.linalg.LinearOperator.T` (transpose) and
  :attr:`~mlx_sparse.linalg.LinearOperator.H` (Hermitian / conjugate
  transpose) properties, both propagate the backing
  :class:`~mlx_sparse.CSRArray` when available so the native C++/Metal code
  paths remain active.  :func:`mlx_sparse.linalg.aslinearoperator` converts
  a :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, SciPy
  sparse matrix, or ``(shape, matvec)`` tuple into a
  :class:`~mlx_sparse.linalg.LinearOperator`.

* Added :func:`mlx_sparse.linalg.lanczos`, the underlying Lanczos
  tridiagonalisation primitive used by :func:`~mlx_sparse.linalg.eigsh`,
  exposed for advanced users who need the raw tridiagonal decomposition.

Improvements
~~~~~~~~~~~~

* :class:`~mlx_sparse.CSRArray` gained :meth:`~mlx_sparse.CSRArray.conj` /
  :meth:`~mlx_sparse.CSRArray.conjugate` convenience methods and a
  :attr:`~mlx_sparse.CSRArray.H` (Hermitian transpose) property that composes
  :meth:`~mlx_sparse.CSRArray.T` with element-wise conjugation.

* :class:`~mlx_sparse.CSRArray` now has a
  :attr:`~mlx_sparse.CSRArray.index_dtype` property that reflects the integer
  dtype of the stored index arrays.

* Added :meth:`~mlx_sparse.CSRArray.sort_indices` to sort column indices
  within each row in-place (returns ``self`` when already sorted, avoiding a
  copy).

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* None.

Documentation
~~~~~~~~~~~~~

* Added four new Jupyter notebooks under *Sparse linear algebra*:
  :doc:`notebooks/13_linalg_solvers`,
  :doc:`notebooks/14_linalg_factorizations`,
  :doc:`notebooks/15_linalg_spectral`, and
  :doc:`notebooks/16_linalg_operators`.  Each notebook walks through
  a worked example with correctness checks and timing comparisons against the
  MLX dense baseline.

* Added the :doc:`tutorials/sparse_linear_systems` tutorial, which assembles
  a 2-D Poisson (Laplacian) system, solves it with CG, Cholesky, and
  ``spsolve``, and discusses when to prefer each approach.

* Added three benchmark scripts under ``benchmarks/``:
  ``bench_linalg_solvers.py``, ``bench_linalg_factorizations.py``, and
  ``bench_linalg_spectral.py``.  Each script reports raw timing, speedup
  versus the MLX dense equivalent, and a relative-error correctness check
  against a SciPy reference.

* Reorganised :doc:`notebooks/index` into two captioned sections,
  *Primitives* (notebooks 01–12) and *Sparse linear algebra* (notebooks
  13–16), for easier navigation.
