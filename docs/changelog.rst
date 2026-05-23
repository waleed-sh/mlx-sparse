Changelog
=========

mlx-sparse v0.0.3b0 (Unreleased)
----------------------------------

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
  performance documentation to explain batched sparse-dense dispatch, atomic
  scatter-add kernels, native transpose-product lowering, symbolic/numeric
  CSR x CSR assembly, and dynamic-output synchronization points.


mlx-sparse v0.0.2b0 (21.05.2026)
----------------------------------

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
  transpose) properties; both propagate the backing
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
