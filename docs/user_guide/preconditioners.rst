.. _preconditioners:

Preconditioners
===============

``mlx_sparse.linalg.preconditioners`` contains native-backed inverse-apply
objects for sparse iterative solvers. A preconditioner ``M`` applies an
approximation to ``A^{-1}``, it is not interpreted as a sparse matrix to invert
implicitly.

Python preconditioner objects are API containers and dispatch helpers. Diagonal
application, Jacobi-preconditioned CG, diagonal/Jacobi-preconditioned GMRES,
ILU(0)-preconditioned GMRES, IC(0)-preconditioned CG,
Chebyshev-preconditioned CG, diagonal/Jacobi-preconditioned MINRES, and exact
LU/Cholesky preconditioner application run through native primitives under
``src/preconditioners``. These paths do not execute a Python callback inside
native Krylov iterations.
Exact-factor preconditioners wrap existing direct solve objects and reuse typed
native or guarded Accelerate apply paths.

Solver diagnostics are available with ``return_info=True`` on ``cg``,
``gmres``, and ``minres``. The returned ``SolverInfo`` records the final true
residual norm, iteration count, status reason, and preconditioner kind. Native
callbacks are exit callbacks only: they run once after the native loop has
finished, avoiding per-iteration CPU/GPU synchronization by default. For
``gmres``, ``callback_type="x"`` receives the final solution, and
``"pr_norm"`` or ``"legacy"`` receives the final reported residual norm. A
full per-restart or per-inner-iteration Python callback stream is intentionally
not enabled for native preconditioned paths because it would change the
CPU/Metal synchronization model.

Notebook examples
-----------------

The notebook gallery has a separate *Sparse preconditioners* section with
self-contained examples for each supported strategy:

* :doc:`../notebooks/21_preconditioner_identity`
* :doc:`../notebooks/22_preconditioner_diagonal`
* :doc:`../notebooks/23_preconditioner_jacobi`
* :doc:`../notebooks/24_preconditioner_ilu0`
* :doc:`../notebooks/25_preconditioner_ichol0`
* :doc:`../notebooks/26_preconditioner_chebyshev`
* :doc:`../notebooks/27_preconditioner_exact`
* :doc:`../notebooks/28_preconditioner_callable`

Current support
---------------

.. list-table::
   :widths: 24 36 40
   :header-rows: 1

   * - Constructor
     - Use
     - Execution boundary
   * - ``identity(A_or_shape)``
     - Baseline no-op preconditioner.
     - ``cg(..., M=identity(...))`` and ``gmres(..., M=identity(...))`` use
       the existing native unpreconditioned paths.
   * - ``diagonal(values)``
     - Explicit diagonal inverse-apply preconditioner.
     - Application uses a native mlx-sparse CPU/Metal primitive for rank-1 and
       rank-2 right-hand sides. ``cg`` and ``gmres`` dispatch to native
       diagonal-preconditioned Krylov paths. ``minres`` also accepts it when
       the inverse diagonal is finite and strictly positive.
   * - ``jacobi(A, check=False)``
     - Jacobi preconditioner built from the summed sparse diagonal.
     - Diagonal extraction uses existing sparse native kernels. ``cg(...,
       M=jacobi(A))`` and ``gmres(..., M=jacobi(A))`` dispatch to native
       C++/Metal Jacobi-preconditioned primitives. ``minres`` requires the
       shifted Jacobi inverse diagonal to be symmetric positive-definite.
   * - ``aspreconditioner(callable, A)``
     - Custom inverse-apply callable or object with ``solve(x)``.
     - ``gmres`` supports these through a host fallback loop with
       left-preconditioning semantics. ``cg`` remains native-only.
   * - ``from_factorized(solver)``
     - Exact inverse-apply wrapper around ``FactorizedSolve``, ``SparseLU``,
       or ``SparseCholesky``.
     - Native explicit factors apply through permutation/triangular-solve
       kernels on CPU or Metal, and GMRES uses typed native exact-factor
       entrypoints. Accelerate-backed ``FactorizedSolve`` objects apply through
       Apple's CPU sparse solver when guarded support is available.
   * - ``exact(A, method="auto")``
     - Convenience wrapper around ``linalg.factorized`` for diagnostics,
       testing, and small exact-preconditioner baselines.
     - Setup follows ``linalg.factorized``: guarded Accelerate when available
       and appropriate, otherwise native square LU/Cholesky fallback.
   * - ``ilu0(A)``
     - Natural-order, no-fill incomplete LU preconditioner for general square
       systems.
     - Setup runs in native C++ on CPU and preserves the input CSR sparsity
       pattern for ``L`` and ``U``. Application uses two native CSR triangular
       solves and can run on CPU or Metal. ``gmres(..., M=ilu0(A))`` dispatches
       to a native left-preconditioned GMRES entrypoint.
   * - ``ichol0(A)``
     - Natural-order, no-fill incomplete Cholesky preconditioner for SPD
       systems.
     - Setup runs in native C++ on CPU and preserves the symmetric lower CSR
       sparsity pattern with no fill. Application uses two native CSR
       triangular solves and can run on CPU or Metal. ``cg(..., M=ichol0(A))``
       dispatches to a native IC(0)-preconditioned CG entrypoint.
   * - ``chebyshev(A, degree=2)``
     - GPU-friendly polynomial preconditioner/smoother for SPD systems.
     - Setup runs in native C++ on CPU to compute Gershgorin bounds and
       optional Lanczos Ritz estimates. Application and
       ``cg(..., M=chebyshev(A))`` use native CPU/Metal kernels with only
       sparse matrix products and vector updates.

``cg`` currently supports native-backed ``identity``, ``diagonal``,
``jacobi``, ``ichol0``, and ``chebyshev`` preconditioners. ``gmres`` supports
``identity``, ``diagonal``/``jacobi``, ``ilu0``, exact-factor preconditioners,
and explicit inverse-apply callables or objects. For ``diagonal``/``jacobi``,
``ilu0``, and exact native or Accelerate factors, GMRES builds the Krylov basis for
``M^{-1} A`` and checks convergence against the true residual ``b - A @ x``
inside native solver entrypoints. Custom callable preconditioners use the
documented host GMRES fallback for inverse applications.

``minres`` accepts ``identity`` and finite strictly positive ``diagonal`` or
``jacobi`` preconditioners. This restriction is intentional: preconditioned
MINRES requires a symmetric positive-definite preconditioner. By default
``minres`` validates this for diagonal/Jacobi objects before entering the native
solver. Passing ``check_preconditioner=False`` disables the Python validation,
but the native recurrence still reports numerical breakdown for invalid inverse
diagonal entries.

Jacobi and diagonal preconditioner application do not use Accelerate because
the current mlx-sparse Accelerate integration is for direct sparse
factorization/solve objects. The native diagonal/Jacobi Krylov paths use the
selected MLX CPU or Metal device, including PCG and preconditioned MINRES solver
kernels. ILU(0) and IC(0) setup are native CPU incomplete-factorization passes,
application uses the existing native CSR triangular-solve kernels, so the apply
phase can run on CPU or Metal depending on the selected MLX device. The native
``cg(..., M=ichol0(A))`` loop is CPU-hosted because IC(0) application is a
triangular dependency chain rather than an elementwise preconditioner. ILU(0)
and IC(0) do not use Accelerate because Apple's sparse solver APIs do not
expose incomplete-factor setup or explicit factors compatible with
``CSRArray``. Chebyshev setup does not use Accelerate because it is spectral
interval estimation over CSR data, Chebyshev application is SpMV plus vector
updates and follows the native CPU or Metal sparse kernels. Exact-factor
preconditioners preserve the existing Accelerate guards from
``linalg.factorized`` and use Accelerate only on Apple builds where it is
available and helpful, otherwise they reuse the native sparse LU/Cholesky solve
path. Python preconditioner objects are metadata and dispatch containers, the
exact LU/Cholesky apply sequence itself is exposed as native bindings so future
solver integrations can reuse the same primitive without adding Python callbacks
to their iterations.

Jacobi
------

``jacobi(A, check=False)`` accepts ``CSRArray``, ``COOArray``, ``CSCArray``, and
sparse-backed ``LinearOperator`` inputs. Inputs are normalized to canonical CSR
for diagonal extraction, so duplicate diagonal entries are summed according to
normal sparse-array semantics.

The inverse diagonal is:

.. code-block:: text

   omega / (diag(A) + shift)

Zero and near-zero shifted diagonal entries are rejected by default:

.. code-block:: python

   import mlx_sparse as ms

   M = ms.linalg.preconditioners.jacobi(A)
   x, info = ms.linalg.cg(A, b, M=M)

Use ``shift`` for explicit regularization. Use ``zero_policy="unit"`` only
when replacing zero shifted diagonal entries with ``1`` before inversion is the
intended behavior. No diagonal shift or pivot perturbation is applied silently.

Pass ``check=True`` when the preconditioner should advertise itself as positive
definite. This performs the cheap necessary check that ``omega`` is positive
and that every shifted diagonal entry is strictly positive. It does not prove
that the original matrix is SPD. The check is applied before any
``zero_policy="unit"`` replacement, so a zero shifted diagonal is still
rejected.

Example: PCG with Jacobi
------------------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   n = 64
   T = scipy.sparse.diags(
       [-np.ones(n - 1), 2.5 * np.ones(n), -np.ones(n - 1)],
       offsets=[-1, 0, 1],
       format="csr",
       dtype=np.float32,
   )
   scale = np.geomspace(1.0e-2, 1.0e2, n).astype(np.float32)
   D = scipy.sparse.diags(scale, format="csr", dtype=np.float32)
   A_sp = (D @ T @ D).astype(np.float32).tocsr()

   A = ms.from_scipy(A_sp)
   x_true = mx.sin(mx.linspace(0.0, np.pi, n))
   b = A @ x_true

   M = ms.linalg.preconditioners.jacobi(A, check=True)
   x, info = ms.linalg.cg(A, b, M=M, rtol=1e-6, maxiter=512)

   assert info == 0

Example: GMRES with Jacobi
--------------------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   A_sp = scipy.sparse.csr_array(
       np.array(
           [
               [7.0, -2.0, 0.5, 0.0],
               [1.0, 5.5, -1.0, 0.25],
               [0.0, -0.5, 4.5, 1.0],
               [0.25, 0.0, -0.75, 3.75],
           ],
           dtype=np.float32,
       )
   )
   A = ms.from_scipy(A_sp)
   b = mx.array([1.0, -2.0, 0.5, 3.0], dtype=mx.float32)

   M = ms.linalg.preconditioners.jacobi(A)
   x, info = ms.linalg.gmres(A, b, M=M, rtol=1e-6, restart=4, maxiter=32)

   assert info == 0

ILU(0)
------

``ilu0(A, shift=0.0, check=True, reuse_analysis=False)`` accepts square
``CSRArray``, ``COOArray``, ``CSCArray``, and sparse-backed ``LinearOperator``
inputs. Inputs are normalized to canonical CSR before setup. The setup is
natural-order ILU(0): no fill-reducing ordering, no pivoting, and no new
off-diagonal entries are introduced. ``L`` keeps the original lower sparsity
pattern plus an implicit unit diagonal, while ``U`` keeps the original upper
pattern including the diagonal.

Every row must contain an explicit diagonal entry. ``shift`` is added only to
existing diagonal entries and is never used to create missing structure. With
``check=True`` the setup rejects zero or near-zero pivots using a scale-aware
guard. ``check=False`` disables the near-zero guard but still rejects exact
zero and non-finite pivots.

``reuse_analysis=True`` caches the triangular-solve diagonal-position and
level-schedule analysis objects for repeated application. It is opt-in because
the best choice depends on matrix shape, device, and RHS rank.

Example: GMRES with ILU(0)
--------------------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   n = 64
   lower = -0.25 * np.ones(n - 1, dtype=np.float32)
   diag = 2.5 * np.ones(n, dtype=np.float32)
   upper = -1.0 * np.ones(n - 1, dtype=np.float32)
   A_sp = scipy.sparse.diags(
       [lower, diag, upper],
       offsets=[-1, 0, 1],
       format="csr",
       dtype=np.float32,
   )
   A = ms.from_scipy(A_sp)
   b = mx.ones((n,), dtype=mx.float32)

   M = ms.linalg.preconditioners.ilu0(A)
   x, info = ms.linalg.gmres(A, b, M=M, rtol=5e-4, restart=8, maxiter=128)

   assert info == 0

IC(0)
-----

``ichol0(A, shift=0.0, check=True)`` accepts square ``CSRArray``,
``COOArray``, ``CSCArray``, and sparse-backed ``LinearOperator`` inputs that
represent symmetric positive-definite systems. Inputs are normalized to
canonical CSR before setup. The setup is natural-order IC(0): no
fill-reducing ordering, no pivoting, and no new off-diagonal entries are
introduced. The stored ``L`` factor uses the symmetric lower pattern of the
input, so upper-only symmetric storage is mirrored into the lower factor
without densifying the matrix.

Every row must contain an explicit diagonal entry. ``shift`` is a non-negative
scalar added only to existing diagonal entries and is never used to create
missing structure. With ``check=True`` the setup rejects non-symmetric mirrored
numeric entries and non-positive or near-zero pivots using a scale-aware guard.
``check=False`` relaxes the symmetry and near-zero pivot checks, but non-finite
and non-positive pivots remain errors because the Cholesky square root would be
invalid.

Example: PCG with IC(0)
-----------------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   grid = 16
   main = 4.0 * np.ones(grid, dtype=np.float32)
   off = -1.0 * np.ones(grid - 1, dtype=np.float32)
   T = scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr")
   I = scipy.sparse.eye(grid, format="csr", dtype=np.float32)
   Y = scipy.sparse.diags([off, off], [-1, 1], shape=(grid, grid), format="csr")
   A_sp = (scipy.sparse.kron(I, T) + scipy.sparse.kron(Y, I)).astype(np.float32)

   A = ms.from_scipy(A_sp)
   b = mx.ones((A.shape[0],), dtype=mx.float32)

   M = ms.linalg.preconditioners.ichol0(A)
   x, info = ms.linalg.cg(A, b, M=M, rtol=1e-4, maxiter=512)

   assert info == 0

Example: MINRES with Jacobi
---------------------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   A_sp = scipy.sparse.csr_array(
       np.array(
           [
               [2.0, 3.0, 0.0],
               [3.0, 2.0, 0.5],
               [0.0, 0.5, 4.0],
           ],
           dtype=np.float32,
       )
   )
   A = ms.from_scipy(A_sp)
   b = mx.array([1.0, -2.0, 0.5], dtype=mx.float32)

   M = ms.linalg.preconditioners.jacobi(A, check=True)
   x, info = ms.linalg.minres(A, b, M=M, rtol=1e-6, maxiter=64)

   assert info == 0

Chebyshev
---------

``chebyshev(A, degree=2, lambda_min=None, lambda_max=None, estimate=True)``
constructs a fixed-degree polynomial inverse approximation for SPD matrices.
It is useful as a GPU-friendly smoother because each application uses only
``A @ x`` and vector updates. When explicit spectral bounds are omitted, setup
uses native Gershgorin bounds and, by default, native Lanczos Ritz estimates to
obtain a positive interval. If no valid interval can be established, setup
raises and asks for explicit ``lambda_min``/``lambda_max`` values.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   grid = 16
   main = 4.0 * np.ones(grid, dtype=np.float32)
   off = -1.0 * np.ones(grid - 1, dtype=np.float32)
   T = scipy.sparse.diags([off, main, off], [-1, 0, 1], format="csr")
   I = scipy.sparse.eye(grid, format="csr", dtype=np.float32)
   Y = scipy.sparse.diags([off, off], [-1, 1], shape=(grid, grid), format="csr")
   A = ms.from_scipy((scipy.sparse.kron(I, T) + scipy.sparse.kron(Y, I)).astype(np.float32))
   b = mx.ones((A.shape[0],), dtype=mx.float32)

   M = ms.linalg.preconditioners.chebyshev(A, degree=2)
   x, info = ms.linalg.cg(A, b, M=M, rtol=1e-4, maxiter=512)

   assert info == 0

Chebyshev is not an incomplete factorization and does not require triangular
solves, so it is attractive on Metal for Poisson-like SPD problems. The
tradeoff is that quality depends on the spectral interval and polynomial
degree, each preconditioner application costs ``degree`` sparse matrix-vector
products.

Example: GMRES with an Exact Factor
-----------------------------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   A_sp = scipy.sparse.csr_array(
       np.array(
           [
               [5.0, -1.0, 0.0],
               [0.5, 4.0, -1.5],
               [0.0, 1.0, 3.0],
           ],
           dtype=np.float32,
       )
   )
   A = ms.from_scipy(A_sp)
   b = mx.array([2.0, -1.0, 0.5], dtype=mx.float32)

   M = ms.linalg.preconditioners.exact(A, method="lu")
   x, info = ms.linalg.gmres(A, b, M=M, rtol=1e-6, restart=2, maxiter=4)

   assert info == 0

Exact-factor preconditioners are primarily for diagnostics, small systems, and
validating solver/preconditioner plumbing. They include direct factorization
setup cost, so they should not be presented as a performance replacement for
incomplete preconditioners such as ILU(0) or IC(0).

Preconditioner metadata
-----------------------

Preconditioner objects expose ``shape``, ``dtype``, ``kind``,
``is_symmetric``, ``is_positive_definite``, ``setup_device``,
``apply_device``, ``nnz``, and ``setup_info``. ILU(0) exposes ``nnz_L`` and
``nnz_U``, IC(0) exposes ``nnz_L``. For unchecked Jacobi,
``is_positive_definite`` is conservative and remains ``False`` even when the
diagonal is positive. Use ``check=True`` to request the cheap positive-diagonal
validation described above.

Choosing a preconditioner
-------------------------

For the current v0.0.5b0 support:

* Use ``identity`` as a baseline.
* Use ``jacobi`` for cheap SPD or diagonally dominant systems with ``cg``,
  ``gmres``, or ``minres`` when the MINRES preconditioner is SPD.
* Use ``diagonal(..., inverse=True)`` when a safe inverse diagonal is already
  available.
* Use ``ilu0`` with ``gmres`` for general nonsymmetric systems when a stronger
  native preconditioner is worth the CPU setup and two triangular solves per
  application.
* Use ``ichol0`` with ``cg`` for SPD systems when a stronger native
  preconditioner than Jacobi is worth the CPU setup and two triangular solves
  per application.
* Use ``chebyshev`` with ``cg`` for SPD systems when a GPU-friendly polynomial
  smoother is preferred over triangular solves. It needs a positive spectral
  interval and costs ``degree`` sparse matrix-vector products per application.
* Use ``exact`` or ``from_factorized`` as a diagnostic baseline or when a
  reusable direct factorization already exists.
* Use custom callables with ``gmres`` only when the convenience of a host
  fallback outweighs the Python callback cost.

ILUT, ILU(k), sparse approximate inverse methods, and AMG are not part of the
current support surface.
