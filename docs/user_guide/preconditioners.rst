.. _preconditioners:

Preconditioners
===============

``mlx_sparse.linalg.preconditioners`` contains native-backed inverse-apply
objects for sparse iterative solvers. A preconditioner ``M`` applies an
approximation to ``A^{-1}``; it is not interpreted as a sparse matrix to invert
implicitly.

Python preconditioner objects are API containers and dispatch helpers. Diagonal
application and Jacobi-preconditioned CG run through native C++/Metal
primitives under ``src/preconditioners``; they do not execute a Python callback
inside the Krylov iteration.

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
     - ``cg(..., M=identity(...))`` uses the existing native unpreconditioned
       CG path.
   * - ``diagonal(values)``
     - Explicit diagonal inverse-apply preconditioner.
     - Application uses a native mlx-sparse CPU/Metal primitive for rank-1 and
       rank-2 right-hand sides.
   * - ``jacobi(A)``
     - Jacobi preconditioner built from the summed sparse diagonal.
     - Diagonal extraction uses existing sparse native kernels. ``cg(...,
       M=jacobi(A))`` dispatches to a native C++/Metal
       Jacobi-preconditioned CG primitive.

``cg`` currently supports ``identity``, ``diagonal``, and ``jacobi``
preconditioners. ``gmres`` and ``minres`` preconditioner support are still
future work.

Jacobi and diagonal preconditioner application do not use Accelerate because
the current mlx-sparse Accelerate integration is for direct sparse
factorization/solve objects. Future exact-factor preconditioners will preserve
the existing Accelerate guards and use Accelerate only on Apple builds where it
is available and helpful.

Jacobi
------

``jacobi(A)`` accepts ``CSRArray``, ``COOArray``, ``CSCArray``, and
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

Choosing a preconditioner
-------------------------

For the current v0.0.5b0 support:

* Use ``identity`` as a baseline.
* Use ``jacobi`` for cheap SPD or diagonally dominant systems.
* Use ``diagonal(..., inverse=True)`` when a safe inverse diagonal is already
  available.

ILU(0), IC(0), exact-factor wrappers, and GMRES preconditioning are planned
separately so each native solver path can be tested and benchmarked directly.
