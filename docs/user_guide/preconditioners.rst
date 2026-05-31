.. _preconditioners:

Preconditioners
===============

``mlx_sparse.linalg.preconditioners`` contains native-backed inverse-apply
objects for sparse iterative solvers. A preconditioner ``M`` applies an
approximation to ``A^{-1}``, it is not interpreted as a sparse matrix to invert
implicitly.

Python preconditioner objects are API containers and dispatch helpers. Diagonal
application and Jacobi-preconditioned CG run through native C++/Metal
primitives under ``src/preconditioners``, they do not execute a Python callback
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
   * - ``jacobi(A, check=False)``
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

   A = ms.csr_array(
       (
           mx.array(A_sp.data, dtype=mx.float32),
           mx.array(A_sp.indices, dtype=mx.int32),
           mx.array(A_sp.indptr, dtype=mx.int32),
       ),
       shape=A_sp.shape,
       canonical=True,
   )
   x_true = mx.sin(mx.linspace(0.0, np.pi, n))
   b = A @ x_true

   M = ms.linalg.preconditioners.jacobi(A, check=True)
   x, info = ms.linalg.cg(A, b, M=M, rtol=1e-6, maxiter=512)

   assert info == 0

Preconditioner metadata
-----------------------

Preconditioner objects expose ``shape``, ``dtype``, ``kind``,
``is_symmetric``, ``is_positive_definite``, ``setup_device``,
``apply_device``, ``nnz``, and ``setup_info``. For unchecked Jacobi,
``is_positive_definite`` is conservative and remains ``False`` even when the
diagonal is positive. Use ``check=True`` to request the cheap positive-diagonal
validation described above.

Choosing a preconditioner
-------------------------

For the current v0.0.5b0 support:

* Use ``identity`` as a baseline.
* Use ``jacobi`` for cheap SPD or diagonally dominant systems.
* Use ``diagonal(..., inverse=True)`` when a safe inverse diagonal is already
  available.

ILU(0), IC(0), exact-factor wrappers, and GMRES preconditioning are planned
separately so each native solver path can be tested and benchmarked directly.
