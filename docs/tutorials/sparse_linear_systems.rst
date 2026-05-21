Solving sparse linear systems
==============================

This tutorial walks through the full workflow of assembling and solving a
sparse linear system using ``mlx_sparse.linalg``.  The running example is the
2D Poisson equation — a classical benchmark for sparse solvers that arises
naturally in heat conduction, electrostatics, and image processing.

The goal is to solve ``A x = b`` where ``A`` is the 2D discrete Laplacian on
an ``n×n`` grid, a sparse symmetric positive-definite (SPD) matrix with at
most 5 non-zeros per row, and ``b`` is a source term.

Assembling the system
---------------------

The 2D Laplacian on an ``n×n`` grid has ``N = n²`` unknowns.  Grid point
``(i, j)`` maps to index ``k = i*n + j``.  Each interior point connects to
its four neighbors with coefficient ``−1`` and has a self-coupling of ``+4``.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms
   from mlx_sparse import linalg


   def build_poisson_2d(n: int) -> ms.CSRArray:
       """2D 5-point Laplacian on an n×n grid, returned as a CSR matrix."""
       rows, cols, vals = [], [], []
       for i in range(n):
           for j in range(n):
               k = i * n + j
               rows.append(k); cols.append(k); vals.append(4.0)
               for ni, nj in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                   if 0 <= ni < n and 0 <= nj < n:
                       rows.append(k); cols.append(ni*n+nj); vals.append(-1.0)

       import scipy.sparse
       csr = scipy.sparse.coo_matrix(
           (np.array(vals, np.float32),
            (np.array(rows, np.int32), np.array(cols, np.int32))),
           shape=(n*n, n*n),
       ).tocsr()
       return ms.csr_array(
           (mx.array(csr.data), mx.array(csr.indices), mx.array(csr.indptr)),
           shape=csr.shape, canonical=True,
       )

   n = 32           # 32×32 grid → 1024×1024 system
   A = build_poisson_2d(n)
   print(f"shape={A.shape}, nnz={A.nnz}, density={A.nnz/A.shape[0]**2:.5f}")

This produces a ``1024×1024`` matrix with 4992 non-zeros — a density of
``0.0048``.  The sparsity grows as ``O(n²)`` rows but only ``O(n²)``
non-zeros (5 per row), so the density decreases as ``5/n²``.

Choosing a right-hand side
--------------------------

For this tutorial we use a smooth source term: the constant vector ``b = 1``.
In a real PDE solver, ``b`` encodes boundary conditions and volumetric sources.

.. code-block:: python

   b = mx.ones((n * n,), dtype=mx.float32)

Solving with Conjugate Gradients
---------------------------------

CG is the standard choice for large sparse SPD systems.  It requires only
matrix-vector products (one per iteration) and converges in at most ``N``
steps.  For the Poisson operator, the condition number grows as ``O(n²)``,
so a plain CG solve needs ``O(n)`` iterations for the ``n×n`` grid.

.. code-block:: python

   x_cg, info = linalg.cg(A, b, rtol=1e-6, maxiter=5000)
   mx.eval(x_cg)

   print(f"CG converged: {info == 0}  (info={info})")
   residual = mx.sqrt(mx.sum((A @ x_cg - b) ** 2))
   mx.eval(residual)
   print(f"||Ax - b|| / ||b|| = {float(residual) / float(mx.sqrt(mx.sum(b**2))):.2e}")

For a 32×32 grid (``n=32``, condition number ~1000), CG typically converges
in under 200 iterations with ``rtol=1e-6``.

Comparing solvers
-----------------

The Laplacian is symmetric so all three solvers apply.  GMRES works on any
system but uses more memory per iteration.  MINRES also handles symmetric
indefinite systems, making it useful when the Laplacian is shifted.

.. code-block:: python

   x_gmres,  info_gm = linalg.gmres( A, b, rtol=1e-6, restart=50)
   x_minres, info_mr = linalg.minres(A, b, rtol=1e-6)
   mx.eval(x_gmres, x_minres)

   for name, x in [("CG", x_cg), ("GMRES", x_gmres), ("MINRES", x_minres)]:
       rel = float(mx.sqrt(mx.sum((A @ x - b)**2))) / float(mx.sqrt(mx.sum(b**2)))
       print(f"{name:8s}  rel_residual={rel:.2e}")

Direct factorization with Cholesky
------------------------------------

For problems where the same matrix must be solved with many right-hand sides,
factorizing once and reusing the factors is more efficient.  ``sparse_cholesky``
computes a sparse lower-triangular factor ``L`` satisfying ``A = L @ L.T``.

.. code-block:: python

   chol = linalg.sparse_cholesky(A)

   # Solve three right-hand sides with one factorization
   rhs_list = [
       mx.ones((n*n,),  dtype=mx.float32),
       mx.zeros((n*n,), dtype=mx.float32).at[n*n//2].add(1.0),
       mx.array(np.random.default_rng(0).normal(size=n*n).astype(np.float32)),
   ]
   for i, rhs in enumerate(rhs_list):
       xi = chol.solve(rhs)
       mx.eval(xi)
       rel = float(mx.sqrt(mx.sum((A @ xi - rhs)**2))) / (float(mx.sqrt(mx.sum(rhs**2))) + 1e-12)
       print(f"RHS {i}: rel_residual={rel:.2e}")

The Cholesky path is faster than running CG three times when ``n`` is small
enough that the fill-in fits in memory.

Convenience: ``spsolve``
-------------------------

For a single right-hand side and no need to reuse factors, ``spsolve`` is the
most concise path.  It performs sparse LU internally.

.. code-block:: python

   x_direct = linalg.spsolve(A, b)
   mx.eval(x_direct)
   rel = float(mx.sqrt(mx.sum((A @ x_direct - b)**2))) / float(mx.sqrt(mx.sum(b**2)))
   print(f"spsolve: rel_residual={rel:.2e}")

Performance guidance
--------------------

* **Grid size and density**: the 2D Laplacian has density ``5/n²`` — it is
  extremely sparse even at large ``n``.  Sparse solvers exploit this; dense
  ``mx.linalg.solve`` pays ``O(n^6)`` for the full grid size.
* **CG vs direct**: CG converges in ``O(n)`` iterations for the 2D Laplacian.
  For large grids (``n ≥ 64``) CG beats direct factorization because
  Cholesky fill-in can be dense in the reordered system.
* **Device selection**: call ``ms.use_gpu()`` before building the matrix to
  keep all sparse-dense products on the Metal path.  The native solver kernels
  dispatch through MLX's backend automatically.
* **Preconditioners**: none of the current native solvers accept an external
  preconditioner.  For large ill-conditioned systems, consider reducing ``n``
  or using a domain-decomposition split before passing to the solver.

See also
--------

* :doc:`finite_difference` — assembling the Laplacian stencil
* :doc:`../user_guide/linalg` — design contract and numerical scope
* :doc:`../notebooks/13_linalg_solvers` — interactive solver notebook
* :doc:`../notebooks/14_linalg_factorizations` — factorization notebook
