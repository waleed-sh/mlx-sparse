Sparse linalg API
=================

.. module:: mlx_sparse.linalg

The linalg namespace contains sparse-native solvers and factorizations. Public
functions accept ``CSRArray`` and ``COOArray`` inputs, dense arrays are not
silently converted because dense linear algebra belongs in ``mlx.linalg``.

**GPU coverage summary**: call ``ms.use_gpu()`` to enable Metal dispatch:

* **Full GPU**: :func:`cg`, :func:`dot`, :func:`vdot`,
  :meth:`SparseCholesky.solve`, :meth:`SparseLU.solve`, :func:`spsolve`
  (triangular-solve and permutation steps).
* **Partial GPU** (Krylov step on GPU, small dense post-processing on CPU):
  :func:`gmres`, :func:`minres`, :func:`eigsh`, :func:`eigs`,
  :func:`lanczos`.
* **CPU only**: :func:`svds`, :func:`sparse_cholesky` / :func:`cholesky`
  (factorisation step), :func:`sparse_lu` / :func:`splu` (factorisation
  step).

See :doc:`../user_guide/linalg` and :doc:`../supported` for the detailed
breakdown and the planned GPU paths.

Iterative solvers
-----------------

.. autofunction:: cg
.. autofunction:: gmres
.. autofunction:: minres

Spectral routines
-----------------

.. autofunction:: lanczos
.. autofunction:: eigsh
.. autofunction:: eigs
.. autofunction:: svds

Sparse direct factorizations
----------------------------

.. autofunction:: sparse_cholesky
.. autofunction:: cholesky
.. autofunction:: sparse_lu
.. autofunction:: splu
.. autofunction:: spsolve

.. autoclass:: SparseCholesky
   :members:

.. autoclass:: SparseLU
   :members:

Operators and sparse reductions
-------------------------------

.. autoclass:: LinearOperator
   :members:

.. autofunction:: aslinearoperator
.. autofunction:: dot
.. autofunction:: vdot
