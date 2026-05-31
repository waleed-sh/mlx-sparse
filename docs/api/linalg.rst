Sparse linalg API
=================

.. module:: mlx_sparse.linalg

The linalg namespace contains sparse-native solvers and factorizations. Public
functions accept ``CSRArray``, ``COOArray``, and ``CSCArray`` inputs, dense
arrays are not silently converted because dense linear algebra belongs in
``mlx.linalg``. COO and CSC inputs are normalized once at solver entry. The
native sparse kernels use CSR, Accelerate-enabled direct solves normalize
supported real inputs to canonical CSC before calling Apple's framework.

.. seealso::

   See :doc:`../user_guide/linalg_solvers` for the solver support matrix,
   including Full CPU + GPU, CPU only, GPU only, Partial, and Accelerate
   coverage labels.

Iterative solvers
-----------------

.. autofunction:: cg
.. autofunction:: gmres
.. autofunction:: minres

Preconditioners
---------------

.. automodule:: mlx_sparse.linalg.preconditioners
   :members:

.. currentmodule:: mlx_sparse.linalg

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
.. autofunction:: factorized
.. autofunction:: spsolve

.. autoclass:: FactorizedSolve
   :members:

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
