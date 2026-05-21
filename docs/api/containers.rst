Containers
==========

.. currentmodule:: mlx_sparse

Sparse array containers in mlx-sparse are immutable frozen dataclasses. They
hold ``mlx.core.array`` buffers and format metadata but do not subclass
``mx.array``. Structural operations return new instances and nothing is mutated
in place.

CSRArray
--------

.. autoclass:: CSRArray
   :members:
   :undoc-members: False
   :special-members: __matmul__
   :show-inheritance:

   .. rubric:: Properties

   .. autosummary::

      ~CSRArray.nnz
      ~CSRArray.dtype
      ~CSRArray.index_dtype
      ~CSRArray.ndim
      ~CSRArray.T
      ~CSRArray.H

   .. rubric:: Methods

   .. autosummary::

      ~CSRArray.todense
      ~CSRArray.sort_indices
      ~CSRArray.sum_duplicates
      ~CSRArray.canonicalize
      ~CSRArray.transpose
      ~CSRArray.conj
      ~CSRArray.conjugate

COOArray
--------

.. autoclass:: COOArray
   :members:
   :undoc-members: False
   :show-inheritance:

   .. rubric:: Properties

   .. autosummary::

      ~COOArray.nnz
      ~COOArray.dtype
      ~COOArray.index_dtype
      ~COOArray.ndim

   .. rubric:: Methods

   .. autosummary::

      ~COOArray.tocsr
      ~COOArray.todense

Utility functions
-----------------

.. autofunction:: issparse
