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
   :special-members: __matmul__, __getitem__
   :show-inheritance:

   .. rubric:: Selection

   ``A[rows]`` selects rows, ``A[:, s:e]`` selects a column range, and
   ``A[rows, s:e]`` does both, each returning a new :class:`CSRArray`. Rows are
   gathered whole, so the ``sorted_indices`` and ``has_canonical_format`` flags
   carry over rather than being dropped. See :meth:`CSRArray.__getitem__` for
   the supported keys and for what an out-of-range index does.

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

      ~CSRArray.__getitem__
      ~CSRArray.todense
      ~CSRArray.tocsc
      ~CSRArray.sort_indices
      ~CSRArray.sum_duplicates
      ~CSRArray.canonicalize
      ~CSRArray.transpose
      ~CSRArray.conj
      ~CSRArray.conjugate

CSCArray
--------

.. autoclass:: CSCArray
   :members:
   :undoc-members: False
   :special-members: __matmul__
   :show-inheritance:

   .. rubric:: Properties

   .. autosummary::

      ~CSCArray.nnz
      ~CSCArray.dtype
      ~CSCArray.index_dtype
      ~CSCArray.ndim
      ~CSCArray.T
      ~CSCArray.H

   .. rubric:: Methods

   .. autosummary::

      ~CSCArray.todense
      ~CSCArray.tocsr
      ~CSCArray.sort_indices
      ~CSCArray.sum_duplicates
      ~CSCArray.canonicalize
      ~CSCArray.transpose
      ~CSCArray.conj
      ~CSCArray.conjugate

COOArray
--------

.. autoclass:: COOArray
   :members:
   :undoc-members: False
   :special-members: __matmul__
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
      ~COOArray.tocsc
      ~COOArray.todense

Utility functions
-----------------

.. autofunction:: issparse
