Operations
==========

.. currentmodule:: mlx_sparse

These module-level functions wrap the native sparse primitives. In most cases
the ``@`` operator on :class:`CSRArray` is the preferred spelling. These
functions exist for explicit dispatch and for callers who prefer a functional
style.

All operations return lazy ``mlx.core.array`` values or new
:class:`CSRArray` instances and do not call ``mx.eval`` internally, with the
exception of :func:`csr_matmat` which must synchronize to host because its
output sparsity pattern is data-dependent.

csr\_matvec
-----------

.. autofunction:: csr_matvec

csr\_matmul
-----------

.. autofunction:: csr_matmul

csr\_matmat
-----------

.. autofunction:: csr_matmat

todense
-------

.. autofunction:: todense

identity\_like
--------------

.. autofunction:: identity_like

is\_available
-------------

.. autofunction:: is_available

Dispatch summary
-----------------

The ``@`` operator on :class:`CSRArray` dispatches based on the type and rank
of ``rhs``:

.. code-block:: python

   C = A @ B  # rhs is CSRArray -> csr_matmat(A, B) returns CSRArray
   y = A @ x  # rhs.ndim == 1 -> csr_matvec(A, x) returns mx.array
   Y = A @ X  # rhs.ndim == 2 -> csr_matmul(A, X) returns mx.array
   Y = A @ Xb  # rhs.ndim > 2 -> csr_matmul(A, Xb) returns mx.array

The explicit function calls accept the same arguments:

.. code-block:: python

   y = ms.csr_matvec(A, x)
   Y = ms.csr_matmul(A, X)
   C = ms.csr_matmat(A, B)

Both :func:`csr_matvec` and :func:`csr_matmul` validate that
``rhs.dtype == A.data.dtype``. There is no implicit type promotion. See
:doc:`../user_guide/dtype_policy` for the full dtype matrix.
