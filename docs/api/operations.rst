Operations
==========

.. currentmodule:: mlx_sparse

These module-level functions wrap the native sparse primitives. In most cases
the ``@`` operator on :class:`COOArray`, :class:`CSRArray`, and
:class:`CSCArray` is the preferred spelling. These functions exist for
explicit dispatch and for callers who prefer a functional style.

Sparse-dense operations return lazy ``mlx.core.array`` values. Sparse-sparse
``matmat`` operations return new sparse arrays and may synchronize structure to
host because their output sparsity pattern is data-dependent.

csr\_matvec
-----------

.. autofunction:: csr_matvec

coo\_matvec
-----------

.. autofunction:: coo_matvec

csc\_matvec
-----------

.. autofunction:: csc_matvec

csc\_matvec\_transpose
----------------------

.. autofunction:: csc_matvec_transpose

csr\_matmul
-----------

.. autofunction:: csr_matmul

coo\_matmul
-----------

.. autofunction:: coo_matmul

csc\_matmul
-----------

.. autofunction:: csc_matmul

csr\_batched\_matvec
--------------------

.. autofunction:: csr_batched_matvec

coo\_batched\_matvec
--------------------

.. autofunction:: coo_batched_matvec

csc\_batched\_matvec
--------------------

.. autofunction:: csc_batched_matvec

csr\_batched\_matmul
--------------------

.. autofunction:: csr_batched_matmul

coo\_batched\_matmul
--------------------

.. autofunction:: coo_batched_matmul

csc\_batched\_matmul
--------------------

.. autofunction:: csc_batched_matmul

csr\_matmat
-----------

.. autofunction:: csr_matmat

coo\_matmat
-----------

.. autofunction:: coo_matmat

csc\_matmat
-----------

.. autofunction:: csc_matmat

Reductions
----------

All sparse containers expose reduction methods as well as module-level helper
functions. ``row_sums`` / ``col_sums`` return the same dtype as the sparse
values, ``row_norms`` / ``col_norms`` return ``float32``, and ``diagonal`` /
``trace`` sum duplicate diagonal entries.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Format
     - Functions
   * - COO
     - :func:`coo_row_sums`, :func:`coo_col_sums`,
       :func:`coo_row_norms`, :func:`coo_col_norms`,
       :func:`coo_diagonal`, :func:`coo_trace`
   * - CSR
     - :func:`csr_row_sums`, :func:`csr_col_sums`,
       :func:`csr_row_norms`, :func:`csr_diagonal`, :func:`csr_trace`
   * - CSC
     - :func:`csc_row_sums`, :func:`csc_col_sums`,
       :func:`csc_row_norms`, :func:`csc_col_norms`,
       :func:`csc_diagonal`, :func:`csc_trace`

COO and CSC reductions are native C++/Metal paths. Norm reductions use dense
matrix semantics, so non-canonical COO/CSC inputs are canonicalized before
norming to ensure duplicate coordinates are summed before the square is taken.

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
   Yb = A @ Xb  # rhs.ndim > 2 -> csr_matmul(A, Xb) returns mx.array

The explicit function calls accept the same arguments:

.. code-block:: python

   y = ms.csr_matvec(A, x)
   Y = ms.csr_matmul(A, X)
   yb = ms.csr_batched_matvec(A, xb)
   Yb = ms.csr_batched_matmul(A, Xb)
   C = ms.csr_matmat(A, B)

For :class:`COOArray` and :class:`CSCArray`, dense RHS dispatch mirrors CSR:
rank-1 RHS uses format-native matvec, rank-2 RHS uses format-native matmul,
and higher-rank RHS is flattened into the corresponding native batched
primitive. Same-format sparse-sparse products are also native:
``COOArray @ COOArray`` returns canonical COO via :func:`coo_matmat`, and
``CSCArray @ CSCArray`` returns canonical CSC via :func:`csc_matmat`.
Mixed-format sparse-sparse products remain explicit: convert the operand
yourself when a different storage format is acceptable.

All sparse-dense products validate that ``rhs.dtype == A.data.dtype``. There
is no implicit type promotion. See :doc:`../user_guide/dtype_policy` for the
full dtype matrix.

Native dispatch notes
---------------------

Sparse-dense matvec and matmul for COO, CSR, and CSC are fixed-output
primitives and stay lazy in the MLX graph. Explicit batched helpers use native
C++/Metal kernels for leading batch dimensions rather than materializing dense
matrices in Python.

Transpose products used by autodiff are also native. On Metal, ``float32``
transpose matvec/matmul use atomic scatter-add kernels. Other GPU value dtypes
lower through native ``csr_transpose`` followed by the ordinary native product.
Sparse-sparse ``matmat`` is different: its output shape depends on the input
structure, so it performs symbolic/count work and synchronizes enough structure
to allocate compact output buffers. CSR uses row symbolic/numeric assembly, COO
groups coordinate rows without routing through CSR, and CSC walks right-hand
columns against left-hand compressed columns to produce sorted output columns.

``coo_matvec`` / ``coo_matmul`` are native coordinate scatter products. On
Metal, ``float32`` uses atomic scatter-add over stored coordinates, other
value dtypes stay native through a serial scatter kernel because Metal does
not provide storage-compatible atomic adds for ``float16``, ``bfloat16``, or
``complex64``.

``csc_matvec`` / ``csc_matmul`` are native compressed-column scatter products.
Forward CSC products walk columns and scatter into output rows, on Metal,
``float32`` uses atomic scatter-add while other value dtypes use native serial
scatter. CSC transpose products are the layout's reduction fast path: each
output entry is one compressed-column dot product.
