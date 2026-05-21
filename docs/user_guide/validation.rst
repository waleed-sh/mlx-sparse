Validation
==========

The constructors :func:`~mlx_sparse.csr_array` and
:func:`~mlx_sparse.coo_array` both accept a ``validate`` keyword that controls
how thoroughly the input arrays are checked. Three levels are available.

Validation levels
-----------------

``"metadata"`` (default)
~~~~~~~~~~~~~~~~~~~~~~~~~

Checks ranks, lengths, and dtypes without reading any array values. Safe to
call on device-resident arrays.

Checks performed for CSR:

.. code-block:: text

   data.ndim == 1
   indices.ndim == 1
   indptr.ndim == 1
   data.shape[0] == indices.shape[0]
   indptr.shape[0] == n_rows + 1
   indices.dtype in {int32, int64}
   indptr.dtype in {int32, int64}
   indices.dtype == indptr.dtype
   data.dtype in {float32, float16, bfloat16, complex64}

Checks performed for COO:

.. code-block:: text

   data.ndim == 1
   row.ndim == 1
   col.ndim == 1
   data.shape[0] == row.shape[0] == col.shape[0]
   row.dtype in {int32, int64}
   col.dtype in {int32, int64}
   row.dtype == col.dtype
   data.dtype in {float32, float16, bfloat16, complex64}

``"full"`` / ``True``
~~~~~~~~~~~~~~~~~~~~~~

Performs all metadata checks and additionally reads the array values. For CSR
this verifies:

.. code-block:: text

   indptr[0] == 0
   indptr[-1] == nnz
   indptr is monotonically nondecreasing
   0 <= indices[j] < n_cols  for all j

For COO this verifies:

.. code-block:: text

   0 <= row[i] < n_rows  for all i
   0 <= col[i] < n_cols  for all i

**Value-level checks require materializing the arrays.** This
calls ``mx.eval`` and transfers data over the device-to-CPU bridge. For large
matrices that are already on the GPU, this synchronization can be expensive.
Use full validation:

* When constructing from host arrays (NumPy, Python lists) where
  synchronization cost is zero.
* In test code to catch malformed inputs early.
* When debugging unexpected results (wrong indices, wrong indptr, etc.).

``False`` / ``"none"``
~~~~~~~~~~~~~~~~~~~~~~~

Skips all validation. Use only when the inputs are known correct and
construction performance is critical (e.g. constructing the same sparse
structure thousands of times in a benchmark loop).

Setting validation on a constructor call
-----------------------------------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   data = mx.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
   indices = mx.array(np.array([0, 1, 2], dtype=np.int32))
   indptr = mx.array(np.array([0, 2, 3], dtype=np.int32))

   # Default: metadata only (no synchronization).
   A = ms.csr_array((data, indices, indptr), shape=(2, 3))

   # Full validation from a host context.
   A = ms.csr_array((data, indices, indptr), shape=(2, 3), validate="full")

   # No checks. Caller guarantees correctness.
   A = ms.csr_array((data, indices, indptr), shape=(2, 3), validate=False)

Native operations do not re-run full validation. They check ranks, shapes, and
dtypes at the C++ level, but they do not scan index bounds or verify indptr
monotonicity. The assumption is that the Python constructor already ran at
least metadata validation when the array was built.

Typical workflow
----------------

For matrices assembled on the host (from NumPy, SciPy, or Python data):

1. Construct with ``validate="full"`` or ``validate="metadata"`` (default).
2. The full check is cheap because the data is already on the host.
3. Subsequent sparse operations on the validated object skip value-level checks.

For matrices transferred from a device computation or from another framework:

1. Use the default ``validate="metadata"`` to check structural consistency
   without triggering host synchronization.
2. Run full validation once in a development/debug build to confirm correctness.
3. Remove it or set ``validate=False`` in production.

.. warning::

   Using ``validate=False`` with incorrectly formed inputs (e.g. ``indptr``
   that doesn't start at 0, out-of-bounds column indices) will produce
   undefined behavior at the C++ level and may cause crashes or silently wrong
   results. Only use it when you are certain the inputs are valid.
