Validation
==========

The constructors :func:`~mlx_sparse.csr_array`, :func:`~mlx_sparse.csc_array`,
and :func:`~mlx_sparse.coo_array` accept a ``validate`` keyword that controls
buffer validation. Three levels are available.

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

CSC has the same checks, with ``indptr.shape[0] == n_cols + 1``.

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

Performs all metadata checks and additionally reads the index buffers. For CSR
this verifies:

.. code-block:: text

   indptr[0] == 0
   indptr[-1] == nnz
   indptr is monotonically nondecreasing
   0 <= indices[j] < n_cols  for all j

CSC checks the same pointer invariants and requires
``0 <= indices[j] < n_rows`` for all stored entries.

For COO this verifies:

.. code-block:: text

   0 <= row[i] < n_rows  for all i
   0 <= col[i] < n_cols  for all i

Full validation also checks any declared sorted or canonical format, as
described below. It raises ``ValueError`` on a false declaration and does not
repair the buffers.

**Value-level checks materialize index buffers on the host.** This calls
``mx.eval`` and copies the indices into NumPy arrays. For GPU inputs, the host
synchronization can be expensive. Format checks reuse those copies and do not
read the numerical data buffer. Compressed validation takes
``O(nnz + n_rows)`` for CSR or ``O(nnz + n_cols)`` for CSC. COO validation takes
``O(nnz)``. These scans do not sort the buffers or iterate over rows in Python.

Use full validation:

* When accepting buffers whose values and format declarations need checking.
* In test code to catch malformed inputs early.
* When debugging unexpected results (wrong indices, wrong indptr, etc.).

``False`` / ``"none"``
~~~~~~~~~~~~~~~~~~~~~~~

``False`` and ``"none"`` select the same mode. Both skip buffer rank, length,
dtype, and value checks without reading array values or synchronizing to host.
Inputs must already satisfy the sparse format invariants.

Shape normalization, buffer conversion to MLX arrays, and input unpacking
still apply. A negative dimension, a shape that is not rank 2, or an invalid
buffer tuple is rejected in every validation mode. Unknown modes raise
``ValueError``.

Sorted and canonical declarations
---------------------------------

``sorted_indices=True`` asserts that CSR column indices are nondecreasing
within each row, or CSC row indices are nondecreasing within each column.
Duplicate entries are allowed. Ordering does not extend across row or column
boundaries, and empty segments impose no ordering requirement.

``canonical=True`` asserts sorted, duplicate-free storage. CSR and CSC require
strictly increasing indices within each compressed segment. COO requires
coordinates sorted by row, then by column, with no repeated ``(row, col)``
pairs. Explicit zero values are permitted in canonical storage.

These are declarations about the supplied buffers. They do not request a sort
or duplicate summation. Use ``sort_indices()`` on a compressed array to sort
its indices while retaining duplicates, or ``canonicalize()`` to sort and sum
duplicates. Full validation leaves undeclared properties alone. Unsorted
buffers and duplicate entries remain valid when neither property is asserted.
It also leaves false flags unchanged when the buffers happen to be canonical.

The format definitions follow SciPy's
`CSR <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html>`_,
`CSC <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html>`_,
and `COO <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_array.html>`_
contracts. The validation policy here checks declarations under ``"full"``.
``"metadata"`` and ``"none"`` trust them without reading the index buffers.

.. warning::

   A false canonical declaration can produce incorrect results. Two values
   ``3`` and ``4`` at the same coordinate represent ``7``. A row norm computed
   without first combining those duplicates would be ``5``. Full validation
   rejects that input when ``canonical=True``. The default metadata check does
   not verify the declaration.

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

   # Skip buffer validation for known-valid inputs.
   A = ms.csr_array((data, indices, indptr), shape=(2, 3), validate="none")

Passing an existing sparse array to its constructor runs the selected checks
before returning the same instance. Its stored flags are retained and are
checked under full validation. The ``sorted_indices`` and ``canonical``
keywords apply to raw buffers, not to existing arrays.

.. code-block:: python

   # Check an array that was created with metadata validation.
   A_checked = ms.csr_array(A, shape=A.shape, validate="full")
   assert A_checked is A

Direct construction of ``CSRArray``, ``CSCArray``, or ``COOArray`` only
normalizes the shape. Use the corresponding constructor function when buffer
validation is required.

Native operations do not re-run full validation. They check ranks, shapes, and
dtypes at the C++ level, but they do not scan index bounds or verify indptr
monotonicity on every call. Callers must supply valid buffers and truthful
format declarations.

Typical workflow
----------------

For matrices assembled on the host (from NumPy, SciPy, or Python data):

1. Construct with ``validate="full"`` when importing buffers of uncertain
   provenance or asserting a sorted or canonical layout.
2. Account for the host index copies and linear scans, even for host inputs.
3. Subsequent sparse operations on the validated object skip value-level checks.

For matrices transferred from a device computation or from another framework:

1. Use the default ``validate="metadata"`` to check structural consistency
   without triggering host synchronization.
2. Run full validation once in a development/debug build to confirm correctness.
3. Use metadata validation in repeated construction paths once the buffer
   producer guarantees the format invariants.

.. warning::

   Using ``validate="metadata"`` or disabled validation with incorrect index
   values, such as ``indptr`` that does not start at 0 or out-of-bounds indices,
   may cause crashes or silently wrong results in native operations. Use these
   modes only when the buffer producer guarantees valid index values.
