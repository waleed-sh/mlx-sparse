Sparse formats
==============

mlx-sparse supports two 2D sparse matrix formats: COO (Coordinate) and CSR
(Compressed Sparse Row). Both store only the non-zero values and enough
structural information to reconstruct the full matrix. The choice of format
matters for performance: COO is the right choice for assembly, CSR is the
right choice for repeated products.

Coordinate (COO) format
------------------------

A COO matrix stores three parallel rank-1 arrays of length ``nnz``:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Field
     - Shape
     - Description
   * - ``data``
     - ``(nnz,)``
     - Non-zero values.
   * - ``row``
     - ``(nnz,)``
     - Row coordinate of each value. Integer dtype.
   * - ``col``
     - ``(nnz,)``
     - Column coordinate of each value. Same integer dtype as ``row``.

COO has no ordering requirement. Duplicate ``(row, col)`` pairs are permitted
and interpreted as a value that should be summed with its duplicates when the
matrix is converted or materialised. This makes COO ideal for:

* Assembling matrices from element-level contributions (finite elements, graph
  algorithms, Hamiltonians) where entries accumulate.
* One-shot construction followed by a single ``tocsr`` call.
* Interoperability with SciPy's ``coo_array`` and ``coo_matrix``, which share
  the same convention.

**Memory layout**::

   entry i:  (data[i], row[i], col[i])

For the 2x3 matrix ``[[1, 2, 0], [0, 0, 3]]``:

.. code-block:: text

   data = [1.0, 2.0, 3.0]
   row = [0, 0, 1]
   col = [0, 1, 2]

Compressed Sparse Row (CSR) format
------------------------------------

A CSR matrix stores three rank-1 arrays:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Field
     - Shape
     - Description
   * - ``data``
     - ``(nnz,)``
     - Non-zero values.
   * - ``indices``
     - ``(nnz,)``
     - Column index of each stored value. Integer dtype.
   * - ``indptr``
     - ``(n_rows + 1,)``
     - Row pointer array. ``indptr[i]`` is the position in ``data`` and
       ``indices`` where row ``i`` begins. ``indptr[i+1]`` is where it ends.
       Integer dtype, same as ``indices``.

Row ``i`` occupies the slice ``data[indptr[i] : indptr[i+1]]`` with column
indices ``indices[indptr[i] : indptr[i+1]]``.

**Format invariants** (validated by the constructor unless ``validate=False``):

.. code-block:: text

   data.ndim == indices.ndim == indptr.ndim == 1
   data.shape[0] == indices.shape[0]  (nnz)
   indptr.shape[0] == n_rows + 1
   indices.dtype == indptr.dtype  (int32 or int64)
   data.dtype in {float32, float16, bfloat16, complex64}

Additional **value-level** invariants (checked only with ``validate="full"``):

.. code-block:: text

   indptr[0] == 0
   indptr[-1] == nnz
   indptr is monotonically nondecreasing
   0 <= indices[j] < n_cols  for all j

For the same 2x3 matrix ``[[1, 2, 0], [0, 0, 3]]``:

.. code-block:: text

   data = [1.0, 2.0, 3.0]
   indices = [0, 1, 2]
   indptr = [0, 2, 3]   # row 0: [0,2), row 1: [2,3)

CSR is the format for all sparse-dense products. Its compressed row structure
makes sequential row access cache-friendly on CPU and maps naturally to a
one-thread-per-row GPU kernel.

Canonical form
--------------

An CSRArray is in **canonical form** when:

1. Column indices within each row are sorted in ascending order.
2. No row contains duplicate column indices.

The constructor never enforces or silently produces canonical form. The flags
``sorted_indices`` and ``has_canonical_format`` are hints. They are only
meaningful if the caller guarantees the corresponding property. To compute the
canonical form from data that may not satisfy it, call
:meth:`~mlx_sparse.CSRArray.canonicalize`:

.. code-block:: python

   # Assemble with potential duplicates (e.g. from finite-element assembly).
   coo = ms.coo_array((data, (row, col)), shape=(m, n))
   csr = coo.tocsr(canonical=True)  # sorts and sums duplicates

   # Or in two steps:
   csr = coo.tocsr()
   csr = csr.canonicalize()

Canonicalization sorts column indices per row (``sort_indices``) and then
merges duplicate columns by summation (``sum_duplicates``). The ``nnz`` after
canonicalization may be smaller than before.

Converting between formats
--------------------------

.. code-block:: python

   # COO -> CSR
   csr = coo.tocsr()  # preserves duplicates
   csr = coo.tocsr(canonical=True)  # sums duplicates

   # CSR -> dense (mx.array)
   dense = csr.todense()
   dense = ms.todense(csr)  # module-level alias

   # COO -> dense (via CSR internally)
   dense = coo.todense()

   # CSR structural operations
   csr_t = csr.T  # transposed CSRArray, shape (n_cols, n_rows)
   csr_h = csr.H  # Hermitian transpose (conj + T)

Dense-to-sparse conversion is available via :func:`~mlx_sparse.fromdense`
(:func:`~mlx_sparse.from_dense` and :func:`~mlx_sparse.from_numpy` are aliases
for readability):

.. code-block:: python

   import mlx.core as mx
   import mlx_sparse as ms

   dense = mx.array(some_numpy_array)
   csr = ms.fromdense(dense)  # all non-zeros
   csr = ms.fromdense(dense, threshold=1e-4)  # drop near-zeros

``fromdense`` synchronizes to host to discover which entries are non-zero, so
it is intended for one-shot construction rather than use inside a hot loop.
When importing from SciPy, prefer :func:`~mlx_sparse.from_scipy` or the generic
:func:`~mlx_sparse.asarray` helper:

.. code-block:: python

   import scipy.sparse
   sp = scipy.sparse.csr_array(dense_numpy)
   csr = ms.from_scipy(sp)
   csr = ms.asarray(sp)

Transpose semantics
-------------------

:attr:`~mlx_sparse.CSRArray.T` returns a new CSRArray with shape
``(n_cols, n_rows)`` built by treating each stored entry ``(row, col, val)``
as a new entry at ``(col, row, val)``. The result always has
``sorted_indices=True``. If the source had ``has_canonical_format=True``, the
transposed result inherits that flag (the structural property transfers
correctly under transpose).

:attr:`~mlx_sparse.CSRArray.H` (Hermitian transpose) applies
:meth:`~mlx_sparse.CSRArray.conj` first, then :meth:`~mlx_sparse.CSRArray.transpose`.
For real dtypes, ``H`` and ``T`` are equivalent.

:meth:`~mlx_sparse.CSRArray.conj` and :meth:`~mlx_sparse.CSRArray.conjugate`
apply ``mx.conjugate`` to ``data`` and share the existing ``indices`` and
``indptr`` arrays.

Empty rows and zero-nnz matrices
---------------------------------

Both formats support matrices with empty rows or zero non-zeros. An empty row
``i`` in CSR is represented by ``indptr[i] == indptr[i+1]``. A zero-nnz
CSRArray has ``data.shape == indices.shape == (0,)`` and
``indptr == [0] * (n_rows + 1)``.

.. code-block:: python

   # 3x3 identity matrix via COO
   data = mx.array([1.0, 1.0, 1.0], dtype=mx.float32)
   row = mx.array([0, 1, 2], dtype=mx.int32)
   col = mx.array([0, 1, 2], dtype=mx.int32)
   eye = ms.coo_array((data, (row, col)), shape=(3, 3)).tocsr(canonical=True)

   # 5x5 matrix with only the first row populated
   data = mx.array([1.0, 2.0], dtype=mx.float32)
   indices = mx.array([0, 4], dtype=mx.int32)
   indptr = mx.array([0, 2, 2, 2, 2, 2], dtype=mx.int32)
   csr = ms.csr_array((data, indices, indptr), shape=(5, 5))
