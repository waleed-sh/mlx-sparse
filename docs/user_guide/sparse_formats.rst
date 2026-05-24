Sparse formats
==============

mlx-sparse supports three 2D sparse matrix formats: COO (Coordinate), CSR
(Compressed Sparse Row), and CSC (Compressed Sparse Column). All store only
the non-zero values and enough
structural information to reconstruct the full matrix. The choice of format
matters for performance: COO is the right choice for assembly, CSR is the
row-oriented product format, and CSC is the column-oriented companion used
when column access or transpose products dominate.

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
* Native one-shot sparse-dense products when converting to a compressed format
  would cost more than the product itself.

**Memory layout**::

   entry i:  (data[i], row[i], col[i])

For the 2x3 matrix ``[[1, 2, 0], [0, 0, 3]]``:

.. code-block:: text

   data = [1.0, 2.0, 3.0]
   row = [0, 0, 1]
   col = [0, 1, 2]

COO sparse-dense products use the explicit coordinates directly:
``coo_matvec(A, x)`` and ``coo_matmul(A, X)`` scatter each stored
contribution into the output. On Metal, ``float32`` uses atomic scatter-add,
other value dtypes remain native through a serial scatter path because Metal
does not expose compatible atomic adds for those storage types.

COO also has a same-format sparse-sparse product. ``COOArray @ COOArray``
dispatches to native ``coo_matmat``: it groups coordinates by row, performs a
symbolic pass to size each output row, fills sorted coordinates, sums duplicate
contributions, and prunes exact zero cancellations. It returns canonical COO
without silently converting through CSR.

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

CSR is the row-compressed fast path for row-oriented sparse-dense products.
Its compressed row structure makes sequential row access cache-friendly on CPU
and maps naturally to one-thread-per-row or row/reduction GPU kernels. COO and
CSC also have native sparse-dense products, choose the storage format that
matches how the matrix is assembled and reused.

Compressed Sparse Column (CSC) format
-------------------------------------

A CSC matrix stores the same three logical arrays as CSR, but compresses by
column instead of row:

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
     - Row index of each stored value. Integer dtype.
   * - ``indptr``
     - ``(n_cols + 1,)``
     - Column pointer array. ``indptr[j]`` is where column ``j`` begins and
       ``indptr[j+1]`` is where it ends.

Column ``j`` occupies ``data[indptr[j] : indptr[j+1]]`` with row indices
``indices[indptr[j] : indptr[j+1]]``. For
``[[1, 2, 0], [0, 0, 3]]``:

.. code-block:: text

   data = [1.0, 2.0, 3.0]
   indices = [0, 0, 1]
   indptr = [0, 1, 2, 3]   # col 0: [0,1), col 1: [1,2), col 2: [2,3)

CSC is not just a spelling of CSR in mlx-sparse. It has native C++/Metal
entrypoints for COO/CSR conversion, dense materialization, sorting,
duplicate summation, canonicalization, and sparse-dense products. Forward
products ``csc_matvec(A, x)`` and ``csc_matmul(A, X)`` walk compressed columns
and scatter into output rows, on Metal, the ``float32`` path uses parallel
atomic scatter and other value dtypes use a correctness-preserving native
serial scatter path. Transpose products are the layout's fast path: each
output column is an independent compressed-column dot product.

Same-format sparse-sparse products are also column-native. ``CSCArray @
CSCArray`` dispatches to ``csc_matmat``, which walks each right-hand compressed
column, gathers the needed left-hand compressed columns, and writes canonical
CSC output with sorted row indices per column.

Canonical form
--------------

An CSRArray is in **canonical form** when:

1. Column indices within each row are sorted in ascending order.
2. No row contains duplicate column indices.

A CSCArray is canonical under the column-dual rule:

1. Row indices within each column are sorted in ascending order.
2. No column contains duplicate row indices.

The constructor never enforces or silently produces canonical form. The flags
``sorted_indices`` and ``has_canonical_format`` are hints. They are only
meaningful if the caller guarantees the corresponding property. To compute the
canonical form from data that may not satisfy it, call
:meth:`~mlx_sparse.CSRArray.canonicalize` or
:meth:`~mlx_sparse.CSCArray.canonicalize`:

.. code-block:: python

   # Assemble with potential duplicates (e.g. from finite-element assembly).
   coo = ms.coo_array((data, (row, col)), shape=(m, n))
   csr = coo.tocsr(canonical=True)  # sorts and sums duplicates

   # Or in two steps:
   csr = coo.tocsr()
   csr = csr.canonicalize()

   csc = coo.tocsc(canonical=True)

Canonicalization sorts indices within the compressed dimension
(``sort_indices``) and then merges adjacent duplicates by summation
(``sum_duplicates``). For CSR the compressed dimension is a row and the
stored index is a column, for CSC the compressed dimension is a column and the
stored index is a row. The ``nnz`` after canonicalization may be smaller than
before.

Converting between formats
--------------------------

.. code-block:: python

   # COO -> CSR
   csr = coo.tocsr()  # preserves duplicates
   csr = coo.tocsr(canonical=True)  # sums duplicates

   # COO -> CSC
   csc = coo.tocsc()
   csc = coo.tocsc(canonical=True)

   # CSR <-> CSC
   csc = csr.tocsc()
   csr = csc.tocsr()

   # CSR -> dense (mx.array)
   dense = csr.todense()
   dense = ms.todense(csr)  # module-level alias

   # CSC and COO -> dense
   dense = csc.todense()
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

:attr:`~mlx_sparse.CSCArray.T` returns a zero-copy CSRArray view of the
transposed structure because CSC(A) is already CSR(A.T). Conversely,
:meth:`~mlx_sparse.CSRArray.tocsc` and :meth:`~mlx_sparse.CSCArray.tocsr`
perform native structural transpose when the target orientation needs to
represent the original matrix.

Empty rows and zero-nnz matrices
---------------------------------

All formats support matrices with empty rows/columns or zero non-zeros. An
empty row ``i`` in CSR is represented by ``indptr[i] == indptr[i+1]``. An
empty column ``j`` in CSC uses the same convention on the column pointer.
A zero-nnz CSRArray has ``data.shape == indices.shape == (0,)`` and
``indptr == [0] * (n_rows + 1)``, a zero-nnz CSCArray uses
``indptr == [0] * (n_cols + 1)``.

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
