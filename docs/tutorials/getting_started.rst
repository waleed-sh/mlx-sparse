Getting started: end-to-end sparse workflow
============================================

This tutorial walks through a complete sparse matrix workflow from scratch:
construction, conversion, products, inspection, and differentiation. It
assumes you have completed the installation step and have ``mlx.core``,
``numpy``, and ``mlx_sparse`` available.

Setup
-----

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   # Confirm the native extension compiled successfully.
   print("native extension:", ms.is_available())

   # Choose a device. Fixed-shape sparse primitives run on CPU or GPU.
   ms.use_gpu()

Step 1: build a small matrix in COO format
------------------------------------------

We will construct the 4x5 matrix:

.. code-block:: text

   [[ 3,  0, -1,  0,  0],
    [ 0,  2,  0,  0,  0],
    [ 0,  0,  0,  4, -2],
    [ 0,  0,  0,  0,  0]]

COO is the right format for construction because it accepts entries in any
order and allows duplicates.

.. code-block:: python

   rows_np = np.array([0, 0, 1, 2, 2], dtype=np.int32)
   cols_np = np.array([0, 2, 1, 3, 4], dtype=np.int32)
   vals_np = np.array([3.0, -1.0, 2.0, 4.0, -2.0], dtype=np.float32)

   data = mx.array(vals_np)
   row = mx.array(rows_np)
   col = mx.array(cols_np)

   coo = ms.coo_array((data, (row, col)), shape=(4, 5))
   print(coo)
   # COOArray(shape=(4, 5), nnz=5, dtype=float32, index_dtype=int32,
   #          has_canonical_format=False)

The ``nnz`` count reports five stored entries. Row 3 is empty. It has no
entries and contributes nothing to ``nnz``.

Step 2: convert to CSR and canonicalize
-----------------------------------------

COO can run native sparse-dense products directly. This tutorial converts to
CSR here to show the compressed row layout and canonicalization path used by
row-oriented products and solvers. Pass ``canonical=True`` to sort column
indices and sum duplicates in one call. For this matrix there are no
duplicates, so the nnz stays at 5.

.. code-block:: python

   csr = coo.tocsr(canonical=True)
   print(csr)
   # CSRArray(shape=(4, 5), nnz=5, dtype=float32, index_dtype=int32,
   #          sorted_indices=True, has_canonical_format=True)

Inspect the CSR buffers:

.. code-block:: python

   mx.eval(csr.data, csr.indices, csr.indptr)
   print("data:   ", np.array(csr.data))  # [ 3. -1.  2.  4. -2.]
   print("indices:", np.array(csr.indices))  # [0 2 1 3 4]
   print("indptr: ", np.array(csr.indptr))  # [0 2 3 5 5]

The ``indptr`` encodes:

* Row 0: entries at positions 0–2 -> columns 0 and 2 (values 3 and -1).
* Row 1: entries at positions 2–3 -> column 1 (value 2).
* Row 2: entries at positions 3–5 -> columns 3 and 4 (values 4 and -2).
* Row 3: entries at positions 5–5 -> empty.

Step 3: sparse-dense matrix-vector product
-------------------------------------------

.. code-block:: python

   x = mx.array(np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))
   y = csr @ x
   mx.eval(y)
   print(np.array(y))  # [ 2.  2.  2.  0.]

Row 0: 3x1 + (−1)x1 = 2. Row 1: 2x1 = 2. Row 2: 4x1 + (−2)x1 = 2.
Row 3: empty -> 0.

Step 4: sparse-dense matrix-matrix product
--------------------------------------------

The ``@`` operator dispatches to :func:`~mlx_sparse.csr_matmul` when the
right-hand side is a 2D array.

.. code-block:: python

   B = mx.array(np.eye(5, dtype=np.float32))  # 5x5 identity
   Y = csr @ B
   mx.eval(Y)
   print(Y.shape)  # (4, 5)
   # Y should equal the dense matrix itself (A @ I = A).

Step 5: materialise as a dense array
--------------------------------------

.. code-block:: python

   dense = csr.todense()
   mx.eval(dense)
   print(np.array(dense))
   # [[ 3.  0. -1.  0.  0.]
   #  [ 0.  2.  0.  0.  0.]
   #  [ 0.  0.  0.  4. -2.]
   #  [ 0.  0.  0.  0.  0.]]

Step 6: transpose and Hermitian transpose
------------------------------------------

.. code-block:: python

   At = csr.T
   print(At)
   # CSRArray(shape=(5, 4), ...)
   print(np.array(At.todense()))
   # [[ 3.  0.  0.  0.]
   #  [ 0.  2.  0.  0.]
   #  [-1.  0.  0.  0.]
   #  [ 0.  0.  4.  0.]
   #  [ 0.  0. -2.  0.]]

For real-valued matrices, ``csr.H`` is the same as ``csr.T``.

Step 7: building a matrix with duplicate entries
-------------------------------------------------

COO construction allows duplicate coordinates. Duplicates are summed when you
call ``tocsr(canonical=True)`` or ``canonicalize()``.

.. code-block:: python

   # Two contributions to (0, 0): 1.5 and 2.5 -> should sum to 4.0.
   data_dup = mx.array([1.5, 2.5, 3.0], dtype=mx.float32)
   row_dup = mx.array([0, 0, 1], dtype=mx.int32)
   col_dup = mx.array([0, 0, 1], dtype=mx.int32)

   coo_dup = ms.coo_array((data_dup, (row_dup, col_dup)), shape=(2, 2))
   csr_dup = coo_dup.tocsr(canonical=True)

   mx.eval(csr_dup.data)
   print(np.array(csr_dup.data))  # [4.0, 3.0]
   print(np.array(csr_dup.todense()))
   # [[4.  0.]
   #  [0.  3.]]

Step 8: autodiff through sparse values and the dense operand
-------------------------------------------------------------

.. code-block:: python

   x_grad = mx.array(np.ones(5, dtype=np.float32))

   def loss(values, x):
       A = ms.csr_array((values, csr.indices, csr.indptr), shape=csr.shape)
       y = A @ x
       return mx.sum(y ** 2)  # scalar output

   grad_values, grad_x = mx.grad(loss, argnums=(0, 1))(csr.data, x_grad)
   mx.eval(grad_values, grad_x)
   print(np.array(grad_values))
   print(np.array(grad_x))

The dense-input gradient is ``2 * A.T @ (A @ x)``. Verify against dense:

.. code-block:: python

   dense_ref = csr.todense()

   def dense_loss(x):
       y = dense_ref @ x
       return mx.sum(y ** 2)

   grad_dense = mx.grad(dense_loss)(x_grad)
   mx.eval(grad_dense)

   np.testing.assert_allclose(
       np.array(grad_x), np.array(grad_dense), rtol=1e-5, atol=1e-5
   )
   print("gradients match!")

What to explore next
---------------------

* :doc:`finite_difference` - a realistic 2D PDE stencil assembled as a sparse
  Laplacian.
* :doc:`scipy_interop` - importing SciPy sparse matrices and comparing results.
* :doc:`graph_laplacian` - building graph Laplacians and applying them on GPU.
* :doc:`physics_workloads` - quantum Hamiltonians and sparse linear layers.
* :doc:`../user_guide/autodiff` - full autodiff semantics and known limits.
* :doc:`../performance` - when sparse is faster than dense, and how to measure
  it.
