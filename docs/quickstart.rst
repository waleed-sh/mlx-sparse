Quick start
===========

This page covers the most common usage patterns in order. By the end you will
have assembled a sparse matrix, run sparse-dense products on the selected MLX
device, computed a gradient through sparse values and dense operands, and seen
the structured and interchange constructors.

Selecting a device
------------------

Call :func:`~mlx_sparse.use_gpu` or :func:`~mlx_sparse.use_cpu` once before
any computation. This sets MLX's default device for all subsequent operations.

.. code-block:: python

   import mlx_sparse as ms

   ms.use_cpu()  # portable default, including Linux CPU-only wheels
   ms.use_gpu()  # Apple Silicon GPU (Metal), when available

If no device is selected, MLX uses its own default (usually GPU on Apple
Silicon). Fixed-shape sparse primitives support all value and index dtype
combinations on CPU and on the Apple Silicon Metal backend. Linux wheels are
CPU-only in this release.

Constructing a COO matrix
--------------------------

:func:`~mlx_sparse.coo_array` accepts a ``(data, (row, col))`` tuple plus a
``shape``. All arrays can be ``mlx.core.array`` or anything convertible via
``mx.array()``, including NumPy arrays.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   # Represents the 3x4 matrix:
   #  [[ 2,  0, -1,  0],
   #   [ 0,  0,  0,  0],
   #   [ 0,  4,  0,  5]]

   data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
   row = mx.array(np.array([0, 0, 2, 2], dtype=np.int32))
   col = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))

   coo = ms.coo_array((data, (row, col)), shape=(3, 4))
   print(coo)
   # COOArray(shape=(3, 4), nnz=4, dtype=float32, index_dtype=int32, ...)

Converting to CSR
-----------------

COO is the assembly format. Convert to CSR before running repeated products.
Pass ``canonical=True`` to sort column indices and sum any duplicate entries
in the same conversion call.

.. code-block:: python

   csr = coo.tocsr(canonical=True)
   print(csr)
   # CSRArray(shape=(3, 4), nnz=4, dtype=float32, index_dtype=int32,
   #          sorted_indices=True, has_canonical_format=True)

For random sparse inputs, use the native :mod:`mlx_sparse.random` namespace:

.. code-block:: python

   csr = ms.random.rand(
       256,
       256,
       density=0.01,
       format="csr",
       dtype=mx.float32,
       rng=0,
       index_dtype=mx.int32,
   )

Structured constructors
-----------------------

For common structured matrices and interchange, use :func:`~mlx_sparse.eye`,
:func:`~mlx_sparse.diags`, :func:`~mlx_sparse.fromdense`,
:func:`~mlx_sparse.from_scipy`, or :func:`~mlx_sparse.asarray` instead of
assembling a COO triple by hand.

.. code-block:: python

   import numpy as np

   # 4x4 identity
   I = ms.eye(4)

   # Tridiagonal Laplacian: diagonals [−1, 2, −1] at offsets [−1, 0, 1]
   n = 6
   L = ms.diags(
       [np.full(n - 1, -1.0), np.full(n, 2.0), np.full(n - 1, -1.0)],
       offsets=[-1, 0, 1],
   )

   # Dense-to-sparse conversion
   dense = mx.array(np.eye(4, dtype=np.float32) * 3.0)
   csr = ms.fromdense(dense)

   # Generic conversion: CSR inputs pass through, SciPy sparse and dense arrays
   # become canonical CSRArray instances.
   csr = ms.asarray(np.eye(4, dtype=np.float32))

Sparse-dense matrix-vector product
------------------------------------

Use the ``@`` operator. The result is a lazy MLX array and no computation
happens until ``mx.eval`` is called.

.. code-block:: python

   x = mx.array(np.ones(4, dtype=np.float32))
   y = csr @ x  # lazy, shape (3,)
   mx.eval(y)  # evaluate on the active device
   print(np.array(y))  # [ 1. 0. 9.]

Sparse-dense matrix-matrix product
------------------------------------

The same ``@`` operator dispatches to :func:`~mlx_sparse.csr_matmul` when the
right-hand side is rank-2, and handles batched dense operands as well.

.. code-block:: python

   B = mx.array(np.random.randn(4, 8).astype(np.float32))
   Y = csr @ B  # shape (3, 8)
   mx.eval(Y)

   # Batched: rhs shape (batch, n_cols, k) -> output shape (batch, n_rows, k)
   B_batch = mx.array(np.random.randn(2, 4, 8).astype(np.float32))
   Y_batch = csr @ B_batch  # shape (2, 3, 8)

Sparse-sparse matrix product
-------------------------------

When both operands are :class:`~mlx_sparse.CSRArray` instances, ``@``
dispatches to :func:`~mlx_sparse.csr_matmat` and returns a new
:class:`~mlx_sparse.CSRArray`:

.. code-block:: python

   A = ms.eye(4)
   B = ms.diags([1.0, 2.0, 3.0, 4.0])
   C = A @ B  # CSRArray, sparse-sparse product

Converting to dense
--------------------

.. code-block:: python

   dense = csr.todense()  # mx.array, shape (3, 4)
   # or, using the module-level helper:
   dense = ms.todense(csr)

Transpose and Hermitian transpose
-----------------------------------

:attr:`~mlx_sparse.CSRArray.T` returns a structural transpose as a new
CSRArray with shape ``(n_cols, n_rows)``. :attr:`~mlx_sparse.CSRArray.H`
additionally conjugates the values.

.. code-block:: python

   At = csr.T  # CSRArray(shape=(4, 3), ...)
   Ah = csr.H  # conjugate transpose. Relevant for complex64 matrices.

Computing gradients
--------------------

``mx.grad`` differentiates through :func:`~mlx_sparse.csr_matvec` and
:func:`~mlx_sparse.csr_matmul` with respect to both sparse ``data`` values and
the dense operand. CPU and GPU are supported for real and ``complex64`` dtypes.

.. code-block:: python

   ms.use_gpu()

   csr = ms.coo_array((data, (row, col)), shape=(3, 4)).tocsr()
   x = mx.array(np.ones(4, dtype=np.float32))

   def loss(values, x):
       A = ms.csr_array((values, csr.indices, csr.indptr), shape=csr.shape)
       y = A @ x
       return mx.sum(y * y)

   grad_values, grad_x = mx.grad(loss, argnums=(0, 1))(csr.data, x)
   mx.eval(grad_values, grad_x)

The gradient matches ``mx.grad`` applied to the equivalent dense matrix
multiply, up to floating-point rounding.

What to read next
-----------------

* :doc:`user_guide/sparse_formats` - detailed COO and CSR format invariants.
* :doc:`user_guide/validation` - when validation reads values and when to
  disable it.
* :doc:`user_guide/dtype_policy` - supported value and index dtypes, Metal vs
  CPU coverage.
* :doc:`user_guide/autodiff` - full autodiff semantics, limitations, and
  future plans.
* :doc:`tutorials/getting_started` - end-to-end worked example from scratch.
