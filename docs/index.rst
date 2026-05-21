mlx-sparse
==========

**mlx-sparse** is a sparse matrix library for Apple Silicon, built as a native
MLX extension. It provides COO and CSR sparse containers backed by
``mlx.core.array``, with C++ MLX primitives for sparse-dense products on both
CPU and Metal GPU. It also supports some linear algebra operations for sparse matrices.

The design follows MLX's operation/primitive split. Python containers own the
user API, C++ primitives own graph construction and backend evaluation, and
Metal kernels run through MLX's command encoder without a separate
synchronization point. Sparse operations participate in MLX's lazy computation
graph. ``mx.eval`` is called only when results are needed, and autodiff through
sparse values and dense operands works with ``mx.grad`` on both CPU and GPU.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   ms.use_gpu()

   # Assemble a sparse matrix in COO format, then convert to CSR.
   data = mx.array(np.array([2.0, -1.0, 4.0], dtype=np.float32))
   row = mx.array(np.array([0, 0, 1], dtype=np.int32))
   col = mx.array(np.array([0, 2, 1], dtype=np.int32))
   A = ms.coo_array((data, (row, col)), shape=(2, 3)).tocsr(canonical=True)

   x = mx.array(np.array([3.0, 10.0, 7.0], dtype=np.float32))
   y = A @ x  # sparse-dense product, lazy
   mx.eval(y)  # materialise on device

   print(A.todense()) # [[2. 0. -1.], [0. 4. 0.]]

Key characteristics
-------------------

* **COO and CSR containers**: immutable frozen dataclasses. Structural
  operations return new instances without in-place mutation.
* **Lazy execution**: sparse operations add nodes to MLX's computation graph.
  No ``mx.eval`` is called inside any sparse operation.
* **Metal GPU kernels**: ``csr_matvec`` and ``csr_matmul`` dispatch through
  MLX's Metal backend for all supported value and index dtypes. No separate
  command queue or synchronization point.
* **CPU backends**: all operations have C++ CPU implementations. Conversions,
  transpose, and canonicalization run on CPU or GPU.
* **Autodiff**: ``mx.grad`` / ``mx.vjp`` / ``mx.jvp`` work for sparse values
  and dense operands of CSR matvec and matmul on both CPU and GPU, including
  ``complex64``.
* **Sparse linalg**: operations like ``eigsh``, ``eigs``, ``cholesky``, ``splu`` (sparse LU),
  as well as SciPy like ``LinearOperator`` are available through ``mlx_sparse.linalg``.
* **Value dtypes**: ``float32``, ``float16``, ``bfloat16``, and ``complex64``
  on CPU and Metal GPU.
* **Index dtypes**: ``int32`` and ``int64``. Mixed dtypes are rejected at
  construction time.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   notebooks/index

.. toctree::
   :maxdepth: 1
   :caption: Reference

   supported
   performance
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog
