mlx-sparse
==========

**mlx-sparse** is a sparse matrix library for MLX, built as a native MLX
extension. It provides COO, CSR, and CSC sparse containers backed by
``mlx.core.array``, with C++ MLX primitives for sparse-dense products on CPU,
and Metal GPU kernels on supported Apple Silicon systems.

.. warning::

   ``mlx-sparse`` supports macOS and Linux. Linux support is CPU-only in this
   release: CUDA and ROCm are not implemented, Metal is Apple-only, and Linux
   builds do not use Accelerate, BLAS, or Sparse BLAS backends. See
   :ref:`currently-supported` for the full capability map.

.. note::

   GPU support in this version is Apple Silicon Metal only. CUDA is not
   currently supported.

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

   # Assemble a sparse matrix directly in COO format.
   data = mx.array(np.array([2.0, -1.0, 4.0], dtype=np.float32))
   row = mx.array(np.array([0, 0, 1], dtype=np.int32))
   col = mx.array(np.array([0, 2, 1], dtype=np.int32))
   A = ms.coo_array((data, (row, col)), shape=(2, 3))

   x = mx.array(np.array([3.0, 10.0, 7.0], dtype=np.float32))
   y = A @ x  # sparse-dense product, lazy
   mx.eval(y)  # materialise on device

   print(A.todense()) # [[2. 0. -1.], [0. 4. 0.]]

Key characteristics
-------------------

* **COO, CSR, and CSC containers**: immutable frozen dataclasses. Structural
  operations return new instances without in-place mutation.
* **Lazy execution**: sparse operations add nodes to MLX's computation graph.
  No ``mx.eval`` is called inside any sparse operation.
* **Metal GPU kernels**: COO, CSR, and CSC sparse-dense products dispatch
  through MLX's Metal backend for supported value and index dtypes. No separate
  command queue or synchronization point.
* **CPU backends**: all operations have C++ CPU implementations. Conversions,
  transpose, and canonicalization run on CPU or GPU.
* **Autodiff**: ``mx.grad`` / ``mx.vjp`` / ``mx.jvp`` work for sparse values
  and dense operands of COO, CSR, and CSC sparse-dense products on both CPU and
  GPU, including ``complex64`` where the corresponding forward primitive
  supports it.
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
   :maxdepth: 2
   :caption: Parallelism and performance

   parallelism/index

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
