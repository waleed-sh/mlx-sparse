Autodiff
========

mlx-sparse integrates with MLX's automatic differentiation system (``mx.grad``,
``mx.vjp``, ``mx.jvp``) for CSR sparse-dense products. The differentiable
numerical inputs are the sparse value buffer (``data``) and the dense right-hand
side. Structural buffers (``indices`` and ``indptr``) are integer topology and
are intentionally non-differentiable.

What is differentiable
----------------------

**Implemented and tested:**

* JVP and VJP for sparse values ``data`` in ``A @ x`` and ``A @ X``.
* JVP and VJP for the dense vector ``x`` in ``A @ x``.
* JVP and VJP for the dense matrix ``X`` in ``A @ X``.
* JVP and VJP for explicit batched sparse-dense products,
  :func:`mlx_sparse.csr_batched_matvec` and
  :func:`mlx_sparse.csr_batched_matmul`.
* Real dtypes and ``complex64`` on CPU and Metal GPU.

**Not implemented:**

* Gradients with respect to ``indices`` or ``indptr``. They are discrete
  structure, not differentiable values.

Dense-RHS VJP
-------------

For the matvec case ``y = A @ x``, the vector-Jacobian product with respect
to ``x`` is:

.. math::

   \overline{x} = A^H \overline{y}

where :math:`A^H` is the Hermitian adjoint. For real dtypes this is simply
``A.T``. For ``complex64`` the VJP conjugates ``data`` before dispatching the
transpose primitive, matching MLX's complex VJP convention for dense matmul.
For ``float32`` on Metal this is implemented as a parallel scatter-add kernel
using ``atomic_float`` output updates. Other GPU value dtypes lower through
native ``csr_transpose`` and ``csr_matvec`` so the implementation stays native
without relying on unsupported complex or low-precision atomic adds.

For the matmul case ``Y = A @ X``, the VJP with respect to ``X`` is:

.. math::

   \overline{X} = A^H \overline{Y}

which dispatches to the transpose matmul primitive. The ``float32`` Metal path
uses atomic scatter-add over source rows and RHS columns. Other value dtypes
lower through native ``csr_transpose`` followed by ``csr_matmul``.

Both operations have CPU and Metal implementations for all supported value
dtypes.

Sparse-value VJP
----------------

For the matvec case, each stored value ``data[p]`` belongs to exactly one row
``r`` and column ``c = indices[p]``. The VJP with respect to that value is:

.. math::

   \overline{\mathrm{data}}[p] = \overline{y}[r] \cdot \overline{x[c]}

For matmul, the right-hand side has columns ``k`` and the VJP sums over them:

.. math::

   \overline{\mathrm{data}}[p] =
   \sum_k \overline{Y}[r, k] \cdot \overline{X[c, k]}

The bar over ``x`` / ``X`` denotes complex conjugation. For real inputs it is a
no-op. These are fixed-output primitives with CPU and Metal implementations.

Dense-RHS JVP
-------------

For a tangent ``\dot{x}`` at ``x``, the JVP through ``y = A @ x`` is:

.. math::

   \dot{y} = A \cdot \dot{x}

JVP with respect to sparse values uses the same formula with ``data`` replaced
by ``dot(data)``. JVP with respect to the dense RHS replaces ``x`` / ``X`` with
the corresponding tangent. These reuse the forward primitives and therefore
have the same device and dtype coverage as the forward operation.

Batched RHS
-----------

For batched vector RHS ``X`` with shape ``(..., n_cols)``,
:func:`mlx_sparse.csr_batched_matvec` computes ``A @ X[b]`` for every leading
batch element and returns ``(..., n_rows)``. For batched matrix RHS
``(..., n_cols, k)``, :func:`mlx_sparse.csr_batched_matmul` returns
``(..., n_rows, k)``. Their JVP and VJP rules flatten the leading batch
dimensions only inside the native primitive, then reshape the gradients back to
the user's batch shape. Sparse-value VJPs reuse the same fixed-output
``csr_matmul_data_vjp`` kernel over the flattened RHS/cotangent columns, and
dense-RHS VJPs reuse the native transpose-product path.

Using ``mx.grad``
-----------------

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   ms.use_gpu()

   data = mx.array(np.array([2.0, -1.0, 4.0, 5.0], dtype=np.float32))
   indices = mx.array(np.array([0, 2, 1, 3], dtype=np.int32))
   indptr = mx.array(np.array([0, 2, 2, 4], dtype=np.int32))
   A = ms.csr_array((data, indices, indptr), shape=(3, 4))

   x = mx.array(np.ones(4, dtype=np.float32))

   def loss(values, x):
       A_values = ms.csr_array((values, A.indices, A.indptr), shape=A.shape)
       y = A_values @ x
       return mx.sum(y * y)

   grad_values, grad_x = mx.grad(loss, argnums=(0, 1))(A.data, x)
   mx.eval(grad_values, grad_x)
   print(np.array(grad_values), np.array(grad_x))

The gradients match dense MLX matmul gradients up to floating-point rounding.
For value gradients, the dense reference is the full dense gradient sampled at
the sparse coordinates. This is verified in ``tests/test_grad.py``.

Using ``mx.vjp`` and ``mx.jvp`` directly
-----------------------------------------

.. code-block:: python

   # VJP: given a cotangent for the output, compute the cotangent for x.
   primals = (x,)
   cotangents = (mx.ones(3, dtype=mx.float32),)
   outputs, grad_x = mx.vjp(lambda x: A @ x, primals, cotangents)

   # JVP: given a tangent for x, compute the tangent for the output.
   tangent_x = mx.ones_like(x)
   outputs, tangent_out = mx.jvp(lambda x: A @ x, (x,), (tangent_x,))

Verifying against dense MLX
-----------------------------

A reliable correctness check is to compare the sparse gradient to the dense
gradient:

.. code-block:: python

   dense = A.todense()

   def sparse_loss(x): return mx.sum((A @ x) ** 2)
   def dense_loss(x): return mx.sum((dense @ x) ** 2)

   np.testing.assert_allclose(
       np.array(mx.grad(sparse_loss)(x)),
       np.array(mx.grad(dense_loss)(x)),
       rtol=1e-5, atol=1e-5,
   )

Complex autodiff
----------------

``complex64`` forward and autodiff paths are implemented for CSR matvec and
matmul. The VJP rules use Hermitian adjoints: dense-RHS gradients conjugate
sparse values, and sparse-value gradients conjugate the dense RHS. The test
suite compares complex sparse gradients, ``mx.vjp``, and ``mx.jvp`` directly
against equivalent dense MLX matmul computations.
