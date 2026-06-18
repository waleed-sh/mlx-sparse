Autodiff
========

mlx-sparse integrates with MLX's automatic differentiation and batching system
(``mx.grad``, ``mx.vjp``, ``mx.jvp``, ``mx.vmap``) for COO, CSR, and CSC
sparse-dense products. The differentiable numerical inputs are the sparse
value buffer (``data``) and the dense right-hand side. Structural buffers
(``row`` / ``col`` / ``indices`` / ``indptr``) are integer topology and are
intentionally non-differentiable.

What is differentiable
----------------------

**Implemented and tested:**

* JVP and VJP for sparse values ``data`` in ``A @ x`` and ``A @ X`` for COO,
  CSR, and CSC.
* JVP and VJP for the dense vector ``x`` in ``A @ x``.
* JVP and VJP for the dense matrix ``X`` in ``A @ X``.
* JVP and VJP for explicit batched sparse-dense products:
  ``coo_*``, ``csr_*``, and ``csc_*`` batched matvec/matmul helpers.
* ``mx.vmap`` over dense vector and dense matrix RHS for COO, CSR, and CSC
  sparse arrays with fixed sparse structure.
* JVP and VJP for sparse values in fixed-topology block and stack assembly:
  ``block_array``, ``bmat``, ``block_diag``, ``vstack``, and ``hstack`` when
  the block structures and placements are fixed.
* Real dtypes and ``complex64`` on CPU and Metal GPU.

**Not implemented:**

* Gradients with respect to ``indices`` or ``indptr``. They are discrete
  structure, not differentiable values.
* Gradients with respect to block shapes, ``None`` placement, row/column
  offsets, or sparse coordinate buffers in structural constructors.
* ``mx.vmap`` over sparse ``data`` batches. This fixed-topology sparse-data
  batch mode is a named v0.0.6b0 limitation because the current native batched
  kernels assume one shared value buffer and batch only the dense RHS.
* Autodiff through sparse-sparse ``matmat``. Its output structure is
  data-dependent and returned as a sparse container, so the differentiable API
  is restricted to fixed-output sparse-dense products.

Dense-RHS VJP
-------------

For the matvec case ``y = A @ x``, the vector-Jacobian product with respect
to ``x`` is:

.. math::

   \overline{x} = A^H \overline{y}

where :math:`A^H` is the Hermitian adjoint. For real dtypes this is simply
``A.T``. For ``complex64`` the VJP conjugates ``data`` before dispatching the
transpose primitive, matching MLX's complex VJP convention for dense matmul.
CSR uses native transpose-product kernels, COO reuses the coordinate product
with swapped ``row``/``col`` buffers, CSC uses its compressed-column transpose
reduction path. These are native paths, not hidden dense materializations.

For the matmul case ``Y = A @ X``, the VJP with respect to ``X`` is:

.. math::

   \overline{X} = A^H \overline{Y}

which dispatches to the format-native transpose matmul primitive. CSR and CSC
have dedicated compressed transpose products, COO reuses its coordinate kernel
with swapped topology. The ``float32`` Metal scatter paths use
``atomic_float`` output updates where the format requires scatter. Other value
dtypes stay native but use serial scatter where Metal lacks compatible atomic
adds.

Both operations have CPU and Metal implementations for all supported value
dtypes.

Sparse-value VJP
----------------

For the matvec case, each stored value ``data[p]`` belongs to exactly one row
``r`` and column ``c``. For CSR, ``r`` is implicit in ``indptr`` and
``c = indices[p]``. For CSC, ``c`` is implicit in ``indptr`` and
``r = indices[p]``. For COO, both coordinates are explicit. The VJP with
respect to that value is:

.. math::

   \overline{\mathrm{data}}[p] = \overline{y}[r] \cdot \overline{x[c]}

For matmul, the right-hand side has columns ``k`` and the VJP sums over them:

.. math::

   \overline{\mathrm{data}}[p] =
   \sum_k \overline{Y}[r, k] \cdot \overline{X[c, k]}

The bar over ``x`` / ``X`` denotes complex conjugation. For real inputs it is a
no-op. These are fixed-output primitives with CPU and Metal implementations.

Fixed-Topology Constructors
---------------------------

Block and stack constructors concatenate stored values while applying fixed
coordinate offsets to the integer structure. When the block structures and
placements are fixed, the value transform is just a native concatenation, so
``mx.jvp`` concatenates value tangents and ``mx.vjp`` splits output cotangents
back to the corresponding input ``data`` buffers.

.. code-block:: python

   row = mx.array([0, 1], dtype=mx.int32)
   col = mx.array([0, 1], dtype=mx.int32)

   def assembled_values(left_data, right_data):
       A = ms.coo_array((left_data, (row, col)), shape=(2, 2), canonical=True)
       B = ms.coo_array((right_data, (row, col)), shape=(2, 2), canonical=True)
       return ms.block_array([[A, B]], format="coo").data

   _, tangent = mx.jvp(
       assembled_values,
       (mx.ones(2), mx.ones(2)),
       (mx.ones(2), 2 * mx.ones(2)),
   )

Dynamic-topology operations such as ``fromdense``, random structure sampling,
canonicalization that sums duplicates, and sparse-sparse products remain
outside the structural autodiff contract.

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

For batched vector RHS ``X`` with shape ``(..., n_cols)``, the explicit
batched matvec helpers compute ``A @ X[b]`` for every leading batch element
and return ``(..., n_rows)``. For batched matrix RHS ``(..., n_cols, k)``, the
batched matmul helpers return ``(..., n_rows, k)``. Their JVP and VJP rules
flatten the leading batch dimensions only inside the native primitive, then
reshape the gradients back to the user's batch shape. Sparse-value VJPs reuse
fixed-output data-VJP kernels over the flattened RHS/cotangent columns, and
dense-RHS VJPs reuse the native transpose-product path.

Using ``mx.vmap``
-----------------

For fixed sparse structure, MLX vectorization over the dense RHS dispatches to
native batched sparse-dense work rather than a Python loop:

.. code-block:: python

   vectors = mx.ones((8, A.shape[1]), dtype=A.dtype)
   matrices = mx.ones((8, A.shape[1], 4), dtype=A.dtype)

   ys = mx.vmap(lambda x: A @ x)(vectors)      # shape (8, A.shape[0])
   Ys = mx.vmap(lambda X: A @ X)(matrices)     # shape (8, A.shape[0], 4)

``in_axes`` may point at any dense RHS axis that represents the mapped batch,
and MLX ``out_axes`` is honored from the primitive's reported mapped output
axis. Sparse buffers must be unmapped. Mapping ``data``, ``row`` / ``col``, or
``indices`` / ``indptr`` raises a precise limitation error instead of silently
falling back to host-side looping.

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

``complex64`` forward and autodiff paths are implemented for COO, CSR, and CSC
matvec and dense-matrix matmul. The VJP rules use Hermitian adjoints:
dense-RHS gradients conjugate sparse values, and sparse-value gradients
conjugate the dense RHS. The test suite compares complex sparse gradients,
``mx.vjp``, and ``mx.jvp`` directly against equivalent dense MLX matmul
computations.
