Constructors
============

.. currentmodule:: mlx_sparse

These functions create sparse array instances from raw buffers or structured
definitions. All array inputs accept MLX arrays, NumPy arrays, or any sequence
convertible via ``mx.array()``.

csr\_array
----------

.. autofunction:: csr_array

coo\_array
----------

.. autofunction:: coo_array

eye
---

.. autofunction:: eye

identity
--------

.. autofunction:: identity

diags
-----

.. autofunction:: diags

block\_array
------------

.. autofunction:: block_array

bmat
----

.. autofunction:: bmat

block\_diag
-----------

.. autofunction:: block_diag

vstack
------

.. autofunction:: vstack

hstack
------

.. autofunction:: hstack

tril
----

.. autofunction:: tril

triu
----

.. autofunction:: triu

fromdense
---------

.. autofunction:: fromdense

from\_dense
-----------

.. autofunction:: from_dense

from\_numpy
-----------

.. autofunction:: from_numpy

from\_scipy
-----------

.. autofunction:: from_scipy

asarray
-------

.. autofunction:: asarray

Validation modes
----------------

The ``validate`` parameter on :func:`csr_array` and :func:`coo_array` accepts:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Value
     - Behaviour
   * - ``"metadata"``
     - Checks ranks, array lengths, and dtypes. No value reads. Default.
   * - ``"full"`` or ``True``
     - Full metadata checks plus value-level checks (bounds, monotonicity).
       May call ``mx.eval`` to read index values from device.
   * - ``False`` or ``"none"``
     - No checks. Use only when inputs are known valid.

See :doc:`../user_guide/validation` for a detailed discussion.

Structured constructors
-----------------------

:func:`eye`, :func:`identity`, :func:`diags`, :func:`block_array`,
:func:`bmat`, :func:`block_diag`, :func:`vstack`, :func:`hstack`,
:func:`tril`, :func:`triu`, :func:`fromdense`, :func:`from_dense`,
:func:`from_numpy`, :func:`from_scipy`, and :func:`asarray` all make common
construction paths explicit.

Block assembly and triangular extraction are native-backed and avoid Python
loops over stored entries. ``None`` entries in :func:`block_array` represent
implicit zero blocks. Dense blocks are converted with the native
:func:`fromdense` path before sparse assembly. CSR and CSC format requests use
native compressed conversion and canonicalization where duplicate coordinates
may need to be summed.

.. code-block:: python

   import numpy as np
   import mlx.core as mx
   import mlx_sparse as ms

   # 5x5 identity
   I = ms.eye(5)

   # 4x4 tridiagonal
   T = ms.diags(
       [np.full(3, -1.0), np.full(4, 2.0), np.full(3, -1.0)],
       offsets=[-1, 0, 1],
   )

   # Convert a dense weight matrix
   W = mx.array(np.random.randn(8, 8).astype(np.float32))
   W_sparse = ms.fromdense(W, threshold=0.1)

   # Convert SciPy sparse or dense NumPy inputs without hand-building buffers
   W_from_np = ms.asarray(np.eye(8, dtype=np.float32))

   # Assemble a saddle-point-style block matrix
   K = ms.block_array([[T, None], [None, ms.identity(4)]], format="csr")

   # Extract a triangular sparse view without densifying
   L = ms.tril(K, format="csr")
