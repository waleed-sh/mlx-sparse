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

diags
-----

.. autofunction:: diags

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

:func:`eye`, :func:`diags`, :func:`fromdense`, :func:`from_dense`,
:func:`from_numpy`, :func:`from_scipy`, and :func:`asarray` all make common
construction paths explicit. Dense conversions return a canonical
:class:`CSRArray` (``has_canonical_format=True``, ``sorted_indices=True``).
They are host-side assembly operations and call ``mx.eval`` internally when
necessary to determine the output size.

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
