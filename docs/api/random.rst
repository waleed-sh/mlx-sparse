Random
======

.. currentmodule:: mlx_sparse.random

The ``mlx_sparse.random`` namespace contains the SciPy-compatible sparse random
constructor surface for v0.0.6b0. The public API, validation, RNG policy, and
documentation are present; native CPU/Metal generation kernels are still under
implementation and the functions raise ``NotImplementedError`` after
validation until those kernels are connected.

random\_array
-------------

.. autofunction:: random_array

random
------

.. autofunction:: random

rand
----

.. autofunction:: rand
