Random
======

.. currentmodule:: mlx_sparse.random

The ``mlx_sparse.random`` namespace contains the SciPy-compatible sparse random
constructor surface for v0.0.6b0. Structure generation is native C++/Metal and
uses a deterministic keyed permutation sampler that produces exactly the
documented ``nnz`` count without replacement, dense masks, or Python loops over
stored entries. CSR and CSC formats use direct compressed-structure generation
rather than COO conversion. Default values are sampled uniformly on ``[0, 1)``
with MLX random vector operations, and custom value samplers are called at most
once for custom ranges or distributions.

random\_array
-------------

.. autofunction:: random_array

random
------

.. autofunction:: random

rand
----

.. autofunction:: rand
