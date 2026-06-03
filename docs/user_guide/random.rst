Sparse Random Generation
========================

``mlx_sparse.random`` is the public namespace for SciPy-like sparse random
constructors:

.. code-block:: python

   import mlx.core as mx
   import mlx_sparse as ms

   key = mx.random.key(0)
   A = ms.random.random_array((1024, 1024), density=0.01, format="csr", rng=key)

Development status
------------------

The v0.0.6b0 exposes and documents the namespace, validates the public
argument surface, and rejects unsupported host RNGs and formats. The production
generation path is intentionally not implemented in Python until native
C++/Metal structure and value kernels are wired in by the end of this release, the public constructors
raise ``NotImplementedError`` after validation.

RNG policy
----------

Use MLX PRNG keys for reproducible sparse structures and values:

.. code-block:: python

   key = mx.random.key(123)
   A = ms.random.rand(64, 64, density=0.05, rng=key)

An integer ``rng`` seed is accepted and normalized to ``mx.random.key(seed)``.
``random_state`` is accepted only as a SciPy-compatibility alias for ``rng``;
passing both is an error. NumPy ``Generator`` and ``RandomState`` objects are
rejected because they imply host-side random structure generation. Future
compatibility paths may accept host-generated values from a user sampler, but
they will not be the production or benchmark path.

Density and formats
-------------------

For shape ``(m, n)``, the structural count is
``round(m * n * density)`` clipped to ``[0, m * n]``. This matches SciPy's
``int(round(...))`` convention, including Python's ties-to-even rounding.

The first native release targets ``"coo"``, ``"csr"``, and ``"csc"`` output.
``None`` is treated as ``"coo"``. SciPy-only storage formats such as ``"bsr"``,
``"dia"``, ``"dok"``, and ``"lil"`` are rejected.

Dtypes, device, and canonicalization
------------------------------------

``dtype=None`` defaults to ``mx.float32``. The current sparse containers accept
``mx.float32``, ``mx.float16``, ``mx.bfloat16``, and ``mx.complex64`` stored
values. Native generation support for ``mx.float64`` where available, integer
dtypes, and ``mx.bool_`` remains part of the generation-kernel implementation
and is not silently emulated with host code.

The constructors use the active MLX device. Linux remains CPU-only, while Metal
is Apple-only. The reproducibility contract is same key plus same arguments
within the same mlx-sparse version; no SciPy or NumPy bitstream equivalence is
promised.

The default ``canonical=True`` requests duplicate-free structure in the
requested format. Noncanonical duplicate-preserving random output is not part of
the skeleton.
