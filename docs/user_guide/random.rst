Sparse Random Generation
========================

``mlx_sparse.random`` is the public namespace for SciPy-like sparse random
constructors:

.. code-block:: python

   import mlx.core as mx
   import mlx_sparse as ms

   key = mx.random.key(0)
   A = ms.random.random_array((1024, 1024), density=0.01, format="csr", rng=key)

Native generation
-----------------

The v0.0.6b0 random constructors validate the public argument surface, reject
unsupported host RNGs and formats, and generate sparse structure with native
C++/Metal kernels. The structure kernel uses a deterministic keyed permutation
over the flattened matrix domain and takes the first ``nnz`` permuted
coordinates. This gives sampling without replacement, exact ``nnz`` output,
duplicate-free canonical coordinates, CPU/Metal parity for the same key, and no
full dense mask. The permutation is not SciPy bitstream-compatible.

COO output generates coordinate buffers directly. CSR and CSC output generate
compressed ``indices`` and ``indptr`` buffers directly in the requested
orientation rather than generating COO first and converting. The compressed
paths use ``nnz``-sized structural keys, native sort, native segment counts,
and cumulative ``indptr`` construction; they do not allocate a dense mask.

Default stored values are sampled uniformly on ``[0, 1)`` with MLX random
vector operations on the active device. Custom ``data_sampler`` / ``data_rvs``
callables are called at most once with ``size=nnz`` and are the public way to
request custom value ranges or distributions.

RNG policy
----------

Use MLX PRNG keys for reproducible sparse structures and values:

.. code-block:: python

   key = mx.random.key(123)
   A = ms.random.rand(64, 64, density=0.05, rng=key)

An integer ``rng`` seed is accepted and normalized to ``mx.random.key(seed)``.
``random_state`` is accepted only as a SciPy-compatibility alias for ``rng``;
passing both is an error. NumPy ``Generator`` and ``RandomState`` objects are
rejected because they imply host-side random structure generation. Host values
from a user sampler are accepted only as an explicit value-data compatibility
path; they are not the production or benchmark path.

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
dtypes, and ``mx.bool_`` remains part of a future package-wide sparse storage
policy expansion and is not silently emulated with host code.

The constructors use the active MLX device. Linux remains CPU-only, while Metal
is Apple-only. The reproducibility contract is same key plus same arguments
within the same mlx-sparse version; no SciPy or NumPy bitstream equivalence is
promised.

The default ``canonical=True`` requests duplicate-free structure in the
requested format. Noncanonical duplicate-preserving random output is not part of
this release.
