SciPy interoperability
=======================

SciPy's ``scipy.sparse`` module is the most common source of sparse matrices
in Python scientific computing. This tutorial shows how to transfer SciPy
sparse matrices to mlx-sparse, run products on Apple Silicon, and compare
results.

SciPy is an optional dependency of mlx-sparse. It is listed under the ``dev``
and ``bench`` extras. Any code that imports it will need a guard:

.. code-block:: python

   import pytest
   scipy_sparse = pytest.importorskip("scipy.sparse")  # in tests
   # or:
   try:
       import scipy.sparse
   except ImportError:
       scipy_sparse = None

Converting a SciPy CSR matrix
-------------------------------

SciPy CSR arrays expose ``.data``, ``.indices``, and ``.indptr`` as NumPy
arrays. Wrap them in ``mx.array`` and pass to :func:`~mlx_sparse.csr_array`.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import scipy.sparse
   import mlx_sparse as ms

   # Generate a random 512x512 sparse matrix with 1% density.
   rng = np.random.default_rng(42)
   sp = scipy.sparse.random(512, 512, density=0.01, format="csr",
                             dtype=np.float32, random_state=rng)

   # Convert indices to int32 (SciPy defaults to int32 on 64-bit platforms,
   # but check explicitly to be safe).
   csr = ms.csr_array(
       (mx.array(sp.data.astype(np.float32)),
        mx.array(sp.indices.astype(np.int32)),
        mx.array(sp.indptr.astype(np.int32))),
       shape=sp.shape,
       sorted_indices=True,   # SciPy CSR has sorted indices by default
       canonical=True,        # SciPy CSR is canonical by default
       validate="metadata",   # skip full validation. scipy is correct by construction
   )

   print(csr)
   # CSRArray(shape=(512, 512), nnz=..., dtype=float32, index_dtype=int32,
   #          sorted_indices=True, has_canonical_format=True)

Verifying products against SciPy
----------------------------------

Once you have a :class:`~mlx_sparse.CSRArray`, run a product and compare to
SciPy's reference result.

.. code-block:: python

   ms.use_cpu()   # run on CPU for fair comparison

   x_np = rng.standard_normal(512).astype(np.float32)
   x    = mx.array(x_np)

   # mlx-sparse result
   y_ms = csr @ x
   mx.eval(y_ms)
   y_ms_np = np.array(y_ms)

   # SciPy reference
   y_sp = sp @ x_np

   np.testing.assert_allclose(y_ms_np, y_sp, rtol=1e-5, atol=1e-5)
   print("results match SciPy!")

Converting a SciPy COO matrix
-------------------------------

SciPy COO arrays expose ``.data``, ``.row``, and ``.col``. Because SciPy COO
may contain duplicate entries, converting via :func:`~mlx_sparse.coo_array`
preserves them. Call ``tocsr(canonical=True)`` to sum duplicates.

.. code-block:: python

   sp_coo = scipy.sparse.random(128, 256, density=0.05, format="coo",
                                 dtype=np.float32, random_state=rng)

   coo = ms.coo_array(
       (mx.array(sp_coo.data.astype(np.float32)),
        (mx.array(sp_coo.row.astype(np.int32)),
         mx.array(sp_coo.col.astype(np.int32)))),
       shape=sp_coo.shape,
   )
   csr = coo.tocsr(canonical=True)

Converting back to SciPy
--------------------------

To go in the other direction (for example, to pass an mlx-sparse result back
to SciPy), evaluate the MLX arrays on host and construct a SciPy object:

.. code-block:: python

   mx.eval(csr.data, csr.indices, csr.indptr)

   sp_from_ms = scipy.sparse.csr_array(
       (np.array(csr.data),
        np.array(csr.indices),
        np.array(csr.indptr)),
       shape=csr.shape,
   )

Running on GPU and comparing to SciPy CPU
------------------------------------------

This pattern is common for benchmarks: generate data with SciPy (CPU), run
on both, compare accuracy.

.. code-block:: python

   import time

   ms.use_gpu()

   csr_gpu = ms.csr_array(
       (mx.array(sp.data.astype(np.float32)),
        mx.array(sp.indices.astype(np.int32)),
        mx.array(sp.indptr.astype(np.int32))),
       shape=sp.shape,
       sorted_indices=True,
       canonical=True,
       validate=False,
   )
   x = mx.array(x_np)

   # Warmup
   for _ in range(5):
       mx.eval(csr_gpu @ x)

   # Timed run
   t0 = time.perf_counter()
   for _ in range(100):
       mx.eval(csr_gpu @ x)
   gpu_ms = 1000 * (time.perf_counter() - t0) / 100

   # SciPy CPU reference timing
   t0 = time.perf_counter()
   for _ in range(100):
       _ = sp @ x_np
   scipy_ms = 1000 * (time.perf_counter() - t0) / 100

   print(f"mlx-sparse GPU: {gpu_ms:.3f} ms")
   print(f"SciPy CPU:      {scipy_ms:.3f} ms")

The GPU advantage over SciPy CPU depends heavily on matrix density and row
length uniformity. See
:doc:`../performance` for a detailed discussion.

Using SciPy in tests
---------------------

The test suite uses SciPy as a correctness reference. Tests guard against
SciPy being absent:

.. code-block:: python

   import pytest

   def test_csr_matvec_matches_scipy(mx, scipy_sparse):
       sp = scipy_sparse.random(128, 256, density=0.02, format="csr",
                                 dtype=np.float32, random_state=0)
       x_np = np.random.randn(256).astype(np.float32)

       csr = ms.csr_array(
           (mx.array(sp.data),
            mx.array(sp.indices, dtype=mx.int32),
            mx.array(sp.indptr,  dtype=mx.int32)),
           shape=sp.shape,
       )
       x = mx.array(x_np)

       y = csr @ x
       mx.eval(y)
       np.testing.assert_allclose(np.array(y), sp @ x_np, rtol=1e-5, atol=1e-5)

The ``scipy_sparse`` fixture in ``tests/conftest.py`` calls
``pytest.importorskip("scipy.sparse")``, so the test is automatically skipped
if SciPy is not installed.
