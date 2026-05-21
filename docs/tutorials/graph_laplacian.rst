Graph Laplacians
================

This tutorial builds a cycle graph Laplacian using COO assembly, converts it
to CSR, and applies it to a signal on GPU. The example is based on
``examples/graph_laplacian.py``.

The graph Laplacian of an undirected graph is the matrix ``L = D - A``, where
``D`` is the diagonal degree matrix and ``A`` is the adjacency matrix. For a
cycle graph on ``n`` nodes (each node connected to its two immediate
neighbours), the Laplacian has:

* **Diagonal entries**: 2 (each node has degree 2).
* **Off-diagonal entries**: -1 at positions ``(i, i+1)`` and ``(i, i-1)``,
  with wrap-around at the boundary.

The result is an ``n x n`` matrix with exactly ``3n`` stored values.

Assembling the COO triplets
----------------------------

Rather than indexing a dense array, assemble the Laplacian directly as COO
triplets. Three bands are concatenated in one shot using NumPy.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   def cycle_graph_laplacian(n: int) -> ms.CSRArray:
       nodes = np.arange(n, dtype=np.int32)

       # Row indices: same node repeated three times for each band
       row = np.concatenate([nodes, nodes, nodes])

       # Column indices: self, next node (mod n), previous node (mod n)
       col = np.concatenate([
           nodes,
           (nodes + 1) % n,
           (nodes - 1) % n,
       ]).astype(np.int32)

       # Values: +2 on diagonal, -1 on both off-diagonals
       data = np.concatenate([
           np.full(n, 2.0, dtype=np.float32),
           np.full(n, -1.0, dtype=np.float32),
           np.full(n, -1.0, dtype=np.float32),
       ])

       return ms.coo_array(
           (mx.array(data), (mx.array(row), mx.array(col))),
           shape=(n, n),
       ).tocsr(canonical=True)

The call to ``tocsr(canonical=True)`` sorts column indices within each row and
sums any duplicate entries. For this construction there are no duplicates, so
the nnz count remains ``3n`` after canonicalization.

Examining the structure
------------------------

.. code-block:: python

   ms.use_cpu()
   L = cycle_graph_laplacian(6)

   print(L)
   # CSRArray(shape=(6, 6), nnz=18, dtype=float32, index_dtype=int32,
   #          sorted_indices=True, has_canonical_format=True)

   mx.eval(L.data, L.indices, L.indptr)
   print(np.array(L.todense()))

For ``n=6`` the dense matrix should be:

.. code-block:: text

   [[ 2, -1,  0,  0,  0, -1],
    [-1,  2, -1,  0,  0,  0],
    [ 0, -1,  2, -1,  0,  0],
    [ 0,  0, -1,  2, -1,  0],
    [ 0,  0,  0, -1,  2, -1],
    [-1,  0,  0,  0, -1,  2]]

The matrix is symmetric (``L == L.T``) and each row sums to zero, which is the
characteristic property of a graph Laplacian.

Applying the Laplacian to a signal
------------------------------------

A graph signal is a function defined at the nodes. The Laplacian applied to a
smooth signal produces a measure of local variation: for a perfectly smooth
(constant) signal, the result is zero everywhere. For a low-frequency signal
like a single period of a sine wave, the response is proportional to the
signal's Fourier frequency.

.. code-block:: python

   ms.use_gpu()

   n = 32
   L = cycle_graph_laplacian(n)

   # One full period of a sine wave on the cycle graph
   signal = mx.sin(2 * mx.pi * mx.arange(n, dtype=mx.float32) / n)

   response = L @ signal
   mx.eval(response)
   print(np.array(response)[:8])

For a discrete-time sine with frequency ``f`` on a cycle graph of ``n`` nodes,
the Laplacian eigenvalue is ``2 - 2*cos(2*pi*f/n)``. For ``f=1`` (one period),
this is approximately ``0.077`` for ``n=32``, so every entry of ``response``
should be close to ``0.077 * signal``.

Verifying the eigenvalue
------------------------

.. code-block:: python

   expected_eigenvalue = 2.0 - 2.0 * np.cos(2.0 * np.pi / n)

   np.testing.assert_allclose(
       np.array(response),
       expected_eigenvalue * np.array(signal),
       rtol=1e-5,
       atol=1e-5,
   )
   print(f"eigenvalue: {expected_eigenvalue:.6f}")

Running the example script
--------------------------

The repository ships a standalone script that runs this computation:

.. code-block:: bash

   python examples/graph_laplacian.py --nodes 32 --device gpu
   python examples/graph_laplacian.py --nodes 1024 --device gpu

Performance notes
-----------------

The cycle Laplacian has exactly 3 non-zeros per row (uniform row lengths). On
Metal, the scalar-per-row kernel handles this pattern efficiently because:

* Each thread processes a short fixed-length loop of 3 iterations.
* The indptr access pattern is sequential.
* The column gather into the dense signal vector is non-sequential but has
  a stride of roughly 1, 2, and ``n-1``, which maps to cache lines well for
  smaller ``n``.

For large ``n`` (above roughly 65,536 nodes), the cache for the signal vector
may start to spill, and the effective bandwidth will fall. Benchmark with
:doc:`../performance` guidance.
