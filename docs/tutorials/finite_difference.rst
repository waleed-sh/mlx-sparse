Finite difference Laplacian
============================

This tutorial demonstrates how to assemble the 2D discrete Laplacian on an
``n x n`` grid and use it for a sparse matrix-vector product. The example is
based on ``examples/finite_difference_2d.py`` in the repository.

The 2D Laplacian appears in heat conduction, Poisson solvers, graph diffusion,
and image processing. Its stencil connects each interior grid point to its four
neighbors with coefficient −1 and a self-coupling of +4 (for a 5-point
stencil on a uniform grid). The resulting matrix is an ``n²xn²`` sparse
symmetric positive-definite matrix with at most 5 non-zeros per row.

Understanding the structure
----------------------------

For an ``n x n`` grid with index function ``idx(i, j) = i * n + j``, the
row corresponding to grid point ``(i, j)`` has entries:

.. code-block:: text

   (idx(i,   j),   idx(i,   j))  <- self: coefficient +4
   (idx(i-1, j),   idx(i,   j))  <- north neighbor: coefficient -1
   (idx(i+1, j),   idx(i,   j))  <- south neighbor: coefficient -1
   (idx(i,   j-1), idx(i,   j))  <- west neighbor: coefficient  -1
   (idx(i,   j+1), idx(i,   j))  <- east neighbor: coefficient  -1

Boundary neighbors that fall outside the grid are omitted. Corner points have
2 neighbors, edge points have 3, and interior points have 4. This produces the
COO entry list directly.

Assembling with COO
--------------------

The COO format is natural for stencil assembly: loop over grid points and
push entries into lists.

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms


   def laplacian_2d(n: int) -> ms.CSRArray:
       rows: list[int] = []
       cols: list[int] = []
       vals: list[float] = []

       def idx(i: int, j: int) -> int:
           return i * n + j

       for i in range(n):
           for j in range(n):
               center = idx(i, j)
               # Self connection.
               rows.append(center)
               cols.append(center)
               vals.append(4.0)
               # Neighbors (only if in bounds).
               for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                   if 0 <= ni < n and 0 <= nj < n:
                       rows.append(center)
                       cols.append(idx(ni, nj))
                       vals.append(-1.0)

       data = mx.array(np.asarray(vals, dtype=np.float32))
       row  = mx.array(np.asarray(rows, dtype=np.int32))
       col  = mx.array(np.asarray(cols, dtype=np.int32))

       return ms.coo_array((data, (row, col)), shape=(n*n, n*n)).tocsr(canonical=True)

The stencil produces no duplicate entries (each ``(i, j)`` pair appears at
most once in the loop), so ``canonical=True`` here only sorts the column
indices. The nnz count does not change.

Examining the matrix
---------------------

.. code-block:: python

   ms.use_cpu()   # construction always runs on CPU
   L = laplacian_2d(4)   # 16x16 matrix for a 4x4 grid

   print(f"shape:  {L.shape}")    # (16, 16)
   print(f"nnz:    {L.nnz}")      # 52 = 16*4 minus boundary connections
   print(f"density: {L.nnz / (16**2):.3f}")   # ~0.20

   mx.eval(L.data)
   print("unique values:", set(float(v) for v in np.array(L.data)))
   # {4.0, -1.0}

Running on GPU
--------------

Select GPU before construction to keep fixed-shape conversion and product
operations on the Metal path.

.. code-block:: python

   ms.use_gpu()
   L = laplacian_2d(16)              # 256x256 sparse Laplacian

   x = mx.array(np.ones(256, dtype=np.float32))
   y = L @ x
   mx.eval(y)
   print(np.array(y)[:8])
   # Expected: interior points yield 0 (sum over 4 neighbors each -1, self +4,
   # but with all ones as input: 4 - num_neighbors = 0 for interior).

Verifying the result
--------------------

The Laplacian applied to a constant vector produces a boundary-indicator vector:
interior points give 0 (their 4 neighbors each contribute −1 while the diagonal
is +4), while boundary points give a positive value proportional to how many
neighbors they are missing.

.. code-block:: python

   y_np = np.array(y)
   n = 16
   for i in range(n):
       for j in range(n):
           k = i * n + j
           num_neighbors = (
               (i > 0) + (i < n-1) + (j > 0) + (j < n-1)
           )
           expected = 4 - num_neighbors   # 0 for interior, 1/2 for boundary
           assert abs(y_np[k] - expected) < 1e-5, f"failed at ({i},{j})"
   print("all entries correct!")

Running the example script
--------------------------

The repository ships the same example as a standalone script:

.. code-block:: bash

   python examples/finite_difference_2d.py --device gpu
   python examples/finite_difference_2d.py --device cpu

The script assembles the 16x16 grid Laplacian and computes ``L @ ones`` on the
selected MLX device.

Performance notes
-----------------

For the 2D Laplacian, each row has exactly 3, 4, or 5 non-zeros, so row
lengths are very uniform. This is the best case for the scalar row Metal
kernel: load imbalance is minimal and GPU utilization is strong. For a 128x128
grid (16,384x16,384 matrix with ~82K non-zeros), the Metal kernel should
comfortably outperform both the CPU kernel and dense MLX matmul.

See :doc:`../performance` for guidance on benchmarking this workload.
