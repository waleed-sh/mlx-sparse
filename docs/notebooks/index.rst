Jupyter notebooks
=================

The notebooks below can be downloaded and executed locally. Pre-computed
outputs are shown inline where available so you can read the results without
running anything.

All timing results are collected on an **Apple M5, 10-core GPU**,
macOS 26.0, MLX 0.31, mlx-sparse 0.0.1b0.

Primitives
----------

These notebooks cover the core sparse containers and operations: construction,
matrix-vector products, matrix-matrix products, autodiff, dtype/device handling,
and integration with third-party libraries.

.. toctree::
   :maxdepth: 1

   01_first_steps
   02_csr_matvec
   03_csr_matmul
   04_constructors
   05_sparse_sparse
   06_autodiff
   07_dtype_device
   08_canonicalization
   09_scipy_bridge
   18_csc_format
   10_benchmarks
   11_graph_algorithms
   12_neural_layers

Sparse linear algebra
---------------------

These notebooks cover ``mlx_sparse.linalg``: iterative solvers, direct
factorizations, spectral routines, and matrix-free linear operators.

.. toctree::
   :maxdepth: 1

   13_linalg_solvers
   14_linalg_factorizations
   15_linalg_spectral
   16_linalg_operators

Sparse preconditioners
----------------------

These notebooks focus on ``mlx_sparse.linalg.preconditioners``. They keep the
preconditioner material separate from the general solver notebook so each
inverse-apply strategy can show its setup cost, apply path, solver effect, and
failure policy in context.

.. toctree::
   :maxdepth: 1

   21_preconditioner_identity
   22_preconditioner_diagonal
   23_preconditioner_jacobi
   24_preconditioner_ilu0
   25_preconditioner_ichol0
   26_preconditioner_chebyshev
   27_preconditioner_exact
   28_preconditioner_callable

Accelerate direct solvers
--------------------------

These notebooks focus on optional Apple Accelerate sparse direct solvers:
opaque reusable factorized solves for square systems and rectangular
least-squares workloads. Source/editable builds must enable Accelerate as
described in :doc:`../installation`.

.. toctree::
   :maxdepth: 1

   19_accelerate_square_solvers
   20_accelerate_rectangular_solvers
