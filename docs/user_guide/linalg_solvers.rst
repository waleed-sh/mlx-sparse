.. _linalg-solvers:

Solvers
=======

This page summarizes the public ``mlx_sparse.linalg`` solver surface and where
each solver runs. It is meant to answer two questions quickly:

* Which sparse solver should I call?
* Does that path run on CPU, Metal GPU, Apple Accelerate, or a mix?

Support labels
--------------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Label
     - Meaning
   * - Full CPU + GPU
     - Native CPU and Metal GPU implementations are available for the solver's
       main numerical work. Calling ``ms.use_gpu()`` or setting MLX's default
       device to GPU keeps the solve on the GPU path.
   * - Partial
     - The solver is available from CPU and GPU contexts, but one important
       phase still runs on CPU. Common examples are CPU sparse factorization
       followed by GPU triangular solves, or GPU Krylov projection followed by
       a small CPU dense solve/eigendecomposition.
   * - CPU only
     - The solver runs on CPU. Apple Accelerate sparse direct solvers are in
       this category even when MLX's default device is GPU.
   * - GPU only
     - A GPU implementation exists without a CPU implementation. No public
       ``mlx_sparse.linalg`` solver currently has this label.

All native sparse linalg entrypoints accept ``CSRArray``, ``COOArray``, and
``CSCArray`` inputs unless the function documentation says otherwise. Native
solver kernels normalize sparse inputs to canonical CSR at solver entry.
Accelerate-backed direct solves normalize supported real inputs to canonical
CSC because Apple's sparse solver API is CSC-oriented.

Solver support matrix
---------------------

.. list-table::
   :widths: 24 24 20 18 34
   :header-rows: 1

   * - API
     - Use case
     - Native CPU/GPU coverage
     - Accelerate coverage
     - Notes
   * - ``linalg.cg``
     - Iterative solve for square symmetric positive-definite systems.
     - Full CPU + GPU
     - No
     - The unpreconditioned and Jacobi-preconditioned CG iterations run in
       native kernels. On GPU, each path uses a Metal solver kernel.
   * - ``linalg.gmres``
     - Iterative solve for square general systems.
     - Partial
     - No
     - Unpreconditioned, diagonal/Jacobi-preconditioned,
       ILU(0)-preconditioned, and exact-factor preconditioned entries are
       native. Arnoldi work can run on GPU when selected, restart bookkeeping,
       true-residual checks, and the small least-squares solve run on CPU.
       Custom callable preconditioners use a documented host fallback.
   * - ``linalg.minres``
     - Iterative solve for square symmetric indefinite systems.
     - Full CPU + GPU
     - No
     - Unpreconditioned and diagonal/Jacobi-preconditioned shifted MINRES use
       native Paige-Saunders recurrence kernels. Diagonal/Jacobi
       preconditioners must be SPD.
   * - ``linalg.spsolve``
     - One-shot direct solve for square systems.
     - Partial
     - CPU only, optional
     - Native fallback uses CPU LU factorization and can use GPU triangular
       solves. Accelerate-enabled Apple builds may use opaque Accelerate LU
       for supported real systems.
   * - ``linalg.factorized(..., method="auto" | "lu")``
     - Reusable direct solve object for square general systems.
     - Partial
     - CPU only, optional
     - Native LU is available everywhere. Accelerate LU is available only when
       the build and Apple runtime expose ``SparseFactorizationLU``.
   * - ``linalg.factorized(..., method="cholesky")``
     - Reusable direct solve object for square SPD systems.
     - Partial
     - CPU only, optional
     - Native explicit Cholesky is available everywhere. Accelerate Cholesky
       uses an opaque factorization object and does not expose sparse factors.
   * - ``linalg.factorized(..., method="ldlt")``
     - Reusable direct solve object for square symmetric indefinite systems.
     - No native path
     - CPU only, optional
     - Available only in Accelerate-enabled Apple builds for supported real
       inputs.
   * - ``linalg.factorized(..., method="qr")``
     - Reusable least-squares solve object for rectangular systems.
     - No native path
     - CPU only, optional
     - Available only in Accelerate-enabled Apple builds for supported real
       inputs.
   * - ``linalg.factorized(..., method="cholesky_ata")``
     - Reusable normal-equation solve object for rectangular systems.
     - No native path
     - CPU only, optional
     - Available only in Accelerate-enabled Apple builds for supported real
       inputs.
   * - ``linalg.sparse_cholesky`` / ``linalg.cholesky``
     - Explicit sparse Cholesky factorization for SPD systems.
     - CPU only for factorization
     - No
     - Returns explicit mlx-sparse CSR factors. The returned
       ``SparseCholesky.solve`` can use native GPU triangular solves. The
       native CPU factorization keeps natural-order semantics and uses an
       allocation-light sparse accumulator implementation.
   * - ``linalg.sparse_lu`` / ``linalg.splu``
     - Explicit sparse LU factorization for square general systems.
     - CPU only for factorization
     - No
     - Returns explicit mlx-sparse CSR factors and pivots. The returned
       ``SparseLU.solve`` can use native GPU permutation and triangular solves.
       The native CPU factorization preserves the existing partial pivoting
       semantics and does not apply fill-reducing ordering.
   * - ``SparseCholesky.solve`` / ``SparseLU.solve``
     - Reuse explicit native factors for one or more right-hand sides.
     - Full CPU + GPU for solve phase
     - No
     - This row covers the solve phase only, the factors were produced by the
       CPU factorization APIs above.  CPU matrix-RHS solves use one native
       triangular-solve sequence instead of a Python loop over RHS columns.
   * - ``linalg.eigsh``
     - A few eigenpairs of a square symmetric/Hermitian sparse matrix.
     - Partial
     - No
     - Lanczos projection can run on GPU, the small projected eigensolve runs
       on CPU.
   * - ``linalg.eigs``
     - A few eigenpairs of a square general sparse matrix.
     - Partial
     - No
     - Arnoldi projection can run on GPU, the small Hessenberg eigensolve runs
       on CPU.
   * - ``linalg.svds``
     - A few singular values/vectors of a sparse matrix.
     - Partial
     - No
     - The native normal-operator Lanczos step can run on GPU, the small
       eigensolve and singular-vector assembly run on CPU.
   * - ``linalg.lanczos``
     - Low-level Lanczos projection helper.
     - Partial
     - No
     - The projection can run on GPU, small projected post-processing is CPU
       work in the higher-level solvers that consume it.

Accelerate direct solves
------------------------

Accelerate support is opt-in at build time with
``MLX_SPARSE_ENABLE_ACCELERATE=ON``. Portable wheels that are not built with
this option report ``ms.capabilities.ACCELERATE == False`` and use the native
paths above.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Method
     - Matrix requirement
     - Availability
   * - ``"cholesky"``
     - Square, real, symmetric positive-definite.
     - Accelerate-enabled Apple build.
   * - ``"ldlt"``
     - Square, real, symmetric indefinite.
     - Accelerate-enabled Apple build.
   * - ``"lu"``
     - Square, real, general nonsingular.
     - Accelerate-enabled Apple build with LU available in the Apple SDK and
       runtime.
   * - ``"qr"``
     - Real rectangular least-squares system.
     - Accelerate-enabled Apple build.
   * - ``"cholesky_ata"``
     - Real rectangular normal-equation solve.
     - Accelerate-enabled Apple build.

Accelerate direct solves currently operate on ``float32`` factorization
objects. ``float16`` and ``bfloat16`` inputs are promoted to ``float32`` for
the solve. Complex sparse inputs are rejected. ``int32`` and ``int64`` sparse
indices are accepted after validation against the matrix shape and the limits
of the Accelerate sparse API.

Choosing a solver
-----------------

* Use ``cg`` for large SPD systems when an iterative method is appropriate.
* Use ``gmres`` for large general square systems when LU factorization is too
  expensive or too memory-heavy.
* Use ``minres`` for large symmetric indefinite systems.
* Use ``spsolve`` for a one-shot square direct solve.
* Use ``factorized`` when solving the same sparse system against multiple
  right-hand sides.  Native explicit-factor solves accept rank-2 RHS arrays on
  CPU, and Accelerate-enabled builds can use opaque framework solves for
  supported methods.
* Use ``sparse_cholesky`` or ``sparse_lu`` only when you need explicit
  mlx-sparse factor objects.
* Use ``eigsh``, ``eigs``, or ``svds`` when you need only a few spectral
  values/vectors rather than a dense decomposition.

Preconditioners
---------------

``linalg.cg`` accepts native-backed ``identity``, ``diagonal``, and ``jacobi``
preconditioners from ``mlx_sparse.linalg.preconditioners``. ``identity`` uses
the existing unpreconditioned CG path. ``diagonal`` and ``jacobi`` dispatch to
native Jacobi-preconditioned CG on CPU or Metal depending on the selected MLX
device and still test convergence against the true residual
``||b - A @ x||``.

``linalg.gmres`` accepts ``identity``, ``diagonal``/``jacobi``, ``ilu0``,
exact-factor preconditioners, and explicit inverse-apply callables or objects.
The diagonal/Jacobi, ILU(0), and exact-factor paths build Krylov vectors for
``M^{-1} A`` through native solver entrypoints and test convergence against the
true residual ``b - A @ x``. ILU(0) setup runs on CPU and applies through native
CSR triangular solves on CPU or Metal. Explicit native LU/Cholesky factors apply
through native permutation/triangular-solve bindings, guarded Accelerate
factorized objects use Apple's CPU sparse solver when that support is built in.
Custom callable/object preconditioners still use a slower host fallback because
arbitrary Python cannot be called from native solver kernels.

``linalg.minres`` accepts ``identity`` plus finite strictly positive
``diagonal``/``jacobi`` preconditioners. This is stricter than GMRES because
preconditioned MINRES requires a symmetric positive-definite preconditioner.
``minres(..., shift=s)`` solves ``(A - s I) x = b``.
