Sparse linear algebra
=====================

``mlx_sparse.linalg`` is the sparse counterpart to ``mlx.linalg``. It does not
densify inputs. Solver, factorization, and spectral routines dispatch through
native C++/Metal sparse kernels and require sparse containers.

Design contract
---------------

* ``CSRArray`` is the primary execution format.
* ``COOArray`` inputs are canonicalized to CSR before dispatch.
* ``CSCArray`` inputs are converted once to canonical CSR before dispatch.
  This is intentional for the current solver layer: Krylov iterations and
  triangular solves are row-output workloads, so the existing CSR kernels avoid
  repeated CSC scatter-add work inside every iteration.
* Dense MLX arrays are rejected by sparse linalg APIs.
* Python owns validation and object packaging only, numerical kernels live in
  the native extension.

Iterative solvers
-----------------

``cg(A, b)`` solves symmetric positive-definite systems using native conjugate
gradients. ``gmres(A, b)`` uses restarted Arnoldi/GMRES for nonsymmetric
systems. ``minres(A, b)`` uses Lanczos projection for symmetric indefinite
systems. All three return ``(x, info)`` where ``info == 0`` means convergence.

Sparse direct factorizations
----------------------------

``sparse_cholesky(A)`` computes a sparse lower factor ``L`` with
``A = L @ L.T`` for positive-definite real matrices. ``sparse_lu(A)`` computes
``P @ A = L @ U`` with sparse CSR factors and row pivoting. ``spsolve(A, b)``
uses sparse LU and sparse triangular solves.

Direct factorization still produces CSR factors. CSC input support is an input
format convenience, not a CSC-native supernodal factorization path.
Accelerate direct-solver support is being built behind a feature gate; the
native infrastructure can validate and normalize real ``float32`` CSR, COO, and
CSC inputs into canonical CSC storage for future framework calls, but public
``sparse_cholesky``, ``sparse_lu``, and ``spsolve`` dispatch still use the
existing sparse factorization path.

Spectral routines
-----------------

``eigsh`` uses native Lanczos projection for Hermitian sparse matrices. ``eigs``
uses native Arnoldi projection. ``svds`` applies Lanczos to the sparse normal
operator without materializing ``A.T @ A``.

Current ``svds`` execution audit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current ``svds`` path is conservative about output assembly but now has a
dedicated native normal-operator Lanczos step:

* ``CSRArray`` inputs are canonicalized. ``COOArray`` and ``CSCArray`` inputs
  are converted once to canonical CSR before dispatch.
* ``float16`` and ``bfloat16`` inputs are promoted to ``float32``. Complex
  sparse inputs are not currently accepted by the spectral routines.
* The native ``csr_svds`` implementation runs Lanczos over the normal operator
  ``A.T @ A`` without materializing the dense or sparse normal matrix.
* Each normal-operator application uses a dedicated fused native step: for a
  CSR row, it computes that row's ``A @ v`` contribution and immediately
  accumulates the matching ``A.T @ (...)`` contribution into the right-vector
  workspace. This avoids a Python-level pair of sparse products and avoids
  host materialization of the intermediate ``A @ v`` vector.
* On Metal, the Lanczos recurrence stays in a native GPU kernel. The small
  tridiagonal eigensolve, Ritz-vector back transformation, and final
  singular-vector assembly still run on CPU after synchronizing the Lanczos
  basis.

Sparse reductions
-----------------

``A.vdot(B)``, ``A.dot(B)``, ``mlx_sparse.linalg.vdot(A, B)``, and
``mlx_sparse.linalg.dot(A, B)`` compute sparse Frobenius inner products by
merging canonical CSR rows in native code. No dense intermediate is created.
``vdot`` follows NumPy/MLX convention and conjugates the left operand for
``complex64`` inputs, ``dot`` does not conjugate either operand.

GPU coverage
------------

Calling ``ms.use_gpu()`` (or ``mx.set_default_device(mx.gpu)``) before a
solver call routes the compute-heavy Krylov step to Metal.

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Function
     - GPU path
     - What runs on GPU
   * - ``linalg.cg``
     - Full
     - Entire CG iteration inside a single Metal threadgroup kernel.
   * - ``linalg.gmres``
     - Partial
     - Arnoldi factorisation (``csr_arnoldi`` kernel) per restart, the
       small least-squares solve and convergence check run on CPU.
   * - ``linalg.minres``
     - Partial
     - Lanczos tridiagonalisation (``csr_lanczos`` kernel), the
       tridiagonal least-squares solve runs on CPU.
   * - ``linalg.eigsh``
     - Partial
     - Lanczos tridiagonalisation (``csr_lanczos`` kernel), Jacobi
       eigendecomposition of the small tridiagonal matrix runs on CPU.
   * - ``linalg.eigs``
     - Partial
     - Arnoldi factorisation (``csr_arnoldi`` kernel), QR iteration on
       the small Hessenberg matrix runs on CPU.
   * - ``linalg.svds``
     - Partial
     - Dedicated normal-operator Lanczos step
       (``A.T @ (A @ v)`` without host intermediate), small eigensolve and
       singular-vector assembly run on CPU.
   * - ``sparse_cholesky`` (factorisation)
     - None
     - Sequential fill-in algorithm runs on CPU.
   * - ``SparseCholesky.solve`` / ``SparseLU.solve`` / ``spsolve``
     - Full
     - Forward/back-substitution via ``csr_triangular_solve`` kernel,
       row permutation via ``csr_permute_vector`` kernel.
   * - ``CSRArray.vdot`` / ``CSRArray.dot``
     - Full
     - Row-merge sparse inner product via ``csr_vdot`` kernel.

The GPU advantage grows with matrix size.  At ``n ≲ 1 000`` the kernel
launch and ``mx.eval()`` synchronisation overhead can exceed the parallel
speedup, the break-even point is typically around ``n ≈ 2 000–5 000``
depending on density.

Numerical scope
---------------

The solver, factorization, and spectral kernels operate on real floating-point
sparse matrices. ``float16`` and ``bfloat16`` inputs are promoted to ``float32``
for solver stability. Sparse ``dot`` and ``vdot`` additionally support
``complex64`` because their conjugation convention is unambiguous.
