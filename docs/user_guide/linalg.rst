Sparse linear algebra
=====================

``mlx_sparse.linalg`` is the sparse counterpart to ``mlx.linalg``. It does not
densify inputs. Solver, factorization, and spectral routines dispatch through
native C++/Metal sparse kernels and require sparse containers.

For a solver-by-solver map of CPU, Metal GPU, and Accelerate coverage, see
:doc:`linalg_solvers`.

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
systems. ``minres(A, b)`` uses a native Paige-Saunders recurrence for symmetric
indefinite systems and supports the shifted system ``(A - shift * I) x = b``.
All three return ``(x, info)`` where ``info == 0`` means convergence.

``cg`` accepts native-backed identity, diagonal, and Jacobi preconditioners
through ``M``. ``gmres`` accepts identity, diagonal/Jacobi, ILU(0),
exact-factor wrappers, and explicit inverse-apply callables or objects,
diagonal/Jacobi, ILU(0), and exact-factor GMRES use native left-preconditioned
solver entrypoints and true-residual convergence checks. ``minres`` accepts
only identity and symmetric positive-definite diagonal/Jacobi preconditioners.
See :doc:`preconditioners` for the current support matrix and
CPU/Metal/Accelerate boundaries.

Sparse direct factorizations
----------------------------

``sparse_cholesky(A)`` computes a sparse lower factor ``L`` with
``A = L @ L.T`` for positive-definite real matrices. ``sparse_lu(A)`` computes
``P @ A = L @ U`` with sparse CSR factors and row pivoting. These APIs return
explicit sparse factors and therefore stay on the native mlx-sparse path.
They preserve natural-order native semantics: no fill-reducing ordering,
supernodal factorization, native QR, native LDLT, or rectangular native direct
solver is introduced by the fallback path.

``factorized(A, method="auto")`` returns a reusable solve object without
exposing explicit factors. On Accelerate-enabled Apple builds it uses opaque
Accelerate ``float32`` factorization objects for supported methods:
``"cholesky"``, ``"ldlt"``, ``"qr"``, ``"cholesky_ata"``, and, on macOS 15.5
or newer SDK/runtimes, ``"lu"``. CSR, CSC, and COO inputs are validated and
normalized to canonical CSC before the framework call. ``spsolve(A, b)`` uses
this transparent Accelerate LU fast path for supported square real systems and
falls back to the native LU path otherwise.

The explicit-factor APIs are intentionally not Accelerate-backed: Accelerate
does not return mlx-sparse ``CSRArray`` factors, so using it there would change
the public contract.

Spectral routines
-----------------

``eigsh`` uses native Lanczos projection for Hermitian sparse matrices. ``eigs``
uses native Arnoldi projection. ``svds`` applies Lanczos to the sparse normal
operator without materializing ``A.T @ A``.

``svds`` execution model
~~~~~~~~~~~~~~~~~~~~~~~~

The ``svds`` path uses a dedicated native normal-operator Lanczos step while
keeping the small post-processing phases on CPU:

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
  singular-vector assembly run on CPU after synchronizing the Lanczos basis.

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
solver call routes supported native kernels to Metal. The exact behavior is
solver-specific: some paths are full GPU paths, some use GPU for the dominant
Krylov or triangular-solve phase, and Accelerate direct solves are CPU-only.
See :doc:`linalg_solvers` for the complete support matrix.

The GPU advantage grows with matrix size. At ``n < 1 000`` the kernel launch
and ``mx.eval()`` synchronization overhead can exceed the parallel speedup.
The break-even point is typically around ``n = 2 000`` to ``5 000`` depending
on density.

Numerical scope
---------------

The solver, factorization, and spectral kernels operate on real floating-point
sparse matrices. ``float16`` and ``bfloat16`` inputs are promoted to ``float32``
for solver stability. Sparse ``dot`` and ``vdot`` additionally support
``complex64`` because their conjugation convention is unambiguous.
