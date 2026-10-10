Sparse linear algebra
=====================

``mlx_sparse.linalg`` provides iterative solves, direct factorizations, and
spectral approximations without converting the input sparse matrix to dense.
Native paths use C++/Metal sparse kernels. CG, GMRES, and BiCGSTAB also accept
fully matrix-free operators through host fallback loops.

For a solver-by-solver map of CPU, Metal GPU, and Accelerate coverage, see
:doc:`linalg_solvers`. See :doc:`solver_limitations` for numerical assumptions,
convergence limits, residual checks, and cases where results can be misleading.

Design contract
---------------

* ``CSRArray`` is the primary execution format.
* ``COOArray`` inputs are canonicalized to CSR before dispatch.
* ``CSCArray`` inputs are converted once to canonical CSR before dispatch.
  This is intentional for the current solver layer: Krylov iterations and
  triangular solves are row-output workloads, so the existing CSR kernels avoid
  repeated CSC scatter-add work inside every iteration.
* Dense MLX arrays are rejected by sparse linalg APIs.
* Native solver paths keep numerical loops in the extension. Matrix-free
  operators and custom callable preconditioners use host fallback loops.

Iterative solvers
-----------------

``cg(A, b)`` solves symmetric positive-definite systems using native conjugate
gradients. ``gmres(A, b)`` uses restarted Arnoldi/GMRES for nonsymmetric
systems. ``minres(A, b)`` uses a native Paige-Saunders recurrence for symmetric
indefinite systems and supports the shifted system ``(A - shift * I) x = b``.
All three return ``(x, info)`` where ``info == 0`` means convergence.

``cg`` accepts native-backed identity, diagonal/Jacobi, IC(0), and Chebyshev
preconditioners through ``M``. ``gmres`` accepts identity, diagonal/Jacobi,
ILU(0), exact-factor wrappers, and explicit inverse-apply callables or objects.
Diagonal/Jacobi, ILU(0), and exact-factor GMRES use native left-preconditioned
solver entrypoints and true-residual convergence checks. ``minres`` accepts
only identity and symmetric positive-definite diagonal/Jacobi preconditioners.
See :doc:`preconditioners` for the current support matrix and
CPU/Metal/Accelerate boundaries.

Fully matrix-free :class:`~mlx_sparse.linalg.LinearOperator` inputs are
accepted by ``cg``, ``gmres``, and ``bicgstab`` through a host fallback loop.
This path supports user-provided matvecs and arbitrary inverse-apply ``M``
objects. Sparse-backed operators use the native CSR CPU/Metal solver paths.

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

``spsolve_triangular(A, b, lower=True, unit_diagonal=False)`` exposes the
native CSR triangular-solve path directly for vector or matrix right-hand
sides. It is useful for checking explicit factors and factor-like
preconditioners. A public triangular-analysis object is intentionally not
exposed in v0.0.5b0 because repeated-apply benchmarks do not yet show a
consistent win over the default native solve path.

Spectral routines
-----------------

``eigsh`` uses native Lanczos projection for real symmetric sparse matrices.
``eigs`` uses native Arnoldi projection. ``svds`` uses native Golub-Kahan
bidiagonalization with fully reorthogonalized left and right bases. All three
routines accept real sparse inputs and compute in float32.

All four spectral entrypoints that build Krylov bases accept a user start
vector ``v0``. The current ``eigsh``, ``eigs``, and ``svds`` routines still
perform a single ``ncv``-bounded Ritz extraction, non-default ``tol`` or
``maxiter`` values are rejected until an implicitly restarted convergence loop
is implemented.

Starting vectors and breakdown
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``v0=None``, ``lanczos``, ``eigsh``, ``eigs``, and ``svds`` draw standard
normal samples from a fixed MLX key. Repeated calls on the same device use the
same start without consuming the global random stream. A random start avoids
the all-ones vector's invariant subspace on graph Laplacians and stochastic
matrices. It does not guarantee overlap with every eigenspace for every input.

Explicit starts must be finite and nonzero after conversion to float32.
Native CPU and Metal normalization scale values before squaring, preserving
the direction of small and large starts. Choose a start with components in
the eigenspaces you want to approximate. For ``svds``, these are eigenspaces
of the right singular subspace. A start in the right nullspace is valid.

An eigenvector start or a matrix with only a few distinct eigenvalues can
close the Krylov subspace before the requested basis dimension is reached.
``eigsh``, ``eigs``, and ``svds`` continue from a new random direction
orthogonalized against the existing basis. Native CPU and Metal code use
two orthogonalization passes and up to three attempts to find that direction.
The original start remains the first basis vector. This allows independent
directions for repeated eigenvalues within the existing ``ncv`` budget.
Breakdown checks are relative to the operator product's norm.

This continuation does not add an implicitly restarted convergence loop.
The routines still perform one ``ncv``-bounded Ritz extraction. They raise
``RuntimeError`` if orthogonal continuation fails or numerical exhaustion
leaves fewer than ``k`` independent Ritz candidates. They do not duplicate
a candidate to fill the result.

The low-level ``lanczos`` routine returns fixed-size arrays. After early
breakdown, unused coefficients and basis columns remain zero.

Chebyshev setup uses a separate fixed-seed random start for its native Lanczos
spectral estimate. Gershgorin bounds retain their existing role in selecting
the preconditioner interval.

The ``which`` selector is case-insensitive and accepts the following values:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Routine
     - Selectors
     - Selection criterion
   * - ``eigsh``
     - ``LM``, ``SM``, ``LA``, ``SA``
     - Largest or smallest magnitude, or largest or smallest algebraic value.
   * - ``eigs``
     - ``LM``, ``SM``, ``LR``, ``SR``
     - Largest or smallest magnitude, or largest or smallest real part.
   * - ``svds``
     - ``LM``, ``SM``
     - Largest or smallest singular value.

Unsupported selectors and non-string values raise ``ValueError`` before
matrix preparation or native execution. SciPy's ``BE`` selector for ``eigsh``
and ``LI`` and ``SI`` selectors for ``eigs`` are not implemented. ``LA`` and
``SA`` apply to ``eigsh`` only. Use ``LR`` and ``SR`` with ``eigs``.

``eigs`` execution model
~~~~~~~~~~~~~~~~~~~~~~~~

The small Arnoldi projection is solved with MLX's native dense eigensolver on
the CPU. Returned vectors are the Ritz vectors ``Q @ y``, normalized on the
selected device. The large basis stays on that device. Eigenvalues and
vectors always have dtype ``complex64``, including for real eigenvalues.
A real matrix can have complex conjugate eigenpairs. ``LM`` and ``SM`` compare
complex magnitudes, while ``LR`` and ``SR`` compare real parts. Ties retain
the projected solver's order. Requesting an odd number of eigenpairs can
select just one member of a conjugate pair.

General eigenvectors need not be orthogonal. A defective matrix may not have
a complete independent eigenbasis. Check the residual
``A @ vectors - vectors * values`` before using an approximation. A small
``ncv`` does not guarantee convergence, especially for nonnormal problems.
Use ``return_eigenvectors=False`` to skip the projected eigenvectors and
large vector reconstruction.

``svds`` execution model
~~~~~~~~~~~~~~~~~~~~~~~~

* ``CSRArray`` inputs are canonicalized. ``COOArray`` and ``CSCArray`` inputs
  are converted once to canonical CSR before dispatch.
* ``float16`` and ``bfloat16`` inputs are promoted to ``float32``. Complex
  sparse inputs are not currently accepted by the spectral routines.
* Native CPU and Metal code apply ``A`` and ``A.T`` separately. The method
  avoids forming ``A.T @ A`` and avoids the accuracy loss from squaring the
  singular values.
* Small GPU problems use a fused recurrence. Larger problems use parallel
  sparse products and two-pass Gram-Schmidt projections through native MLX
  matrix products. Only scalar breakdown checks synchronize each step.
* Both bases use two orthogonalization passes. At breakdown, an independent
  random direction completes the affected basis. The left basis has at most
  ``min(A.shape)`` columns. The right basis retains an additional column when
  possible, including the terminal residual for a right-nullspace start.
* The small matrix ``Q_left.T @ A @ Q_right`` retains projection corrections
  and couplings introduced by continuation. It is solved with MLX's native
  dense SVD on the CPU. Sparse products already computed during basis
  construction are reused in the projection. Projection and vector
  reconstruction run on the
  selected device without copying the large bases to the host.
* Both singular vector families are reconstructed from orthonormal bases.
  Zero singular values have unit left and right vectors. Near rank
  deficiency, values below float32 resolution can have large relative error.
  Bases for repeated and zero singular subspaces are not unique.
* ``return_singular_vectors=False`` skips the small singular vectors and all
  large vector reconstruction. ``"u"`` and ``"vh"`` skip reconstruction of
  the unrequested vector family.

Singular values are nonnegative. ``LM`` returns them in descending order and
``SM`` in ascending order. Check both residuals
``A @ vh.T - u * s`` and ``A.T @ u - vh.T * s``. Increase ``ncv`` if the
residuals are too large. The current single projection does not guarantee
convergence of the smallest singular triplets with a small basis. A near-zero
projected singular value can have a nonzero residual in the full matrix.
It does not establish the matrix's numerical rank.

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
