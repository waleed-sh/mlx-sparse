.. _solver-limitations:

Solver limitations
==================

This page describes the solver behavior in v0.0.6b1, including assumptions
that are not checked at entry and cases where a returned result needs an
independent check. See :doc:`linalg_solvers` for available methods and device
coverage, and :doc:`preconditioners` for preconditioner construction.

The comparisons below use established implementations such as PETSc,
SLEPc, SuperLU, CHOLMOD, PROPACK, and PRIMME. They identify specific capabilities
and accuracy controls. They do not imply that one method is fastest for every
matrix.

.. contents:: On this page
   :local:
   :depth: 2

Inputs, precision, and result checks
------------------------------------

Real float32 computation
~~~~~~~~~~~~~~~~~~~~~~~~

The solvers compute with real ``float32`` matrix values. ``float16`` and
``bfloat16`` values are promoted before computation. Promotion does not recover
precision already lost when the matrix was stored. Complex matrix inputs and
double-precision solver computation are not supported. ``eigs`` returns
``complex64`` results because a real matrix can have complex eigenvalues.

Some CPU paths use double-precision reductions or small dense calculations.
The matrix products, stored Krylov vectors, and returned solutions still have
float32 accuracy. CPU and Metal reductions can follow different orders, so
iteration counts, breakdown decisions, and vector bases need not agree exactly.

Float32 machine epsilon is about ``1.19e-7``. A tolerance below that value
does not provide double-precision accuracy. Conditioning matters as well as
precision. A small residual can coexist with a large solution error when the
matrix is ill-conditioned. [#templates]_

.. warning::

   ``spsolve``, ``SparseLU.solve``, ``SparseCholesky.solve``, and
   ``FactorizedSolve.solve`` currently convert right-hand sides to float32
   without rejecting complex MLX arrays.
   Their imaginary parts can be discarded. Supply real floating right-hand
   sides explicitly. These methods do not solve complex systems.

Storage and mathematical assumptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse containers are normalized to canonical CSR for native execution, or
canonical CSC for Accelerate. This may allocate new buffers and combine
duplicate entries. A container marked canonical is trusted by conversion
helpers. Incorrect index buffers or false canonical metadata are not repaired
by choosing a solver. See :doc:`validation`.

Structural validation does not prove symmetry, positive definiteness,
nonsingularity, or numerical rank. Checks for NaN and infinity also vary between
entry points. Use finite matrix values, right-hand sides, and initial guesses.
Do not rely on a numerical breakdown status to detect every invalid input.

All iterative linear solvers require a square operator and a rank-1 right-hand
side. Rank-2 right-hand sides are supported by direct solves, not by block
Krylov methods. Spectral and direct routines require explicit sparse containers.
They do not accept a fully matrix-free ``LinearOperator``.

Checking a linear solution
~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``A @ x = b``, check the residual of the returned solution against the
same tolerance used in the solve:

.. code-block:: python

   import mlx.core as mx
   import mlx_sparse as ms

   def check_solution(A, x, b, *, rtol, atol=0.0, shift=0.0):
       residual = b - (A @ x - shift * x)
       residual_norm = mx.linalg.norm(residual)
       target = mx.maximum(rtol * mx.linalg.norm(b), mx.array(atol))
       finite = mx.all(mx.isfinite(x)) & mx.all(mx.isfinite(residual))
       passed = finite & (residual_norm <= target)
       mx.eval(residual_norm, target, passed)
       return bool(passed.item()), float(residual_norm.item()), float(target.item())

   # A and b are a real sparse system prepared by the application.
   A = ms.asarray(A, dtype=mx.float32)
   b = b.astype(mx.float32)
   x, info = ms.linalg.cg(A, b, rtol=1e-5, atol=1e-7, return_info=True)
   passed, residual_norm, target = check_solution(A, x, b, rtol=1e-5, atol=1e-7)

For shifted MINRES, pass the same ``shift`` to the check. For matrix right-hand
sides, check each column separately. One aggregate norm can hide a poor solve
for a column with a small right-hand side. Norms can overflow or underflow for
extreme float32 scales. Rescale such problems and their absolute tolerances
before using this example.

This check measures the residual in float32. Applications needing a more
accurate assessment can evaluate the residual with a higher-precision sparse
operator in a separate validation step. A residual check does not estimate the
condition number or certify forward accuracy.

Iterative linear solvers
------------------------

Status, budgets, and callbacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cg``, ``gmres``, ``bicgstab``, and ``minres`` return ``(x, info)``. An integer
``info == 0`` means that the implementation's stopping test passed. Positive
values indicate an exhausted iteration budget. Negative values indicate
breakdown, an invalid preconditioned inner product, or non-finite arithmetic.
The returned ``x`` may still be useful after a nonzero status, but it needs a
residual check. Breakdown alone does not prove that the matrix is singular.

``return_info=True`` exposes the reported residual, iteration count, and reason
in ``SolverInfo``. It does not run an additional accuracy check. In particular,
the generic success message mentions a true-residual tolerance even on CG
paths that stop using the recursive residual described below.

The default ``maxiter`` is ``10 * n``. For GMRES, it counts inner Arnoldi steps,
not restart cycles. The default restart length is ``min(20, n)``. A zero budget
can succeed only if the starting solution already satisfies the stopping test.

Callbacks run once at solver exit, including on host fallback paths. They
cannot monitor intermediate convergence or stop an iteration early. GMRES
``"pr_norm"`` and ``"legacy"`` callbacks receive the final reported norm,
without SciPy's relative normalization or per-iteration callback stream.
``"legacy"`` does not change budget accounting. These names should not be
treated as full SciPy callback compatibility. [#scipy-gmres]_

CG: positive-definite systems and residual drift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CG requires a real symmetric positive-definite operator. A preconditioner
must be fixed, linear, symmetric, and positive definite. The solver does not
verify these properties for the full matrix. A positive diagonal alone does
not prove that the matrix is positive definite. An indefinite or nonsymmetric
input can break down or produce an unreliable approximation. [#petsc-cg]_

.. warning::

   CG stopping tests use the recursively updated residual. Rounding error can
   separate this vector from ``b - A @ x``. The current native and matrix-free
   CG paths can report ``info == 0`` while the returned solution fails the
   requested true-residual tolerance. ``SolverInfo.residual_norm`` does not
   certify that tolerance either. Recompute the residual before accepting a
   result, especially for ill-conditioned systems or long solves.

The native implementation does not periodically replace the recursive residual
with a fresh matrix product. Increasing ``maxiter`` or tightening ``rtol`` does
not resolve residual drift by itself. A better-conditioned formulation or
preconditioner may help. If a verified float32 solve cannot meet the required
accuracy, use a solver with suitable precision and residual controls.

Some preconditioned CG paths use absolute floors in denominator checks.
For example, Jacobi-preconditioned CG can break down on an identity matrix
with a right-hand side whose entries are ``1e-4``, although unpreconditioned CG
solves the same problem. Such a status is an implementation scaling limitation.
It is not evidence that the identity matrix is singular.

GMRES: restart size and stagnation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GMRES accepts general real square systems and uses left preconditioning.
It checks the true residual before reporting convergence. Restarting limits
basis storage, but discards information that can be needed for convergence.
A small restart length can stagnate on a nonsymmetric or ill-conditioned
problem. A larger restart increases storage and orthogonalization work.

Native projected least-squares problems use Givens QR. This avoids solving
that small problem through normal equations, but does not remove float32
Arnoldi limitations. A five-by-five Hilbert-like regression still fails to
meet ``rtol=1e-6`` with a full restart length and 64 inner steps. Increasing
the budget is not a universal remedy for loss of accuracy.

There is no flexible GMRES, Krylov recycling, or automatic deflation. A callable
``M`` must apply the same linear map throughout a solve. A changing inner
solver or nonlinear preconditioner requires a flexible method such as PETSc's
FGMRES, which has a different algorithmic contract. [#fgmres]_

BiCGSTAB: low storage, irregular convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BiCGSTAB accepts general real square systems and uses a short recurrence.
It checks the true residual before reporting success. Residuals need not
decrease monotonically, and near-zero recurrence scalars can cause breakdown
even for a nonsingular matrix. Preconditioning can change this behavior, but
does not guarantee convergence. [#templates]_

There is no BiCGSTAB(l) variant or automatic switch to GMRES after breakdown.
Its preconditioner must remain a fixed linear inverse application. A BiCGSTAB
iteration can require two operator products plus residual checks and
preconditioner applications, so iteration counts alone are not a useful
comparison with CG or GMRES.

MINRES: symmetry, shifts, and singular systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MINRES accepts a real symmetric operator, including an indefinite one, and
solves ``(A - shift * I) @ x = b``. Symmetry is assumed rather than checked.
The true residual of this shifted equation is tested before reporting success.
Supported nontrivial preconditioners are diagonal or Jacobi and must be
positive definite. Disabling ``check_preconditioner`` bypasses checks. It does
not make an indefinite preconditioner valid.

The recurrence includes absolute float32 breakdown floors. For example,
an identity system with right-hand-side entries of ``1e-8`` can break down
with the default relative tolerance and zero absolute tolerance. A
near-singular diagonal regression also remains a known convergence limitation.
Scaling or a suitable positive diagonal preconditioner can help, but each
result still needs verification.

This is MINRES, without the QLP extension. There is no numerical-rank report or
guarantee of a minimum-norm solution for singular or inconsistent systems.
MINRES-QLP explicitly addresses those cases. Use an implementation with that
contract when the minimum-norm least-squares solution is required. [#minresqlp]_

Matrix-free operators
~~~~~~~~~~~~~~~~~~~~~

Fully matrix-free ``LinearOperator`` inputs are supported by CG, GMRES, and
BiCGSTAB through host fallback loops. Sparse-backed operators use the native
CSR paths. MINRES does not have a fully matrix-free fallback.

The fallback synchronizes operator and preconditioner results with the host
each iteration. Its NumPy bookkeeping does not turn the operator into a
double-precision solve. Matvecs and returned solutions remain float32. Operator
callbacks must return finite vectors of the expected shape and implement a
fixed linear map. A callback exception may be reported as a non-finite failure
status, so that status does not uniquely identify an arithmetic overflow.

There is no iterative rectangular least-squares solver such as LSQR or LSMR.
Constructing normal equations as a workaround changes the conditioning and
can increase storage. LSMR works directly with products by ``A`` and its
transpose and provides a different problem contract. [#lsmr]_

Preconditioners
---------------

Inverse application and solver compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``M(v)`` applies an approximation to ``A^{-1} @ v``. It is not a matrix that
the solver factorizes automatically. ``cg`` accepts identity, diagonal/Jacobi,
IC(0), and Chebyshev on native sparse paths. ``minres`` accepts identity and
positive diagonal/Jacobi preconditioners. GMRES and BiCGSTAB additionally
support ILU(0), exact factors, and callable inverse applications. Arbitrary
callables use host fallbacks. See :doc:`preconditioners` for the full support
matrix.

An exact factor can reduce the iteration count while costing as much to
construct as a direct solve. Reuse amortizes setup only when the matrix stays
unchanged. There are no built-in algebraic multigrid, ILUT, or ILU(k)
preconditioners.

Jacobi and incomplete factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jacobi uses the diagonal and cannot detect poorly conditioned couplings that
are absent from that diagonal. Positive-diagonal checks apply to the
preconditioner, not to the full operator. ``zero_policy="unit"`` substitutes
for a small or missing diagonal. This avoids division by that entry, but does
not establish that the resulting preconditioner is useful or valid for CG.

ILU(0) and IC(0) retain the existing sparsity pattern in natural order. They
do not pivot, add fill, or choose a fill-reducing permutation. ILU(0) can
encounter a zero pivot on a nonsingular matrix. IC(0) can encounter a
non-positive pivot on an SPD matrix. Neither failure proves that an exact
factorization is impossible. [#templates]_

Their ``shift`` is applied to existing diagonal entries. It does not insert
missing diagonal structure. ``check=False`` relaxes selected checks, but does
not implement pivot repair or a robust modified incomplete factorization.
A preconditioner shift changes the inverse approximation. It does not change
the equation, unlike the ``shift`` argument to MINRES.

Chebyshev spectral bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

Chebyshev is a fixed-degree polynomial inverse approximation for SPD matrices.
Its effectiveness depends on the spectral interval and degree. Each
application costs ``degree`` sparse matrix-vector products.

.. warning::

   Automatic Chebyshev setup combines Gershgorin bounds with short Lanczos
   Ritz estimates. A positive estimated interval is not a certificate that
   the matrix is SPD or that all eigenvalues lie inside it. In particular,
   the estimated lower endpoint is heuristic when the Gershgorin lower bound
   is non-positive. Incorrect bounds can produce a poor or invalid CG
   preconditioner. Supply justified bounds when the matrix spectrum is known
   and check the final solve.

Explicit ``lambda_min`` and ``lambda_max`` are checked for a finite positive
ordering, not for containment of the matrix spectrum. A smoothing interval
that targets only part of the spectrum is also a different choice from a
bound used for a full inverse approximation. PETSc documents that distinction
in its Chebyshev implementation. [#chebyshev]_

Eigenvalues and singular values
-------------------------------

One projection, without a convergence loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``eigsh``, ``eigs``, and ``svds`` build one bounded Krylov basis and extract
``k`` candidates from a small dense projection. The original sparse matrix
is not converted to dense. Dense basis vectors and projected matrices are
part of the algorithm.

.. warning::

   These routines do not test residuals and iterate until the requested pairs
   converge. They can return ``k`` approximations with large residuals and
   no convergence warning. The default ``tol=0.0`` does not request
   machine-precision convergence. Non-default ``tol`` and any non-``None``
   ``maxiter`` raise ``NotImplementedError``. Check residuals before using
   the results as eigenpairs or singular triplets.

The default ``ncv`` is ``2 * k + 1``, capped at the available dimension.
Explicit ``ncv`` is clamped between ``k + 1`` and that dimension. The cap is
``n`` for eigenproblems and ``min(m, n)`` for SVD. A larger basis often helps,
but has no general accuracy guarantee. ``k`` must be smaller than the relevant
matrix dimension, so these APIs do not return a full decomposition.

There is no restart, locking of converged vectors, spectral shift-and-invert,
generalized eigenproblem, or user preconditioner for these routines.
``which="SM"`` selects small-magnitude candidates from the current projection.
It does not transform the problem to target the matrix's smallest values.
Interior values and tight clusters can require a much larger subspace.
Restarted Krylov-Schur and targeted transformations in SLEPc address these
limitations with explicit convergence controls. [#slepc-eps]_

Starts, multiplicity, and basis storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default start uses a fixed random key and does not advance MLX's global
random stream. Repeating a call with the same inputs and ``ncv`` on the same
device is not an independent convergence experiment. Supply a different
``v0`` if you want to assess sensitivity to the start. For SVD, ``v0`` is a
right-space vector of length ``n``, including for a wide matrix.

A start can have little or no overlap with a wanted subspace. The high-level
spectral routines continue from an orthogonal random direction after invariant
breakdown. This avoids duplicating pairs when a subspace closes. It is not a
convergence restart and does not guarantee complete multiplicities within a
small budget. Failed continuation or too few independent candidates raises
``RuntimeError``.

With basis dimension ``r``, eigenproblems store roughly ``n * r`` basis
entries. SVD stores left and right bases and operator images. Full
reorthogonalization work grows roughly as ``n * r**2`` for eigenproblems and
``(m + n) * r**2`` for SVD, in addition to sparse products and dense extraction.
Setting ``ncv`` to the full dimension can remove the memory advantage of a
partial decomposition. Signs, complex phases, and bases within a repeated
subspace are not unique. Compare residuals and subspaces rather than exact
vector entries.

eigsh: real symmetry and projected accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``eigsh`` assumes real symmetry and uses Lanczos with full reorthogonalization.

.. warning::

   ``eigsh`` does not verify that ``A == A.T``. It also does not infer the
   missing half of a matrix stored with only one triangle. Store the full
   symmetric operator for sparse products. A nonsymmetric input invalidates
   the symmetric projection and can return misleading real eigenvalues.

The projected tridiagonal problem is currently solved by a CPU Jacobi routine.
It has a fixed stopping threshold of ``1e-6`` on the largest off-diagonal
entry after scaling and a finite rotation budget. It does not report
nonconvergence of that small solve. Increasing ``ncv`` does not change this
threshold. Eigenvector reconstruction reads the full basis on CPU, and
``return_eigenvectors=False`` still computes those vectors internally.

eigs: complex results and nonnormal matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``eigs`` uses real Arnoldi and a complex dense eigensolve of the projected
Hessenberg matrix. Values and right eigenvectors are always ``complex64``.
``LM`` and ``SM`` use complex magnitude, while ``LR`` and ``SR`` use the real
part. Imaginary-part selectors and left eigenvectors are not supported.
An odd ``k`` can select one member of a conjugate pair.

The projected solve runs on CPU. Requested vectors are reconstructed on the
selected device. There is no guarantee that right eigenvectors of a general
matrix are mutually orthogonal. For a defective or strongly nonnormal matrix,
a small eigenpair residual alone can give a weak bound on eigenvalue error.
Eigenvector conditioning and separation matter. [#slepc-eps]_

svds: direct projection and small singular values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``svds`` uses Golub-Kahan bases with full reorthogonalization on both sides
and takes a direct SVD of the projected operator. It does not form ``A.T @ A``
or recover left vectors by dividing ``A @ v`` by a singular value. This avoids
those sources of squared-conditioning error and division by zero. The small
dense SVD runs on CPU, and requested vectors are reconstructed on the selected
device.

These choices do not ensure convergence of the singular triplets. An
orthonormal returned basis can still have large residuals. ``which="SM"`` is
particularly limited by the single projection, and a small returned value
does not certify numerical rank or a nullspace. It should not be used as the
only justification for inverting a singular value.

Established partial-SVD implementations provide additional controls. PROPACK
uses partial reorthogonalization and offers an implicitly restarted driver.
SLEPc provides thick-restart Lanczos and residual-based stopping. PRIMME_SVDS
uses a preconditioned two-stage method for accurate extreme singular triplets.
Those capabilities are not present here. [#propack]_ [#slepc-svd]_ [#primme]_

Checking spectral results
~~~~~~~~~~~~~~~~~~~~~~~~~

For an eigenpair, compute ``A @ v - value * v``. For a singular triplet,
check both ``A @ v - s * u`` and ``A.T @ u - s * v``. Checking only the first
SVD equation can miss an invalid right singular vector. [#slepc-svd]_

.. code-block:: python

   import mlx.core as mx
   import mlx_sparse as ms

   A = ms.asarray(A, dtype=mx.float32)
   values, vectors = ms.linalg.eigs(A, k=2, ncv=20)
   A_complex = ms.asarray(A, dtype=mx.complex64)
   eigen_residuals = mx.stack([
       mx.linalg.norm(A_complex @ vectors[:, j] - values[j] * vectors[:, j])
       for j in range(values.shape[0])
   ])

   U, s, Vh = ms.linalg.svds(A, k=2, ncv=20)
   left_residuals = mx.stack([
       mx.linalg.norm(A @ Vh[j, :] - s[j] * U[:, j])
       for j in range(s.shape[0])
   ])
   right_residuals = mx.stack([
       mx.linalg.norm(A.T @ U[:, j] - s[j] * Vh[j, :])
       for j in range(s.shape[0])
   ])
   mx.eval(eigen_residuals, left_residuals, right_residuals)

Sparse products require matching matrix and vector dtypes. The complex copy
above is for checking ``eigs`` outputs, not for input to the solver. For
``eigsh``, the matrix and vectors can both remain float32.

The example reports absolute norms. Set application tolerances using the
operator scale. Dividing by ``abs(value)`` or ``s`` is unsuitable near zero.
Small residuals certify approximate pairs, not that all wanted values were
found. Values-only calls cannot perform these vector checks, so request
vectors when validating accuracy or rank-sensitive results.

lanczos: a projection helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lanczos`` returns coefficients and, optionally, a basis. It has no
eigenpair convergence test. Symmetry is assumed. Unlike the higher-level
spectral routines, it does not continue after invariant breakdown.

Returned arrays retain the requested size, with zero padding after early
breakdown. The public tuple does not include the actual basis dimension.
Do not interpret padded tridiagonal entries as additional zero eigenvalues
or padded basis columns as normalized vectors. A zero coefficient can also
occur in a valid recurrence, so coefficients alone cannot identify the used
dimension. ``return_basis=False`` omits the basis from the result, without
removing all internal basis storage.

With ``reorthogonalize=False``, finite-precision loss of orthogonality can
produce repeated approximations to the same eigenvalue. Retain the default
reorthogonalization when interpreting the projection spectrally. [#templates]_

Direct and triangular solves
----------------------------

Reuse and backend selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``spsolve`` factorizes on every call. ``factorized``, ``sparse_lu``/``splu``,
and ``sparse_cholesky``/``cholesky`` reuse factors for subsequent right-hand
sides. Changing the matrix does not update those factors. There is no public
symbolic-analysis reuse, numerical refactorization, or factor update API.

``factorized(method="auto")`` chooses LU for square matrices and QR for
rectangular matrices. It does not detect SPD structure or choose Cholesky
automatically. The selected backend depends on build capabilities and runtime
availability. Inspect the returned object's ``method`` and ``backend`` when
comparing machines.

Native LU and Cholesky
~~~~~~~~~~~~~~~~~~~~~~

Native explicit factorization runs on CPU. LU uses partial row pivoting.
Cholesky uses a lower factor and does not support ``upper=True``. Neither
applies a fill-reducing ordering, equilibration, supernodal blocking, iterative
refinement, or a condition-number estimate. Sparse input does not ensure sparse
factors. Fill can approach dense storage.

These are material differences from mature sparse direct packages. SuperLU
uses sparse supernodal LU, while CHOLMOD provides fill-reducing analysis,
supernodal Cholesky, and reuse of symbolic analysis for numerical factorization.
[#superlu]_ [#cholmod]_

.. warning::

   Native LU and Cholesky use absolute float32 thresholds for pivots and
   dropping small factor entries. Even a well-conditioned matrix such as
   ``1e-8 * I`` can be rejected as singular or not positive definite.
   Small entries can also be removed during factorization. An error message
   is therefore not a scale-independent diagnosis of the matrix. Rescale
   before factorization and check the residual after solving. There is no
   public drop-threshold or pivot-tolerance control.

.. warning::

   Native Cholesky uses lower-triangle entries and supplies missing lower
   entries from the upper triangle. It does not check that mirrored entries
   agree. A nonsymmetric input can be factored as a different symmetric
   matrix and return a solution with a large residual for the original input.
   Verify symmetry before using Cholesky. If storing only one triangle, check
   the solution against the intended full symmetric operator, since sparse
   ``A @ x`` itself does not mirror the stored triangle.

Direct solves do not return ``SolverInfo`` or an accuracy certificate.
Successful factor construction and a finite solution do not replace a
residual check. Reusable factor solves do not perform iterative refinement.

spsolve_triangular: triangle and diagonal handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lower`` selects which stored off-diagonal entries participate in the solve.
Entries in the opposite triangle are ignored. ``unit_diagonal=True`` assumes
ones on the diagonal regardless of stored diagonal values. Verify that these
options describe the factor you intend to solve.

.. warning::

   A missing diagonal entry is currently treated as one even when
   ``unit_diagonal=False``. This can return a plausible finite solution to a
   different system. With a stored zero or sufficiently small diagonal,
   CPU execution raises an error, while the Metal kernel can divide by zero
   and return infinity or NaN. For a non-unit triangular solve, require an
   explicitly stored, finite, nonzero diagonal and verify the solution.
   CPU and GPU failure behavior is not interchangeable.

This applies to the triangular kernels used by native factor solve objects
as well as the public triangular solver. Matrix right-hand sides avoid a
Python loop over columns, but the Metal implementation follows row
dependencies serially within each right-hand side. A vector solve does not
distribute those dependencies across the GPU. The public ``analyzed`` argument
currently accepts only ``None`` and does not expose a level-scheduled solve.

Accelerate methods and rectangular systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accelerate support must be enabled at build time. QR, LDLT, and
``cholesky_ata`` have no native fallback. Accelerate LU additionally requires
a supporting Apple SDK and runtime, including macOS 15.5 or newer. Framework
factorization and solves run on CPU, including when Metal is the default device.
The explicit native factor APIs do not select Accelerate.

``method="ldlt"`` uses Apple's default symmetric indefinite factorization.
The current SDK defines that default as threshold partial pivoting, with
possible additional fill. The wrapper does not expose a pivoting strategy,
ordering choice, or numeric tolerance controls. LDLT is a direct linear solve,
without a minimum-norm least-squares contract for singular systems.
[#apple-direct]_

For a full-rank rectangular matrix, QR handles overdetermined least squares
and underdetermined minimum-norm solves. The wrapper does not expose a rank
estimate, a user rank threshold, or a truncated pseudoinverse. Do not assume
SVD-style rank-deficient behavior. [#apple-qr]_

.. warning::

   ``factorized(A, method="cholesky_ata")`` solves
   ``(A.T @ A) @ x = c``. For an ``m`` by ``n`` matrix, it expects a
   right-hand side of length ``n``, not the original length-``m`` observation
   vector ``b``. It does not compute ``c = A.T @ b`` for you. A square input
   can hide this difference because both shapes are the same. Use QR for
   the direct least-squares problem ``min ||A @ x - b||``.

Apple's CholeskyAtA factorization obtains the QR factorization without storing
``Q``. It does not require explicitly assembling the normal matrix. The solve
still has a normal-equation contract, with squared condition number for a
full-column-rank matrix. It requires ``m >= n`` and suitable column rank.
[#apple-ata]_

Symmetric Accelerate factorizations interpret the supplied lower triangle as
symmetric data. They do not validate agreement with the upper triangle.
The one-shot Accelerate ``spsolve`` path includes a residual sanity check.
It is not a condition or rank certificate, and it is not applied to every
``FactorizedSolve.solve`` call.

GPU execution and scaling
-------------------------

Selecting Metal does not make every stage a GPU calculation. Factorization
runs on CPU. Spectral methods use dense projected solves on CPU. ``eigsh``
and unpreconditioned native GMRES also read the full Krylov basis on CPU for
reconstruction. ``eigs`` and ``svds`` reconstruct requested vectors on the
selected device.

Using a dense projected problem is standard in sparse Krylov methods. Its
size is controlled by the basis dimension rather than by the full matrix
dimension. [#slepc-eps]_ On Apple Silicon, MLX shares memory between CPU and
GPU, which avoids a separate device-memory copy requirement. Host reads
still wait for GPU producers, so synchronization and CPU work remain real
costs. [#mlx-memory]_

Several fused Metal recurrences, including CG, MINRES, BiCGSTAB, Lanczos, and
Arnoldi, use one cooperative threadgroup for an entire solve or basis build.
This reduces host interaction inside those loops, but limits parallelism for
large matrices. The large-matrix SVD path distributes sparse products and
orthogonalization across GPU work, with host synchronization for scalar norm
and breakdown decisions between directions. No path has a general guarantee
of being faster than CPU.

Returning an integer iterative status requires evaluating the native status
array. Solver calls therefore synchronize before returning even when
``return_info=False``. User callbacks and structured diagnostics can add
host reads. They should not be treated as fully lazy sparse array operations.

There are no distributed solvers, multi-GPU Krylov methods, or CUDA/ROCm
backends. Solver and factorization outputs also have no supported autodiff
contract. Differentiable sparse products do not imply differentiable solves.
See :doc:`autodiff` for the supported operation surface.

When assessing performance, include factor or preconditioner setup unless
it will be amortized over unchanged matrices. Compare elapsed time at the same
verified residual and precision. Fixed iteration or ``ncv`` timings can compare
work while returning very different accuracy.

References
----------

.. [#templates] R. Barrett et al. *Templates for the Solution of Linear
   Systems: Building Blocks for Iterative Methods*, 2nd edition, SIAM, 1994.
   `Book and algorithms <https://netlib.org/templates/templates.pdf>`_.

.. [#scipy-gmres] SciPy. `GMRES API and callback semantics
   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html>`_.

.. [#petsc-cg] PETSc. `CG operator and preconditioner requirements
   <https://petsc.org/release/manualpages/KSP/KSPCG/>`_.

.. [#fgmres] PETSc. `Flexible GMRES and preconditioner requirements
   <https://petsc.org/release/manualpages/KSP/KSPFGMRES/>`_.

.. [#minresqlp] Stanford Systems Optimization Laboratory.
   `MINRES-QLP: sparse symmetric equations and least squares
   <https://web.stanford.edu/group/SOL/software/minresqlp/>`_.

.. [#lsmr] D. C. L. Fong and M. A. Saunders.
   `LSMR: sparse equations and least squares
   <https://web.stanford.edu/group/SOL/software/lsmr/>`_.

.. [#chebyshev] PETSc. `Chebyshev iteration and spectral estimation
   <https://petsc.org/release/manualpages/KSP/KSPCHEBYSHEV/>`_.

.. [#slepc-eps] SLEPc. `Eigenvalue solvers, transformations, and error checks
   <https://slepc.upv.es/release/documentation/manual/eps.html>`_.

.. [#slepc-svd] SLEPc. `SVD methods and singular-triplet residuals
   <https://slepc.upv.es/release/documentation/manual/svd.html>`_.

.. [#propack] R. M. Larsen. *Lanczos bidiagonalization with partial
   reorthogonalization*, DAIMI PB-537, 1998.
   `PROPACK documentation <https://rmlarsen.github.io/propack/>`_.

.. [#primme] L. Wu, E. Romero, and A. Stathopoulos.
   `PRIMME_SVDS: A High-Performance Preconditioned SVD Solver for Accurate
   Large-Scale Computations <https://arxiv.org/abs/1607.01404>`_, 2017.

.. [#superlu] SuperLU developers. `SuperLU documentation and user guide
   <https://portal.nersc.gov/project/sparse/superlu/>`_.

.. [#cholmod] T. A. Davis et al. `CHOLMOD User Guide
   <https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/CHOLMOD/Doc/CHOLMOD_UserGuide.tex>`_.

.. [#apple-qr] Apple. `SparseFactorizationQR
   <https://developer.apple.com/documentation/accelerate/sparsefactorizationqr>`_.

.. [#apple-direct] Apple. `Sparse factorization types
   <https://developer.apple.com/documentation/accelerate/sparsefactorization_t>`_
   and the pivoting definitions in the Accelerate ``Sparse/Solve.h`` SDK header.

.. [#apple-ata] Apple. `SparseFactorizationCholeskyAtA
   <https://developer.apple.com/documentation/accelerate/sparsefactorizationcholeskyata>`_
   and its factorization and solve contract in the Accelerate ``Sparse/Solve.h``
   SDK header.

.. [#mlx-memory] MLX. `Unified memory and CPU/GPU execution
   <https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html>`_.
