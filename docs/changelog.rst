Changelog
=========

mlx-sparse v0.0.5b0 (Unreleased)
--------------------------------

.. note::

    This release focuses on making sparse iterative solves materially more
    robust and complete by adding first-class native-backed preconditioners
    under :mod:`mlx_sparse.linalg.preconditioners`, wiring them into the Krylov
    solver surface, and tightening true-residual and numerical-breakdown
    behavior across the iterative linalg stack. See the roadmap
    `here <https://github.com/waleed-sh/mlx-sparse/issues/26>`_.

New Features
~~~~~~~~~~~~

* Added the public :mod:`mlx_sparse.linalg.preconditioners` namespace with
  identity, diagonal, and Jacobi preconditioner objects. Diagonal application
  uses a native CPU/Metal primitive, and ``cg(..., M=jacobi(A))`` dispatches to
  a native Jacobi-preconditioned CG primitive rather than a Python solver loop
  (`PR #27 <https://github.com/waleed-sh/mlx-sparse/pull/27>`_).

* Added native ``csr_pcg_jacobi`` and ``diagonal_preconditioner_apply`` entry
  points under ``src/preconditioners/pcg`` and
  ``src/preconditioners/diagonal``, with Python bindings through
  :mod:`mlx_sparse._native`
  (`PR #27 <https://github.com/waleed-sh/mlx-sparse/pull/27>`_).

* Added left-preconditioned GMRES support for ``identity``,
  ``diagonal``/``jacobi``, and explicit inverse-apply callables or objects.
  Diagonal/Jacobi GMRES dispatches to native C++ with CPU/Metal
  preconditioned Arnoldi kernels under ``src/preconditioners/gmres``, custom
  callables use a documented host fallback
  (`PR #30 <https://github.com/waleed-sh/mlx-sparse/pull/30>`_).

* Added exact-factor preconditioners through ``from_factorized(solver)`` and
  ``exact(A, method="auto")``. The wrappers compose with ``FactorizedSolve``,
  ``SparseLU``, and ``SparseCholesky`` without refactorizing on application,
  preserve native or Accelerate backend metadata, and support rank-1/rank-2
  inverse application where the underlying solver supports it. Explicit native
  LU/Cholesky factors now use native exact-apply bindings, and
  ``gmres(..., M=exact(...))`` routes through typed native exact-factor GMRES
  entrypoints instead of the Python host fallback
  (`PR #31 <https://github.com/waleed-sh/mlx-sparse/pull/31>`_).

* Added native shifted MINRES support and native diagonal/Jacobi-preconditioned
  MINRES under ``src/preconditioners/minres``. ``minres(..., shift=s)`` now
  solves ``(A - s I) x = b``, and ``minres(..., M=jacobi(A, check=True))``
  accepts only symmetric positive-definite diagonal preconditioners by default
  (`PR #32 <https://github.com/waleed-sh/mlx-sparse/pull/32>`_).

* Added native ILU(0) preconditioners through
  ``preconditioners.ilu0(A, shift=0.0, check=True, reuse_analysis=False)``.
  Setup is natural-order, no-fill, CPU-native C++ over canonical CSR input,
  application reuses native CSR triangular solves for rank-1/rank-2 right-hand
  sides on CPU or Metal. ``gmres(..., M=ilu0(A))`` routes through a native
  left-preconditioned GMRES entrypoint
  (`PR #33 <https://github.com/waleed-sh/mlx-sparse/pull/33>`_).

* Added native IC(0) preconditioners through
  ``preconditioners.ichol0(A, shift=0.0, check=True)``. Setup is
  natural-order, no-fill, CPU-native C++ over canonical CSR input, application
  reuses native CSR triangular solves for rank-1/rank-2 right-hand sides on
  CPU or Metal. ``cg(..., M=ichol0(A))`` routes through a native
  IC(0)-preconditioned CG entrypoint
  (`PR #34 <https://github.com/waleed-sh/mlx-sparse/pull/34>`_).

* Added native Chebyshev polynomial preconditioners through
  ``preconditioners.chebyshev(A, degree=2, lambda_min=None,
  lambda_max=None, estimate=True)``. Setup computes native Gershgorin bounds
  and optional Lanczos Ritz estimates over canonical CSR input, application and
  ``cg(..., M=chebyshev(A))`` use native CPU/Metal kernels with only sparse
  matrix-vector products and vector updates
  (`PR #35 <https://github.com/waleed-sh/mlx-sparse/pull/35>`_).

* Added optional ``return_info=True`` diagnostics for ``cg``, ``gmres``, and
  ``minres``. The default return remains ``(x, int)``, while the opt-in path
  returns a structured ``SolverInfo`` with status, final true residual norm,
  iteration count, convergence reason, breakdown reason, and preconditioner
  metadata where applicable
  (`PR #36 <https://github.com/waleed-sh/mlx-sparse/pull/36>`_).

* Added public ``linalg.spsolve_triangular(A, b, lower=True,
  unit_diagonal=False, analyzed=None)`` around the native CSR triangular-solve
  primitive for rank-1 and rank-2 right-hand sides. Public triangular analysis
  remains deferred because repeated-apply benchmarks do not yet show a
  consistent advantage over the default native solve path
  (`PR #37 <https://github.com/waleed-sh/mlx-sparse/pull/37>`_).

* Added user start-vector support for ``lanczos``, ``eigsh``, ``eigs``, and
  ``svds``. ``eigsh`` and ``eigs`` now thread ``v0`` through the native
  Lanczos/Arnoldi projection calls, and ``svds`` threads ``v0`` through the
  native CPU/Metal normal-operator Lanczos primitive
  (`PR #37 <https://github.com/waleed-sh/mlx-sparse/pull/37>`_).

* Added bounded matrix-free host fallback loops for ``cg`` and ``gmres`` when
  ``A`` is a fully matrix-free ``LinearOperator``. Sparse-backed
  ``LinearOperator`` inputs continue to use the native CSR CPU/Metal paths,
  while matrix-free fallbacks accept arbitrary inverse-apply ``M`` objects or
  callables and check convergence against the true residual
  (`PR #37 <https://github.com/waleed-sh/mlx-sparse/pull/37>`_).

Improvements
~~~~~~~~~~~~

* Normalized iterative-solver status handling for zero-iteration budgets and
  numerical breakdowns. Native CG, PCG, GMRES, and preconditioned GMRES paths
  now avoid reporting ``info == 0`` for an unconverged solve when
  ``maxiter=0``
  (`PR #36 <https://github.com/waleed-sh/mlx-sparse/pull/36>`_).

* Added opt-in final callbacks for native ``cg``, ``gmres``, and ``minres``
  without adding Python calls inside CPU/Metal Krylov loops. GMRES
  ``callback_type="x"`` receives the final solution, while ``"pr_norm"`` and
  ``"legacy"`` receive the final reported residual norm
  (`PR #38 <https://github.com/waleed-sh/mlx-sparse/pull/38>`_).

* Added scale-aware denominator and finite-value checks to the native
  unpreconditioned CG CPU/Metal paths, matching the robustness policy already
  used by the preconditioned CG implementations
  (`PR #36 <https://github.com/waleed-sh/mlx-sparse/pull/36>`_).

* Replaced GMRES' projected normal-equation solve with a native
  upper-Hessenberg QR solve using Givens rotations, avoiding the condition
  number squaring of ``H.T @ H`` during restarted GMRES updates while keeping
  the existing CPU/Metal Arnoldi dispatch
  (`PR #29 <https://github.com/waleed-sh/mlx-sparse/pull/29>`_).

* Tightened GMRES status handling so a solve that reaches the true residual
  tolerance on the final allowed restart reports success instead of returning
  the iteration budget as ``info``
  (`PR #29 <https://github.com/waleed-sh/mlx-sparse/pull/29>`_).

* Replaced the native MINRES Lanczos-projection plus normal-equation
  least-squares path with a Paige-Saunders-style streaming recurrence. The new
  path keeps constant Krylov-vector storage, checks the true shifted residual
  before reporting success, and has matching native CPU and Metal kernels
  (`PR #32 <https://github.com/waleed-sh/mlx-sparse/pull/32>`_).

* Added true-residual convergence checks and finite inverse-diagonal handling
  to native Jacobi-preconditioned GMRES. The preconditioned basis is built for
  ``M^{-1} A``, while success is still determined from ``b - A @ x``
  (`PR #30 <https://github.com/waleed-sh/mlx-sparse/pull/30>`_).

* Extended preconditioner normalization so reusable sparse direct-solve objects
  are recognized as exact inverse-apply preconditioners instead of generic
  Python callables
  (`PR #31 <https://github.com/waleed-sh/mlx-sparse/pull/31>`_).

* Added guarded Accelerate exact-factor GMRES routing for real Accelerate
  factorized objects, preserving the optional Apple CPU sparse-solver boundary
  when the package is built with Accelerate support and falling back to native
  explicit factors otherwise
  (`PR #31 <https://github.com/waleed-sh/mlx-sparse/pull/31>`_).

* Hardened diagonal and Jacobi preconditioner validation with explicit finite
  setup checks, finite RHS checks for standalone inverse application,
  conservative positive-definite metadata, and an opt-in ``check=True`` path
  for positive shifted Jacobi diagonals
  (`PR #28 <https://github.com/waleed-sh/mlx-sparse/pull/28>`_).

* Hardened ILU(0) setup with explicit failures for rectangular systems, missing
  diagonal structure, non-finite input or shift values, unsupported dtypes, and
  zero or near-zero pivots. Diagonal shifts are explicit and are applied only to
  existing diagonal entries
  (`PR #33 <https://github.com/waleed-sh/mlx-sparse/pull/33>`_).

* Kept ILU(0) triangular analysis caching opt-in through ``reuse_analysis``
  after focused CPU/GPU apply benchmarks showed mixed benefits across RHS rank
  and device (`PR #33 <https://github.com/waleed-sh/mlx-sparse/pull/33>`_).

* Hardened IC(0) setup with explicit failures for rectangular systems, missing
  diagonal structure, non-symmetric mirrored values when ``check=True``,
  non-finite input or shift values, unsupported dtypes, negative shifts, and
  non-positive or near-zero pivots. Diagonal shifts are explicit and are
  applied only to existing diagonal entries
  (`PR #34 <https://github.com/waleed-sh/mlx-sparse/pull/34>`_).

* Added Chebyshev spectral-interval validation with clear failures for missing
  positive lower/upper bounds, invalid explicit intervals, non-positive
  degrees, non-finite CSR values, and unsupported dtypes. Gershgorin lower
  bounds are used when positive, otherwise the default setup uses native
  Lanczos estimates to obtain a conservative positive lower bound for
  Poisson-like SPD systems
  (`PR #35 <https://github.com/waleed-sh/mlx-sparse/pull/35>`_).

* Refactored shared sparse linalg helpers into the
  :mod:`mlx_sparse.linalg.utils` subpackage, separating array/dtype handling,
  sparse input normalization, iterative solver bookkeeping, direct-solver
  validation, spectral sizing, operator construction, and preconditioner
  validation from the public algorithm modules
  (`PR #28 <https://github.com/waleed-sh/mlx-sparse/pull/28>`_).

* Centralized spectral iteration-control validation so non-default ``tol`` and
  ``maxiter`` for ``eigsh``, ``eigs``, and ``svds`` now fail with an explicit
  explanation that the current native routines perform one ``ncv``-bounded
  Ritz extraction rather than an implicitly restarted convergence loop
  (`PR #37 <https://github.com/waleed-sh/mlx-sparse/pull/37>`_).

* Clarified native iterative-solver callback semantics for preconditioned paths:
  callbacks remain opt-in final callbacks, GMRES ``callback_type`` payload names
  mirror SciPy's ``"x"``, ``"pr_norm"``, and ``"legacy"`` vocabulary where the
  payload is available without per-iteration Python synchronization, and
  ``"legacy"`` keeps mlx-sparse's normal ``maxiter`` accounting
  (`PR #38 <https://github.com/waleed-sh/mlx-sparse/pull/38>`_).

* Added Linux CPU-only build, CI, and publishing support. Linux builds use the
  native CPU backend, leave Apple-only Metal and Accelerate paths unavailable,
  and continue to report CUDA and ROCm as reserved future capabilities. The
  native extension now builds as C++20 to match current MLX Linux headers.
  Source distributions now prune generated benchmark artifacts and local
  matplotlib cache files while retaining benchmark source files.

Tests
~~~~~

* Added GMRES robustness tests for final-restart true-residual success,
  nonsymmetric diagonal-dominant systems against SciPy and dense NumPy solves,
  convection-diffusion-like systems against SciPy, and the remaining
  Hilbert-like float32 limitation with a bounded-residual assertion
  (`PR #29 <https://github.com/waleed-sh/mlx-sparse/pull/29>`_).

* Expanded preconditioner coverage for rank-1/rank-2 RHS application,
  low-precision RHS promotion, CSR/COO/CSC Jacobi setup, input immutability,
  SciPy PCG comparison, dense NumPy residual checks, and pathological
  near-singular or singular PCG behavior
  (`PR #28 <https://github.com/waleed-sh/mlx-sparse/pull/28>`_).

* Added GMRES preconditioner tests covering native identity dispatch,
  native Jacobi/diagonal left-preconditioned GMRES against SciPy, direct native
  true-residual reporting, non-finite inverse-diagonal breakdown handling,
  callable inverse-apply host fallback, and sparse-matrix ``M`` rejection
  (`PR #30 <https://github.com/waleed-sh/mlx-sparse/pull/30>`_).

* Expanded preconditioner normalization and callback coverage for ``None``,
  existing preconditioner objects, ``SparseLU``, ``SparseCholesky``,
  ``FactorizedSolve``, callable inverse-apply functions, objects with
  ``solve(x)``, native Jacobi-PCG exit callbacks, preconditioned GMRES callback
  payloads, and CSR/COO/CSC Jacobi-GMRES correctness on CPU/GPU-capable paths
  (`PR #38 <https://github.com/waleed-sh/mlx-sparse/pull/38>`_).

* Added exact-factor preconditioner tests covering native LU and Cholesky
  metadata, native exact-apply bindings against dense NumPy solves, rank-1/rank-2
  application, ``FactorizedSolve`` backend metadata preservation, ``exact(A)``
  convenience construction, rectangular factorization rejection, non-finite
  output rejection, and GMRES composition with assertions that exact LU/Cholesky
  do not use the Python host fallback
  (`PR #31 <https://github.com/waleed-sh/mlx-sparse/pull/31>`_).

* Added MINRES recurrence tests for symmetric indefinite systems, singular
  compatible systems, shifted solves against SciPy's convention,
  Jacobi-preconditioned indefinite systems against SciPy ``LinearOperator``
  preconditioning, near-singular diagonal preconditioning, and strict rejection
  of non-SPD or non-native MINRES preconditioners
  (`PR #32 <https://github.com/waleed-sh/mlx-sparse/pull/32>`_).

* Added ILU(0) tests against an internal natural-order ILU(0) reference,
  dense NumPy triangular-solve results, SciPy ``spilu`` quality comparisons,
  native rank-1/rank-2 apply paths on CPU and GPU-capable devices, explicit
  failure modes, input immutability, and GMRES iteration reduction on
  nonsymmetric diagonal-dominant and convection-diffusion-like systems
  (`PR #33 <https://github.com/waleed-sh/mlx-sparse/pull/33>`_).

* Added IC(0) tests against an internal natural-order no-fill reference,
  dense NumPy triangular-solve results, native rank-1/rank-2 apply paths on CPU
  and GPU-capable devices, explicit failure modes, input immutability, and PCG
  coverage for 2-D Poisson, anisotropic diffusion, scaled diagonal, and
  shifted near-singular SPD systems
  (`PR #34 <https://github.com/waleed-sh/mlx-sparse/pull/34>`_).

* Added Chebyshev tests against dense NumPy polynomial-apply references,
  native CPU and Metal rank-1/rank-2 apply paths, spectral-estimate metadata,
  explicit invalid-interval failures, public ``cg(..., M=chebyshev(A))``
  routing, and native Chebyshev-PCG iteration reduction against Jacobi on
  Poisson-like SPD systems
  (`PR #35 <https://github.com/waleed-sh/mlx-sparse/pull/35>`_).

* Added linalg completeness tests for public sparse triangular solves against
  dense NumPy references, matrix-free ``LinearOperator`` CG/GMRES with
  callable inverse-apply preconditioners, and spectral ``v0`` threading through
  Lanczos, Arnoldi, and normal-operator Lanczos paths
  (`PR #37 <https://github.com/waleed-sh/mlx-sparse/pull/37>`_).

* Added validation-matrix benchmark tests covering the v0.0.5b0
  preconditioner matrix-family builders, SciPy Jacobi ``LinearOperator``
  contract, report schema, runtime metadata, residual thresholds, and summary
  aggregation
  (`PR #39 <https://github.com/waleed-sh/mlx-sparse/pull/39>`_).

Benchmarks
~~~~~~~~~~

* Added ``benchmarks/bench_jacobi_pcg_validation.py`` for Jacobi PCG validation
  with setup time, solve time, true residual, matrix metadata, preconditioner
  metadata, and SciPy CG/Jacobi reference measurements
  (`PR #28 <https://github.com/waleed-sh/mlx-sparse/pull/28>`_).

* Added focused native Jacobi-GMRES benchmark coverage comparing
  preconditioned GMRES against the existing unpreconditioned solver benchmark
  (`PR #30 <https://github.com/waleed-sh/mlx-sparse/pull/30>`_).

* Added focused MINRES recurrence benchmark coverage around the shifted
  Paige-Saunders implementation using the existing linalg solver benchmark
  (`PR #32 <https://github.com/waleed-sh/mlx-sparse/pull/32>`_).

* Added ``benchmarks/bench_ilu0_preconditioner.py`` for ILU(0) setup,
  rank-1/rank-2 apply, analyzed-apply timing, GMRES iteration counts, true
  residuals, fill ratio, CPU/GPU device metadata, and SciPy ``spilu``
  comparison context
  (`PR #33 <https://github.com/waleed-sh/mlx-sparse/pull/33>`_).

* Added ``benchmarks/bench_gmres_suitesparse_normal_equations.py`` for GMRES
  non-regression coverage on the SuiteSparse ``well1033`` and ``illc1033``
  normal-equation fixtures, reporting setup metadata, solve timing, iteration
  count, status, final reported residual, true relative residual, and SciPy
  GMRES comparison context
  (`PR #29 <https://github.com/waleed-sh/mlx-sparse/pull/29>`_).

* Added ``benchmarks/bench_ic0_preconditioner.py`` for IC(0) setup,
  rank-1/rank-2 apply, PCG iteration counts, true residuals, fill ratio, and
  CPU/GPU device metadata against Jacobi and unpreconditioned CG baselines
  (`PR #34 <https://github.com/waleed-sh/mlx-sparse/pull/34>`_).

* Added ``benchmarks/bench_chebyshev_preconditioner.py`` for Chebyshev setup,
  rank-1/rank-2 apply, PCG iteration counts, true residuals, spectral interval
  metadata, CPU/GPU device metadata, and comparison against Jacobi and
  unpreconditioned CG baselines on Poisson-like SPD systems
  (`PR #35 <https://github.com/waleed-sh/mlx-sparse/pull/35>`_).

* Added ``benchmarks/bench_preconditioner_validation_matrix.py`` as the
  release validation matrix for 1-D/2-D/3-D Poisson SPD, anisotropic diffusion
  SPD, badly scaled diagonal and block-diagonal SPD, nonsymmetric
  convection-diffusion-like systems, random sparse diagonal-dominant systems,
  Hilbert-like dense-ish CSR systems, and bundled SuiteSparse normal-equation
  fixtures. Each record names matrix metadata, solver, preconditioner, setup
  time, apply time, solve time, iteration count, final true residual,
  preconditioned residual when available, fill metrics, runtime device/thread
  settings, SciPy comparison context, and residual/iteration threshold status
  (`PR #39 <https://github.com/waleed-sh/mlx-sparse/pull/39>`_).

Documentation
~~~~~~~~~~~~~

* Added a Preconditioners user-guide page and updated linalg solver docs to
  describe the current native CG and GMRES preconditioner support,
  exact-factor wrappers, callable GMRES host fallback behavior, native
  exact-factor GMRES routing, native shifted/diagonal-preconditioned MINRES,
  native ILU(0) GMRES routing, native IC(0) PCG routing, native Chebyshev PCG
  routing, CPU/Metal and Accelerate boundaries, and the remaining
  incomplete-factor preconditioner gaps
  (`PR #27 <https://github.com/waleed-sh/mlx-sparse/pull/27>`_,
  `PR #28 <https://github.com/waleed-sh/mlx-sparse/pull/28>`_,
  `PR #30 <https://github.com/waleed-sh/mlx-sparse/pull/30>`_,
  `PR #31 <https://github.com/waleed-sh/mlx-sparse/pull/31>`_,
  `PR #32 <https://github.com/waleed-sh/mlx-sparse/pull/32>`_,
  `PR #33 <https://github.com/waleed-sh/mlx-sparse/pull/33>`_,
  `PR #34 <https://github.com/waleed-sh/mlx-sparse/pull/34>`_,
  `PR #35 <https://github.com/waleed-sh/mlx-sparse/pull/35>`_).

* Updated linalg docs and API references for ``spsolve_triangular``,
  matrix-free ``LinearOperator`` CG/GMRES fallbacks, spectral ``v0`` support,
  and the current ``tol``/``maxiter`` limitation for one-shot
  ``ncv``-bounded Ritz extraction
  (`PR #37 <https://github.com/waleed-sh/mlx-sparse/pull/37>`_).

* Added a separate *Sparse preconditioners* notebook track with dense,
  self-contained notebooks for identity, diagonal, Jacobi, ILU(0), IC(0),
  Chebyshev, exact-factor, and custom callable preconditioners. The notebooks
  show setup/apply boundaries, solver diagnostics, expected benefits, and
  tradeoffs separately from the general linalg solver notebook
  (`PR #39 <https://github.com/waleed-sh/mlx-sparse/pull/39>`_).

mlx-sparse v0.0.4b1 (31.05.2026)
----------------------------------

.. note::

    Unlike the previous release which targetted Accelerate integration, this release targeted
    native CPU performance when Metal and Accelerate are unavailable, disabled, or intentionally avoided.
    See the roadmap `here <https://github.com/waleed-sh/mlx-sparse/issues/13>`_.

New Features
~~~~~~~~~~~~~~~~~~~~~

* Added enum-backed :mod:`mlx_sparse.runtime` controls for package-wide CPU
  worker settings, per-family SpGEMM/solver worker overrides, and separate
  SpGEMM/solver parallel gates, including direct attribute reads/writes such as
  ``ms.runtime.N_THREADS = 8``, temporary context-manager overrides, and
  structured ``runtime.info()`` diagnostics for performance tooling
  (`PR #15 <https://github.com/waleed-sh/mlx-sparse/pull/15>`_).

* Added documented runtime configuration variables for ``CPU_THREADS``,
  ``SPGEMM_PARALLEL``, ``SPGEMM_THREADS``, ``SOLVER_PARALLEL``, and
  ``SOLVER_THREADS`` alongside the existing experimental Metal SpGEMM flag
  (`PR #15 <https://github.com/waleed-sh/mlx-sparse/pull/15>`_).

Benchmarks
~~~~~~~~~~

* Added shared benchmark helpers that evaluate dense MLX results and force all
  structural buffers for sparse containers before timing
  (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Added a native CPU direct-solver benchmark that records runtime capability,
  device, hardware, worker-count, matrix-structure, dtype, warmup, and
  iteration metadata for reproducible non-Accelerate baselines
  (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Split native LU and Cholesky direct-solver timings into ``factor_only``,
  ``solve_only``, and ``factor_plus_solve`` phases
  (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Added native CPU sparse-operation benchmark suites for ``fromdense``,
  compressed ``sort_indices`` and ``sum_duplicates``, COO-to-CSR/CSC
  conversion, CSR/COO/CSC sparse-sparse products, CSR/CSC transpose products,
  and COO/CSC dense products
  (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Added benchmark matrix families for uniformly short rows, highly imbalanced
  rows, banded matrices, diagonal-dominant matrices, duplicate-heavy COO and
  compressed inputs, exact-cancellation SpGEMM, and output-density sweeps
  (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Expanded native CPU benchmarks to sweep matrix dimensions, target nonzeros
  per row, and short-row occupancies, with a hard 32k maximum dimension guard,
  dense materialization limits for ``fromdense`` cases, and explicit-density
  compatibility flags for ad-hoc runs
  (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Added SciPy reference timings to every benchmark entrypoint, with
  speedup-versus-SciPy fields in machine-readable reports and text summaries.
  Native Cholesky records explicitly mark SciPy sparse Cholesky as unavailable
  instead of using a misleading substitute, while LU records compare against
  SciPy SuperLU (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Added structured before/after report support with loose local-regression
  comparison thresholds for optimized native CPU families
  (`PR #14 <https://github.com/waleed-sh/mlx-sparse/pull/14>`_).

* Expanded the reductions benchmark so standalone CPU runs honor
  ``MLX_SPARSE_TEST_DEVICE=cpu`` and include CSR row reductions plus CSR/CSC
  dense-conversion timings alongside SciPy references
  (`PR #18 <https://github.com/waleed-sh/mlx-sparse/pull/18>`_).

Improvements
~~~~~~~~~~~~

* Added fixed-worker native CPU parallel SpGEMM implementations for same-format
  CSR, COO, and CSC sparse-sparse products. The host paths split independent
  output rows or columns across the configured ``SPGEMM_THREADS`` workers,
  reuse private per-worker accumulator workspaces, stitch results
  deterministically, and preserve canonical ordering plus exact
  zero-cancellation semantics. Setting ``SPGEMM_THREADS=1`` or disabling
  ``SPGEMM_PARALLEL`` keeps the serial Gustavson/SPA path
  (`PR #15 <https://github.com/waleed-sh/mlx-sparse/pull/15>`_).

* Improved serial host CSR/COO/CSC SpGEMM assembly by writing only final
  nonzero entries after each row or column accumulation, avoiding the previous
  candidate-value materialization and separate prune pass on the native CPU
  host path while preserving canonical ordering and exact zero-cancellation
  semantics (`PR #15 <https://github.com/waleed-sh/mlx-sparse/pull/15>`_).

* Improved the serial host CSR/COO/CSC SpGEMM hot path by removing the default
  symbolic upper-bound pass, initializing newly touched accumulator slots
  directly with the first product, using insertion sort for tiny touched lists,
  and adaptively scanning dense markers for disordered high-output rows to
  reduce memory traffic and canonical-output sorting overhead
  (`PR #15 <https://github.com/waleed-sh/mlx-sparse/pull/15>`_).

* Improved native CPU CSR sparse-dense products. ``csr_matvec`` now has a
  short-row serial path, ``csr_matmul`` and ``csr_batched_matmul`` specialize
  common RHS widths with stack accumulators, and row-owned CSR matvec/matmul
  batch-row work can use the fixed package-wide ``CPU_THREADS`` worker count
  (`PR #16 <https://github.com/waleed-sh/mlx-sparse/pull/16>`_).

* Improved native CPU COO/CSC batched sparse-dense products. Batched COO and
  CSC matmul now use fixed-worker batch-owned CPU partitions when
  ``CPU_THREADS`` resolves above one, while non-batched COO/CSC forward dense
  products remain serial because they scatter into shared dense output rows.
  The native CPU sparse-operation benchmark suite now records COO/CSC batched
  matvec and matmul timings with SciPy references
  (`PR #17 <https://github.com/waleed-sh/mlx-sparse/pull/17>`_).

* Added fixed-worker native CPU partitions for additional race-free
  disjoint-output kernels: CSR row sums/norms/diagonal/dense conversion, CSC
  column sums/norms/diagonal/dense conversion, staged ``fromdense`` row
  count/fill, compressed CSR/CSC ``sort_indices`` and ``sum_duplicates``, and
  CSR/COO/CSC sparse-value VJP kernels.  The serial path remains the
  ``CPU_THREADS=1`` regression target
  (`PR #18 <https://github.com/waleed-sh/mlx-sparse/pull/18>`_).

* Reworked CPU CSR-to-CSC and CSC-to-CSR conversion fills to use private
  per-worker histograms and per-worker write offsets when more than one CPU
  worker is configured.  The implementation avoids shared mutable
  ``next`` counters and preserves deterministic compressed output ordering
  (`PR #18 <https://github.com/waleed-sh/mlx-sparse/pull/18>`_).

* Reworked the remaining native CPU compressed transpose/conversion and
  scatter-style reduction/product paths. COO-to-CSR/CSC conversion and CSR
  transpose now use histogram-prefix-scatter style assembly with private
  worker write offsets, CSC transpose products use output-column ownership,
  CSR transpose products and axis-mismatched reductions use deterministic
  private accumulators, and non-batched COO/CSC forward dense products remain
  measured serial fallbacks rather than introducing unsynchronized scatter
  writes (`PR #19 <https://github.com/waleed-sh/mlx-sparse/pull/19>`_).

* Added a CPU-only immediate host assembly fast path for ``fromdense``. CPU
  streams now scan dense rows directly into canonical CSR buffers, while GPU
  streams keep the staged count/prefix/fill implementation. Immediate host
  assembly was also measured for compressed ``sum_duplicates`` and deliberately
  not adopted because it did not improve over the existing staged CPU path
  (`PR #20 <https://github.com/waleed-sh/mlx-sparse/pull/20>`_).

* Added native rank-2 RHS support for explicit-factor CPU solves.  Native
  Cholesky and LU reuse now apply CSR triangular solves to dense RHS matrices
  in one native call sequence, LU row permutation accepts matrix RHS inputs,
  and solver CPU parallelism is separated behind ``MLX_SPARSE_SOLVER_PARALLEL``
  / ``MLX_SPARSE_SOLVER_THREADS``
  (`PR #21 <https://github.com/waleed-sh/mlx-sparse/pull/21>`_).

* Added native CSR triangular-solve structural analysis primitives for
  diagonal-position lookup and dependency-level schedules, plus an analyzed
  solve path used by the benchmark suite.  The analysis path is guarded by
  graph structure and falls back to row-order solves when no useful level
  parallelism exists.  Production explicit-factor solves keep the measured
  row-order path because cached diagonal positions and fixed-worker level
  scheduling did not improve the single-thread regression target.  Cholesky
  solves cache the transposed upper factor without changing the public factor
  dataclass constructor
  (`PR #22 <https://github.com/waleed-sh/mlx-sparse/pull/22>`_).

* Improved native CPU explicit-factor construction for the portable direct
  solver fallback.  Cholesky now uses sorted sparse rows plus reusable dense
  marker/work arrays instead of map-heavy working storage, while LU uses sorted
  sparse row vectors to reduce per-pivot map churn without changing partial
  pivoting semantics, factor structure, natural ordering, or error behavior.
  The direct-solver benchmark now reports input import/canonicalization
  separately and records that native factorization fuses symbolic construction,
  numeric updates, and CSR materialization
  (`PR #23 <https://github.com/waleed-sh/mlx-sparse/pull/23>`_).

* Added thresholded deterministic tree reductions for native CPU scalar
  reductions.  CSR/COO/CSC trace and CSR sparse ``dot``/``vdot`` keep the
  serial path for small inputs, then split larger inputs into configured
  ``CPU_THREADS`` row/nonzero ranges with worker-local partials and a stable
  final tree reduction.  The implementation keeps low-precision and complex
  accumulator semantics intact and does not introduce architecture-specific
  SIMD intrinsics or dense temporaries for these paths
  (`PR #24 <https://github.com/waleed-sh/mlx-sparse/pull/24>`_).

Documentation
~~~~~~~~~~~~~

* Added a dedicated ``Parallelism and performance`` documentation section with
  a detailed native CPU execution model, runtime and environment control
  examples, fair sparse benchmarking guidance, and local v0.0.4b1 performance
  figures comparing the optimized native CPU paths against the v0.0.4b0-style
  baseline and SciPy (`PR #25 <https://github.com/waleed-sh/mlx-sparse/pull/25>`_).

* Clarified the remaining serial or dependency-bound native CPU paths in the
  release notes and performance docs.  Natural-order Cholesky/LU
  factorization, production row-order triangular solves, non-batched COO/CSC
  scatter dense products, small scalar reductions, and rejected
  iterative/spectral helper parallelism are not described as broadly parallel
  native backend coverage (`PR #25 <https://github.com/waleed-sh/mlx-sparse/pull/25>`_).



mlx-sparse v0.0.4b0 (28.05.2026)
----------------------------------

.. note::

    This release focused on improving the performance of the existing sparse linear algebra solvers, specifically for the
    CPU case, by providing Accelerate based solvers when appropriate and optimised versions of the current C++ native solvers
    otherwise. See the roadmap `here <https://github.com/waleed-sh/mlx-sparse/issues/1>`_.

New features
~~~~~~~~~~~~

* Support multiplying all sparse array types by numbers.

* Added user-friendly native capability reporting via
  :data:`mlx_sparse.capabilities` and :func:`mlx_sparse.has_capability`. Users
  can check booleans such as ``ms.capabilities.METAL`` or query status strings
  for CPU, Metal, and reserved Accelerate/CUDA/ROCm backend capabilities
  (`PR #3 <https://github.com/waleed-sh/mlx-sparse/pull/3>`_).

* Added a CMake feature gate, ``MLX_SPARSE_ENABLE_ACCELERATE``, that detects
  and links Apple's Accelerate framework on Darwin builds for future sparse
  solver integration. No Accelerate-backed solver dispatch is enabled yet
  (`PR #4 <https://github.com/waleed-sh/mlx-sparse/pull/4>`_).

* Added a native Accelerate status/error mapping layer for future sparse solver
  integration. Factorization, Sparse BLAS, and iterative status codes now
  normalize to predictable Python ``ValueError`` or ``RuntimeError`` exceptions
  while preserving operation names and optional diagnostic detail
  (`PR #9 <https://github.com/waleed-sh/mlx-sparse/pull/9>`_).

* Added shared native Accelerate CSC adapter infrastructure for future direct
  solver dispatch. The adapter validates ``float32`` values, shape constraints,
  ``int32``/``int64`` compressed or coordinate indices, row/column bounds, and
  Accelerate ``int``/``long`` overflow limits. CSR and COO inputs normalize
  through owned canonical CSC buffers before any framework call
  (`PR #10 <https://github.com/waleed-sh/mlx-sparse/pull/10>`_).

* Added RAII wrappers for Accelerate symbolic, ``float32`` numeric, and
  ``float32`` subfactor objects. The wrappers are move-only, release resources
  through ``SparseCleanup``, retain shared opaque objects explicitly, route
  Accelerate parameter callbacks into Python exceptions, and expose solve and
  refactor helpers for future direct-solver dispatch
  (`PR #11 <https://github.com/waleed-sh/mlx-sparse/pull/11>`_).

* Added optional Accelerate-backed sparse direct solves for Apple builds that
  opt into ``MLX_SPARSE_ENABLE_ACCELERATE``. Supported real ``float32`` CSR,
  CSC, and COO inputs normalize through the shared CSC adapter and use opaque
  Accelerate Cholesky, LDLT, QR, Cholesky-at-A, and runtime-gated LU
  factorization objects. ``linalg.spsolve`` now takes the Accelerate LU fast
  path for supported square systems, while explicit-factor APIs stay on the
  native path because they promise CSR factors
  (`PR #12 <https://github.com/waleed-sh/mlx-sparse/pull/12>`_).

* Added :func:`mlx_sparse.linalg.factorized` and
  :class:`mlx_sparse.linalg.FactorizedSolve` for reusable opaque solves with
  ``backend``, ``method``, ``rhs_size``, and ``solution_size`` metadata
  (`PR #12 <https://github.com/waleed-sh/mlx-sparse/pull/12>`_).

Improvements
~~~~~~~~~~~~

* Added an experimental staged Metal path for ``COOArray @ COOArray`` behind
  ``ms.config.EXPERIMENTAL_METAL_SPGEMM``. The path row-buckets explicit COO
  coordinates for scheduling, then uses COO-specific symbolic, numeric-fill,
  and zero-prune kernels to return canonical COO output without calling CSR
  sparse-sparse multiplication
  (`PR #5 <https://github.com/waleed-sh/mlx-sparse/pull/5>`_).

* Added an experimental staged Metal path for ``CSCArray @ CSCArray`` behind
  ``ms.config.EXPERIMENTAL_METAL_SPGEMM``. The path stays column-native and
  uses CSC-specific symbolic, numeric-fill, and zero-prune kernels to return
  canonical CSC output without calling CSR sparse-sparse multiplication
  (`PR #6 <https://github.com/waleed-sh/mlx-sparse/pull/6>`_).

* Added a dedicated native normal-operator Lanczos path for
  :func:`mlx_sparse.linalg.svds`. The CSR implementation now evaluates
  ``A.T @ (A @ v)`` as a fused native step instead of decomposing each Lanczos
  application into separate host SpMVs, and the Metal path keeps the recurrence
  on GPU before synchronizing the small Ritz post-processing back to CPU
  (`PR #7 <https://github.com/waleed-sh/mlx-sparse/pull/7>`_).

* Improved sparse Metal reductions for reduction-heavy workloads. Large CSR,
  CSC, and COO traces now use staged partial reductions, CSR/CSC diagonal
  extraction uses threadgroup reductions for long compressed segments, and
  non-``float32`` COO/CSC norm reductions lower through native compressed
  storage-aligned reductions instead of scatter-heavy norm atomics
  (`PR #8 <https://github.com/waleed-sh/mlx-sparse/pull/8>`_).

Packaging
~~~~~~~~~

* Updated the PyPI publishing workflow so macOS wheels are built with
  ``MLX_SPARSE_ENABLE_ACCELERATE=ON`` and verified after wheel installation
  before upload (`PR #12 <https://github.com/waleed-sh/mlx-sparse/pull/12>`_).

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* None.

Documentation
~~~~~~~~~~~~~

* Added sparse execution documentation covering ``svds`` sparse execution,
  Metal reduction coverage by dtype and format, and
  serial/scatter-heavy/fallback boundaries
  (`PR #2 <https://github.com/waleed-sh/mlx-sparse/pull/2>`_).

* Updated sparse format, supported-feature, and performance documentation to
  describe the COO/CSC sparse-sparse execution paths and the experimental
  Metal gate accurately (`PR #5 <https://github.com/waleed-sh/mlx-sparse/pull/5>`_,
  `PR #6 <https://github.com/waleed-sh/mlx-sparse/pull/6>`_).

* Updated sparse linear algebra docs to describe the new ``svds`` fused
  normal-operator Lanczos execution path and its remaining CPU post-processing
  boundary (`PR #7 <https://github.com/waleed-sh/mlx-sparse/pull/7>`_).

* Updated dtype and performance documentation with the reduction accumulation
  policy, staged trace behavior, and remaining scatter-heavy ``float32`` norm
  limitations (`PR #8 <https://github.com/waleed-sh/mlx-sparse/pull/8>`_).

* Updated sparse linalg, installation, capability, supported-feature, and
  performance docs for the Accelerate direct-solver fast path.

* Added a dedicated solver support page documenting each public
  ``mlx_sparse.linalg`` solver, its CPU/GPU coverage label, and whether an
  Accelerate-enabled build can use an Apple sparse direct-solver path
  (`PR #12 <https://github.com/waleed-sh/mlx-sparse/pull/12>`_).

mlx-sparse v0.0.3b0 (25.05.2026)
----------------------------------

.. note::

    This release focuses on expanding the supported sparse formats to include COO and CSC arrays, with native C++/Metal
    kernels for basic operations.

New features
~~~~~~~~~~~~

* Added a typed runtime configuration manager exposed as
  :data:`mlx_sparse.config`, with attribute access, ``get_config`` /
  ``set_config``, context-manager overrides, environment-variable sync, and
  forced environment overrides.

* Added the ``EXPERIMENTAL_METAL_SPGEMM`` configuration flag for opting into
  the staged Metal CSR x CSR implementation while keeping the optimized native
  host implementation as the default.

* Added explicit batched sparse-dense APIs:
  :func:`mlx_sparse.csr_batched_matvec` for RHS shape ``(..., n_cols)`` and
  :func:`mlx_sparse.csr_batched_matmul` for RHS shape ``(..., n_cols, k)``.
  ``CSRArray @ dense`` with rank greater than 2 now dispatches through these
  native batched primitives.

* Added native staged ``fromdense`` and ``CSRArray.sum_duplicates`` /
  ``canonicalize`` implementations. These replace NumPy fallback behavior in
  the native path with count/prefix/fill C++ and Metal primitives.

* Added native CSR x CSR multiplication with a symbolic pass, prefix-sum
  output allocation, and numeric fill pass returning canonical CSR output.

* Added native COO x COO and CSC x CSC sparse-sparse multiplication. The new
  paths use format-specific symbolic/count passes, prefix allocation, sorted
  numeric fill, and zero pruning without routing through CSR.

* Added first-class ``CSCArray`` support with explicit constructors,
  validation, repr/metadata flags, COO/CSR conversion paths, dense
  materialization, sorting, duplicate summation, canonicalization, and native
  ``csc_matvec`` / ``csc_matvec_transpose`` entrypoints.

* Added native COO and CSC sparse-dense matrix products for dense vector,
  dense matrix, batched vector, and batched matrix right-hand sides. ``COOArray
  @ dense`` and ``CSCArray @ dense`` now dispatch through format-specific
  C++/Metal primitives instead of converting through CSR.

* Added native COO and CSC reductions: row sums, column sums, row norms,
  column norms, diagonal extraction, and trace. CSC column sums and column
  norms are storage-aligned compressed-column reductions.

* Added CSC input support to sparse linalg entrypoints. CSC matrices are
  converted once to canonical CSR at solver entry so existing CSR-native Krylov,
  direct factorization, triangular solve, spectral, and sparse inner-product
  kernels remain the execution path.

Improvements
~~~~~~~~~~~~

* Reorganized the native source tree so sparse and linalg operations live in
  operation-specific directories containing their C++, header, and Metal files.
  The previous monolithic sparse/linalg source layout has been split into
  localized implementation units.

* Improved CSR transpose. The CPU path now uses a counting transpose, and the
  Metal path performs parallel counts and prefix construction followed by a
  deterministic fill that preserves sorted row indices in the transposed CSR.

* Improved transpose-product kernels used by autodiff. ``float32`` Metal
  transpose matvec/matmul now use parallel atomic scatter-add kernels.
  Non-``float32`` GPU transpose products lower through native transpose plus
  native sparse-dense product to avoid unsupported Metal atomic semantics while
  staying out of NumPy.

* Extended JVP/VJP coverage through the new batched sparse-dense primitives,
  including sparse-value and dense-RHS gradients.

* Extended JVP/VJP coverage to COO and CSC sparse-dense products, including
  batched dense RHS gradients and fixed-output sparse-value VJP kernels.

* Added dedicated CSC native kernels instead of hidden CSR routing for the
  first CSC surface: column-major COO conversion, CSR/CSC conversion,
  dense materialization, per-column sorting, duplicate summation, forward
  matvec scatter-add, and transpose matvec segmented reductions.

* Added dedicated COO and CSC dense-RHS Metal kernels. COO uses coordinate
  scatter, CSC uses compressed-column scatter for forward products and
  compressed-column reductions for transpose products. ``float32`` scatter
  paths use ``atomic_float`` and non-``float32`` scatter paths remain native
  through serial GPU kernels where Metal lacks compatible atomic add support.

* Added reduction-specific Metal kernels for COO and CSC. COO coordinate
  scatter reductions use ``atomic_float`` where storage-compatible, COO/CSC
  norm scatters accumulate squared magnitudes into ``float32`` atomics, and
  CSC column reductions use scalar or threadgroup vector reductions over
  contiguous compressed columns.

* Broadened native correctness and regression tests against dense MLX and
  SciPy references, including GPU dtype coverage, complex gradients,
  pathological linalg cases, and performance regression checks.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* Removed several hidden NumPy fallback paths from native structural operations
  so canonicalization, dense conversion, and sparse-sparse multiplication use
  native implementations when the extension is available.

* Fixed GPU transpose correctness for solver paths by replacing the previous
  fragile transpose fill behavior with a deterministic native fill.

Documentation
~~~~~~~~~~~~~

* Added the :doc:`api/configuration` reference page.

* Updated operation, autodiff, device-execution, supported-feature, and
  performance documentation to explain COO/CSR/CSC batched sparse-dense
  dispatch, atomic scatter-add kernels, native transpose-product lowering,
  symbolic/numeric sparse-sparse assembly, and dynamic-output synchronization
  points.

* Documented COO/CSC reduction semantics, including duplicate-aware norm
  canonicalization and why CSC column reductions are the storage-aligned fast
  path.

* Added CSC container, conversion, and native matvec documentation plus a CSC
  notebook covering SciPy interop and CSR/CSC conversion semantics.


mlx-sparse v0.0.2b0 (21.05.2026)
----------------------------------

.. note::

    This release focuses on providing basic sparse linear algebra operations via native C++ and Metal kernels. The
    focus here is not performance but rather functionality first.

New features
~~~~~~~~~~~~

* Added :mod:`mlx_sparse.linalg`, a sparse linear algebra sub-package with
  three solver families, a spectral module, and a matrix-free operator interface.

* **Iterative solvers**: :func:`mlx_sparse.linalg.cg`,
  :func:`mlx_sparse.linalg.gmres`, and :func:`mlx_sparse.linalg.minres` solve
  sparse linear systems natively on CPU and Metal GPU.  All three accept
  :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, and any
  CSR-backed :class:`~mlx_sparse.linalg.LinearOperator` as the system matrix.

* **Direct factorizations**: :func:`mlx_sparse.linalg.cholesky` (SPD systems),
  :func:`mlx_sparse.linalg.splu` (general systems via sparse LU), and the
  convenience wrapper :func:`mlx_sparse.linalg.spsolve` for single right-hand
  sides.  Factor objects expose a ``.solve(b)`` method for multiple RHS
  without re-factorising.

* **Spectral methods**: :func:`mlx_sparse.linalg.eigsh` computes a few
  eigenvalues and eigenvectors of a real symmetric sparse matrix via a
  native Lanczos iteration.  :func:`mlx_sparse.linalg.eigs` handles general
  non-symmetric matrices (Arnoldi), and :func:`mlx_sparse.linalg.svds`
  computes a partial SVD via randomised bidiagonalisation.

* **LinearOperator interface**: :class:`mlx_sparse.linalg.LinearOperator`
  wraps any callable matvec (or a sparse array) into a uniform operator
  object accepted throughout the linalg sub-package.  The operator exposes
  :attr:`~mlx_sparse.linalg.LinearOperator.T` (transpose) and
  :attr:`~mlx_sparse.linalg.LinearOperator.H` (Hermitian / conjugate
  transpose) properties, both propagate the backing
  :class:`~mlx_sparse.CSRArray` when available so the native C++/Metal code
  paths remain active.  :func:`mlx_sparse.linalg.aslinearoperator` converts
  a :class:`~mlx_sparse.CSRArray`, :class:`~mlx_sparse.COOArray`, SciPy
  sparse matrix, or ``(shape, matvec)`` tuple into a
  :class:`~mlx_sparse.linalg.LinearOperator`.

* Added :func:`mlx_sparse.linalg.lanczos`, the underlying Lanczos
  tridiagonalisation primitive used by :func:`~mlx_sparse.linalg.eigsh`,
  exposed for advanced users who need the raw tridiagonal decomposition.

Improvements
~~~~~~~~~~~~

* :class:`~mlx_sparse.CSRArray` gained :meth:`~mlx_sparse.CSRArray.conj` /
  :meth:`~mlx_sparse.CSRArray.conjugate` convenience methods and a
  :attr:`~mlx_sparse.CSRArray.H` (Hermitian transpose) property that composes
  :meth:`~mlx_sparse.CSRArray.T` with element-wise conjugation.

* :class:`~mlx_sparse.CSRArray` now has a
  :attr:`~mlx_sparse.CSRArray.index_dtype` property that reflects the integer
  dtype of the stored index arrays.

* Added :meth:`~mlx_sparse.CSRArray.sort_indices` to sort column indices
  within each row in-place (returns ``self`` when already sorted, avoiding a
  copy).

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* None.

Deprecations
~~~~~~~~~~~~

* None.

Bug fixes
~~~~~~~~~

* None.

Documentation
~~~~~~~~~~~~~

* Added four new Jupyter notebooks under *Sparse linear algebra*:
  :doc:`notebooks/13_linalg_solvers`,
  :doc:`notebooks/14_linalg_factorizations`,
  :doc:`notebooks/15_linalg_spectral`, and
  :doc:`notebooks/16_linalg_operators`.  Each notebook walks through
  a worked example with correctness checks and timing comparisons against the
  MLX dense baseline.

* Added the :doc:`tutorials/sparse_linear_systems` tutorial, which assembles
  a 2-D Poisson (Laplacian) system, solves it with CG, Cholesky, and
  ``spsolve``, and discusses when to prefer each approach.

* Added three benchmark scripts under ``benchmarks/``:
  ``bench_linalg_solvers.py``, ``bench_linalg_factorizations.py``, and
  ``bench_linalg_spectral.py``.  Each script reports raw timing, speedup
  versus the MLX dense equivalent, and a relative-error correctness check
  against a SciPy reference.

* Reorganised :doc:`notebooks/index` into two captioned sections,
  *Primitives* (notebooks 01–12) and *Sparse linear algebra* (notebooks
  13–16), for easier navigation.
