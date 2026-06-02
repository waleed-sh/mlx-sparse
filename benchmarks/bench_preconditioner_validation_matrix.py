# Copyright (c) 2026 The mlx-sparse contributors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Preconditioner validation matrix for v0.0.5b0.

The focused preconditioner benchmarks measure individual implementations.  This
benchmark complements them with one shared report schema across matrix families,
solvers, and comparison baselines.  Timing fields are informational; regression
checks are based on iteration counts and true residuals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(BENCHMARK_DIR)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

mx.set_default_device(mx.Device(mx.cpu, 0))

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._host import to_numpy
from mlx_sparse.linalg import preconditioners

try:
    from benchmarks.benchmark_utils import (
        force_eval,
        force_scipy_eval,
        sparse_matrix_metadata,
        time_result,
    )
except ModuleNotFoundError:  # pragma: no cover - direct execution from benchmarks/
    from benchmark_utils import (  # type: ignore
        force_eval,
        force_scipy_eval,
        sparse_matrix_metadata,
        time_result,
    )

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "suitesparse" / "HB"
DEFAULT_FAMILIES = (
    "poisson_1d",
    "poisson_2d",
    "poisson_3d",
    "anisotropic_diffusion_2d",
    "badly_scaled_diagonal",
    "block_diagonal_spd",
    "convection_diffusion_1d",
    "random_diagonal_dominant",
    "hilbert_like",
    "suitesparse_well1033_normal",
    "suitesparse_illc1033_normal",
)
SPD_FAMILIES = {
    "poisson_1d",
    "poisson_2d",
    "poisson_3d",
    "anisotropic_diffusion_2d",
    "badly_scaled_diagonal",
    "block_diagonal_spd",
    "hilbert_like",
    "suitesparse_well1033_normal",
    "suitesparse_illc1033_normal",
}


@dataclass(frozen=True, slots=True)
class MatrixCase:
    """A sparse linear-system benchmark case."""

    family: str
    matrix: sp.csr_array
    rhs: np.ndarray
    kind: str
    size_parameter: int
    description: str

    @property
    def norm_rhs(self) -> float:
        """Euclidean norm of the right-hand side."""

        return float(np.linalg.norm(self.rhs))


def poisson_1d(n: int) -> sp.csr_array:
    """Return a 1-D Poisson SPD tridiagonal matrix."""

    main = 2.0 * np.ones(n, dtype=np.float32)
    off = -1.0 * np.ones(n - 1, dtype=np.float32)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr", dtype=np.float32)


def poisson_2d(grid: int) -> sp.csr_array:
    """Return a 5-point 2-D Poisson SPD matrix."""

    T = poisson_1d(grid)
    I = sp.eye(grid, format="csr", dtype=np.float32)
    return (sp.kron(I, T, format="csr") + sp.kron(T, I, format="csr")).astype(
        np.float32
    )


def poisson_3d(grid: int) -> sp.csr_array:
    """Return a 7-point 3-D Poisson SPD matrix."""

    T = poisson_1d(grid)
    I = sp.eye(grid, format="csr", dtype=np.float32)
    return (
        sp.kron(sp.kron(I, I, format="csr"), T, format="csr")
        + sp.kron(sp.kron(I, T, format="csr"), I, format="csr")
        + sp.kron(sp.kron(T, I, format="csr"), I, format="csr")
    ).astype(np.float32)


def anisotropic_diffusion_2d(
    grid: int, *, ax: float = 0.08, ay: float = 1.25
) -> sp.csr_array:
    """Return an SPD anisotropic 2-D diffusion operator with a mass shift."""

    diag = (2.0 * ax + 2.0 * ay + 0.2) * np.ones(grid, dtype=np.float32)
    off_x = -ax * np.ones(grid - 1, dtype=np.float32)
    off_y = -ay * np.ones(grid - 1, dtype=np.float32)
    Tx = sp.diags([off_x, diag, off_x], [-1, 0, 1], format="csr", dtype=np.float32)
    Ty = sp.diags(
        [off_y, off_y], [-1, 1], shape=(grid, grid), format="csr", dtype=np.float32
    )
    I = sp.eye(grid, format="csr", dtype=np.float32)
    return (sp.kron(I, Tx, format="csr") + sp.kron(Ty, I, format="csr")).astype(
        np.float32
    )


def badly_scaled_diagonal(n: int, *, condition: float = 1.0e8) -> sp.csr_array:
    """Return an SPD diagonal matrix with a large diagonal range."""

    diag = np.geomspace(1.0, condition, n).astype(np.float32)
    return sp.diags(diag, 0, format="csr", dtype=np.float32)


def block_diagonal_spd(blocks: int, *, block_size: int = 4) -> sp.csr_array:
    """Return an SPD block-diagonal matrix with differently scaled blocks."""

    mats = []
    for block in range(blocks):
        scale = np.float32(1.0 + 0.5 * block)
        diag = (4.0 * scale) * np.ones(block_size, dtype=np.float32)
        off = (-1.0 * scale) * np.ones(block_size - 1, dtype=np.float32)
        mats.append(sp.diags([off, diag, off], [-1, 0, 1], format="csr"))
    return sp.block_diag(mats, format="csr").astype(np.float32)


def convection_diffusion_1d(n: int) -> sp.csr_array:
    """Return a nonsymmetric convection-diffusion-like tridiagonal matrix."""

    diffusion = np.float32(0.18)
    convection = np.float32(0.55)
    h = np.float32(1.0 / (n + 1))
    lower = (-diffusion / h**2 - convection / h) * np.ones(n - 1, dtype=np.float32)
    diag = (2.0 * diffusion / h**2 + convection / h + 1.0) * np.ones(
        n, dtype=np.float32
    )
    upper = (-diffusion / h**2) * np.ones(n - 1, dtype=np.float32)
    return sp.diags([lower, diag, upper], [-1, 0, 1], format="csr", dtype=np.float32)


def random_diagonal_dominant(
    n: int, *, density: float = 0.04, seed: int = 0
) -> sp.csr_array:
    """Return a reproducible nonsymmetric sparse diagonally dominant matrix."""

    rng = np.random.default_rng(seed)
    base = sp.random(
        n,
        n,
        density=density,
        format="csr",
        random_state=rng,
        data_rvs=lambda count: rng.uniform(-0.25, 0.25, size=count).astype(np.float32),
    ).astype(np.float32)
    base.setdiag(0.0)
    base.eliminate_zeros()
    row_abs = np.asarray(np.abs(base).sum(axis=1)).reshape(-1).astype(np.float32)
    diag = row_abs + np.float32(1.5)
    return (base + sp.diags(diag, 0, format="csr", dtype=np.float32)).tocsr()


def hilbert_like(n: int) -> sp.csr_array:
    """Return a small dense-ish SPD Hilbert-like matrix as CSR."""

    i = np.arange(n, dtype=np.float32)[:, None]
    j = np.arange(n, dtype=np.float32)[None, :]
    dense = 1.0 / (i + j + 1.0)
    dense += 0.02 * np.eye(n, dtype=np.float32)
    return sp.csr_array(dense.astype(np.float32))


def suitesparse_normal(name: str) -> sp.csr_array:
    """Return ``A.T @ A`` for one bundled SuiteSparse fixture."""

    design = scipy.io.mmread(FIXTURE_DIR / f"{name}.mtx").astype(np.float32).tocsr()
    normal = (design.T @ design).astype(np.float32).tocsr()
    normal.sum_duplicates()
    normal.sort_indices()
    return sp.csr_array(normal)


def make_matrix_case(family: str, size: int) -> MatrixCase:
    """Build a named benchmark case with a deterministic right-hand side."""

    if family == "poisson_1d":
        matrix = poisson_1d(size)
        kind = "spd"
        description = "1-D Poisson SPD"
    elif family == "poisson_2d":
        matrix = poisson_2d(size)
        kind = "spd"
        description = "2-D Poisson SPD"
    elif family == "poisson_3d":
        matrix = poisson_3d(size)
        kind = "spd"
        description = "3-D Poisson SPD"
    elif family == "anisotropic_diffusion_2d":
        matrix = anisotropic_diffusion_2d(size)
        kind = "spd"
        description = "anisotropic diffusion SPD"
    elif family == "badly_scaled_diagonal":
        matrix = badly_scaled_diagonal(size)
        kind = "spd"
        description = "badly scaled diagonal SPD"
    elif family == "block_diagonal_spd":
        matrix = block_diagonal_spd(size)
        kind = "spd"
        description = "block-diagonal SPD"
    elif family == "convection_diffusion_1d":
        matrix = convection_diffusion_1d(size)
        kind = "general"
        description = "nonsymmetric convection-diffusion-like"
    elif family == "random_diagonal_dominant":
        matrix = random_diagonal_dominant(size)
        kind = "general"
        description = "random sparse diagonal-dominant general"
    elif family == "hilbert_like":
        matrix = hilbert_like(size)
        kind = "spd"
        description = "Hilbert-like dense-ish SPD CSR"
    elif family == "suitesparse_well1033_normal":
        matrix = suitesparse_normal("well1033")
        kind = "spd"
        description = "SuiteSparse well1033 normal equations"
    elif family == "suitesparse_illc1033_normal":
        matrix = suitesparse_normal("illc1033")
        kind = "spd"
        description = "SuiteSparse illc1033 normal equations"
    else:
        raise ValueError(f"unknown family {family!r}.")

    matrix = matrix.astype(np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    x_true = np.sin(np.linspace(0.1, 1.1, matrix.shape[1], dtype=np.float32))
    rhs = np.asarray(matrix @ x_true, dtype=np.float32).reshape(-1)
    return MatrixCase(
        family=family,
        matrix=sp.csr_array(matrix),
        rhs=rhs,
        kind=kind,
        size_parameter=size,
        description=description,
    )


def _select_mlx_device(device: str) -> mx.Stream:
    """Pin subsequent MLX allocations to the requested benchmark device."""

    if device == "cpu":
        mlx_device = mx.Device(mx.cpu, 0)
        mx.set_default_device(mlx_device)
        stream = mx.default_stream(mlx_device)
        mx.set_default_stream(stream)
        ms.use_cpu(require_available=False)
        return stream
    elif device == "gpu":
        mlx_device = ms.use_device("gpu")
        stream = mx.default_stream(mlx_device)
        mx.set_default_stream(stream)
        return stream
    else:
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}.")


def to_mlx_csr(matrix: sp.csr_array, *, device: str = "cpu") -> ms.CSRArray:
    """Convert a SciPy CSR array to canonical mlx-sparse CSR."""

    stream = _select_mlx_device(device)
    matrix = matrix.astype(np.float32).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    with mx.stream(stream):
        data = mx.asarray(np.asarray(matrix.data, dtype=np.float32), dtype=mx.float32)
        indices = mx.asarray(
            np.asarray(matrix.indices, dtype=np.int32, order="C"), dtype=mx.int32
        )
        indptr = mx.asarray(
            np.asarray(matrix.indptr, dtype=np.int32, order="C"), dtype=mx.int32
        )
        mx.eval(data, indices, indptr)
    return ms.csr_array(
        (
            data,
            indices,
            indptr,
        ),
        shape=matrix.shape,
        sorted_indices=True,
        canonical=True,
    )


def to_mlx_vector(values: np.ndarray, *, device: str = "cpu") -> mx.array:
    """Convert a NumPy vector to an MLX float32 vector on ``device``."""

    stream = _select_mlx_device(device)
    with mx.stream(stream):
        vector = mx.asarray(np.asarray(values, dtype=np.float32), dtype=mx.float32)
        mx.eval(vector)
    return vector


def scipy_jacobi_operator(matrix: sp.csr_array) -> spla.LinearOperator:
    """Return SciPy's equivalent inverse-diagonal ``LinearOperator``."""

    diagonal = matrix.diagonal().astype(np.float32, copy=False)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal == 0.0):
        raise ValueError("Jacobi diagonal must be finite and nonzero.")
    inverse = 1.0 / diagonal
    return spla.LinearOperator(
        matrix.shape, matvec=lambda x: inverse * x, dtype=np.float32
    )


def true_residual(matrix: sp.csr_array, x: np.ndarray, rhs: np.ndarray) -> float:
    """Return the host true residual norm for ``matrix @ x = rhs``."""

    return float(np.linalg.norm(matrix @ x - rhs))


def relative_residual(matrix: sp.csr_array, x: np.ndarray, rhs: np.ndarray) -> float:
    """Return the host true relative residual for ``matrix @ x = rhs``."""

    return true_residual(matrix, x, rhs) / max(float(np.linalg.norm(rhs)), 1.0)


def _timing_or_none(
    fn: Callable[[], Any], *, warmup: int, iters: int
) -> dict[str, Any]:
    return time_result(fn, warmup=warmup, iters=iters, evaluator=force_eval).as_dict()


def _preconditioner_stats(M, matrix_nnz: int) -> dict[str, Any]:
    if M is None:
        return {
            "kind": "none",
            "nnz": None,
            "nnz_L": None,
            "nnz_U": None,
            "fill_ratio": None,
            "setup_info": None,
        }
    nnz = getattr(M, "nnz", None)
    return {
        "kind": getattr(M, "kind", type(M).__name__),
        "nnz": int(nnz) if nnz is not None else None,
        "nnz_L": int(getattr(M, "nnz_L", 0)) or None,
        "nnz_U": int(getattr(M, "nnz_U", 0)) or None,
        "fill_ratio": float(nnz / max(matrix_nnz, 1)) if nnz is not None else None,
        "setup_info": dict(getattr(M, "setup_info", {}) or {}),
    }


def _thresholds(*, rtol: float, maxiter: int, norm_rhs: float) -> dict[str, Any]:
    max_relative = max(25.0 * rtol, 1.0e-5 / max(norm_rhs, 1.0), 5.0e-5)
    return {
        "max_relative_true_residual": float(max_relative),
        "max_iterations": int(maxiter),
    }


def _threshold_status(
    *, relative_true_residual: float, iterations: int, thresholds: dict[str, Any]
) -> dict[str, Any]:
    residual_ok = relative_true_residual <= thresholds["max_relative_true_residual"]
    iterations_ok = iterations <= thresholds["max_iterations"]
    return {
        "residual_ok": bool(residual_ok),
        "iterations_ok": bool(iterations_ok),
        "passed": bool(residual_ok and iterations_ok),
    }


def _base_record(
    *,
    case: MatrixCase,
    mlx_A: ms.CSRArray,
    solver: str,
    implementation: str,
    preconditioner_name: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    matrix_metadata = sparse_matrix_metadata(mlx_A)
    matrix_metadata.update(
        {
            "family": case.family,
            "description": case.description,
            "kind": case.kind,
            "size_parameter": int(case.size_parameter),
        }
    )
    return {
        "suite": "preconditioner_validation_matrix",
        "matrix": matrix_metadata,
        "solver": solver,
        "implementation": implementation,
        "preconditioner": preconditioner_name,
        "settings": dict(settings),
        "runtime": ms.runtime.info(),
    }


def _solve_mlx(
    *,
    case: MatrixCase,
    mlx_A: ms.CSRArray,
    b_mx: mx.array,
    solver: str,
    preconditioner_name: str,
    make_preconditioner: Callable[[], Any] | None,
    settings: dict[str, Any],
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    setup_timing = None
    apply_timing = None
    M = None
    setup_error = None

    if make_preconditioner is not None:
        try:
            setup_timing = _timing_or_none(
                make_preconditioner, warmup=warmup, iters=iters
            )
            M = force_eval(make_preconditioner())
            apply_timing = _timing_or_none(lambda: M(b_mx), warmup=warmup, iters=iters)
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

    record = _base_record(
        case=case,
        mlx_A=mlx_A,
        solver=solver,
        implementation="mlx_sparse",
        preconditioner_name=preconditioner_name,
        settings=settings,
    )
    record["setup_time_ms"] = setup_timing
    record["apply_time_ms"] = apply_timing
    record["preconditioner_stats"] = _preconditioner_stats(M, mlx_A.nnz)
    if setup_error is not None:
        record.update({"status": "setup_error", "error": setup_error})
        return record

    if solver == "cg":
        solve_fn = lambda: linalg.cg(
            mlx_A,
            b_mx,
            M=M,
            rtol=settings["rtol"],
            atol=settings["atol"],
            maxiter=settings["maxiter"],
            return_info=True,
        )
    else:
        solve_fn = lambda: linalg.gmres(
            mlx_A,
            b_mx,
            M=M,
            rtol=settings["rtol"],
            atol=settings["atol"],
            restart=settings["restart"],
            maxiter=settings["maxiter"],
            return_info=True,
        )

    solve_timing = _timing_or_none(solve_fn, warmup=warmup, iters=iters)
    start_ns = time.perf_counter_ns()
    x, info = solve_fn()
    mx.eval(x)
    single_solve_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    x_np = np.asarray(to_numpy(x), dtype=np.float64)
    rel = relative_residual(case.matrix, x_np, case.rhs)
    thresholds = _thresholds(
        rtol=settings["rtol"], maxiter=settings["maxiter"], norm_rhs=case.norm_rhs
    )
    record.update(
        {
            "status": int(info.status),
            "converged": bool(info.converged),
            "iterations": int(info.iterations),
            "final_true_residual": true_residual(case.matrix, x_np, case.rhs),
            "relative_true_residual": rel,
            "final_preconditioned_residual": info.preconditioned_residual_norm,
            "reported_residual_norm": float(info.residual_norm),
            "solve_time_ms": solve_timing,
            "single_solve_ms": single_solve_ms,
            "thresholds": thresholds,
            "threshold_status": _threshold_status(
                relative_true_residual=rel,
                iterations=int(info.iterations),
                thresholds=thresholds,
            ),
        }
    )
    return record


def _solve_scipy(
    *,
    case: MatrixCase,
    mlx_A: ms.CSRArray,
    solver: str,
    preconditioner_name: str,
    scipy_M,
    settings: dict[str, Any],
    warmup: int,
    iters: int,
    setup_time_ms: dict[str, Any] | None = None,
    preconditioner_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    iterations = 0

    def callback(_value):
        nonlocal iterations
        iterations += 1

    def solve_fn():
        nonlocal iterations
        iterations = 0
        if solver == "cg":
            return spla.cg(
                case.matrix,
                case.rhs,
                M=scipy_M,
                rtol=settings["rtol"],
                atol=settings["atol"],
                maxiter=settings["maxiter"],
                callback=callback,
            )
        return spla.gmres(
            case.matrix,
            case.rhs,
            M=scipy_M,
            rtol=settings["rtol"],
            atol=settings["atol"],
            restart=settings["restart"],
            maxiter=settings["maxiter"],
            callback=callback,
            callback_type="pr_norm",
        )

    solve_timing = time_result(
        solve_fn, warmup=warmup, iters=iters, evaluator=force_scipy_eval
    ).as_dict()
    x, status = solve_fn()
    x_np = np.asarray(x, dtype=np.float64)
    rel = relative_residual(case.matrix, x_np, case.rhs)
    thresholds = _thresholds(
        rtol=settings["rtol"], maxiter=settings["maxiter"], norm_rhs=case.norm_rhs
    )
    record = _base_record(
        case=case,
        mlx_A=mlx_A,
        solver=solver,
        implementation="scipy",
        preconditioner_name=preconditioner_name,
        settings=settings,
    )
    record.update(
        {
            "setup_time_ms": setup_time_ms,
            "apply_time_ms": None,
            "preconditioner_stats": preconditioner_stats
            or _preconditioner_stats(None, mlx_A.nnz),
            "status": int(status),
            "converged": bool(status == 0),
            "iterations": int(iterations),
            "final_true_residual": true_residual(case.matrix, x_np, case.rhs),
            "relative_true_residual": rel,
            "final_preconditioned_residual": None,
            "reported_residual_norm": None,
            "solve_time_ms": solve_timing,
            "single_solve_ms": None,
            "thresholds": thresholds,
            "threshold_status": _threshold_status(
                relative_true_residual=rel,
                iterations=int(iterations),
                thresholds=thresholds,
            ),
        }
    )
    return record


def _scipy_spilu(case: MatrixCase, *, warmup: int, iters: int):
    setup_timing = time_result(
        lambda: spla.spilu(
            case.matrix.tocsc(),
            drop_tol=0.0,
            fill_factor=1.0,
            permc_spec="NATURAL",
            diag_pivot_thresh=0.0,
        ),
        warmup=warmup,
        iters=iters,
        evaluator=force_scipy_eval,
    ).as_dict()
    factor = spla.spilu(
        case.matrix.tocsc(),
        drop_tol=0.0,
        fill_factor=1.0,
        permc_spec="NATURAL",
        diag_pivot_thresh=0.0,
    )
    stats = {
        "kind": "scipy_spilu",
        "nnz": int(factor.L.nnz + factor.U.nnz),
        "nnz_L": int(factor.L.nnz),
        "nnz_U": int(factor.U.nnz),
        "fill_ratio": float((factor.L.nnz + factor.U.nnz) / max(case.matrix.nnz, 1)),
        "setup_info": {"permc_spec": "NATURAL", "drop_tol": 0.0, "fill_factor": 1.0},
    }
    operator = spla.LinearOperator(
        case.matrix.shape, matvec=factor.solve, dtype=np.float32
    )
    return setup_timing, operator, stats


def benchmark_case(
    *,
    family: str,
    size: int,
    device: str = "cpu",
    warmup: int,
    iters: int,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    include_scipy: bool,
    selected_preconditioners: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Benchmark one matrix family and return JSON-safe records."""

    case = make_matrix_case(family, size)
    mlx_A = to_mlx_csr(case.matrix, device=device)
    b_mx = to_mlx_vector(case.rhs, device=device)
    settings = {
        "rtol": float(rtol),
        "atol": float(atol),
        "restart": int(restart),
        "maxiter": int(maxiter),
        "warmup": int(warmup),
        "iters": int(iters),
        "device": str(mx.default_device()),
    }
    selected = set(selected_preconditioners or ())
    records: list[dict[str, Any]] = []

    def use(name: str) -> bool:
        return not selected or name in selected

    if case.kind == "spd":
        specs: tuple[tuple[str, str, Callable[[], Any] | None], ...] = (
            ("none", "cg", None),
            ("jacobi", "cg", lambda: preconditioners.jacobi(mlx_A, check=True)),
            ("ichol0", "cg", lambda: preconditioners.ichol0(mlx_A)),
            ("chebyshev", "cg", lambda: preconditioners.chebyshev(mlx_A, degree=2)),
            (
                "exact_cholesky",
                "cg",
                lambda: preconditioners.exact(mlx_A, method="cholesky"),
            ),
        )
    else:
        specs = (
            ("none", "gmres", None),
            ("jacobi", "gmres", lambda: preconditioners.jacobi(mlx_A)),
            ("ilu0", "gmres", lambda: preconditioners.ilu0(mlx_A)),
            ("exact_lu", "gmres", lambda: preconditioners.exact(mlx_A, method="lu")),
        )

    for name, solver, make_M in specs:
        if use(name):
            records.append(
                _solve_mlx(
                    case=case,
                    mlx_A=mlx_A,
                    b_mx=b_mx,
                    solver=solver,
                    preconditioner_name=name,
                    make_preconditioner=make_M,
                    settings=settings,
                    warmup=warmup,
                    iters=iters,
                )
            )

    if include_scipy:
        scipy_solver = "cg" if case.kind == "spd" else "gmres"
        if use("scipy_none"):
            records.append(
                _solve_scipy(
                    case=case,
                    mlx_A=mlx_A,
                    solver=scipy_solver,
                    preconditioner_name="none",
                    scipy_M=None,
                    settings=settings,
                    warmup=warmup,
                    iters=iters,
                )
            )
        if use("scipy_jacobi"):
            records.append(
                _solve_scipy(
                    case=case,
                    mlx_A=mlx_A,
                    solver=scipy_solver,
                    preconditioner_name="jacobi",
                    scipy_M=scipy_jacobi_operator(case.matrix),
                    settings=settings,
                    warmup=warmup,
                    iters=iters,
                    preconditioner_stats={
                        "kind": "scipy_jacobi_linear_operator",
                        "nnz": int(case.matrix.shape[0]),
                        "nnz_L": None,
                        "nnz_U": None,
                        "fill_ratio": float(
                            case.matrix.shape[0] / max(case.matrix.nnz, 1)
                        ),
                        "setup_info": {"contract": "inverse diagonal LinearOperator"},
                    },
                )
            )
        if case.kind == "general" and use("scipy_spilu"):
            try:
                spilu_setup, spilu_M, spilu_stats = _scipy_spilu(
                    case, warmup=warmup, iters=iters
                )
                records.append(
                    _solve_scipy(
                        case=case,
                        mlx_A=mlx_A,
                        solver="gmres",
                        preconditioner_name="spilu",
                        scipy_M=spilu_M,
                        settings=settings,
                        warmup=warmup,
                        iters=iters,
                        setup_time_ms=spilu_setup,
                        preconditioner_stats=spilu_stats,
                    )
                )
            except Exception as exc:
                record = _base_record(
                    case=case,
                    mlx_A=mlx_A,
                    solver="gmres",
                    implementation="scipy",
                    preconditioner_name="spilu",
                    settings=settings,
                )
                record.update(
                    {"status": "setup_error", "error": f"{type(exc).__name__}: {exc}"}
                )
                records.append(record)

    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact validation status for a full benchmark run."""

    solved = [record for record in records if isinstance(record.get("status"), int)]
    return {
        "record_count": len(records),
        "solved_count": len(solved),
        "threshold_pass_count": sum(
            1 for record in solved if record["threshold_status"]["passed"]
        ),
        "max_relative_true_residual": max(
            (float(record["relative_true_residual"]) for record in solved), default=None
        ),
        "families": sorted({record["matrix"]["family"] for record in records}),
        "implementations": sorted({record["implementation"] for record in records}),
    }


def main() -> None:
    """Run the validation matrix and print a JSON report."""

    parser = argparse.ArgumentParser(
        description="Benchmark the v0.0.5b0 preconditioner validation matrix."
    )
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--restart", type=int, default=16)
    parser.add_argument("--maxiter", type=int, default=256)
    parser.add_argument(
        "--skip-scipy",
        action="store_true",
        help="Skip SciPy comparison baselines.",
    )
    parser.add_argument(
        "--preconditioners",
        nargs="+",
        default=None,
        help="Optional filter such as none jacobi ilu0 scipy_jacobi.",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        ms.use_cpu(require_available=False)
    else:
        ms.use_device(args.device)
    selected = tuple(args.preconditioners) if args.preconditioners else None
    records: list[dict[str, Any]] = []
    for family in args.families:
        sizes = [args.sizes[0]] if family.startswith("suitesparse_") else args.sizes
        for size in sizes:
            records.extend(
                benchmark_case(
                    family=family,
                    size=size,
                    device=args.device,
                    warmup=args.warmup,
                    iters=args.iters,
                    rtol=args.rtol,
                    atol=args.atol,
                    restart=args.restart,
                    maxiter=args.maxiter,
                    include_scipy=not args.skip_scipy,
                    selected_preconditioners=selected,
                )
            )
    print(
        json.dumps(
            {
                "suite": "preconditioner_validation_matrix",
                "summary": summarize_records(records),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
