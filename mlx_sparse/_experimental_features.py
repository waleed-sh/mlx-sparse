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

"""Notices for public APIs with audited numerical correctness risks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_sparse.experimental import ExperimentalRegistry

_DOCS = (
    "https://github.com/waleed-sh/mlx-sparse/blob/main/"
    "docs/user_guide/solver_limitations.rst"
)


def register_features(registry: ExperimentalRegistry) -> None:
    notices = {
        "_iterative.cg": (
            "CG can report success from its recursive residual even when the "
            "true residual exceeds the requested tolerance. Recompute b - A @ x "
            "and check its norm before accepting the solution."
        ),
        "_eigen.eigsh": (
            "This routine extracts eigenpairs from one bounded Krylov projection "
            "without checking convergence. The projected eigensolve also has "
            "fixed accuracy limits. Supply a real symmetric operator and check "
            "each eigenpair residual against the original matrix."
        ),
        "_eigen.eigs": (
            "This routine extracts eigenpairs from one bounded Arnoldi projection "
            "without checking convergence. Returned candidates may not represent "
            "the requested part of the full spectrum. Check each eigenpair "
            "residual against the original matrix."
        ),
        "_eigen.svds": (
            "This routine extracts singular triplets from one bounded projection "
            "without checking convergence. Returned candidates may not represent "
            "the requested part of the full spectrum. Check both A @ v = s * u "
            "and A.T @ u = s * v against the original matrix."
        ),
        "_triangular.spsolve_triangular": (
            "A missing stored diagonal is treated as one even when "
            "unit_diagonal=False. Zero or tiny pivots have device-dependent "
            "failure behavior. Supply explicit finite nonzero diagonals unless "
            "using a unit diagonal and check the true residual."
        ),
    }
    lu = (
        "Native LU uses absolute float32 pivot and drop thresholds that can "
        "reject scaled systems or discard entries. There is no accuracy "
        "certificate or iterative refinement. Check the factorization and "
        "true solve residual against the original matrix."
    )
    cholesky = (
        "Cholesky does not verify agreement between the two matrix triangles. "
        "An asymmetric input can be factored as a different symmetric operator. "
        "Native pivot and drop thresholds also depend on scale. Supply a "
        "real symmetric positive-definite matrix and check the true residual."
    )
    rhs = (
        "Complex MLX right-hand sides can lose their imaginary parts during "
        "float32 conversion. Use real floating right-hand sides. Native factors "
        "also have scale-dependent pivot, drop, and triangular-solve limits. "
        "Check the true residual against the intended operator."
    )
    for name in ("sparse_lu", "splu"):
        notices[f"_factorizations.{name}"] = lu
    for name in ("sparse_cholesky", "cholesky"):
        notices[f"_factorizations.{name}"] = cholesky
    notices["_factorizations.factorized"] = (
        "Factorization and solve guarantees depend on the selected backend and "
        "method. Native factors have scale-dependent accuracy limits. "
        "cholesky_ata solves A.T @ A rather than the original system. Complex "
        "MLX right-hand sides can lose their imaginary parts. Check the method's "
        "operator and the true residual using real floating right-hand sides."
    )
    notices["_factorizations.spsolve"] = rhs
    notices["_factorizations.SparseLU.solve"] = rhs
    notices["_factorizations.SparseCholesky.solve"] = cholesky + " " + rhs
    notices["_factorizations.FactorizedSolve.solve"] = (
        "cholesky_ata solves A.T @ A rather than the original system. " + rhs
    )
    for implementation, message in notices.items():
        public_name = implementation.split(".", 1)[1]
        registry.register(
            f"mlx_sparse.linalg.{implementation}",
            display_name=f"mlx_sparse.linalg.{public_name}",
            message=message,
            docs_url=_DOCS,
        )
