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

from __future__ import annotations

from pathlib import Path

from mlx_sparse._ext_loader import extension


def test_linalg_python_layer_does_not_use_dense_numerical_shortcuts():
    root = Path(__file__).resolve().parents[1] / "mlx_sparse" / "linalg"
    banned = (
        ".todense(",
        ".toarray(",
        "mx.linalg",
        "np.linalg",
        "scipy.sparse.linalg",
        "mx.sum(",
        "mx.sqrt(",
        "mx.eye(",
        "mx.stack(",
        "mx.take(",
        "mx.matmul(",
    )
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(root.parent.parent)}: {token}")

    assert offenders == []


def test_gmres_paths_do_not_use_projected_normal_equations():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "src" / "linalg" / "gmres" / "gmres.cpp",
        root / "src" / "preconditioners" / "gmres" / "gmres.cpp",
    ]

    offenders = [
        str(path.relative_to(root))
        for path in paths
        if "least_squares_normal_equations" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_native_linalg_symbols_are_exported_when_extension_loads():
    ext = extension()
    if ext is None:
        return

    required = {
        "csr_cg",
        "csr_pcg_jacobi",
        "csr_pcg_ic0",
        "csr_pcg_chebyshev",
        "diagonal_preconditioner_apply",
        "csr_chebyshev_spectral_bounds",
        "csr_chebyshev_preconditioner_apply",
        "csr_gmres",
        "csr_gmres_jacobi",
        "csr_gmres_ilu0",
        "csr_minres",
        "csr_minres_jacobi",
        "csr_lanczos",
        "csr_arnoldi",
        "csr_eigsh",
        "csr_eigs",
        "csr_svds",
        "csr_cholesky",
        "csr_lu",
        "csr_ilu0",
        "csr_ilu0_preconditioner_apply",
        "csr_ic0",
        "csr_ic0_preconditioner_apply",
        "csr_triangular_solve",
        "csr_triangular_diagonal_positions",
        "csr_triangular_level_schedule",
        "csr_triangular_solve_analyzed",
        "csr_vdot",
        "csr_dot",
        "csr_permute_vector",
    }
    missing = sorted(name for name in required if not hasattr(ext, name))
    assert missing == []
