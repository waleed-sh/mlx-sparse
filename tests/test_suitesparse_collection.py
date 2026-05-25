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

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse import linalg
from mlx_sparse._ext_loader import extension_available
from mlx_sparse._host import to_numpy

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "suitesparse" / "HB"
SUITESPARSE_MATRICES = (
    ("well1033", (1033, 320), 4732, "real general", 1.661333e2, 2e-5, 2e-5),
    ("illc1033", (1033, 320), 4732, "real general", 1.888813e4, 2e-5, 2e-5),
    ("bcsstk03", (112, 112), 640, "real symmetric", 6.791333e6, 2e-5, 3.0e6),
)
LEAST_SQUARES_MATRICES = ("well1033", "illc1033")


def _read_suitesparse_matrix(name: str):
    scipy_io = pytest.importorskip("scipy.io")
    matrix = scipy_io.mmread(FIXTURE_DIR / f"{name}.mtx").tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _read_rhs(name: str) -> np.ndarray:
    scipy_io = pytest.importorskip("scipy.io")
    rhs = scipy_io.mmread(FIXTURE_DIR / f"{name}_b.mtx")
    return np.asarray(rhs, dtype=np.float32).reshape(-1)


def _expected_dense(matrix) -> np.ndarray:
    return np.asarray(matrix.toarray(), dtype=np.float32)


def _require_native_linalg():
    if not extension_available():
        pytest.skip("native extension unavailable")


def _relative_residual(scipy_csr, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(scipy_csr @ x - b) / np.linalg.norm(b))


def test_suitesparse_fixture_metadata_is_preserved():
    for name, _, _, matrix_type, _, _, _ in SUITESPARSE_MATRICES:
        text = (FIXTURE_DIR / f"{name}.mtx").read_text(encoding="utf-8")

        assert "% UF Sparse Matrix Collection, Tim Davis" in text
        assert f"% name: HB/{name}" in text
        assert f"%%MatrixMarket matrix coordinate {matrix_type}" in text

    for name in LEAST_SQUARES_MATRICES:
        text = (FIXTURE_DIR / f"{name}_b.mtx").read_text(encoding="utf-8")
        assert f"% name: HB/{name} : b matrix" in text
        assert "%%MatrixMarket matrix array real general" in text


@pytest.mark.parametrize(
    ("name", "shape", "_entries", "_matrix_type", "condition", "_rtol", "_atol"),
    SUITESPARSE_MATRICES,
)
def test_suitesparse_condition_numbers_match_fixture_purpose(
    name,
    shape,
    _entries,
    _matrix_type,
    condition,
    _rtol,
    _atol,
):
    scipy_csr = _read_suitesparse_matrix(name)
    dense = scipy_csr.toarray()
    actual_condition = np.linalg.cond(dense)

    assert scipy_csr.shape == shape
    np.testing.assert_allclose(actual_condition, condition, rtol=5e-5)

    if name == "well1033":
        assert actual_condition < 2e2
    elif name == "illc1033":
        assert actual_condition > 1e4
    else:
        assert actual_condition > 1e6


@pytest.mark.parametrize(
    (
        "name",
        "shape",
        "expected_stored_entries",
        "_matrix_type",
        "_condition",
        "rtol",
        "atol",
    ),
    SUITESPARSE_MATRICES,
)
def test_suitesparse_matrix_market_roundtrip_to_csr_and_coo(
    mx,
    name,
    shape,
    expected_stored_entries,
    _matrix_type,
    _condition,
    rtol,
    atol,
):
    scipy_csr = _read_suitesparse_matrix(name)

    csr = ms.from_scipy(scipy_csr, index_dtype=mx.int64)
    coo = ms.from_scipy(scipy_csr, format="coo", index_dtype=mx.int64)

    assert scipy_csr.shape == shape
    assert scipy_csr.nnz == expected_stored_entries
    assert csr.shape == scipy_csr.shape
    assert csr.nnz == expected_stored_entries
    assert csr.index_dtype == mx.int64
    assert csr.has_canonical_format
    assert coo.shape == scipy_csr.shape
    assert coo.nnz == expected_stored_entries
    assert coo.index_dtype == mx.int64

    expected = _expected_dense(scipy_csr)
    np.testing.assert_allclose(to_numpy(csr.todense()), expected, rtol=2e-5, atol=1e-5)
    np.testing.assert_allclose(to_numpy(coo.todense()), expected, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize(
    (
        "name",
        "_shape",
        "_expected_stored_entries",
        "_matrix_type",
        "_condition",
        "rtol",
        "atol",
    ),
    SUITESPARSE_MATRICES,
)
def test_suitesparse_sparse_dense_products_match_scipy(
    mx,
    name,
    _shape,
    _expected_stored_entries,
    _matrix_type,
    _condition,
    rtol,
    atol,
):
    scipy_csr = _read_suitesparse_matrix(name).astype(np.float32)
    csr = ms.from_scipy(scipy_csr)

    vector = np.linspace(-1.0, 1.0, scipy_csr.shape[1], dtype=np.float32)
    rhs = np.stack(
        [
            vector,
            np.cos(np.arange(scipy_csr.shape[1], dtype=np.float32) / 5.0),
            np.linspace(0.5, -0.25, scipy_csr.shape[1], dtype=np.float32),
        ],
        axis=1,
    )
    batched_vectors = np.stack([vector, vector[::-1]], axis=0)
    batched_matrices = np.stack([rhs, rhs * np.float32(0.25) - np.float32(1.0)])
    transpose_vector = np.linspace(-0.5, 0.75, scipy_csr.shape[0], dtype=np.float32)

    np.testing.assert_allclose(
        to_numpy(csr @ mx.array(vector)),
        scipy_csr @ vector,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(csr @ mx.array(rhs)),
        scipy_csr @ rhs,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(csr.T @ mx.array(transpose_vector)),
        scipy_csr.T @ transpose_vector,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(ms.csr_batched_matvec(csr, mx.array(batched_vectors))),
        batched_vectors @ scipy_csr.toarray().T,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        to_numpy(ms.csr_batched_matmul(csr, mx.array(batched_matrices))),
        np.stack([scipy_csr @ mat for mat in batched_matrices]),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("name", ("well1033", "bcsstk03"))
def test_suitesparse_coo_and_csc_native_products_match_scipy(mx, name):
    scipy_csr = _read_suitesparse_matrix(name).astype(np.float32)
    coo = ms.from_scipy(scipy_csr, format="coo", index_dtype=mx.int64)
    csc = ms.from_scipy(scipy_csr, format="csc", index_dtype=mx.int64)

    vector = np.cos(np.arange(scipy_csr.shape[1], dtype=np.float32) / 11.0)
    dense_rhs = np.arange(scipy_csr.shape[1] * 4, dtype=np.float32).reshape(
        -1, 4
    ) / np.float32(23.0)
    transpose_vector = np.sin(
        np.arange(scipy_csr.shape[0], dtype=np.float32) / np.float32(13.0)
    )

    np.testing.assert_allclose(
        to_numpy(coo @ mx.array(vector)),
        scipy_csr @ vector,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        to_numpy(csc @ mx.array(vector)),
        scipy_csr @ vector,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        to_numpy(csc @ mx.array(dense_rhs)),
        scipy_csr @ dense_rhs,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        to_numpy(csc.T @ mx.array(transpose_vector)),
        scipy_csr.T @ transpose_vector,
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("name", LEAST_SQUARES_MATRICES)
def test_suitesparse_least_squares_rhs_normal_equation_parts_match_scipy(mx, name):
    scipy_csr = _read_suitesparse_matrix(name).astype(np.float32)
    csr = ms.from_scipy(scipy_csr)
    rhs = _read_rhs(name)

    assert rhs.shape == (scipy_csr.shape[0],)
    np.testing.assert_allclose(
        to_numpy(csr.T @ mx.array(rhs)),
        scipy_csr.T @ rhs,
        rtol=2e-5,
        atol=1e-4,
    )


@pytest.mark.parametrize("name", LEAST_SQUARES_MATRICES)
def test_suitesparse_normal_equation_sparse_sparse_product(mx, name):
    scipy_csr = _read_suitesparse_matrix(name).astype(np.float32)

    out = ms.from_scipy(scipy_csr.T) @ ms.from_scipy(scipy_csr)
    expected = scipy_csr.T @ scipy_csr
    expected.sum_duplicates()
    expected.sort_indices()

    assert out.has_canonical_format
    assert out.sorted_indices
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        expected.toarray(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_suitesparse_csc_normal_equation_sparse_sparse_product(mx):
    scipy_csr = _read_suitesparse_matrix("well1033").astype(np.float32)

    left = ms.from_scipy(scipy_csr.T, format="csc", index_dtype=mx.int64)
    right = ms.from_scipy(scipy_csr, format="csc", index_dtype=mx.int64)
    out = left @ right
    expected = scipy_csr.T @ scipy_csr
    expected.sum_duplicates()
    expected.sort_indices()

    assert isinstance(out, ms.CSCArray)
    assert out.has_canonical_format
    assert out.sorted_indices
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        expected.toarray(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_suitesparse_explicit_zeros_preserved_by_from_scipy_but_not_fromdense(mx):
    illc = _read_suitesparse_matrix("illc1033").astype(np.float32)
    illc_without_explicit_zeros = illc.copy()
    illc_without_explicit_zeros.eliminate_zeros()

    assert illc.nnz == 4732
    assert illc.count_nonzero() == 4719
    assert illc_without_explicit_zeros.nnz == 4719

    from_scipy = ms.from_scipy(illc)
    from_dense = ms.fromdense(mx.array(illc.toarray().astype(np.float32)))

    assert from_scipy.nnz == illc.nnz
    assert from_dense.nnz == illc_without_explicit_zeros.nnz
    np.testing.assert_allclose(
        to_numpy(from_dense.todense()),
        illc_without_explicit_zeros.toarray(),
    )


@pytest.mark.native
def test_suitesparse_well1033_normal_equation_cg_and_gmres_solve(mx):
    _require_native_linalg()
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    design = _read_suitesparse_matrix("well1033").astype(np.float32)
    rhs = _read_rhs("well1033")
    normal = (design.T @ design).tocsr()
    normal.sum_duplicates()
    normal.sort_indices()
    normal_rhs = design.T @ rhs
    sparse_normal = ms.from_scipy(normal)

    expected, scipy_info = scipy_linalg.cg(
        normal,
        normal_rhs,
        rtol=1e-5,
        atol=1e-5,
        maxiter=2000,
    )
    assert scipy_info == 0

    cg_x, cg_info = linalg.cg(
        sparse_normal,
        mx.array(normal_rhs),
        rtol=1e-5,
        atol=1e-5,
        maxiter=2000,
    )
    gmres_x, gmres_info = linalg.gmres(
        sparse_normal,
        mx.array(normal_rhs),
        rtol=1e-5,
        atol=1e-5,
        restart=64,
        maxiter=2000,
    )

    assert cg_info == 0
    assert gmres_info == 0
    expected_residual = _relative_residual(normal, expected, normal_rhs)
    assert _relative_residual(normal, to_numpy(cg_x), normal_rhs) <= max(
        2e-5, 5.0 * expected_residual
    )
    assert _relative_residual(normal, to_numpy(gmres_x), normal_rhs) <= max(
        2e-5, 5.0 * expected_residual
    )


@pytest.mark.native
def test_suitesparse_well1033_normal_equation_accepts_csc_solver_input(mx):
    _require_native_linalg()
    design = _read_suitesparse_matrix("well1033").astype(np.float32)
    rhs = _read_rhs("well1033")
    normal = (design.T @ design).tocsr()
    normal.sum_duplicates()
    normal.sort_indices()
    normal_rhs = design.T @ rhs
    sparse_normal = ms.from_scipy(normal, format="csc", index_dtype=mx.int64)

    x, info = linalg.cg(
        sparse_normal,
        mx.array(normal_rhs),
        rtol=1e-5,
        atol=1e-5,
        maxiter=2000,
    )

    assert info == 0
    assert _relative_residual(normal, to_numpy(x), normal_rhs) < 2e-5


@pytest.mark.native
def test_suitesparse_illc1033_normal_equation_solver_residual_is_bounded(mx):
    _require_native_linalg()
    design = _read_suitesparse_matrix("illc1033").astype(np.float32)
    rhs = _read_rhs("illc1033")
    normal = (design.T @ design).tocsr()
    normal.sum_duplicates()
    normal.sort_indices()
    normal_rhs = design.T @ rhs
    sparse_normal = ms.from_scipy(normal)

    x, info = linalg.cg(
        sparse_normal,
        mx.array(normal_rhs),
        rtol=1e-4,
        atol=1e-4,
        maxiter=3000,
    )

    residual = _relative_residual(normal, to_numpy(x), normal_rhs)
    assert np.isfinite(residual)
    if info == 0:
        assert residual < 1e-4
    else:
        assert residual < 5e-2


@pytest.mark.native
def test_suitesparse_bcsstk03_spd_solvers_and_eigenvalue(mx):
    _require_native_linalg()
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    scipy_spd = _read_suitesparse_matrix("bcsstk03").astype(np.float32)
    sparse_spd = ms.from_scipy(scipy_spd)
    x_true = np.sin(np.arange(scipy_spd.shape[0], dtype=np.float32) / 7.0)
    rhs = scipy_spd @ x_true

    cg_x, cg_info = linalg.cg(
        sparse_spd,
        mx.array(rhs),
        rtol=1e-5,
        atol=1e2,
        maxiter=2000,
    )
    minres_x, minres_info = linalg.minres(
        sparse_spd,
        mx.array(rhs),
        rtol=1e-5,
        atol=1e2,
        maxiter=2000,
    )
    got_eig = linalg.eigsh(
        sparse_spd,
        k=1,
        which="LM",
        ncv=20,
        return_eigenvectors=False,
    )
    expected_eig = scipy_linalg.eigsh(
        scipy_spd,
        k=1,
        which="LM",
        return_eigenvectors=False,
    )

    assert cg_info == 0
    assert minres_info == 0
    assert _relative_residual(scipy_spd, to_numpy(cg_x), rhs) < 1e-5
    assert _relative_residual(scipy_spd, to_numpy(minres_x), rhs) < 1e-5
    np.testing.assert_allclose(
        np.sort(to_numpy(got_eig)),
        np.sort(expected_eig),
        rtol=5e-3,
        atol=5e7,
    )


@pytest.mark.native
def test_suitesparse_bcsstk03_direct_factorizations_have_small_residual(mx):
    _require_native_linalg()
    scipy_spd = _read_suitesparse_matrix("bcsstk03").astype(np.float32)
    sparse_spd = ms.from_scipy(scipy_spd, format="csc", index_dtype=mx.int64)
    x_true = np.cos(np.arange(scipy_spd.shape[0], dtype=np.float32) / 9.0)
    rhs = scipy_spd @ x_true

    chol = linalg.sparse_cholesky(sparse_spd)
    lu = linalg.sparse_lu(sparse_spd)
    chol_x = to_numpy(chol.solve(mx.array(rhs)))
    lu_x = to_numpy(lu.solve(mx.array(rhs)))

    assert _relative_residual(scipy_spd, chol_x, rhs) < 5e-5
    assert _relative_residual(scipy_spd, lu_x, rhs) < 5e-5


@pytest.mark.native
@pytest.mark.parametrize("name", LEAST_SQUARES_MATRICES)
def test_suitesparse_lsq_svds_largest_singular_value_matches_scipy(mx, name):
    _require_native_linalg()
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    scipy_csr = _read_suitesparse_matrix(name).astype(np.float32)
    sparse_csr = ms.from_scipy(scipy_csr)

    got = linalg.svds(
        sparse_csr,
        k=1,
        which="LM",
        ncv=32,
        return_singular_vectors=False,
    )
    expected = scipy_linalg.svds(
        scipy_csr,
        k=1,
        which="LM",
        return_singular_vectors=False,
    )

    np.testing.assert_allclose(to_numpy(got), expected, rtol=5e-3, atol=5e-3)


@pytest.mark.native
def test_suitesparse_rectangular_svds_vectors_satisfy_singular_equations(mx):
    _require_native_linalg()
    scipy_csr = _read_suitesparse_matrix("well1033").astype(np.float32)
    sparse_csr = ms.from_scipy(scipy_csr, format="coo", index_dtype=mx.int64)

    u, singular, vh = linalg.svds(
        sparse_csr,
        k=2,
        which="LM",
        ncv=40,
        return_singular_vectors=True,
    )
    u_np = to_numpy(u)
    singular_np = to_numpy(singular)
    vh_np = to_numpy(vh)

    for i, sigma in enumerate(singular_np):
        right = vh_np[i]
        left = u_np[:, i]
        np.testing.assert_allclose(
            scipy_csr @ right,
            sigma * left,
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            scipy_csr.T @ left,
            sigma * right,
            rtol=2e-2,
            atol=2e-2,
        )
