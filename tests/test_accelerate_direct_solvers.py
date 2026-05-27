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

import numpy as np
import pytest

import mlx_sparse as ms
import mlx_sparse._native as _native
from mlx_sparse import linalg
from mlx_sparse._ext_loader import extension_available

pytestmark = [pytest.mark.native, pytest.mark.accelerate, pytest.mark.cpu_only]


def _require_accelerate_solvers():
    if not extension_available():
        pytest.skip("native extension unavailable")
    if not _native.accelerate_solvers_available():
        pytest.skip("Accelerate solver support is not compiled in")


def _require_accelerate_lu():
    _require_accelerate_solvers()
    if not _native.accelerate_lu_solvers_available():
        pytest.skip("Accelerate LU requires a macOS 15.5 SDK and runtime")


def _spd_matrix(scipy_sparse, n: int = 8):
    return scipy_sparse.diags(
        [
            -np.ones(n - 1, dtype=np.float32),
            4.0 * np.ones(n, dtype=np.float32),
            -np.ones(n - 1, dtype=np.float32),
        ],
        offsets=[-1, 0, 1],
        format="csr",
        dtype=np.float32,
    )


def _general_matrix(scipy_sparse, n: int = 7):
    return scipy_sparse.diags(
        [
            0.25 * np.ones(n - 2, dtype=np.float32),
            -0.75 * np.ones(n - 1, dtype=np.float32),
            5.0 * np.ones(n, dtype=np.float32),
            1.25 * np.ones(n - 1, dtype=np.float32),
        ],
        offsets=[-2, -1, 0, 1],
        format="csr",
        dtype=np.float32,
    )


def _rectangular_matrix(scipy_sparse):
    rows = np.array([0, 0, 1, 2, 2, 3, 4, 5, 5, 6], dtype=np.int32)
    cols = np.array([0, 2, 1, 0, 3, 2, 1, 0, 3, 2], dtype=np.int32)
    data = np.array(
        [2.0, -1.0, 1.5, 0.5, 3.0, -2.0, 2.5, 1.0, -0.75, 1.25],
        dtype=np.float32,
    )
    return scipy_sparse.coo_matrix((data, (rows, cols)), shape=(7, 4)).tocsr()


def _to_ms(mx, scipy_matrix, *, fmt="csr", index_dtype=None, dtype=None):
    if index_dtype is None:
        index_dtype = mx.int32
    if dtype is None:
        dtype = mx.float32
    return ms.from_scipy(
        scipy_matrix,
        format=fmt,
        dtype=dtype,
        index_dtype=index_dtype,
        canonical=True,
    )


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
@pytest.mark.parametrize("index_attr", ["int32", "int64"])
def test_accelerate_cholesky_spd_matches_scipy_formats_and_indices(
    mx, scipy_sparse, to_numpy, fmt, index_attr
):
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    _require_accelerate_solvers()
    scipy_a = _spd_matrix(scipy_sparse)
    index_dtype = getattr(mx, index_attr)
    a = _to_ms(mx, scipy_a, fmt=fmt, index_dtype=index_dtype)
    rhs_np = np.column_stack(
        [
            np.linspace(0.5, 1.5, scipy_a.shape[0], dtype=np.float32),
            np.linspace(-1.0, 0.25, scipy_a.shape[0], dtype=np.float32),
        ]
    )

    solver = linalg.factorized(a, method="cholesky")

    assert solver.backend == "accelerate"
    assert solver.method == "cholesky"
    assert solver.shape == scipy_a.shape
    expected = np.column_stack(
        [scipy_linalg.spsolve(scipy_a, rhs_np[:, col]) for col in range(2)]
    )
    np.testing.assert_allclose(
        to_numpy(solver.solve(mx.array(rhs_np))),
        expected,
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.parametrize("index_attr", ["int32", "int64"])
def test_accelerate_lu_spsolve_fast_path_matches_scipy(
    mx, scipy_sparse, to_numpy, index_attr
):
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    _require_accelerate_lu()
    scipy_a = _general_matrix(scipy_sparse)
    a = _to_ms(mx, scipy_a, fmt="csc", index_dtype=getattr(mx, index_attr))
    b_np = np.linspace(-1.0, 2.0, scipy_a.shape[0], dtype=np.float32)

    solver = linalg.factorized(a, method="lu")
    x = linalg.spsolve(a, mx.array(b_np))

    assert solver.backend == "accelerate"
    expected = scipy_linalg.spsolve(scipy_a, b_np)
    np.testing.assert_allclose(
        to_numpy(solver.solve(mx.array(b_np))), expected, rtol=2e-4, atol=2e-4
    )
    np.testing.assert_allclose(to_numpy(x), expected, rtol=2e-4, atol=2e-4)


def test_accelerate_ldlt_symmetric_indefinite_matches_scipy(mx, scipy_sparse, to_numpy):
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    _require_accelerate_solvers()
    dense = np.array(
        [
            [4.0, 1.0, 0.0, 0.5],
            [1.0, -3.0, 1.25, 0.0],
            [0.0, 1.25, 2.5, -0.75],
            [0.5, 0.0, -0.75, 3.0],
        ],
        dtype=np.float32,
    )
    scipy_a = scipy_sparse.csr_matrix(dense)
    a = _to_ms(mx, scipy_a, fmt="csr")
    b_np = np.array([1.0, -2.0, 0.5, 1.5], dtype=np.float32)

    solver = linalg.factorized(a, method="ldlt")

    assert solver.backend == "accelerate"
    expected = scipy_linalg.spsolve(scipy_a, b_np)
    np.testing.assert_allclose(
        to_numpy(solver.solve(mx.array(b_np))),
        expected,
        rtol=3e-4,
        atol=3e-4,
    )


def test_accelerate_qr_rectangular_auto_matches_dense_lstsq(mx, scipy_sparse, to_numpy):
    _require_accelerate_solvers()
    scipy_a = _rectangular_matrix(scipy_sparse)
    a = _to_ms(mx, scipy_a, fmt="coo")
    b_np = np.column_stack(
        [
            np.linspace(0.0, 1.0, scipy_a.shape[0], dtype=np.float32),
            np.linspace(-0.5, 0.5, scipy_a.shape[0], dtype=np.float32),
        ]
    )

    solver = linalg.factorized(a)

    assert solver.backend == "accelerate"
    assert solver.method == "qr"
    assert solver.rhs_size == scipy_a.shape[0]
    assert solver.solution_size == scipy_a.shape[1]
    expected = np.linalg.lstsq(scipy_a.toarray(), b_np, rcond=None)[0]
    np.testing.assert_allclose(
        to_numpy(solver.solve(mx.array(b_np))),
        expected,
        rtol=4e-4,
        atol=4e-4,
    )


def test_accelerate_cholesky_ata_solves_normal_equations(mx, scipy_sparse, to_numpy):
    _require_accelerate_solvers()
    scipy_a = _rectangular_matrix(scipy_sparse)
    a = _to_ms(mx, scipy_a, fmt="csc")
    x_expected = np.array([0.5, -1.0, 1.5, 0.25], dtype=np.float32)
    normal = scipy_a.T @ scipy_a
    rhs_np = normal @ x_expected

    solver = linalg.factorized(a, method="cholesky_ata")

    assert solver.backend == "accelerate"
    assert solver.rhs_size == scipy_a.shape[1]
    assert solver.solution_size == scipy_a.shape[1]
    np.testing.assert_allclose(
        to_numpy(solver.solve(mx.array(rhs_np, dtype=mx.float32))),
        x_expected,
        rtol=5e-4,
        atol=5e-4,
    )


@pytest.mark.parametrize("dtype_attr", ["float16", "bfloat16"])
def test_accelerate_factorized_promotes_half_dtypes(
    mx, scipy_sparse, to_numpy, dtype_attr
):
    scipy_linalg = pytest.importorskip("scipy.sparse.linalg")
    _require_accelerate_solvers()
    scipy_a = _spd_matrix(scipy_sparse, n=6)
    dtype = getattr(mx, dtype_attr)
    a = _to_ms(mx, scipy_a, fmt="csr", dtype=dtype)
    b_np = np.linspace(1.0, 2.0, scipy_a.shape[0], dtype=np.float32)

    solver = linalg.factorized(a, method="cholesky")
    x = solver.solve(mx.array(b_np, dtype=dtype))

    assert solver.backend == "accelerate"
    assert x.dtype == mx.float32
    expected = scipy_linalg.spsolve(scipy_a, b_np)
    np.testing.assert_allclose(to_numpy(x), expected, rtol=2e-2, atol=2e-2)


def test_accelerate_direct_solvers_reject_complex_values(mx, scipy_sparse):
    _require_accelerate_solvers()
    scipy_a = _spd_matrix(scipy_sparse, n=4).astype(np.complex64)
    a = _to_ms(mx, scipy_a, fmt="csr", dtype=mx.complex64)

    with pytest.raises(TypeError, match="real float data"):
        linalg.factorized(a, method="cholesky")


def test_accelerate_singular_systems_raise_python_exceptions(mx, scipy_sparse):
    _require_accelerate_solvers()
    singular_spd = scipy_sparse.diags(
        [np.array([1.0, 0.0, 2.0], dtype=np.float32)],
        offsets=[0],
        format="csr",
    )
    a = _to_ms(mx, singular_spd, fmt="csr")

    with pytest.raises(
        RuntimeError,
        match="SparseFactorizationFailed|SparseMatrixIsSingular|singular|numerical",
    ):
        linalg.factorized(a, method="cholesky")


def test_accelerate_lu_singular_system_raises(mx, scipy_sparse):
    _require_accelerate_lu()
    singular = scipy_sparse.csr_matrix(
        np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    )
    a = _to_ms(mx, singular, fmt="csc")

    with pytest.raises(
        RuntimeError,
        match="SparseFactorizationFailed|SparseMatrixIsSingular|singular|numerical",
    ):
        linalg.spsolve(a, mx.ones((2,), dtype=mx.float32))


@pytest.mark.parametrize(
    "indices, indptr, message",
    [
        ([0, 5], [0, 2, 2], "out of bounds"),
        ([0, 1], [0, 2, 1], "monotonically"),
    ],
)
def test_accelerate_malformed_csr_inputs_are_validated(mx, indices, indptr, message):
    _require_accelerate_solvers()
    bad = ms.csr_array(
        (
            mx.array([1.0, 2.0], dtype=mx.float32),
            mx.array(indices, dtype=mx.int32),
            mx.array(indptr, dtype=mx.int32),
        ),
        shape=(2, 2),
        validate="metadata",
    )

    with pytest.raises(ValueError, match=message):
        linalg.factorized(bad, method="cholesky")


def test_spsolve_keeps_rectangular_inputs_out_of_direct_solve(mx, scipy_sparse):
    _require_accelerate_solvers()
    scipy_a = _rectangular_matrix(scipy_sparse)
    a = _to_ms(mx, scipy_a, fmt="csr")

    with pytest.raises(ValueError, match="square"):
        linalg.spsolve(a, mx.ones((scipy_a.shape[0],), dtype=mx.float32))
