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

from types import SimpleNamespace

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse._ext_loader import extension_available
from mlx_sparse.linalg import preconditioners
from mlx_sparse.linalg._interface import LinearOperator, aslinearoperator
from mlx_sparse.linalg.utils import arrays as array_utils
from mlx_sparse.linalg.utils import factorization as factor_utils
from mlx_sparse.linalg.utils import sparse as sparse_utils


def _csr(mx, *, dtype=None, shape=(2, 2)):
    value_dtype = mx.float32 if dtype is None else dtype
    return ms.csr_array(
        (
            mx.array([2.0, 3.0], dtype=value_dtype),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=shape,
        canonical=True,
    )


def _coo(mx, *, dtype=None):
    value_dtype = mx.float32 if dtype is None else dtype
    return ms.coo_array(
        (
            mx.array([2.0, 3.0], dtype=value_dtype),
            (
                mx.array([0, 1], dtype=mx.int32),
                mx.array([0, 1], dtype=mx.int32),
            ),
        ),
        shape=(2, 2),
    )


def test_host_norm_and_finite_scalar_validation():
    assert array_utils.host_norm([3.0, 4.0]) == pytest.approx(5.0)
    assert array_utils.finite_scalar("omega", "2.5") == pytest.approx(2.5)

    with pytest.raises(ValueError, match="omega must be finite"):
        array_utils.finite_scalar("omega", np.inf)


def test_float32_vector_validation_and_promotion(mx):
    promoted = array_utils.ensure_float32_vector(
        "weights", mx.array([1.0, 2.0], dtype=mx.float16)
    )
    assert promoted.dtype == mx.float32

    with pytest.raises(ValueError, match="rank-1"):
        array_utils.ensure_float32_vector(
            "weights", mx.array([[1.0, 2.0]], dtype=mx.float32)
        )
    with pytest.raises(TypeError, match="real floating dtype"):
        array_utils.ensure_float32_vector(
            "weights", mx.array(np.array([1.0 + 0.0j], dtype=np.complex64))
        )
    with pytest.raises(ValueError, match="finite"):
        array_utils.ensure_float32_vector(
            "weights",
            mx.array([1.0, np.nan], dtype=mx.float32),
            require_finite=True,
        )


def test_float32_sparse_promotes_public_formats_and_rejects_complex(mx):
    csr = _csr(mx, dtype=mx.float16)
    csc = _coo(mx, dtype=mx.bfloat16).tocsc(canonical=True)
    coo = _coo(mx, dtype=mx.float16)

    assert (
        array_utils.ensure_float32_sparse(csr, context="test").data.dtype == mx.float32
    )
    assert (
        array_utils.ensure_float32_sparse(csc, context="test").data.dtype == mx.float32
    )
    assert (
        array_utils.ensure_float32_sparse(coo, context="test").data.dtype == mx.float32
    )

    complex_csr = ms.csr_array(
        (
            mx.array(np.array([1.0 + 0.0j], dtype=np.complex64)),
            mx.array([0], dtype=mx.int32),
            mx.array([0, 1, 1], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    with pytest.raises(TypeError, match="real float"):
        array_utils.ensure_float32_sparse(complex_csr, context="test")


def test_rhs_validation_contracts(mx):
    vector = array_utils.ensure_rank1_or_rank2_rhs(
        mx.array([1.0, 2.0], dtype=mx.float16), leading_dim=2
    )
    assert vector.dtype == mx.float32

    matrix = array_utils.ensure_rank1_or_rank2_rhs(
        mx.array([[1.0], [2.0]], dtype=mx.float32), leading_dim=2, dtype=None
    )
    assert matrix.shape == (2, 1)

    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        array_utils.ensure_rank1_or_rank2_rhs(mx.array(1.0), leading_dim=2)
    with pytest.raises(ValueError, match="leading dimension"):
        array_utils.ensure_rank1_or_rank2_rhs(
            mx.array([1.0, 2.0, 3.0], dtype=mx.float32), leading_dim=2
        )
    with pytest.raises(ValueError, match="at least one column"):
        array_utils.ensure_rank1_or_rank2_rhs(
            mx.ones((2, 0), dtype=mx.float32), leading_dim=2
        )
    with pytest.raises(TypeError, match="real floating dtype"):
        array_utils.ensure_rank1_or_rank2_rhs(
            mx.array(np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64)),
            leading_dim=2,
        )
    with pytest.raises(ValueError, match="finite"):
        array_utils.ensure_rank1_or_rank2_rhs(
            mx.array([1.0, np.inf], dtype=mx.float32),
            leading_dim=2,
            require_finite=True,
        )


def test_sparse_shape_and_input_normalization_contracts(mx):
    csr = _csr(mx)
    assert sparse_utils.square_shape(3) == (3, 3)
    assert sparse_utils.as_sparse(csr, context="ctx", dense_guidance="") is csr
    assert sparse_utils.as_csr(csr, context="ctx") is csr

    with pytest.raises(ValueError, match="square shape"):
        sparse_utils.square_shape((2, 3))
    with pytest.raises(TypeError, match="ctx expects CSRArray"):
        sparse_utils.as_sparse(object(), context="ctx", dense_guidance="dense hint")
    with pytest.raises(TypeError, match="ctx expected CSRArray"):
        sparse_utils.as_csr(object(), context="ctx", dense_guidance="dense hint")


def test_canonical_csr_linear_operator_branches(mx):
    csr = _csr(mx)
    sparse_op = aslinearoperator(csr)
    canonical = sparse_utils.canonical_csr(
        sparse_op,
        context="solver",
        dense_guidance="",
        allow_sparse_linear_operator=True,
    )
    assert canonical.has_canonical_format

    matrix_free = LinearOperator((2, 2), matvec=lambda x: x)
    with pytest.raises(TypeError, match="fully matrix-free"):
        sparse_utils.canonical_csr(
            matrix_free,
            context="solver",
            dense_guidance="",
            allow_sparse_linear_operator=True,
        )
    with pytest.raises(TypeError, match="sparse-backed LinearOperator"):
        sparse_utils.canonical_csr(
            object(),
            context="solver",
            dense_guidance="dense hint",
            allow_sparse_linear_operator=True,
        )


def test_preconditioner_object_validation_branches(mx, to_numpy):
    identity = preconditioners.identity((2, 2))
    assert identity.setup_info["kind"] == "identity"
    np.testing.assert_allclose(
        to_numpy(identity(mx.array([1.0, 2.0], dtype=mx.float32))), [1.0, 2.0]
    )

    with pytest.raises(ValueError, match="inverse_diagonal has length"):
        preconditioners.DiagonalPreconditioner(
            mx.array([1.0], dtype=mx.float32), shape=(2, 2)
        )
    with pytest.raises(ValueError, match="A is required"):
        preconditioners.aspreconditioner(None)
    with pytest.raises(ValueError, match="does not match"):
        preconditioners.aspreconditioner(identity, (3, 3))
    with pytest.raises(TypeError, match="not inverse-apply"):
        preconditioners.aspreconditioner(_csr(mx), (2, 2))


def test_diagonal_constructor_validation_branches(mx):
    with pytest.raises(TypeError, match="float32"):
        preconditioners.diagonal(
            mx.array([1.0, 2.0], dtype=mx.float32), dtype=mx.float16
        )
    with pytest.raises(ValueError, match="expected 3"):
        preconditioners.diagonal(mx.array([1.0, 2.0], dtype=mx.float32), shape=(3, 3))
    with pytest.raises(ValueError, match="zero_atol"):
        preconditioners.diagonal(mx.array([1.0, 2.0], dtype=mx.float32), zero_atol=-1.0)
    with pytest.raises(ValueError, match="zero or near-zero"):
        preconditioners.diagonal(
            mx.array([1.0e-8, 2.0], dtype=mx.float32), zero_atol=1.0e-7
        )


def test_diagonal_matvec_alias_uses_same_native_apply(mx, to_numpy):
    if not extension_available():
        pytest.skip("native extension unavailable")
    diagonal = preconditioners.diagonal(mx.array([2.0, 4.0], dtype=mx.float32))
    rhs = mx.array([2.0, 8.0], dtype=mx.float32)

    assert diagonal.dtype == mx.float32
    np.testing.assert_allclose(
        to_numpy(diagonal.matvec(rhs)), to_numpy(diagonal.solve(rhs)), rtol=1e-6
    )


def test_jacobi_validation_branches(mx):
    with pytest.raises(ValueError, match="zero_policy"):
        preconditioners.jacobi(_csr(mx), zero_policy="ignore")
    with pytest.raises(ValueError, match="square matrix"):
        preconditioners.jacobi(_csr(mx, shape=(2, 3)))
    with pytest.raises(ValueError, match="zero_atol"):
        preconditioners.jacobi(_csr(mx), zero_atol=-1.0)

    nan_diag = ms.csr_array(
        (
            mx.array([np.nan, 2.0], dtype=mx.float32),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    with pytest.raises(ValueError, match="finite"):
        preconditioners.jacobi(nan_diag)


def test_factorization_method_and_accelerate_decision_helpers(mx, monkeypatch):
    assert factor_utils.normalize_factorized_method("chol") == "cholesky"
    assert factor_utils.normalize_factorized_method("least-squares") == "qr"
    with pytest.raises(ValueError, match="factorized method"):
        factor_utils.normalize_factorized_method("bad")

    assert factor_utils.auto_factorized_method(SimpleNamespace(shape=(3, 3))) == "lu"
    assert factor_utils.auto_factorized_method(SimpleNamespace(shape=(4, 3))) == "qr"

    monkeypatch.setattr(
        factor_utils._native, "accelerate_solvers_available", lambda: False
    )
    assert factor_utils.accelerate_method_available("qr") is False

    monkeypatch.setattr(
        factor_utils._native, "accelerate_solvers_available", lambda: True
    )
    monkeypatch.setattr(
        factor_utils._native, "accelerate_lu_solvers_available", lambda: False
    )
    assert factor_utils.accelerate_method_available("lu") is False
    assert factor_utils.accelerate_method_available("qr") is True

    monkeypatch.setattr(
        factor_utils, "accelerate_method_available", lambda method: True
    )
    assert factor_utils.should_use_accelerate(_csr(mx), "lu") is True
    complex_csr = ms.csr_array(
        (
            mx.array(np.array([1.0 + 0.0j], dtype=np.complex64)),
            mx.array([0], dtype=mx.int32),
            mx.array([0, 1, 1], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )
    assert factor_utils.should_use_accelerate(complex_csr, "lu") is False


def test_accelerate_factorize_sparse_dispatches_by_format(mx, monkeypatch):
    calls = []

    def fake_factorize(*args):
        calls.append(args[-2:])
        return SimpleNamespace(rhs_size=2, solution_size=2)

    monkeypatch.setattr(
        factor_utils._native, "accelerate_factorize_csr_float32", fake_factorize
    )
    monkeypatch.setattr(
        factor_utils._native, "accelerate_factorize_csc_float32", fake_factorize
    )
    monkeypatch.setattr(
        factor_utils._native, "accelerate_factorize_coo_float32", fake_factorize
    )

    for sparse in (
        _csr(mx, dtype=mx.float16),
        _coo(mx).tocsc(canonical=True),
        _coo(mx),
    ):
        promoted, solver = factor_utils.accelerate_factorize_sparse(sparse, "lu")
        assert promoted.data.dtype == mx.float32
        assert solver.rhs_size == 2

    assert calls == [((2, 2), "lu"), ((2, 2), "lu"), ((2, 2), "lu")]


def test_factor_rhs_and_native_factorized_solve_validation(mx):
    class RecordingSolver:
        def __init__(self):
            self.last_rhs = None

        def solve(self, rhs):
            self.last_rhs = rhs
            return rhs + 1.0

    solver = RecordingSolver()
    wrapper = factor_utils.NativeFactorizedSolve(solver, rhs_size=2)
    out = wrapper.solve(mx.array([1.0, 2.0], dtype=mx.float32))
    assert out.shape == (2,)
    assert solver.last_rhs.shape == (2,)

    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        factor_utils.ensure_factor_rhs(mx.array(1.0), leading_dim=2)
    with pytest.raises(ValueError, match=r"expected \(2,\)"):
        factor_utils.ensure_factor_rhs(mx.array([1.0], dtype=mx.float32), leading_dim=2)
    with pytest.raises(ValueError, match="first dimension"):
        factor_utils.ensure_factor_rhs(mx.ones((1, 2), dtype=mx.float32), leading_dim=2)
    with pytest.raises(ValueError, match="at least one column"):
        factor_utils.ensure_factor_rhs(mx.ones((2, 0), dtype=mx.float32), leading_dim=2)


def test_accelerate_residual_validation_errors(mx):
    with pytest.raises(RuntimeError, match="non-finite"):
        factor_utils.check_accelerate_direct_residual(
            _csr(mx),
            mx.array([np.nan, 0.0], dtype=mx.float32),
            mx.array([1.0, 1.0], dtype=mx.float32),
        )

    if not extension_available():
        pytest.skip("native extension unavailable")
    with pytest.raises(RuntimeError, match="residual is too large"):
        factor_utils.check_accelerate_direct_residual(
            _csr(mx),
            mx.zeros((2,), dtype=mx.float32),
            mx.ones((2,), dtype=mx.float32),
        )


def test_solve_accelerate_spsolve_checked_vector_and_matrix_paths(mx, monkeypatch):
    calls = []

    def fake_check(A, x, rhs):
        calls.append((x.shape, rhs.shape))

    class EchoSolver:
        rhs_size = 2
        solution_size = 2

        def solve(self, rhs):
            return rhs

    monkeypatch.setattr(factor_utils, "check_accelerate_direct_residual", fake_check)
    singularity_checker = lambda A: object()

    vector = factor_utils.solve_accelerate_spsolve_checked(
        _csr(mx),
        EchoSolver(),
        mx.array([1.0, 2.0], dtype=mx.float32),
        singularity_checker=singularity_checker,
    )
    matrix = factor_utils.solve_accelerate_spsolve_checked(
        _csr(mx),
        EchoSolver(),
        mx.ones((2, 2), dtype=mx.float32),
        singularity_checker=singularity_checker,
    )

    assert vector.shape == (2,)
    assert matrix.shape == (2, 2)
    assert calls == [
        ((2,), (2,)),
        ((2,), (2,)),
        ((2, 2), (2, 2)),
        ((2,), (2,)),
    ]

    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        factor_utils.solve_accelerate_spsolve_checked(
            _csr(mx),
            EchoSolver(),
            mx.array(1.0),
            singularity_checker=singularity_checker,
        )


def test_solve_accelerate_spsolve_checked_preserves_probe_failure(mx, monkeypatch):
    call_count = {"count": 0}
    original = RuntimeError("probe residual failed")

    def fake_check(A, x, rhs):
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise original

    class EchoSolver:
        rhs_size = 2
        solution_size = 2

        def solve(self, rhs):
            return rhs

    def singularity_checker(A):
        raise RuntimeError("native singularity check failed")

    monkeypatch.setattr(factor_utils, "check_accelerate_direct_residual", fake_check)

    with pytest.raises(RuntimeError, match="probe residual failed"):
        factor_utils.solve_accelerate_spsolve_checked(
            _csr(mx),
            EchoSolver(),
            mx.array([1.0, 2.0], dtype=mx.float32),
            singularity_checker=singularity_checker,
        )
