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

import pytest

from mlx_sparse._ext_loader import extension

pytestmark = pytest.mark.native


def _require_ext():
    ext = extension()
    if ext is None:
        pytest.skip("native extension unavailable")
    return ext


@pytest.mark.parametrize(
    "family,status_code",
    [
        ("factorization", 0),
        ("sparse_blas", 0),
        ("iterative", 0),
    ],
)
def test_accelerate_success_statuses_do_not_raise(family, status_code):
    ext = _require_ext()

    assert (
        ext._accelerate_check_status_for_testing(
            family, status_code, "accelerate_unit_test"
        )
        is None
    )


@pytest.mark.parametrize(
    "family,status_code,expected_exception,status_name,description_fragment",
    [
        (
            "factorization",
            -1,
            RuntimeError,
            "SparseFactorizationFailed",
            "numerical issue",
        ),
        (
            "factorization",
            -2,
            RuntimeError,
            "SparseMatrixIsSingular",
            "singular",
        ),
        (
            "factorization",
            -3,
            RuntimeError,
            "SparseInternalError",
            "internal",
        ),
        (
            "factorization",
            -4,
            ValueError,
            "SparseParameterError",
            "parameter",
        ),
        (
            "sparse_blas",
            -1000,
            ValueError,
            "SPARSE_ILLEGAL_PARAMETER",
            "parameter",
        ),
        (
            "sparse_blas",
            -1001,
            RuntimeError,
            "SPARSE_CANNOT_SET_PROPERTY",
            "properties",
        ),
        (
            "sparse_blas",
            -1002,
            RuntimeError,
            "SPARSE_SYSTEM_ERROR",
            "system",
        ),
        (
            "iterative",
            1,
            RuntimeError,
            "SparseIterativeMaxIterations",
            "converge",
        ),
        (
            "iterative",
            -1,
            ValueError,
            "SparseIterativeParameterError",
            "parameter",
        ),
        (
            "iterative",
            -2,
            RuntimeError,
            "SparseIterativeIllConditioned",
            "ill-conditioned",
        ),
        (
            "iterative",
            -99,
            RuntimeError,
            "SparseIterativeInternalError",
            "internal",
        ),
    ],
)
def test_accelerate_statuses_map_to_python_exceptions(
    family,
    status_code,
    expected_exception,
    status_name,
    description_fragment,
):
    ext = _require_ext()

    with pytest.raises(expected_exception) as err:
        ext._accelerate_check_status_for_testing(family, status_code, "csr_spsolve")

    message = str(err.value)
    assert "csr_spsolve" in message
    assert "Accelerate" in message
    assert status_name in message
    assert f"({status_code})" in message
    assert description_fragment in message


def test_accelerate_status_mapper_preserves_reported_detail():
    ext = _require_ext()

    with pytest.raises(RuntimeError) as err:
        ext._accelerate_check_status_for_testing(
            "factorization",
            -1,
            "csr_spsolve",
            "reported pivot breakdown in column 7",
        )

    assert "reported pivot breakdown in column 7" in str(err.value)


def test_accelerate_unknown_statuses_fail_loudly():
    ext = _require_ext()

    with pytest.raises(RuntimeError) as err:
        ext._accelerate_check_status_for_testing("factorization", -12345, "csr_lu")

    message = str(err.value)
    assert "UnknownAccelerateStatus" in message
    assert "(-12345)" in message


def test_accelerate_unknown_status_family_is_value_error():
    ext = _require_ext()

    with pytest.raises(ValueError, match="unknown Accelerate status family"):
        ext._accelerate_check_status_for_testing("not_a_family", -1, "csr_lu")


def test_accelerate_status_name_helper_uses_canonical_names():
    ext = _require_ext()

    assert (
        ext._accelerate_status_name_for_testing("factorization", -2)
        == "SparseMatrixIsSingular"
    )
    assert (
        ext._accelerate_status_name_for_testing("sparse_blas", -1000)
        == "SPARSE_ILLEGAL_PARAMETER"
    )
    assert (
        ext._accelerate_status_name_for_testing("iterative", -99)
        == "SparseIterativeInternalError"
    )
