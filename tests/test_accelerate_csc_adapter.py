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

import mlx.core as mx
import pytest

from mlx_sparse._ext_loader import extension

pytestmark = [pytest.mark.native, pytest.mark.accelerate]


def _require_ext():
    ext = extension()
    if ext is None:
        pytest.skip("native extension unavailable")
    return ext


def _array(values, dtype):
    return mx.array(values, dtype=dtype)


def _assert_summary(summary, *, shape, starts, rows, values):
    assert summary["n_rows"] == shape[0]
    assert summary["n_cols"] == shape[1]
    assert summary["nnz"] == len(values)
    assert list(summary["column_starts"]) == starts
    assert list(summary["row_indices"]) == rows
    assert list(summary["values"]) == pytest.approx(values)
    assert isinstance(summary["accelerate_framework"], bool)
    if summary["accelerate_framework"]:
        assert summary["accelerate_row_count"] == shape[0]
        assert summary["accelerate_column_count"] == shape[1]
        assert summary["accelerate_block_size"] == 1
        assert summary["accelerate_data_points_to_owned_values"] is True


def test_csc_adapter_preserves_canonical_int32_csc():
    ext = _require_ext()

    summary = ext._accelerate_csc_adapter_summary_for_testing(
        _array([2.0, 3.0, 5.0, 7.0], mx.float32),
        _array([0, 0, 2, 1], mx.int32),
        _array([0, 1, 3, 4], mx.int32),
        3,
        3,
        require_square=True,
    )

    _assert_summary(
        summary,
        shape=(3, 3),
        starts=[0, 1, 3, 4],
        rows=[0, 0, 2, 1],
        values=[2.0, 3.0, 5.0, 7.0],
    )


def test_csc_adapter_canonicalizes_unsorted_int64_csc_and_sums_duplicates():
    ext = _require_ext()

    summary = ext._accelerate_csc_adapter_summary_for_testing(
        _array([1.0, 2.0, 3.0, 4.0, 5.0], mx.float32),
        _array([2, 0, 2, 1, 1], mx.int64),
        _array([0, 3, 5], mx.int64),
        3,
        2,
    )

    _assert_summary(
        summary,
        shape=(3, 2),
        starts=[0, 2, 3],
        rows=[0, 2, 1],
        values=[2.0, 4.0, 9.0],
    )


def test_csc_adapter_can_require_canonical_csc_input():
    ext = _require_ext()

    with pytest.raises(ValueError, match="strictly sorted, duplicate-free"):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([1.0, 2.0, 3.0], mx.float32),
            _array([2, 0, 2], mx.int32),
            _array([0, 3], mx.int32),
            3,
            1,
            canonicalize=False,
        )


def test_csr_adapter_validates_and_converts_to_canonical_csc():
    ext = _require_ext()

    summary = ext._accelerate_csr_adapter_summary_for_testing(
        _array([1.0, 2.0, 3.0, 4.0, 5.0], mx.float32),
        _array([2, 0, 2, 1, 0], mx.int32),
        _array([0, 3, 4, 5], mx.int32),
        3,
        3,
        require_square=True,
    )

    _assert_summary(
        summary,
        shape=(3, 3),
        starts=[0, 2, 3, 4],
        rows=[0, 2, 1, 0],
        values=[2.0, 5.0, 4.0, 4.0],
    )


def test_coo_adapter_validates_and_converts_to_canonical_csc():
    ext = _require_ext()

    summary = ext._accelerate_coo_adapter_summary_for_testing(
        _array([1.0, 2.0, 3.0, 4.0], mx.float32),
        _array([1, 0, 1, 0], mx.int64),
        _array([1, 0, 1, 0], mx.int64),
        2,
        2,
        require_square=True,
    )

    _assert_summary(
        summary,
        shape=(2, 2),
        starts=[0, 1, 2],
        rows=[0, 1],
        values=[6.0, 4.0],
    )


def test_adapter_rejects_non_float32_values():
    ext = _require_ext()

    with pytest.raises(ValueError, match="dtype float32"):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([1.0], mx.float16),
            _array([0], mx.int32),
            _array([0, 1], mx.int32),
            1,
            1,
        )


def test_adapter_rejects_non_rank_one_buffers():
    ext = _require_ext()

    with pytest.raises(ValueError, match="rank-1"):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([[1.0]], mx.float32),
            _array([0], mx.int32),
            _array([0, 1], mx.int32),
            1,
            1,
        )


def test_adapter_rejects_mismatched_index_dtypes():
    ext = _require_ext()

    with pytest.raises(ValueError, match="same dtype"):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([1.0], mx.float32),
            _array([0], mx.int32),
            _array([0, 1], mx.int64),
            1,
            1,
        )


def test_adapter_rejects_non_integer_indices():
    ext = _require_ext()

    with pytest.raises(ValueError, match="dtype int32 or int64"):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([1.0], mx.float32),
            _array([0.0], mx.float32),
            _array([0.0, 1.0], mx.float32),
            1,
            1,
        )


@pytest.mark.parametrize(
    "n_rows,n_cols,error,pattern",
    [
        (-1, 1, ValueError, "non-negative"),
        (2**31, 1, OverflowError, "dimension range"),
    ],
)
def test_adapter_rejects_invalid_shape_dimensions(n_rows, n_cols, error, pattern):
    ext = _require_ext()

    with pytest.raises(error, match=pattern):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([], mx.float32),
            _array([], mx.int64),
            _array([0, 0], mx.int64),
            n_rows,
            n_cols,
            require_non_empty=False,
        )


def test_adapter_rejects_empty_shapes_by_default_but_can_allow_them():
    ext = _require_ext()

    with pytest.raises(ValueError, match="non-empty"):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([], mx.float32),
            _array([], mx.int32),
            _array([0], mx.int32),
            0,
            0,
        )

    summary = ext._accelerate_csc_adapter_summary_for_testing(
        _array([], mx.float32),
        _array([], mx.int32),
        _array([0], mx.int32),
        0,
        0,
        require_non_empty=False,
    )

    _assert_summary(summary, shape=(0, 0), starts=[0], rows=[], values=[])


def test_adapter_rejects_rectangular_shape_when_square_required():
    ext = _require_ext()

    with pytest.raises(ValueError, match="square"):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([], mx.float32),
            _array([], mx.int32),
            _array([0, 0, 0], mx.int32),
            3,
            2,
            require_square=True,
        )


@pytest.mark.parametrize(
    "indptr,n_cols,pattern",
    [
        ([0], 1, "size"),
        ([1, 1], 1, r"indptr\[0\]"),
        ([0, 0], 1, r"indptr\[-1\]"),
        ([0, 1, 0], 2, "monotonically"),
        ([0, 3], 1, "exceeds nnz"),
    ],
)
def test_csc_adapter_rejects_malformed_indptr(indptr, n_cols, pattern):
    ext = _require_ext()

    with pytest.raises(ValueError, match=pattern):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([1.0], mx.float32),
            _array([0], mx.int32),
            _array(indptr, mx.int32),
            1,
            n_cols,
            require_non_empty=False,
        )


@pytest.mark.parametrize(
    "indices,error,pattern",
    [
        ([-1], ValueError, "non-negative"),
        ([2], ValueError, "out of bounds"),
        ([2**31], OverflowError, "int index range"),
    ],
)
def test_csc_adapter_rejects_invalid_row_indices(indices, error, pattern):
    ext = _require_ext()

    with pytest.raises(error, match=pattern):
        ext._accelerate_csc_adapter_summary_for_testing(
            _array([1.0], mx.float32),
            _array(indices, mx.int64),
            _array([0, 1], mx.int64),
            2,
            1,
        )


def test_csr_adapter_rejects_invalid_column_indices_before_conversion():
    ext = _require_ext()

    with pytest.raises(ValueError, match="out of bounds"):
        ext._accelerate_csr_adapter_summary_for_testing(
            _array([1.0], mx.float32),
            _array([3], mx.int32),
            _array([0, 1], mx.int32),
            1,
            3,
        )


def test_coo_adapter_rejects_invalid_coordinates():
    ext = _require_ext()

    with pytest.raises(ValueError, match="column index"):
        ext._accelerate_coo_adapter_summary_for_testing(
            _array([1.0], mx.float32),
            _array([0], mx.int64),
            _array([-1], mx.int64),
            1,
            1,
        )
