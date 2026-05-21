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

"""Coverage for _host.to_mx, _host.to_numpy bfloat16 path, and
remaining _validation.py branches."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms


class TestToNumpy:
    def test_float32(self):
        from mlx_sparse._host import to_numpy

        x = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_bfloat16_promoted_to_float32(self):
        from mlx_sparse._host import to_numpy

        x = mx.array([1.0, 2.0], dtype=mx.bfloat16)
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0], rtol=1e-3)

    def test_float16(self):
        from mlx_sparse._host import to_numpy

        x = mx.array([1.0, 2.0], dtype=mx.float16)
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)


class TestToMx:
    def test_no_dtype(self):
        from mlx_sparse._host import to_mx

        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = to_mx(arr)
        assert isinstance(result, mx.array)

    def test_with_dtype(self):
        from mlx_sparse._host import to_mx

        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = to_mx(arr, dtype=mx.float16)
        assert result.dtype == mx.float16


class TestValidateCsrValues:
    def test_indptr_not_starting_zero_raises(self):
        data = mx.array([1.0], dtype=mx.float32)
        indices = mx.array([0], dtype=mx.int32)
        indptr = mx.array([1, 2], dtype=mx.int32)  # indptr[0] != 0
        with pytest.raises(ValueError, match="indptr\\[0\\]"):
            ms.csr_array((data, indices, indptr), shape=(1, 2), validate="full")

    def test_indptr_last_not_nnz_raises(self):
        data = mx.array([1.0, 2.0], dtype=mx.float32)
        indices = mx.array([0, 1], dtype=mx.int32)
        indptr = mx.array([0, 1, 5], dtype=mx.int32)  # indptr[-1]=5 but nnz=2
        with pytest.raises(ValueError, match="indptr\\[-1\\]"):
            ms.csr_array((data, indices, indptr), shape=(2, 2), validate="full")

    def test_non_monotone_indptr_raises(self):
        # 3-row matrix: indptr[0]=0, indptr[-1]=2==nnz, but indptr[2]=1 < indptr[1]=2
        data = mx.array([1.0, 2.0], dtype=mx.float32)
        indices = mx.array([0, 1], dtype=mx.int32)
        indptr = mx.array([0, 2, 1, 2], dtype=mx.int32)
        with pytest.raises(ValueError, match="monotonically"):
            ms.csr_array((data, indices, indptr), shape=(3, 2), validate="full")

    def test_out_of_bounds_index_raises(self):
        data = mx.array([1.0, 2.0], dtype=mx.float32)
        indices = mx.array([0, 5], dtype=mx.int32)  # col=5 but n_cols=2
        indptr = mx.array([0, 1, 2], dtype=mx.int32)
        with pytest.raises(ValueError, match="in bounds"):
            ms.csr_array((data, indices, indptr), shape=(2, 2), validate="full")

    def test_empty_indices_skips_bounds_check(self):
        data = mx.array([], dtype=mx.float32)
        indices = mx.array([], dtype=mx.int32)
        indptr = mx.array([0, 0, 0], dtype=mx.int32)
        csr = ms.csr_array((data, indices, indptr), shape=(2, 2), validate="full")
        assert csr.nnz == 0

class TestValidateCooValues:
    def test_out_of_bounds_row_raises(self):
        data = mx.array([1.0], dtype=mx.float32)
        row = mx.array([5], dtype=mx.int32)  # row 5, but n_rows=2
        col = mx.array([0], dtype=mx.int32)
        with pytest.raises(ValueError, match="row coordinates"):
            ms.coo_array((data, (row, col)), shape=(2, 2), validate="full")

    def test_out_of_bounds_col_raises(self):
        data = mx.array([1.0], dtype=mx.float32)
        row = mx.array([0], dtype=mx.int32)
        col = mx.array([5], dtype=mx.int32)  # col 5, but n_cols=2
        with pytest.raises(ValueError, match="col coordinates"):
            ms.coo_array((data, (row, col)), shape=(2, 2), validate="full")

    def test_empty_coordinates_skips_bounds_check(self):
        data = mx.array([], dtype=mx.float32)
        row = mx.array([], dtype=mx.int32)
        col = mx.array([], dtype=mx.int32)
        coo = ms.coo_array((data, (row, col)), shape=(2, 2), validate="full")
        assert coo.nnz == 0


class TestNormalizeValidationMode:
    def test_true_becomes_full(self):
        from mlx_sparse._validation import normalize_validation_mode

        assert normalize_validation_mode(True) == "full"

    def test_false_becomes_none(self):
        from mlx_sparse._validation import normalize_validation_mode

        assert normalize_validation_mode(False) == "none"

    def test_metadata_passthrough(self):
        from mlx_sparse._validation import normalize_validation_mode

        assert normalize_validation_mode("metadata") == "metadata"

    def test_invalid_raises(self):
        from mlx_sparse._validation import normalize_validation_mode

        with pytest.raises(ValueError, match="validate must be"):
            normalize_validation_mode("invalid")


class TestValidateCsrMatmulInputs:
    def test_dtype_mismatch_raises(self):
        from mlx_sparse._validation import validate_csr_matmul_inputs

        data = mx.array([1.0], dtype=mx.float32)
        indices = mx.array([0], dtype=mx.int32)
        indptr = mx.array([0, 1, 1], dtype=mx.int32)
        rhs = mx.array([[1.0]], dtype=mx.float16)  # dtype mismatch
        with pytest.raises(TypeError, match="same dtype"):
            validate_csr_matmul_inputs(data, indices, indptr, rhs, (2, 1))

    def test_rhs_not_rank2_raises(self):
        from mlx_sparse._validation import validate_csr_matmul_inputs

        data = mx.array([1.0], dtype=mx.float32)
        indices = mx.array([0], dtype=mx.int32)
        indptr = mx.array([0, 1, 1], dtype=mx.int32)
        rhs = mx.array([1.0], dtype=mx.float32)  # rank-1 not rank-2
        with pytest.raises(ValueError, match="rank-2"):
            validate_csr_matmul_inputs(data, indices, indptr, rhs, (2, 1))

    def test_wrong_leading_dim_raises(self):
        from mlx_sparse._validation import validate_csr_matmul_inputs

        data = mx.array([1.0], dtype=mx.float32)
        indices = mx.array([0], dtype=mx.int32)
        indptr = mx.array([0, 1, 1], dtype=mx.int32)
        rhs = mx.array([[1.0, 2.0]], dtype=mx.float32)  # shape (1, 2) but n_cols=2
        # indptr says 2 rows, n_cols=2, rhs.shape[0]=1 != n_cols=2
        with pytest.raises(ValueError, match="leading dimension"):
            validate_csr_matmul_inputs(data, indices, indptr, rhs, (2, 2))


class TestCsrArrayFromExisting:
    def test_shape_mismatch_raises(self):
        csr = ms.csr_array(
            (
                mx.array([1.0], dtype=mx.float32),
                mx.array([0], dtype=mx.int32),
                mx.array([0, 1, 1], dtype=mx.int32),
            ),
            shape=(2, 2),
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            ms.csr_array(csr, shape=(3, 3))

    def test_passthrough_same_shape(self):
        csr = ms.csr_array(
            (
                mx.array([1.0], dtype=mx.float32),
                mx.array([0], dtype=mx.int32),
                mx.array([0, 1, 1], dtype=mx.int32),
            ),
            shape=(2, 2),
        )
        result = ms.csr_array(csr, shape=(2, 2))
        assert result is csr


class TestIssparse:
    def test_csr_is_sparse(self):
        csr = ms.csr_array(
            (
                mx.array([1.0], dtype=mx.float32),
                mx.array([0], dtype=mx.int32),
                mx.array([0, 1, 1], dtype=mx.int32),
            ),
            shape=(2, 2),
        )
        assert ms.issparse(csr)

    def test_coo_is_sparse(self):
        coo = ms.coo_array(
            (
                mx.array([1.0], dtype=mx.float32),
                (mx.array([0], dtype=mx.int32), mx.array([0], dtype=mx.int32)),
            ),
            shape=(2, 2),
        )
        assert ms.issparse(coo)

    def test_dense_is_not_sparse(self):
        assert not ms.issparse(mx.array([[1.0]]))

    def test_none_is_not_sparse(self):
        assert not ms.issparse(None)
