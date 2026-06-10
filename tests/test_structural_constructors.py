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


def _coo_base(mx, *, index_dtype=None):
    index_dtype = mx.int32 if index_dtype is None else index_dtype
    return ms.coo_array(
        (
            mx.array(np.array([1.0, -2.0, 3.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1, 1], dtype=np.int32), dtype=index_dtype),
                mx.array(np.array([1, 0, 2], dtype=np.int32), dtype=index_dtype),
            ),
        ),
        shape=(2, 3),
        canonical=True,
    )


def _csr_rhs(mx, *, index_dtype=None):
    index_dtype = mx.int32 if index_dtype is None else index_dtype
    return ms.csr_array(
        (
            mx.array(np.array([4.0, 5.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int32), dtype=index_dtype),
            mx.array(np.array([0, 1, 2], dtype=np.int32), dtype=index_dtype),
        ),
        shape=(2, 2),
        sorted_indices=True,
        canonical=True,
    )


def _expected_type(format_name):
    if format_name in (None, "coo"):
        return ms.COOArray
    if format_name == "csr":
        return ms.CSRArray
    if format_name == "csc":
        return ms.CSCArray
    raise AssertionError(format_name)


def _assert_canonical_compressed(array, to_numpy):
    assert isinstance(array, (ms.CSRArray, ms.CSCArray))
    assert array.sorted_indices
    assert array.has_canonical_format
    pointer = to_numpy(array.indptr)
    indices = to_numpy(array.indices)
    for segment in range(len(pointer) - 1):
        start = int(pointer[segment])
        end = int(pointer[segment + 1])
        assert np.all(indices[start:end][:-1] < indices[start:end][1:])


@pytest.mark.parametrize("out_format", [None, "coo", "csr", "csc"])
def test_block_array_mixed_inputs_none_and_formats(out_format, mx, to_numpy):
    left = _coo_base(mx, index_dtype=mx.int64)
    right = _csr_rhs(mx)
    dense_np = np.array([[0, 6, 0], [7, 0, 8]], dtype=np.int32)
    dense = mx.array(dense_np)

    out = ms.block_array(
        [
            [left, None],
            [dense, right.tocsc(canonical=True)],
        ],
        format=out_format,
    )

    expected = np.block(
        [
            [
                to_numpy(left.todense()),
                np.zeros((2, 2), dtype=np.float32),
            ],
            [
                dense_np.astype(np.float32),
                to_numpy(right.todense()),
            ],
        ]
    )
    assert isinstance(out, _expected_type(out_format))
    assert out.shape == expected.shape
    assert out.dtype == mx.float32
    assert out.index_dtype == mx.int64
    np.testing.assert_allclose(to_numpy(out.todense()), expected)
    if isinstance(out, (ms.CSRArray, ms.CSCArray)):
        _assert_canonical_compressed(out, to_numpy)


def test_block_array_all_none_and_empty_dimensions(mx, to_numpy):
    empty = ms.csr_array(
        (
            mx.array(np.array([], dtype=np.float32)),
            mx.array(np.array([], dtype=np.int32)),
            mx.array(np.array([0], dtype=np.int32)),
        ),
        shape=(0, 3),
        sorted_indices=True,
        canonical=True,
    )
    tall_empty = ms.csc_array(
        (
            mx.array(np.array([], dtype=np.complex64)),
            mx.array(np.array([], dtype=np.int32)),
            mx.array(np.array([0], dtype=np.int32)),
        ),
        shape=(2, 0),
        sorted_indices=True,
        canonical=True,
    )

    none_only = ms.block_array([[None]])
    assert none_only.shape == (0, 0)
    assert none_only.nnz == 0

    out = ms.block_array([[empty, None], [None, tall_empty]], dtype=mx.complex64)
    assert out.shape == (2, 3)
    assert out.dtype == mx.complex64
    assert out.nnz == 0
    np.testing.assert_allclose(to_numpy(out.todense()), np.zeros((2, 3), np.complex64))


def test_block_array_validation_errors_are_precise(mx):
    base = _coo_base(mx)
    with pytest.raises(ValueError, match="rectangular"):
        ms.block_array([[base], [base, base]])
    with pytest.raises(ValueError, match="heights"):
        ms.block_array([[base, ms.eye(3)]])
    with pytest.raises(ValueError, match="widths"):
        ms.block_array([[base], [ms.eye(2, 4)]])
    with pytest.raises(NotImplementedError, match="supported formats"):
        ms.block_array([[base]], format="bsr")


def test_block_diag_vstack_hstack_match_dense_reference(mx, to_numpy):
    a = _coo_base(mx)
    b = _csr_rhs(mx)
    dense = mx.array(np.array([[9, 0, -1]], dtype=np.int32))

    diag = ms.block_diag([a, b, dense], format="csr")
    expected_diag = np.zeros((5, 8), dtype=np.float32)
    expected_diag[:2, :3] = to_numpy(a.todense())
    expected_diag[2:4, 3:5] = to_numpy(b.todense())
    expected_diag[4:5, 5:8] = np.array([[9, 0, -1]], dtype=np.float32)
    assert isinstance(diag, ms.CSRArray)
    np.testing.assert_allclose(to_numpy(diag.todense()), expected_diag)

    v = ms.vstack([a, dense], format="csc")
    expected_v = np.vstack([to_numpy(a.todense()), np.array([[9, 0, -1]], np.float32)])
    assert isinstance(v, ms.CSCArray)
    np.testing.assert_allclose(to_numpy(v.todense()), expected_v)

    h = ms.hstack([b, mx.array(np.array([[0], [8]], dtype=np.float32))], format="csr")
    expected_h = np.hstack([to_numpy(b.todense()), np.array([[0], [8]], np.float32)])
    assert isinstance(h, ms.CSRArray)
    np.testing.assert_allclose(to_numpy(h.todense()), expected_h)


def test_stack_and_block_diag_validation(mx):
    with pytest.raises(ValueError, match="at least one"):
        ms.vstack([])
    with pytest.raises(ValueError, match="columns"):
        ms.vstack([ms.eye(2), ms.eye(2, 3)])
    with pytest.raises(ValueError, match="rows"):
        ms.hstack([ms.eye(2), ms.eye(3)])
    with pytest.raises(TypeError, match="None"):
        ms.block_diag([ms.eye(1), None])


@pytest.mark.parametrize("input_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("out_format", [None, "coo", "csr", "csc"])
@pytest.mark.parametrize(("fn_name", "k"), [("tril", 0), ("tril", -1), ("triu", 1)])
def test_tril_triu_all_formats_match_numpy(
    input_format, out_format, fn_name, k, mx, to_numpy
):
    dense_np = np.array(
        [
            [1, 2, 0, 4],
            [5, 6, 7, 0],
            [0, 8, 9, 10],
        ],
        dtype=np.float32,
    )
    base = ms.fromdense(mx.array(dense_np))
    if input_format == "coo":
        array = base.tocoo(canonical=True)
    elif input_format == "csr":
        array = base
    else:
        array = base.tocsc(canonical=True)

    out = getattr(ms, fn_name)(array, k=k, format=out_format)
    expected = np.triu(dense_np, k=k) if fn_name == "triu" else np.tril(dense_np, k=k)

    assert isinstance(out, _expected_type(out_format))
    assert out.shape == dense_np.shape
    np.testing.assert_allclose(to_numpy(out.todense()), expected)


def test_tril_triu_dense_complex_and_zero_nnz_inputs(mx, to_numpy):
    dense = mx.array(
        np.array(
            [[1 + 2j, 3 - 1j], [0 + 0j, -4 + 0.5j]],
            dtype=np.complex64,
        )
    )
    upper = ms.triu(dense, format="csc")
    assert isinstance(upper, ms.CSCArray)
    assert upper.dtype == mx.complex64
    np.testing.assert_allclose(to_numpy(upper.todense()), np.triu(to_numpy(dense)))

    zero = ms.coo_array(
        (
            mx.array(np.array([], dtype=np.float32)),
            (
                mx.array(np.array([], dtype=np.int64)),
                mx.array(np.array([], dtype=np.int64)),
            ),
        ),
        shape=(3, 2),
        canonical=True,
    )
    lower = ms.tril(zero, format="csr")
    assert lower.shape == (3, 2)
    assert lower.nnz == 0
    assert lower.index_dtype == mx.int64


def test_identity_square_alias_formats(mx, to_numpy):
    for out_format in [None, "csr", "coo", "csc"]:
        ident = ms.identity(
            4, dtype=mx.complex64, format=out_format, index_dtype=mx.int64
        )
        assert isinstance(
            ident,
            ms.CSRArray if out_format in (None, "csr") else _expected_type(out_format),
        )
        assert ident.shape == (4, 4)
        assert ident.dtype == mx.complex64
        assert ident.index_dtype == mx.int64
        np.testing.assert_allclose(
            to_numpy(ident.todense()), np.eye(4, dtype=np.complex64)
        )


def test_block_array_sparse_value_jvp_and_vjp(mx, to_numpy):
    row_a = mx.array(np.array([0, 1], dtype=np.int32))
    col_a = mx.array(np.array([0, 1], dtype=np.int32))
    row_b = mx.array(np.array([0, 0, 1], dtype=np.int32))
    col_b = mx.array(np.array([0, 2, 1], dtype=np.int32))
    data_a = mx.array(np.array([2.0, -3.0], dtype=np.float32))
    data_b = mx.array(np.array([5.0, 7.0, -11.0], dtype=np.float32))
    tangent_a = mx.array(np.array([0.5, 2.0], dtype=np.float32))
    tangent_b = mx.array(np.array([-1.0, 3.0, 4.0], dtype=np.float32))
    cotangent = mx.array(np.array([1.0, 2.0, -1.0, 0.25, 4.0], dtype=np.float32))

    def assembled_data(values_a, values_b):
        a = ms.coo_array((values_a, (row_a, col_a)), shape=(2, 2), canonical=True)
        b = ms.coo_array((values_b, (row_b, col_b)), shape=(2, 3), canonical=True)
        return ms.block_array([[a, b]], format="coo").data

    _, jvp = mx.jvp(assembled_data, (data_a, data_b), (tangent_a, tangent_b))
    _, vjp = mx.vjp(assembled_data, (data_a, data_b), (cotangent,))

    np.testing.assert_allclose(
        to_numpy(jvp[0]),
        np.concatenate([to_numpy(tangent_a), to_numpy(tangent_b)]),
    )
    np.testing.assert_allclose(to_numpy(vjp[0]), to_numpy(cotangent)[:2])
    np.testing.assert_allclose(to_numpy(vjp[1]), to_numpy(cotangent)[2:])
