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


def _base_a(mx, *, index_dtype=None):
    index_dtype = mx.int32 if index_dtype is None else index_dtype
    return ms.coo_array(
        (
            mx.array(np.array([2.0, -1.0, 4.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 0, 1], dtype=np.int32), dtype=index_dtype),
                mx.array(np.array([1, 2, 0], dtype=np.int32), dtype=index_dtype),
            ),
        ),
        shape=(2, 3),
        canonical=True,
    )


def _base_b(mx, *, index_dtype=None):
    index_dtype = mx.int32 if index_dtype is None else index_dtype
    return ms.coo_array(
        (
            mx.array(np.array([3.0, 5.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1], dtype=np.int32), dtype=index_dtype),
                mx.array(np.array([1, 0], dtype=np.int32), dtype=index_dtype),
            ),
        ),
        shape=(2, 2),
        canonical=True,
    )


def _as_format(array, format_name):
    if format_name == "coo":
        return array
    if format_name == "csr":
        return array.tocsr(canonical=True)
    if format_name == "csc":
        return array.tocsc(canonical=True)
    raise AssertionError(format_name)


def _expected_type(format_name):
    if format_name in (None, "coo"):
        return ms.COOArray
    if format_name == "csr":
        return ms.CSRArray
    if format_name == "csc":
        return ms.CSCArray
    raise AssertionError(format_name)


def _assert_canonical_compressed(array, to_numpy):
    if isinstance(array, ms.CSRArray):
        pointer = to_numpy(array.indptr)
        indices = to_numpy(array.indices)
    elif isinstance(array, ms.CSCArray):
        pointer = to_numpy(array.indptr)
        indices = to_numpy(array.indices)
    else:
        raise AssertionError(type(array))
    assert array.sorted_indices
    assert array.has_canonical_format
    for i in range(len(pointer) - 1):
        start = int(pointer[i])
        end = int(pointer[i + 1])
        segment = indices[start:end]
        assert np.all(segment[:-1] < segment[1:])


@pytest.mark.parametrize("lhs_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("rhs_format", ["coo", "csr", "csc"])
@pytest.mark.parametrize("out_format", [None, "coo", "csr", "csc"])
def test_kron_all_sparse_format_pairs_match_dense_reference(
    lhs_format,
    rhs_format,
    out_format,
    mx,
    to_numpy,
):
    lhs = _as_format(_base_a(mx), lhs_format)
    rhs = _as_format(_base_b(mx), rhs_format)

    out = ms.kron(lhs, rhs, format=out_format)

    assert isinstance(out, _expected_type(out_format))
    assert out.shape == (4, 6)
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        np.kron(to_numpy(lhs.todense()), to_numpy(rhs.todense())),
    )
    if isinstance(out, (ms.CSRArray, ms.CSCArray)):
        _assert_canonical_compressed(out, to_numpy)


def test_kron_coo_preserves_fixed_duplicate_topology_and_csr_sums_it(mx, to_numpy):
    lhs = ms.coo_array(
        (
            mx.array(np.array([1.0, 2.0, -3.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 0, 1], dtype=np.int32)),
                mx.array(np.array([1, 1, 0], dtype=np.int32)),
            ),
        ),
        shape=(2, 2),
        canonical=False,
    )
    rhs = _base_b(mx)

    raw = ms.kron(lhs, rhs, format="coo")
    compressed = ms.kron(lhs, rhs, format="csr")

    assert raw.nnz == lhs.nnz * rhs.nnz
    assert not raw.has_canonical_format
    _assert_canonical_compressed(compressed, to_numpy)
    np.testing.assert_allclose(
        to_numpy(raw.todense()),
        np.kron(to_numpy(lhs.todense()), to_numpy(rhs.todense())),
    )
    np.testing.assert_allclose(to_numpy(compressed.todense()), to_numpy(raw.todense()))


def test_kron_zero_nnz_rectangular_inputs_keep_shape_and_format(mx, to_numpy):
    lhs = ms.csr_array(
        (
            mx.array(np.array([], dtype=np.float32)),
            mx.array(np.array([], dtype=np.int32)),
            mx.array(np.array([0, 0, 0], dtype=np.int32)),
        ),
        shape=(2, 3),
        sorted_indices=True,
        canonical=True,
    )
    rhs = _base_b(mx).tocsc(canonical=True)

    out = ms.kron(lhs, rhs, format="csc")

    assert isinstance(out, ms.CSCArray)
    assert out.shape == (4, 6)
    assert out.nnz == 0
    assert out.has_canonical_format
    np.testing.assert_allclose(
        to_numpy(out.todense()), np.zeros((4, 6), dtype=np.float32)
    )


def test_kron_dense_inputs_use_native_fromdense_path(mx, to_numpy, monkeypatch):
    import mlx_sparse._construct as construct

    calls = []
    original_fromdense = construct.fromdense

    def tracking_fromdense(*args, **kwargs):
        calls.append(kwargs.get("dtype"))
        return original_fromdense(*args, **kwargs)

    monkeypatch.setattr(construct, "fromdense", tracking_fromdense)
    dense = mx.array(
        np.array([[0, 2, 0], [1, 0, 3]], dtype=np.int32),
    )
    rhs = _base_b(mx)

    out = ms.kron(dense, rhs, format="csr")

    assert calls == [mx.float32]
    assert isinstance(out, ms.CSRArray)
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        np.kron(to_numpy(dense).astype(np.float32), to_numpy(rhs.todense())),
    )


@pytest.mark.parametrize(
    ("lhs_dtype", "rhs_dtype", "expected_dtype"),
    [
        ("float16", "float32", "float32"),
        ("bfloat16", "float16", "float32"),
        ("complex64", "float32", "complex64"),
    ],
)
def test_kron_dtype_promotion(lhs_dtype, rhs_dtype, expected_dtype, mx):
    lhs = _base_a(mx)
    rhs = _base_b(mx)
    lhs = ms.coo_array(
        (lhs.data.astype(getattr(mx, lhs_dtype)), (lhs.row, lhs.col)),
        shape=lhs.shape,
        canonical=True,
    )
    rhs = ms.coo_array(
        (rhs.data.astype(getattr(mx, rhs_dtype)), (rhs.row, rhs.col)),
        shape=rhs.shape,
        canonical=True,
    )

    out = ms.kron(lhs, rhs)

    assert out.dtype == getattr(mx, expected_dtype)


def test_kron_complex_values_match_numpy(mx, to_numpy):
    lhs = ms.coo_array(
        (
            mx.array(np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)),
            (
                mx.array(np.array([0, 1], dtype=np.int32)),
                mx.array(np.array([0, 1], dtype=np.int32)),
            ),
        ),
        shape=(2, 2),
        canonical=True,
    )
    rhs = _base_b(mx)

    out = ms.kron(lhs, rhs, format="csr")

    assert out.dtype == mx.complex64
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        np.kron(to_numpy(lhs.todense()), to_numpy(rhs.todense())),
    )


def test_kronsum_matches_dense_reference_for_all_output_formats(mx, to_numpy):
    a = ms.coo_array(
        (
            mx.array(np.array([2.0, -1.0, 3.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 0, 1], dtype=np.int32)),
                mx.array(np.array([0, 1, 1], dtype=np.int32)),
            ),
        ),
        shape=(2, 2),
        canonical=True,
    )
    b = ms.csr_array(
        (
            mx.array(np.array([4.0, 5.0, -2.0], dtype=np.float32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
            mx.array(np.array([0, 1, 2, 3], dtype=np.int32)),
        ),
        shape=(3, 3),
        sorted_indices=True,
        canonical=True,
    )
    expected = np.kron(np.eye(3, dtype=np.float32), to_numpy(a.todense())) + np.kron(
        to_numpy(b.todense()),
        np.eye(2, dtype=np.float32),
    )

    for out_format in (None, "coo", "csr", "csc"):
        out = ms.kronsum(a, b, format=out_format)
        assert isinstance(out, _expected_type(out_format))
        np.testing.assert_allclose(to_numpy(out.todense()), expected)


def test_kron_rejects_unsupported_formats_rank_errors_and_overflow(mx):
    lhs = _base_a(mx)
    rhs = _base_b(mx)

    with pytest.raises(NotImplementedError, match="format='bsr'.*supported formats"):
        ms.kron(lhs, rhs, format="bsr")
    with pytest.raises(ValueError, match="rank-2"):
        ms.kron(mx.array(np.array([1.0, 2.0], dtype=np.float32)), rhs)
    with pytest.raises(ValueError, match="square"):
        ms.kronsum(lhs, rhs)

    empty = ms.coo_array(
        (
            mx.array(np.array([], dtype=np.float32)),
            (
                mx.array(np.array([], dtype=np.int64)),
                mx.array(np.array([], dtype=np.int64)),
            ),
        ),
        shape=(50_000, 1),
    )
    with pytest.raises(OverflowError, match="kron rows"):
        ms.kron(empty, empty)


def test_kron_fixed_topology_jvp_and_vjp_match_manual_product(mx, to_numpy):
    row_a = mx.array(np.array([0, 1], dtype=np.int32))
    col_a = mx.array(np.array([0, 1], dtype=np.int32))
    row_b = mx.array(np.array([0, 1, 1], dtype=np.int32))
    col_b = mx.array(np.array([1, 0, 1], dtype=np.int32))
    lhs_data = mx.array(np.array([2.0, -3.0], dtype=np.float32))
    rhs_data = mx.array(np.array([5.0, 7.0, -11.0], dtype=np.float32))
    lhs_tangent = mx.array(np.array([0.25, 1.5], dtype=np.float32))
    rhs_tangent = mx.array(np.array([-2.0, 3.0, 4.0], dtype=np.float32))
    cotangent = mx.array(np.array([1.0, 2.0, 3.0, -1.0, 0.5, 4.0], dtype=np.float32))

    def kron_data(left_values, right_values):
        left = ms.coo_array(
            (left_values, (row_a, col_a)),
            shape=(2, 2),
            canonical=True,
        )
        right = ms.coo_array(
            (right_values, (row_b, col_b)),
            shape=(2, 2),
            canonical=True,
        )
        return ms.kron(left, right, format="coo").data

    _, jvp = mx.jvp(
        kron_data,
        (lhs_data, rhs_data),
        (lhs_tangent, rhs_tangent),
    )
    _, vjp = mx.vjp(kron_data, (lhs_data, rhs_data), (cotangent,))

    lhs_np = to_numpy(lhs_data)
    rhs_np = to_numpy(rhs_data)
    lhs_tangent_np = to_numpy(lhs_tangent)
    rhs_tangent_np = to_numpy(rhs_tangent)
    cotangent_np = to_numpy(cotangent).reshape(lhs_np.size, rhs_np.size)
    expected_jvp = (
        lhs_tangent_np[:, None] * rhs_np[None, :]
        + lhs_np[:, None] * rhs_tangent_np[None, :]
    ).reshape(-1)
    expected_lhs_vjp = np.sum(cotangent_np * rhs_np[None, :], axis=1)
    expected_rhs_vjp = np.sum(cotangent_np * lhs_np[:, None], axis=0)

    np.testing.assert_allclose(to_numpy(jvp[0]), expected_jvp)
    np.testing.assert_allclose(to_numpy(vjp[0]), expected_lhs_vjp)
    np.testing.assert_allclose(to_numpy(vjp[1]), expected_rhs_vjp)


def test_compressed_tocoo_native_conversions_round_trip(mx, to_numpy):
    csr = _base_a(mx).tocsr(canonical=True)
    csc = _base_a(mx).tocsc(canonical=True)

    csr_coo = csr.tocoo(canonical=True)
    csc_coo = csc.tocoo(canonical=True)

    assert isinstance(csr_coo, ms.COOArray)
    assert isinstance(csc_coo, ms.COOArray)
    assert csr_coo.has_canonical_format
    assert csc_coo.has_canonical_format
    np.testing.assert_allclose(to_numpy(csr_coo.todense()), to_numpy(csr.todense()))
    np.testing.assert_allclose(to_numpy(csc_coo.todense()), to_numpy(csc.todense()))


@pytest.mark.gpu
def test_kron_gpu_matches_cpu_for_structure_and_values(mx, to_numpy):
    gpu_device = mx.Device(mx.gpu, 0)
    if not mx.is_available(gpu_device):
        pytest.skip("MLX GPU device is not available.")
    cpu_device = mx.Device(mx.cpu, 0)

    mx.set_default_device(cpu_device)
    cpu_out = ms.kron(
        _base_a(mx, index_dtype=mx.int64).tocsr(canonical=True),
        _base_b(mx).tocsc(canonical=True),
        format="csc",
    )
    cpu_dense = to_numpy(cpu_out.todense())
    cpu_indices = to_numpy(cpu_out.indices)
    cpu_indptr = to_numpy(cpu_out.indptr)

    mx.set_default_device(gpu_device)
    gpu_out = ms.kron(
        _base_a(mx, index_dtype=mx.int64).tocsr(canonical=True),
        _base_b(mx).tocsc(canonical=True),
        format="csc",
    )

    assert isinstance(gpu_out, ms.CSCArray)
    assert gpu_out.index_dtype == mx.int64
    np.testing.assert_allclose(to_numpy(gpu_out.todense()), cpu_dense)
    np.testing.assert_array_equal(to_numpy(gpu_out.indices), cpu_indices)
    np.testing.assert_array_equal(to_numpy(gpu_out.indptr), cpu_indptr)
