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
import mlx_sparse._native as native
from mlx_sparse._host import to_numpy


def _dense_from_csc(data_np, indices_np, indptr_np, shape):
    dense = np.zeros(shape, dtype=data_np.dtype)
    for col in range(shape[1]):
        start = int(indptr_np[col])
        end = int(indptr_np[col + 1])
        np.add.at(dense[:, col], indices_np[start:end], data_np[start:end])
    return dense


def _sample_csc(mx, dtype_name: str, index_dtype):
    shape = (5, 4)
    data = np.array(
        [2.0, -1.0, 0.5, 3.0, -4.0, 1.5, -2.0, 2.25, 5.0],
        dtype=np.float32,
    )
    if dtype_name == "complex64":
        data = data.astype(np.complex64) + 1j * np.array(
            [0.25, -0.5, 0.75, 0.0, -1.25, 0.5, -0.25, 1.5, 0.0],
            dtype=np.float32,
        )
    indices = np.array([0, 2, 2, 4, 1, 0, 3, 1, 4], dtype=index_dtype)
    indptr = np.array([0, 4, 5, 7, 9], dtype=index_dtype)
    dtype = getattr(mx, dtype_name)
    return ms.csc_array(
        (
            mx.array(data).astype(dtype),
            mx.array(indices),
            mx.array(indptr),
        ),
        shape=shape,
        sorted_indices=False,
        canonical=False,
    )


def test_csc_array_metadata_repr_validation_and_flags(mx):
    csc = _sample_csc(mx, "float32", np.int32)
    text = repr(csc)

    assert csc.shape == (5, 4)
    assert csc.nnz == 9
    assert csc.ndim == 2
    assert csc.dtype == mx.float32
    assert csc.index_dtype == mx.int32
    assert "CSCArray" in text
    assert "nnz=9" in text
    assert not csc.sorted_indices
    assert not csc.has_canonical_format
    assert ms.csc_array(csc, shape=csc.shape) is csc
    assert ms.issparse(csc)

    with pytest.raises(ValueError, match="shape mismatch"):
        ms.csc_array(csc, shape=(4, 5))
    with pytest.raises(ValueError, match="n_cols"):
        ms.csc_array(
            (
                mx.array(np.array([1.0], dtype=np.float32)),
                mx.array(np.array([0], dtype=np.int32)),
                mx.array(np.array([0, 1], dtype=np.int32)),
            ),
            shape=(2, 2),
        )
    with pytest.raises(ValueError, match="bounds"):
        ms.csc_array(
            (
                mx.array(np.array([1.0], dtype=np.float32)),
                mx.array(np.array([2], dtype=np.int32)),
                mx.array(np.array([0, 1, 1], dtype=np.int32)),
            ),
            shape=(2, 2),
            validate="full",
        )


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    ("dtype_name", "rtol", "atol"),
    [
        ("float32", 1e-5, 1e-5),
        ("float16", 6e-3, 6e-3),
        ("bfloat16", 4e-2, 4e-2),
        ("complex64", 1e-5, 1e-5),
    ],
)
def test_csc_todense_conversions_and_scipy_parity(
    mx, scipy_sparse, dtype_name, rtol, atol, index_dtype
):
    csc = _sample_csc(mx, dtype_name, index_dtype)
    data_np = to_numpy(csc.data)
    indices_np = to_numpy(csc.indices)
    indptr_np = to_numpy(csc.indptr)
    dense = _dense_from_csc(data_np, indices_np, indptr_np, csc.shape)
    scipy_csc = scipy_sparse.csc_matrix((data_np, indices_np, indptr_np), csc.shape)

    np.testing.assert_allclose(to_numpy(csc.todense()), dense, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        to_numpy(csc.todense()),
        scipy_csc.astype(np.complex64 if np.iscomplexobj(data_np) else np.float32)
        .toarray()
        .astype(dense.dtype),
        rtol=rtol,
        atol=atol,
    )

    csr = csc.tocsr()
    roundtrip = csr.tocsc()
    np.testing.assert_allclose(to_numpy(csr.todense()), dense, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        to_numpy(roundtrip.todense()), dense, rtol=rtol, atol=atol
    )

    # Build a real COO from SciPy to ensure COO -> CSC sorts by column then row.
    scipy_oracle_dtype = np.complex64 if np.iscomplexobj(data_np) else np.float32
    scipy_coo = scipy_csc.astype(scipy_oracle_dtype).tocoo()
    coo = ms.coo_array(
        (
            mx.array(scipy_coo.data).astype(csc.dtype),
            (
                mx.array(scipy_coo.row.astype(index_dtype)),
                mx.array(scipy_coo.col.astype(index_dtype)),
            ),
        ),
        shape=csc.shape,
    )
    from_coo = coo.tocsc(canonical=False)
    np.testing.assert_allclose(
        to_numpy(from_coo.todense()), dense, rtol=rtol, atol=atol
    )


def test_csc_sort_sum_duplicates_and_canonicalize(mx, scipy_sparse):
    data = mx.array(np.array([1.0, 4.0, 2.0, -1.0, 3.0], dtype=np.float32))
    indices = mx.array(np.array([2, 0, 0, 0, 1], dtype=np.int32))
    indptr = mx.array(np.array([0, 3, 5], dtype=np.int32))
    csc = ms.csc_array(
        (data, indices, indptr),
        shape=(3, 2),
        sorted_indices=False,
        canonical=False,
    )
    scipy_csc = scipy_sparse.csc_matrix(
        (to_numpy(data), to_numpy(indices), to_numpy(indptr)), shape=csc.shape
    )
    scipy_csc.sum_duplicates()
    scipy_csc.sort_indices()

    sorted_csc = csc.sort_indices()
    assert sorted_csc.sorted_indices
    assert not sorted_csc.has_canonical_format
    np.testing.assert_array_equal(to_numpy(sorted_csc.indices[:3]), [0, 0, 2])

    canonical = csc.canonicalize()
    assert canonical.sorted_indices
    assert canonical.has_canonical_format
    np.testing.assert_allclose(to_numpy(canonical.todense()), scipy_csc.toarray())
    assert canonical.canonicalize() is canonical
    assert canonical.sort_indices() is canonical


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_csc_empty_and_rectangular_conversions(mx, scipy_sparse, index_dtype):
    empty = ms.csc_array(
        (
            mx.array(np.array([], dtype=np.float32)),
            mx.array(np.array([], dtype=index_dtype)),
            mx.array(np.zeros(5, dtype=index_dtype)),
        ),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )
    assert empty.nnz == 0
    np.testing.assert_allclose(to_numpy(empty.todense()), np.zeros((3, 4)))
    np.testing.assert_allclose(to_numpy(empty.tocsr().todense()), np.zeros((3, 4)))

    scipy_rect = scipy_sparse.random(
        7,
        3,
        density=0.35,
        format="csc",
        dtype=np.float32,
        random_state=np.random.default_rng(42),
    )
    scipy_rect.sum_duplicates()
    scipy_rect.sort_indices()
    csc = ms.from_scipy(
        scipy_rect, format="csc", index_dtype=getattr(mx, index_dtype.__name__)
    )

    assert isinstance(csc, ms.CSCArray)
    assert csc.sorted_indices
    assert csc.has_canonical_format
    np.testing.assert_allclose(to_numpy(csc.todense()), scipy_rect.toarray())


@pytest.mark.parametrize(
    ("dtype_name", "rtol", "atol"),
    [
        ("float32", 1e-5, 1e-5),
        ("float16", 6e-3, 6e-3),
        ("bfloat16", 4e-2, 4e-2),
        ("complex64", 1e-5, 1e-5),
    ],
)
def test_native_csc_matvec_and_transpose_match_dense_and_scipy(
    mx, scipy_sparse, dtype_name, rtol, atol
):
    csc = _sample_csc(mx, dtype_name, np.int64).canonicalize()
    data_np = to_numpy(csc.data)
    indices_np = to_numpy(csc.indices)
    indptr_np = to_numpy(csc.indptr)
    scipy_csc = scipy_sparse.csc_matrix((data_np, indices_np, indptr_np), csc.shape)

    x_np = np.linspace(-1.5, 2.0, csc.shape[1], dtype=np.float32)
    xt_np = np.linspace(0.5, -1.0, csc.shape[0], dtype=np.float32)
    if dtype_name == "complex64":
        x_np = x_np.astype(np.complex64) + 0.25j
        xt_np = xt_np.astype(np.complex64) - 0.5j
    dtype = getattr(mx, dtype_name)
    x = mx.array(x_np).astype(dtype)
    xt = mx.array(xt_np).astype(dtype)

    y = ms.csc_matvec(csc, x)
    yt = ms.csc_matvec_transpose(csc, xt)
    y_operator = csc @ x

    dense = to_numpy(csc.todense())
    expected = scipy_csc @ x_np
    expected_t = scipy_csc.T @ xt_np

    np.testing.assert_allclose(to_numpy(y), dense @ to_numpy(x), rtol=rtol, atol=atol)
    np.testing.assert_allclose(to_numpy(y), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(to_numpy(y_operator), expected, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        to_numpy(yt), dense.T @ to_numpy(xt), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(to_numpy(yt), expected_t, rtol=rtol, atol=atol)


def test_csc_conjugate_transpose_and_unimplemented_matmul_paths(mx):
    csc = ms.csc_array(
        (
            mx.array(np.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(2, 2),
        sorted_indices=True,
        canonical=True,
    )
    np.testing.assert_allclose(
        to_numpy(csc.H.todense()), np.conj(to_numpy(csc.todense())).T
    )

    rhs = ms.eye(2, dtype=mx.complex64)
    with pytest.raises(NotImplementedError, match="CSC sparse-sparse matmul"):
        csc @ rhs
    with pytest.raises(NotImplementedError, match="CSC dense-matrix matmul"):
        csc @ mx.array(np.eye(2, dtype=np.complex64))
    with pytest.raises(TypeError, match="aslinearoperator"):
        ms.linalg.aslinearoperator(csc)


def test_csc_fallback_wrappers(monkeypatch, mx):
    monkeypatch.setattr(native, "extension", lambda: None)
    csc = _sample_csc(mx, "float32", np.int32)
    dense = to_numpy(csc.todense())
    x = mx.array(np.linspace(-1.0, 1.0, csc.shape[1], dtype=np.float32))
    xt = mx.array(np.linspace(2.0, -1.0, csc.shape[0], dtype=np.float32))

    np.testing.assert_allclose(
        to_numpy(native.csc_matvec(csc.data, csc.indices, csc.indptr, x, csc.shape)),
        dense @ to_numpy(x),
    )
    np.testing.assert_allclose(
        to_numpy(
            native.csc_matvec_transpose(
                csc.data, csc.indices, csc.indptr, xt, csc.shape
            )
        ),
        dense.T @ to_numpy(xt),
    )
