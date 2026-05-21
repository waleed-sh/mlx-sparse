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

import builtins
import sys

import numpy as np
import pytest
from conftest import to_numpy

import mlx_sparse as ms
from mlx_sparse import _ext_loader, _typing


def _complex_csr(mx):
    return ms.csr_array(
        (
            mx.array(np.array([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex64)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(2, 2),
        sorted_indices=True,
        canonical=True,
    )


def test_csr_idempotent_methods_and_conjugate_transpose(mx):
    csr = _complex_csr(mx)

    assert csr.sort_indices() is csr
    assert csr.canonicalize() is csr
    np.testing.assert_allclose(
        to_numpy(csr.conjugate().todense()), np.conj(to_numpy(csr.todense()))
    )
    np.testing.assert_allclose(
        to_numpy(csr.H.todense()), np.conj(to_numpy(csr.todense())).T
    )


def test_csr_and_coo_constructors_accept_existing_instances_and_reject_bad_args(mx):
    csr = ms.eye(2)
    assert ms.csr_array(csr, shape=(2, 2)) is csr
    assert ms.csr_array(
        (csr.data, csr.indices, csr.indptr),
        shape=csr.shape,
        canonical=True,
    ).sorted_indices

    with pytest.raises(ValueError, match="shape mismatch"):
        ms.csr_array(csr, shape=(3, 3))
    with pytest.raises(TypeError, match="csr_array expects"):
        ms.csr_array((csr.data, csr.indices), shape=(2, 2))

    coo = ms.coo_array(
        (
            mx.array(np.array([1.0], dtype=np.float32)),
            (
                mx.array(np.array([0], dtype=np.int32)),
                mx.array(np.array([1], dtype=np.int32)),
            ),
        ),
        shape=(2, 2),
        canonical=True,
    )
    assert coo.has_canonical_format
    assert ms.coo_array(coo, shape=(2, 2)) is coo
    with pytest.raises(ValueError, match="shape mismatch"):
        ms.coo_array(coo, shape=(3, 3))
    with pytest.raises(TypeError, match="coo_array expects"):
        ms.coo_array((coo.data, coo.row), shape=(2, 2))


def test_matmul_dispatch_and_operator_error_paths(mx):
    csr = ms.csr_array(
        (
            mx.array(np.array([2.0, -1.0], dtype=np.float32)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 2], dtype=np.int32)),
        ),
        shape=(1, 2),
        sorted_indices=True,
        canonical=True,
    )
    coo_rhs = ms.coo_array(
        (
            mx.array(np.array([3.0, 5.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1], dtype=np.int32)),
                mx.array(np.array([0, 0], dtype=np.int32)),
            ),
        ),
        shape=(2, 1),
    )

    sparse_out = csr @ coo_rhs
    assert isinstance(sparse_out, ms.CSRArray)
    np.testing.assert_allclose(to_numpy(sparse_out.todense()), [[1.0]])

    with pytest.raises(ValueError, match="rank-1 or higher"):
        csr @ mx.array(1.0)
    with pytest.raises(ValueError, match="rank-2"):
        ms.csr_matmul(csr, mx.array(np.array([1.0, 2.0], dtype=np.float32)))
    with pytest.raises(ValueError, match="sparse dimension"):
        ms.csr_matmul(csr, mx.array(np.ones((2, 3, 1), dtype=np.float32)))
    with pytest.raises(TypeError, match="csr_matvec expects"):
        ms.csr_matvec(object(), mx.array(np.array([1.0], dtype=np.float32)))
    with pytest.raises(TypeError, match="csr_matmul expects"):
        ms.csr_matmul(object(), mx.array(np.ones((1, 1), dtype=np.float32)))
    with pytest.raises(TypeError, match="csr_matmat expects CSRArray lhs"):
        ms.csr_matmat(object(), csr)
    with pytest.raises(TypeError, match="csr_matmat expects CSRArray rhs"):
        ms.csr_matmat(csr, object())
    with pytest.raises(TypeError, match="todense expects"):
        ms.todense(object())


def test_public_dtype_predicates_and_missing_extension_paths(monkeypatch, mx):
    assert _typing.is_index_dtype(mx.int32)
    assert not _typing.is_index_dtype(mx.float32)
    assert _typing.is_supported_value_dtype(mx.complex64)
    assert not _typing.is_supported_value_dtype(mx.int32)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx_sparse" and "_ext" in fromlist:
            raise ImportError("extension intentionally hidden")
        if name == "mlx_sparse._ext":
            raise ImportError("extension intentionally hidden")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("mlx_sparse._ext", None)

    assert _ext_loader.load_extension() is None
    assert not _typing.is_available()
