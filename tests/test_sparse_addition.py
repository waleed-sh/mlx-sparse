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


def _csr_with_duplicates(mx, *, index_dtype=None):
    index_dtype = mx.int32 if index_dtype is None else index_dtype
    return ms.csr_array(
        (
            mx.array(np.array([5.0, 1.0, 2.0, 4.0, -4.0, 7.0], dtype=np.float32)),
            mx.array(np.array([2, 0, 0, 3, 3, 1]), dtype=index_dtype),
            mx.array(np.array([0, 3, 5, 6]), dtype=index_dtype),
        ),
        shape=(3, 4),
        sorted_indices=False,
        canonical=False,
    )


def _csr_rhs(mx, *, index_dtype=None):
    index_dtype = mx.int32 if index_dtype is None else index_dtype
    return ms.csr_array(
        (
            mx.array(np.array([-3.0, 6.0, 1.0, -7.0], dtype=np.float32)),
            mx.array(np.array([0, 1, 2, 1]), dtype=index_dtype),
            mx.array(np.array([0, 1, 3, 4]), dtype=index_dtype),
        ),
        shape=(3, 4),
        sorted_indices=True,
        canonical=True,
    )


def _assert_canonical_csr(array, to_numpy):
    assert isinstance(array, ms.CSRArray)
    assert array.sorted_indices
    assert array.has_canonical_format
    indptr = to_numpy(array.indptr)
    indices = to_numpy(array.indices)
    data = to_numpy(array.data)
    assert not np.any(data == 0)
    for row in range(array.shape[0]):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        segment = indices[start:end]
        assert np.all(segment[:-1] < segment[1:])


def test_csr_add_canonicalizes_duplicates_and_prunes_zero_cancellations(mx, to_numpy):
    lhs = _csr_with_duplicates(mx)
    rhs = _csr_rhs(mx)

    out = ms.add(lhs, rhs)

    _assert_canonical_csr(out, to_numpy)
    np.testing.assert_allclose(
        to_numpy(out.todense()),
        np.array(
            [
                [0.0, 0.0, 5.0, 0.0],
                [0.0, 6.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert out.nnz == 3


def test_csr_subtract_operator_matches_dense_reference(mx, to_numpy):
    lhs = _csr_with_duplicates(mx).canonicalize()
    rhs = _csr_rhs(mx)

    out = lhs - rhs

    _assert_canonical_csr(out, to_numpy)
    np.testing.assert_allclose(
        to_numpy(out.todense()), to_numpy(lhs.todense() - rhs.todense())
    )


def test_subtract_self_returns_structural_zero(mx, to_numpy):
    array = _csr_with_duplicates(mx).canonicalize()

    out = ms.subtract(array, array)

    _assert_canonical_csr(out, to_numpy)
    assert out.nnz == 0
    np.testing.assert_allclose(to_numpy(out.indptr), np.zeros(array.shape[0] + 1))


def test_add_and_subtract_module_functions_match_operators(mx, to_numpy):
    lhs = _csr_with_duplicates(mx).canonicalize()
    rhs = _csr_rhs(mx)

    np.testing.assert_allclose(
        to_numpy(ms.add(lhs, rhs).todense()), to_numpy((lhs + rhs).todense())
    )
    np.testing.assert_allclose(
        to_numpy(ms.subtract(lhs, rhs).todense()),
        to_numpy((lhs - rhs).todense()),
    )


def test_homogeneous_csc_addition_returns_canonical_csc(mx, to_numpy):
    lhs = _csr_with_duplicates(mx).tocsc(canonical=True)
    rhs = _csr_rhs(mx).tocsc(canonical=True)

    out = lhs + rhs

    assert isinstance(out, ms.CSCArray)
    assert out.sorted_indices
    assert out.has_canonical_format
    np.testing.assert_allclose(
        to_numpy(out.todense()), to_numpy(lhs.todense() + rhs.todense())
    )


def test_coo_and_mixed_format_addition_return_sparse_without_densifying(
    mx, to_numpy, monkeypatch
):
    lhs = _csr_with_duplicates(mx).canonicalize().tocsc(canonical=True)
    rhs = _csr_rhs(mx).canonicalize()
    coo_rhs = ms.coo_array(
        (
            rhs.data,
            (
                mx.array(np.array([0, 1, 1, 2], dtype=np.int32)),
                mx.array(np.array([0, 1, 2, 1], dtype=np.int32)),
            ),
        ),
        shape=rhs.shape,
        canonical=True,
    )

    def fail_todense(self):
        raise AssertionError("sparse addition should not densify operands")

    monkeypatch.setattr(ms.CSRArray, "todense", fail_todense)
    monkeypatch.setattr(ms.CSCArray, "todense", fail_todense)
    monkeypatch.setattr(ms.COOArray, "todense", fail_todense)

    out = ms.add(lhs, coo_rhs)

    assert isinstance(out, ms.CSRArray)
    _assert_canonical_csr(out, to_numpy)


@pytest.mark.parametrize("lhs_format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("rhs_format", ["csr", "csc", "coo"])
def test_all_sparse_format_pairs_match_dense_reference(
    lhs_format, rhs_format, mx, to_numpy
):
    base_lhs = _csr_with_duplicates(mx).canonicalize()
    base_rhs = _csr_rhs(mx)

    # Build COO through the public random-free constructor so the test uses
    # explicit coordinates and can verify the mixed-format conversion route.
    coo_lhs = ms.coo_array(
        (
            mx.array(np.array([3.0, 5.0, 6.0, 1.0, 7.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 0, 1, 1, 2], dtype=np.int32)),
                mx.array(np.array([0, 2, 1, 2, 1], dtype=np.int32)),
            ),
        ),
        shape=(3, 4),
        canonical=True,
    )
    coo_rhs = ms.coo_array(
        (
            mx.array(np.array([-3.0, 6.0, 1.0, -7.0], dtype=np.float32)),
            (
                mx.array(np.array([0, 1, 1, 2], dtype=np.int32)),
                mx.array(np.array([0, 1, 2, 1], dtype=np.int32)),
            ),
        ),
        shape=(3, 4),
        canonical=True,
    )
    choices_lhs = {
        "csr": base_lhs,
        "csc": base_lhs.tocsc(canonical=True),
        "coo": coo_lhs,
    }
    choices_rhs = {
        "csr": base_rhs,
        "csc": base_rhs.tocsc(canonical=True),
        "coo": coo_rhs,
    }
    lhs = choices_lhs[lhs_format]
    rhs = choices_rhs[rhs_format]

    out = lhs + rhs

    assert ms.issparse(out)
    if lhs_format == rhs_format == "csc":
        assert isinstance(out, ms.CSCArray)
    else:
        assert isinstance(out, ms.CSRArray)
        _assert_canonical_csr(out, to_numpy)
    np.testing.assert_allclose(
        to_numpy(out.todense()), to_numpy(lhs.todense() + rhs.todense())
    )


def test_index_dtype_promotes_to_int64_when_operands_differ(mx):
    lhs = _csr_with_duplicates(mx, index_dtype=mx.int64).canonicalize()
    rhs = _csr_rhs(mx, index_dtype=mx.int32)

    out = lhs + rhs

    assert out.index_dtype == mx.int64


def test_complex_sparse_addition_and_subtraction(mx, to_numpy):
    lhs = ms.csr_array(
        (
            mx.array(np.array([1.0 + 2.0j, 3.0 - 1.0j], dtype=np.complex64)),
            mx.array(np.array([0, 1], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(2, 2),
        sorted_indices=True,
        canonical=True,
    )
    rhs = ms.csr_array(
        (
            mx.array(np.array([-1.0 - 2.0j, 4.0 + 3.0j], dtype=np.complex64)),
            mx.array(np.array([0, 0], dtype=np.int32)),
            mx.array(np.array([0, 1, 2], dtype=np.int32)),
        ),
        shape=(2, 2),
        sorted_indices=True,
        canonical=True,
    )

    summed = lhs + rhs
    differenced = lhs - rhs

    _assert_canonical_csr(summed, to_numpy)
    _assert_canonical_csr(differenced, to_numpy)
    np.testing.assert_allclose(
        to_numpy(summed.todense()), to_numpy(lhs.todense() + rhs.todense())
    )
    np.testing.assert_allclose(
        to_numpy(differenced.todense()),
        to_numpy(lhs.todense() - rhs.todense()),
    )
    assert summed.nnz == 2


def test_sparse_addition_validation_errors_are_precise(mx):
    lhs = _csr_with_duplicates(mx).canonicalize()
    rhs = _csr_rhs(mx)
    bad_shape = ms.eye(2)
    bad_dtype = ms.csr_array(
        (
            rhs.data.astype(mx.float16),
            rhs.indices,
            rhs.indptr,
        ),
        shape=rhs.shape,
        sorted_indices=True,
        canonical=True,
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        ms.add(lhs, bad_shape)
    with pytest.raises(TypeError, match="matching value dtypes"):
        ms.add(lhs, bad_dtype)
    with pytest.raises(TypeError, match="Sparse\\+dense addition"):
        ms.add(lhs, mx.zeros(lhs.shape, dtype=lhs.dtype))
    with pytest.raises(NotImplementedError, match="nonzero scalar"):
        lhs + 1.0
    with pytest.raises(NotImplementedError, match="nonzero scalar"):
        2.0 - lhs


def test_zero_scalar_is_sparse_preserving(mx, to_numpy):
    lhs = _csr_with_duplicates(mx).canonicalize()

    assert (lhs + 0) is lhs
    assert (0 + lhs) is lhs
    assert (lhs - 0) is lhs
    negated = 0 - lhs

    assert isinstance(negated, ms.CSRArray)
    np.testing.assert_allclose(to_numpy(negated.todense()), -to_numpy(lhs.todense()))


def test_sparse_add_cpu_gpu_parity_when_gpu_available(mx, to_numpy):
    cpu = mx.Device(mx.cpu, 0)
    gpu = mx.Device(mx.gpu, 0)
    if not mx.is_available(gpu):
        pytest.skip("MLX GPU device is not available")

    mx.set_default_device(cpu)
    lhs_cpu = _csr_with_duplicates(mx).canonicalize()
    rhs_cpu = _csr_rhs(mx)
    out_cpu = lhs_cpu + rhs_cpu
    cpu_dense = to_numpy(out_cpu.todense())
    cpu_indices = to_numpy(out_cpu.indices)
    cpu_indptr = to_numpy(out_cpu.indptr)

    mx.set_default_device(gpu)
    lhs_gpu = _csr_with_duplicates(mx).canonicalize()
    rhs_gpu = _csr_rhs(mx)
    out_gpu = lhs_gpu + rhs_gpu
    mx.eval(out_gpu.data, out_gpu.indices, out_gpu.indptr)

    np.testing.assert_allclose(to_numpy(out_gpu.todense()), cpu_dense)
    np.testing.assert_array_equal(to_numpy(out_gpu.indices), cpu_indices)
    np.testing.assert_array_equal(to_numpy(out_gpu.indptr), cpu_indptr)
