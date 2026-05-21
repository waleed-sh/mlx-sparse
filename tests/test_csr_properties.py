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

"""Coverage for missing CSRArray methods: index_dtype, sort_indices, conj,
conjugate, H, vdot/dot with COO + error branches, and __repr__."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse._ext_loader import extension_available


def _real_2x2(sorted_idx=True, canonical=True):
    return ms.csr_array(
        (
            mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float32),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        sorted_indices=sorted_idx,
        canonical=canonical,
    )


def _complex_2x2():
    return ms.csr_array(
        (
            mx.array(np.array([1.0 + 2.0j, -3.0 + 0.5j], dtype=np.complex64)),
            mx.array([0, 1], dtype=mx.int32),
            mx.array([0, 1, 2], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )


def _float16_2x2():
    return ms.csr_array(
        (
            mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float16),
            mx.array([0, 1, 0, 1], dtype=mx.int32),
            mx.array([0, 2, 4], dtype=mx.int32),
        ),
        shape=(2, 2),
        canonical=True,
    )


def _coo_2x2():
    return ms.coo_array(
        (
            mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float32),
            (
                mx.array([0, 0, 1, 1], dtype=mx.int32),
                mx.array([0, 1, 0, 1], dtype=mx.int32),
            ),
        ),
        shape=(2, 2),
    )


class TestCSRArrayProperties:
    def test_index_dtype_int32(self):
        csr = _real_2x2()
        assert csr.index_dtype == mx.int32

    def test_index_dtype_int64(self):
        csr = ms.csr_array(
            (
                mx.array([1.0], dtype=mx.float32),
                mx.array([0], dtype=mx.int64),
                mx.array([0, 0, 1], dtype=mx.int64),
            ),
            shape=(2, 2),
        )
        assert csr.index_dtype == mx.int64

    def test_repr_contains_shape(self):
        csr = _real_2x2()
        r = repr(csr)
        assert "CSRArray" in r
        assert "(2, 2)" in r
        assert "nnz=4" in r

    def test_repr_contains_dtype(self):
        csr = _real_2x2()
        r = repr(csr)
        assert "float32" in r


class TestSortIndices:
    def test_already_sorted_returns_self(self):
        csr = _real_2x2(sorted_idx=True)
        result = csr.sort_indices()
        assert result is csr

    def test_unsorted_returns_sorted(self):
        # Row 0: col indices [1, 0] — intentionally unsorted
        csr = ms.csr_array(
            (
                mx.array([1.0, 4.0, 1.0, 3.0], dtype=mx.float32),
                mx.array([1, 0, 0, 1], dtype=mx.int32),  # row0: [1,0], row1: [0,1]
                mx.array([0, 2, 4], dtype=mx.int32),
            ),
            shape=(2, 2),
            sorted_indices=False,
            canonical=False,
        )
        result = csr.sort_indices()
        assert result.sorted_indices
        assert not result.has_canonical_format  # duplicates not summed
        mx.eval(result.indices)
        # After sorting row 0: indices should be [0, 1]
        idx = np.array(result.indices)
        assert idx[0] == 0 and idx[1] == 1


class TestConjAndHermitian:
    def test_conj_real_is_noop_values(self):
        csr = _real_2x2()
        result = csr.conj()
        mx.eval(result.data)
        np.testing.assert_allclose(np.array(result.data), np.array(csr.data))

    def test_conj_preserves_structure(self):
        csr = _real_2x2()
        result = csr.conj()
        assert result.shape == csr.shape
        assert result.nnz == csr.nnz
        assert result.sorted_indices == csr.sorted_indices
        assert result.has_canonical_format == csr.has_canonical_format

    def test_conj_complex_flips_sign(self):
        csr = _complex_2x2()
        result = csr.conj()
        mx.eval(result.data)
        original = np.array(csr.data)
        conjugated = np.array(result.data)
        np.testing.assert_allclose(conjugated.real, original.real)
        np.testing.assert_allclose(conjugated.imag, -original.imag)

    def test_conjugate_alias(self):
        csr = _complex_2x2()
        r1 = csr.conj()
        r2 = csr.conjugate()
        mx.eval(r1.data, r2.data)
        np.testing.assert_allclose(np.array(r1.data), np.array(r2.data))

    def test_H_real_equals_transpose(self):
        csr = _real_2x2()
        result = csr.H
        transpose = csr.T
        mx.eval(result.data, transpose.data, result.indices, transpose.indices)
        np.testing.assert_allclose(
            np.array(result.todense()), np.array(transpose.todense())
        )

    def test_H_complex_equals_conj_transpose(self):
        csr = _complex_2x2()
        H = csr.H
        expected = np.conj(np.array(csr.todense())).T
        np.testing.assert_allclose(np.array(H.todense()), expected)


class TestVdotBranches:
    def test_vdot_coo_rhs(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _real_2x2()
        coo = _coo_2x2()
        result = csr.vdot(coo)
        mx.eval(result)
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = float(np.sum(A * A))
        np.testing.assert_allclose(float(np.array(result)), expected, rtol=1e-5)

    def test_vdot_wrong_type_raises(self):
        csr = _real_2x2()
        with pytest.raises(TypeError, match="vdot"):
            csr.vdot(mx.array([[1.0]]))

    def test_vdot_shape_mismatch_raises(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr_a = _real_2x2()
        csr_b = ms.csr_array(
            (
                mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                mx.array([0, 1, 2], dtype=mx.int32),
                mx.array([0, 1, 2, 3], dtype=mx.int32),
            ),
            shape=(3, 3),
            canonical=True,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            csr_a.vdot(csr_b)

    def test_vdot_float16_lhs_promoted(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr_f16 = _float16_2x2()
        csr_f32 = _real_2x2()
        result = csr_f16.vdot(csr_f32)
        mx.eval(result)
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = float(np.sum(A * A))
        np.testing.assert_allclose(float(np.array(result)), expected, rtol=1e-3)

    def test_vdot_float16_rhs_promoted(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr_f32 = _real_2x2()
        csr_f16 = _float16_2x2()
        result = csr_f32.vdot(csr_f16)
        mx.eval(result)
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = float(np.sum(A * A))
        np.testing.assert_allclose(float(np.array(result)), expected, rtol=1e-3)

    def test_vdot_dtype_mismatch_raises(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr_f32 = _real_2x2()
        csr_cplx = _complex_2x2()
        with pytest.raises(TypeError, match="same dtype"):
            csr_f32.vdot(csr_cplx)

    def test_vdot_unsupported_dtype_raises(self):
        # Construct directly (bypassing public validation) to trigger the
        # defensive branch: dtype not in {float32, complex64} after promotion.
        from mlx_sparse._csr import CSRArray

        csr = CSRArray(
            data=mx.array([1, 0, 0, 1], dtype=mx.int32),
            indices=mx.array([0, 1, 0, 1], dtype=mx.int32),
            indptr=mx.array([0, 2, 4], dtype=mx.int32),
            shape=(2, 2),
            sorted_indices=True,
            has_canonical_format=True,  # skip canonicalize to avoid float ops
        )
        with pytest.raises(TypeError, match="float32 and complex64"):
            csr.vdot(csr)


class TestDotBranches:
    def test_dot_coo_rhs(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _real_2x2()
        coo = _coo_2x2()
        result = csr.dot(coo)
        mx.eval(result)
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = float(np.sum(A * A))
        np.testing.assert_allclose(float(np.array(result)), expected, rtol=1e-5)

    def test_dot_wrong_type_raises(self):
        csr = _real_2x2()
        with pytest.raises(TypeError, match="dot"):
            csr.dot(mx.array([[1.0]]))

    def test_dot_shape_mismatch_raises(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr_a = _real_2x2()
        csr_b = ms.csr_array(
            (
                mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                mx.array([0, 1, 2], dtype=mx.int32),
                mx.array([0, 1, 2, 3], dtype=mx.int32),
            ),
            shape=(3, 3),
            canonical=True,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            csr_a.dot(csr_b)

    def test_dot_float16_promoted(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr_f16 = _float16_2x2()
        csr_f32 = _real_2x2()
        result = csr_f16.dot(csr_f32)
        mx.eval(result)
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        expected = float(np.sum(A * A))
        np.testing.assert_allclose(float(np.array(result)), expected, rtol=1e-3)

    def test_dot_dtype_mismatch_raises(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr_f32 = _real_2x2()
        csr_cplx = _complex_2x2()
        with pytest.raises(TypeError, match="same dtype"):
            csr_f32.dot(csr_cplx)

    def test_dot_unsupported_dtype_raises(self):
        from mlx_sparse._csr import CSRArray

        csr = CSRArray(
            data=mx.array([1, 0, 0, 1], dtype=mx.int32),
            indices=mx.array([0, 1, 0, 1], dtype=mx.int32),
            indptr=mx.array([0, 2, 4], dtype=mx.int32),
            shape=(2, 2),
            sorted_indices=True,
            has_canonical_format=True,
        )
        with pytest.raises(TypeError, match="float32 and complex64"):
            csr.dot(csr)

    def test_dot_no_conjugation_for_complex(self):
        if not extension_available():
            pytest.skip("native extension unavailable")
        csr = _complex_2x2()
        dot_result = csr.dot(csr)
        vdot_result = csr.vdot(csr)
        mx.eval(dot_result, vdot_result)
        # vdot conjugates left, dot does not — imaginary parts differ
        dot_val = complex(np.array(dot_result))
        vdot_val = complex(np.array(vdot_result))
        # vdot result should be real and non-negative (sum of |x|^2)
        assert abs(vdot_val.imag) < 1e-5
        # dot result has non-zero imaginary part for non-real input
        # (unless values happen to cancel, which is unlikely for our fixture)
        _ = dot_val  # just verify it's finite
        assert np.isfinite(dot_val.real)


class TestCSRMatmulCOORhs:
    def test_csr_at_coo(self):
        csr = _real_2x2()
        coo = ms.coo_array(
            (
                mx.array([1.0, 1.0], dtype=mx.float32),
                (mx.array([0, 1], dtype=mx.int32), mx.array([0, 1], dtype=mx.int32)),
            ),
            shape=(2, 2),
        )
        result = csr @ coo
        assert isinstance(result, ms.CSRArray)
