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

"""``validate="full"`` checks the format flags the caller asserts.

``sorted_indices`` and ``canonical`` are hints that operations read instead of
inspecting the buffers, so an untrue one is answered rather than rejected. The
cell that motivates the check is :func:`test_wrong_canonical_flag_silently_
changes_row_norms`: a matrix with two entries in the same cell reports a row
norm computed over the unsummed entries. Under ``validate="full"`` the caller
now gets a ``ValueError`` at construction instead.
"""

import mlx.core as mx
import numpy as np
import pytest

import mlx_sparse as ms

# 2x2 with two stored entries in cell (0, 0): the row is [3 + 4, 0] = [7, 0].
DUP_DATA = np.array([3.0, 4.0, 5.0], dtype=np.float32)
DUP_INDICES = np.array([0, 0, 1], dtype=np.int32)
DUP_INDPTR = np.array([0, 2, 3], dtype=np.int32)

# 3x4 whose first row stores columns 2 then 0.
UNSORTED_DATA = np.array([2.0, 1.0, 3.0, 4.0], dtype=np.float32)
UNSORTED_INDICES = np.array([2, 0, 1, 3], dtype=np.int32)
UNSORTED_INDPTR = np.array([0, 2, 3, 4], dtype=np.int32)


def _mx(array):
    return mx.array(array.copy())


def test_wrong_canonical_flag_silently_changes_row_norms():
    honest = ms.csr_array(
        (_mx(DUP_DATA), _mx(DUP_INDICES), _mx(DUP_INDPTR)), shape=(2, 2)
    )
    lying = ms.csr_array(
        (_mx(DUP_DATA), _mx(DUP_INDICES), _mx(DUP_INDPTR)),
        shape=(2, 2),
        canonical=True,
    )
    # The row holds 7 in column 0, so its norm is 7. The untrue flag skips the
    # duplicate summation and hypot(3, 4) = 5 comes back instead.
    assert np.allclose(np.array(honest.row_norms()), [7.0, 5.0])
    assert np.allclose(np.array(lying.row_norms()), [5.0, 5.0])

    with pytest.raises(ValueError, match="declared canonical"):
        ms.csr_array(
            (_mx(DUP_DATA), _mx(DUP_INDICES), _mx(DUP_INDPTR)),
            shape=(2, 2),
            canonical=True,
            validate="full",
        )


def test_csr_full_validation_rejects_untrue_sorted_indices():
    with pytest.raises(ValueError, match="sorted indices"):
        ms.csr_array(
            (_mx(UNSORTED_DATA), _mx(UNSORTED_INDICES), _mx(UNSORTED_INDPTR)),
            shape=(3, 4),
            sorted_indices=True,
            validate="full",
        )


def test_csr_full_validation_reports_the_offending_row():
    indices = np.array([0, 1, 5, 2, 6, 7], dtype=np.int32)
    indptr = np.array([0, 2, 4, 6], dtype=np.int32)
    data = np.arange(6, dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        ms.csr_array(
            (_mx(data), _mx(indices), _mx(indptr)),
            shape=(3, 8),
            sorted_indices=True,
            validate="full",
        )
    # Row 1 stores columns 5 then 2; rows 0 and 2 are sorted.
    assert "row 1" in str(excinfo.value)
    assert "5 then 2" in str(excinfo.value)


def test_csr_full_validation_accepts_truthful_flags():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    indices = np.array([0, 2, 1, 3], dtype=np.int32)
    indptr = np.array([0, 2, 3, 4], dtype=np.int32)
    array = ms.csr_array(
        (_mx(data), _mx(indices), _mx(indptr)),
        shape=(3, 4),
        canonical=True,
        validate="full",
    )
    assert array.has_canonical_format
    assert array.sorted_indices


def test_csr_sorted_flag_permits_duplicates_but_canonical_does_not():
    # Columns 0, 0 are nondecreasing, so the row is sorted but not canonical.
    kwargs = dict(shape=(2, 2), validate="full")
    sorted_only = ms.csr_array(
        (_mx(DUP_DATA), _mx(DUP_INDICES), _mx(DUP_INDPTR)),
        sorted_indices=True,
        **kwargs,
    )
    assert sorted_only.sorted_indices
    assert not sorted_only.has_canonical_format
    with pytest.raises(ValueError, match="strictly increasing"):
        ms.csr_array(
            (_mx(DUP_DATA), _mx(DUP_INDICES), _mx(DUP_INDPTR)),
            canonical=True,
            **kwargs,
        )


def test_csr_full_validation_ignores_empty_rows_when_checking_order():
    # Rows 1 and 3 are empty, so indptr repeats. The boundary between rows must
    # not be read as an unsorted pair.
    data = np.array([5.0, 9.0, 1.0], dtype=np.float32)
    indices = np.array([7, 2, 4], dtype=np.int32)
    indptr = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
    array = ms.csr_array(
        (_mx(data), _mx(indices), _mx(indptr)),
        shape=(5, 8),
        canonical=True,
        validate="full",
    )
    assert array.has_canonical_format


@pytest.mark.parametrize("nnz", [0, 1])
def test_csr_full_validation_accepts_tiny_arrays(nnz):
    data = np.arange(nnz, dtype=np.float32)
    indices = np.zeros(nnz, dtype=np.int32)
    indptr = np.array([0] * (2 - nnz) + [nnz] * nnz, dtype=np.int32)
    array = ms.csr_array(
        (_mx(data), _mx(indices), _mx(indptr)),
        shape=(1, 2),
        canonical=True,
        validate="full",
    )
    assert array.nnz == nnz


def test_csc_full_validation_rejects_untrue_flags():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    indices = np.array([2, 0, 1], dtype=np.int32)
    indptr = np.array([0, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="column 0 has row indices 2 then 0"):
        ms.csc_array(
            (_mx(data), _mx(indices), _mx(indptr)),
            shape=(3, 2),
            sorted_indices=True,
            validate="full",
        )
    ordered = ms.csc_array(
        (_mx(data), _mx(np.array([0, 2, 1], dtype=np.int32)), _mx(indptr)),
        shape=(3, 2),
        canonical=True,
        validate="full",
    )
    assert ordered.has_canonical_format


def test_coo_full_validation_rejects_untrue_canonical():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    row = np.array([0, 1, 0], dtype=np.int32)
    col = np.array([0, 0, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="row-major ordered and duplicate-free"):
        ms.coo_array(
            (_mx(data), (_mx(row), _mx(col))),
            shape=(2, 2),
            canonical=True,
            validate="full",
        )
    ordered = ms.coo_array(
        (
            _mx(data),
            (
                _mx(np.array([0, 0, 1], dtype=np.int32)),
                _mx(np.array([0, 1, 0], dtype=np.int32)),
            ),
        ),
        shape=(2, 2),
        canonical=True,
        validate="full",
    )
    assert ordered.has_canonical_format


def test_coo_full_validation_rejects_duplicate_coordinates():
    data = np.array([1.0, 2.0], dtype=np.float32)
    row = np.array([1, 1], dtype=np.int32)
    col = np.array([0, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="duplicate-free"):
        ms.coo_array(
            (_mx(data), (_mx(row), _mx(col))),
            shape=(2, 2),
            canonical=True,
            validate="full",
        )


def test_flags_are_still_taken_on_trust_below_full_validation():
    # The check is opt-in: "metadata" must not pay for a host round trip.
    for mode in ("metadata", False):
        array = ms.csr_array(
            (_mx(UNSORTED_DATA), _mx(UNSORTED_INDICES), _mx(UNSORTED_INDPTR)),
            shape=(3, 4),
            canonical=True,
            validate=mode,
        )
        assert array.has_canonical_format


def test_conversions_produce_arrays_that_pass_their_own_declared_flags():
    rng = np.random.default_rng(0)
    n = 64
    nnz = 400
    row = rng.integers(0, n, size=nnz).astype(np.int32)
    col = rng.integers(0, n, size=nnz).astype(np.int32)
    data = rng.standard_normal(nnz).astype(np.float32)
    coo = ms.coo_array((mx.array(data), (mx.array(row), mx.array(col))), shape=(n, n))

    csr = coo.tocsr()
    assert csr.sorted_indices
    ms.csr_array(
        (csr.data, csr.indices, csr.indptr),
        shape=csr.shape,
        sorted_indices=True,
        validate="full",
    )

    canonical = coo.tocsr(canonical=True)
    assert canonical.has_canonical_format
    ms.csr_array(
        (canonical.data, canonical.indices, canonical.indptr),
        shape=canonical.shape,
        canonical=True,
        validate="full",
    )

    csc = coo.tocsc(canonical=True)
    assert csc.has_canonical_format
    ms.csc_array(
        (csc.data, csc.indices, csc.indptr),
        shape=csc.shape,
        canonical=True,
        validate="full",
    )
