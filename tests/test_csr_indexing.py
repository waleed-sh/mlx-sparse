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

"""Row and column selection on CSRArray, against SciPy as the oracle.

The interesting cases are the ones a shape does not describe: an index that
counts from the end, an index that repeats, an empty selection, and an index
outside the matrix. SciPy answers all four and this follows it, including
raising on the last one.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import scipy.sparse as sp

import mlx_sparse as ms

DEVICES = [pytest.param(mx.cpu, id="cpu"), pytest.param(mx.gpu, id="gpu")]


def _pair(m=6, n=7, density=0.4, seed=0, sorted_indices=True):
    """The same matrix as a SciPy oracle and as a CSRArray."""
    reference = sp.random(
        m, n, density=density, format="csr", random_state=seed, dtype=np.float32
    )
    reference.sort_indices()
    ours = ms.csr_array(
        (
            mx.array(reference.data),
            mx.array(reference.indices.astype(np.int32)),
            mx.array(reference.indptr.astype(np.int32)),
        ),
        shape=(m, n),
        sorted_indices=sorted_indices,
        canonical=sorted_indices,
    )
    return sp.csr_array(reference), ours


def _as_scipy(array):
    mx.eval(array.data, array.indices, array.indptr)
    return sp.csr_matrix(
        (np.asarray(array.data), np.asarray(array.indices), np.asarray(array.indptr)),
        shape=array.shape,
    )


def _same(ours, reference):
    got = _as_scipy(ours)
    want = sp.csr_matrix(reference)
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    assert (got != want).nnz == 0


KEYS = [
    ("rows", [0, 2, 4]),
    ("rows_unordered", [4, 0, 2]),
    ("rows_repeated", [3, 3, 3]),
    ("rows_all", [0, 1, 2, 3, 4, 5]),
    ("rows_negative", [-1, -3]),
    ("rows_mixed_sign", [0, -1, 2]),
    ("rows_empty", []),
    ("row_slice", slice(1, 4)),
    ("row_slice_step", slice(0, 6, 2)),
    ("row_slice_open", slice(None, 3)),
    ("col_slice", (slice(None), slice(2, 5))),
    ("col_slice_step", (slice(None), slice(0, 7, 3))),
    ("col_slice_empty", (slice(None), slice(3, 3))),
    ("col_slice_all", (slice(None), slice(None))),
    ("both", ([1, 3], slice(1, 5))),
    ("both_negative", ([-1, -2], slice(0, 4))),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("label,key", KEYS, ids=lambda v: v)
def test_matches_scipy(device, label, key):
    """Every supported key answers what SciPy answers."""
    with mx.stream(device):
        reference, ours = _pair()
        _same(ours[key], reference[key])


@pytest.mark.parametrize("device", DEVICES)
def test_boolean_mask_selects_the_marked_rows(device):
    with mx.stream(device):
        reference, ours = _pair()
        mask = np.array([True, False, True, True, False, False])
        _same(ours[mask], reference[mask])
        _same(ours[mx.array(mask)], reference[mask])


@pytest.mark.parametrize("device", DEVICES)
def test_an_mlx_index_array_works_like_a_list(device):
    with mx.stream(device):
        reference, ours = _pair()
        _same(ours[mx.array([0, 2, 2], dtype=mx.int32)], reference[[0, 2, 2]])


@pytest.mark.parametrize("device", DEVICES)
def test_selecting_from_an_empty_matrix(device):
    """No stored entries anywhere, which is where an off-by-one would show."""
    with mx.stream(device):
        ours = ms.csr_array(
            (
                mx.zeros((0,), dtype=mx.float32),
                mx.zeros((0,), dtype=mx.int32),
                mx.zeros((5,), dtype=mx.int32),
            ),
            shape=(4, 3),
        )
        picked = ours[[0, 3, 3]]
        assert picked.shape == (3, 3)
        assert picked.nnz == 0
        assert np.asarray(picked.indptr).tolist() == [0, 0, 0, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_selecting_only_empty_rows_from_a_full_matrix(device):
    """The chosen rows are empty even though the matrix is not."""
    with mx.stream(device):
        # Rows 1 and 2 are empty; rows 0 and 3 are not.
        ours = ms.csr_array(
            (
                mx.array([1.0, 2.0], dtype=mx.float32),
                mx.array([0, 1], dtype=mx.int32),
                mx.array([0, 1, 1, 1, 2], dtype=mx.int32),
            ),
            shape=(4, 2),
        )
        picked = ours[[1, 2]]
        assert picked.shape == (2, 2)
        assert picked.nnz == 0
        assert np.asarray(picked.indptr).tolist() == [0, 0, 0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16, mx.complex64])
def test_every_value_dtype_survives_selection(device, dtype):
    """The gather must not be limited to the dtypes MLX will scatter."""
    with mx.stream(device):
        ours = ms.csr_array(
            (
                mx.array([1.0, 2.0, 3.0]).astype(dtype),
                mx.array([0, 1, 2], dtype=mx.int32),
                mx.array([0, 1, 2, 3], dtype=mx.int32),
            ),
            shape=(3, 3),
        )
        rows = ours[[2, 0]]
        cols = ours[:, 1:3]
        mx.eval(rows.data, cols.data)
        assert rows.dtype == dtype and cols.dtype == dtype
        assert np.asarray(rows.data.astype(mx.float32)).tolist() == [3.0, 1.0]


@pytest.mark.parametrize("device", DEVICES)
def test_int64_indices_are_preserved(device):
    with mx.stream(device):
        ours = ms.csr_array(
            (
                mx.array([1.0, 2.0], dtype=mx.float32),
                mx.array([0, 1], dtype=mx.int64),
                mx.array([0, 1, 2], dtype=mx.int64),
            ),
            shape=(2, 2),
        )
        picked = ours[[1, 0]]
        assert picked.index_dtype == mx.int64
        assert picked.indptr.dtype == mx.int64


@pytest.mark.parametrize("device", DEVICES)
def test_the_declared_flags_carry_over(device):
    """Whole rows move, so what was true of the source is true of the result."""
    with mx.stream(device):
        _, ours = _pair(sorted_indices=True)
        for picked in (ours[[2, 0, 2]], ours[1:4], ours[:, 1:5]):
            assert picked.sorted_indices is True
            assert picked.has_canonical_format is True
            # And the flags are honest: full validation re-checks them.
            ms.csr_array(
                (picked.data, picked.indices, picked.indptr),
                shape=picked.shape,
                sorted_indices=True,
                canonical=True,
                validate="full",
            )


@pytest.mark.parametrize("device", DEVICES)
def test_an_unsorted_source_does_not_gain_a_flag(device):
    with mx.stream(device):
        ours = ms.csr_array(
            (
                mx.array([1.0, 2.0], dtype=mx.float32),
                mx.array([1, 0], dtype=mx.int32),
                mx.array([0, 2, 2], dtype=mx.int32),
            ),
            shape=(2, 2),
        )
        assert ours[[0]].sorted_indices is False
        assert ours[[0]].has_canonical_format is False


# --- the awkward inputs, which are the point of the exercise ----------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [6, 7, 9999, -7, -100])
def test_a_row_index_outside_the_shape_raises(device, bad):
    """SciPy raises here and so does this, with the offending index named.

    Unlike the conversion kernels, which cannot raise and therefore drop an
    out-of-range coordinate, this is an operand the caller just supplied and
    the operation already reads a count back to the host, so refusing costs
    nothing that was not already being paid.
    """
    with mx.stream(device):
        reference, ours = _pair()
        with pytest.raises(IndexError, match="out of range"):
            ours[[0, bad]]
        with pytest.raises(IndexError):
            reference[[0, bad]]


@pytest.mark.parametrize("device", DEVICES)
def test_a_bare_integer_is_refused_with_a_usable_message(device):
    """SciPy returns a 1-D array here; this package has no such container."""
    with mx.stream(device):
        _, ours = _pair()
        with pytest.raises(IndexError, match=r"A\[\[i\]\]"):
            ours[2]


@pytest.mark.parametrize("device", DEVICES)
def test_a_reversed_column_slice_is_refused(device):
    with mx.stream(device):
        _, ours = _pair()
        with pytest.raises(IndexError, match="negative step"):
            ours[:, ::-1]


@pytest.mark.parametrize("device", DEVICES)
def test_a_badly_shaped_key_is_refused(device):
    with mx.stream(device):
        _, ours = _pair()
        with pytest.raises(IndexError, match="at most a row and a column"):
            ours[[0], 0:2, 0:2]
        with pytest.raises(IndexError, match="must be a slice"):
            ours[[0], [1, 2]]
        with pytest.raises(IndexError, match="one entry per row"):
            ours[np.array([True, False])]
        with pytest.raises(IndexError, match="one-dimensional"):
            ours[np.zeros((2, 2), dtype=np.int32)]
        with pytest.raises(IndexError, match="integers or a boolean"):
            ours[[0.5, 1.5]]


@pytest.mark.parametrize("device", DEVICES)
def test_selection_composes_with_the_rest_of_the_package(device):
    """A selected matrix is an ordinary one: it converts and multiplies."""
    with mx.stream(device):
        reference, ours = _pair()
        picked = ours[[4, 1, 1]]
        rhs = mx.array(np.arange(7, dtype=np.float32))
        product = picked @ rhs
        dense = picked.todense()
        back = picked.tocsc().tocsr()
        mx.eval(product, dense, back.data)
        want = sp.csr_matrix(reference[[4, 1, 1]])
        assert np.allclose(np.asarray(product), want @ np.arange(7, dtype=np.float32))
        assert np.allclose(np.asarray(dense), want.toarray())
        _same(back, want)


def test_a_randomized_sweep_against_scipy():
    """Shapes and densities the fixed cases do not reach."""
    rng = np.random.default_rng(5)
    for trial in range(40):
        m = int(rng.integers(1, 9))
        n = int(rng.integers(1, 9))
        reference, ours = _pair(
            m=m, n=n, density=float(rng.uniform(0.05, 0.9)), seed=trial
        )
        rows = rng.integers(-m, m, int(rng.integers(0, m + 2)))
        _same(ours[rows], reference[rows])
        start = int(rng.integers(0, n))
        stop = int(rng.integers(start, n + 1))
        _same(ours[:, start:stop], reference[:, start:stop])
        _same(ours[rows, start:stop], reference[rows, start:stop])
