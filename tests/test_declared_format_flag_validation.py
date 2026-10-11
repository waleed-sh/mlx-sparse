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
from mlx_sparse import _validation
from mlx_sparse._host import to_numpy


def _buffers(mx, format, first, second, *, index_dtype="int32", dtype="float32"):
    data = mx.ones((len(first),), dtype=getattr(mx, dtype))
    first = mx.array(first, dtype=getattr(mx, index_dtype))
    second = mx.array(second, dtype=getattr(mx, index_dtype))
    return (data, (first, second)) if format == "coo" else (data, first, second)


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("mode", ["full", True])
@pytest.mark.parametrize(
    ("indices", "flags", "assertion"),
    [
        ([2, 1, 0], {"sorted_indices": True}, "sorted_indices=True"),
        ([2, 1, 0], {"canonical": True}, "canonical=True"),
        ([1, 1, 2], {"canonical": True}, "canonical=True"),
        ([1, 2, 1], {"canonical": True}, "canonical=True"),
    ],
)
def test_compressed_full_validation_rejects_false_flags(
    mx, format, index_dtype, mode, indices, flags, assertion
):
    buffers = _buffers(
        mx,
        format,
        [0, 3, *indices, 0],
        [0, 0, 2, 2, 5, 6, 6],
        index_dtype=index_dtype,
    )
    shape = (6, 4) if format == "csr" else (4, 6)
    segment = "row" if format == "csr" else "column"
    with pytest.raises(ValueError, match=assertion) as error:
        getattr(ms, f"{format}_array")(buffers, shape, validate=mode, **flags)
    assert f"{segment} 3" in str(error.value)
    pair = "entries 3 and 4" if indices == [1, 2, 1] else "entries 2 and 3"
    assert pair in str(error.value)


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("canonical", [False, True])
def test_compressed_full_validation_checks_only_within_segments(
    mx, format, index_dtype, canonical
):
    # Boundaries include descending and equal adjacent indices, plus empty segments.
    indices = [1, 2, 0, 2, 2] if canonical else [1, 1, 0, 2, 2]
    buffers = _buffers(
        mx, format, indices, [0, 0, 2, 2, 4, 5, 5], index_dtype=index_dtype
    )
    shape = (6, 3) if format == "csr" else (3, 6)
    array = getattr(ms, f"{format}_array")(
        buffers, shape, validate="full", sorted_indices=True, canonical=canonical
    )
    assert array.sorted_indices
    assert array.has_canonical_format is canonical
    assert array.data is buffers[0]
    assert array.indices is buffers[1]
    assert array.indptr is buffers[2]


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (4, 0), (3, 4)])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_full_validation_accepts_empty_canonical_arrays(mx, format, shape, index_dtype):
    if format == "coo":
        first, second = [], []
    else:
        first = []
        second = [0] * (shape[0 if format == "csr" else 1] + 1)
    buffers = _buffers(mx, format, first, second, index_dtype=index_dtype)
    array = getattr(ms, f"{format}_array")(
        buffers, shape, validate="full", canonical=True
    )
    assert array.nnz == 0
    assert array.has_canonical_format


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "complex64"])
def test_canonical_validation_accepts_explicit_zeros_and_singletons(mx, format, dtype):
    buffers = _buffers(mx, format, [0], [0] if format == "coo" else [0, 1], dtype=dtype)
    buffers = (mx.zeros((1,), dtype=getattr(mx, dtype)), *buffers[1:])
    array = getattr(ms, f"{format}_array")(
        buffers, (1, 1), validate="full", canonical=True
    )
    assert array.has_canonical_format
    assert array.data is buffers[0]
    if format != "coo":
        assert array.sorted_indices


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "complex64"])
def test_canonical_validation_rejects_duplicate_explicit_zeros(mx, format, dtype):
    buffers = _buffers(
        mx, format, [0, 0], [0, 0] if format == "coo" else [0, 2], dtype=dtype
    )
    buffers = (mx.zeros((2,), dtype=getattr(mx, dtype)), *buffers[1:])
    with pytest.raises(ValueError, match="canonical=True"):
        getattr(ms, f"{format}_array")(buffers, (1, 1), validate="full", canonical=True)


@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("mode", ["full", True])
@pytest.mark.parametrize(
    ("row", "col"),
    [
        ([0, 1, 0], [0, 0, 2]),
        ([0, 0, 1], [2, 1, 0]),
        ([0, 0, 1], [1, 1, 0]),
        ([0, 0, 0], [1, 2, 1]),
    ],
)
def test_coo_full_validation_rejects_false_canonical_flags(
    mx, index_dtype, mode, row, col
):
    buffers = _buffers(mx, "coo", row, col, index_dtype=index_dtype)
    with pytest.raises(ValueError, match="canonical=True") as error:
        ms.coo_array(buffers, (2, 3), validate=mode, canonical=True)
    assert "entries" in str(error.value).lower()


@pytest.mark.parametrize("reverse", [False, True])
def test_coo_canonical_validation_does_not_flatten_coordinates(mx, reverse):
    # A row-major integer key would cross the signed int64 boundary.
    row, col = [0, 2**31, 2**32 - 1], [2**32 - 1, 0, 2**32 - 1]
    if reverse:
        row, col = row[::-1], col[::-1]
    buffers = _buffers(mx, "coo", row, col, index_dtype="int64")
    if reverse:
        with pytest.raises(ValueError, match="canonical=True"):
            ms.coo_array(buffers, (2**32, 2**32), validate="full", canonical=True)
    else:
        array = ms.coo_array(buffers, (2**32, 2**32), validate="full", canonical=True)
        assert array.has_canonical_format


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("mode", ["metadata", False, "none"])
def test_weaker_validation_trusts_flags_without_host_reads(
    mx, monkeypatch, format, mode
):
    buffers = _buffers(mx, format, [1, 0, 0], [0, 0, 0] if format == "coo" else [0, 3])
    shape = (2, 1) if format in ("csc", "coo") else (1, 2)

    def unexpected_read(*args, **kwargs):
        pytest.fail("Validation read array values.")

    monkeypatch.setattr(_validation, "to_numpy", unexpected_read)
    monkeypatch.setattr(mx, "eval", unexpected_read)
    constructor = getattr(ms, f"{format}_array")
    array = constructor(buffers, shape, validate=mode, canonical=True)
    assert array.has_canonical_format
    assert constructor(array, shape, validate=mode) is array


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_full_validation_preserves_unasserted_noncanonical_buffers(mx, format):
    buffers = _buffers(mx, format, [1, 0, 0], [0, 0, 0] if format == "coo" else [0, 3])
    shape = (2, 1) if format in ("csc", "coo") else (1, 2)
    constructor = getattr(ms, f"{format}_array")
    array = constructor(buffers, shape, validate="full", canonical=False)
    assert not array.has_canonical_format
    if format != "coo":
        assert not array.sorted_indices
    assert array.data is buffers[0]
    assert constructor(array, shape, validate="full") is array


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("source", ["constructor", "dataclass"])
def test_full_validation_rechecks_existing_canonical_assertions(mx, format, source):
    buffers = _buffers(mx, format, [0, 0], [0, 0] if format == "coo" else [0, 2])
    constructor = getattr(ms, f"{format}_array")
    if source == "constructor":
        array = constructor(buffers, (1, 1), canonical=True)
    elif format == "coo":
        array = ms.COOArray(buffers[0], *buffers[1], (1, 1), has_canonical_format=True)
    else:
        array = getattr(ms, f"{format.upper()}Array")(
            *buffers, (1, 1), has_canonical_format=True
        )
    with pytest.raises(ValueError, match="canonical=True"):
        constructor(array, (1, 1), validate="full")


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_full_validation_rechecks_existing_sorted_assertions(mx, format):
    buffers = _buffers(mx, format, [1, 0], [0, 2])
    shape = (1, 2) if format == "csr" else (2, 1)
    constructor = getattr(ms, f"{format}_array")
    array = constructor(buffers, shape, sorted_indices=True)
    with pytest.raises(ValueError, match="sorted_indices=True"):
        constructor(array, shape, validate=True)


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
@pytest.mark.parametrize("mode", ["metadata", "full", True])
def test_checked_modes_recheck_existing_buffer_metadata(mx, format, mode):
    buffers = _buffers(mx, format, [0, 0], [0, 0] if format == "coo" else [0, 2])
    buffers = (mx.ones((1,), dtype=mx.float32), *buffers[1:])
    constructor = getattr(ms, f"{format}_array")
    array = constructor(buffers, (1, 1), validate="none")
    with pytest.raises(ValueError, match="same length"):
        constructor(array, (1, 1), validate=mode)


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_full_validation_rechecks_existing_index_bounds(mx, format):
    buffers = _buffers(mx, format, [1], [0] if format == "coo" else [0, 1])
    constructor = getattr(ms, f"{format}_array")
    array = constructor(buffers, (1, 1))
    with pytest.raises(ValueError, match="bounds"):
        constructor(array, (1, 1), validate="full")


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("indptr", [[1, 2], [0, 1], [0, 2, 1, 2], [0, -1, 2, 2]])
def test_pointer_errors_precede_declared_order_checks(mx, format, indptr):
    buffers = _buffers(mx, format, [1, 0], indptr)
    shape = (len(indptr) - 1, 2) if format == "csr" else (2, len(indptr) - 1)
    constructor = getattr(ms, f"{format}_array")
    for arg in (buffers, constructor(buffers, shape, validate=False, canonical=True)):
        with pytest.raises(ValueError, match="indptr"):
            constructor(arg, shape, validate="full", canonical=True)


@pytest.mark.native
def test_duplicate_row_norms_require_canonicalization(mx):
    if not ms.is_available():
        pytest.skip("Native extension is required.")
    buffers = (
        mx.array([3.0, 4.0], dtype=mx.float32),
        mx.array([0, 0], dtype=mx.int32),
        mx.array([0, 2], dtype=mx.int32),
    )
    with pytest.raises(ValueError, match="canonical=True"):
        ms.csr_array(buffers, (1, 1), validate="full", canonical=True)
    array = ms.csr_array(buffers, (1, 1), validate="full", sorted_indices=True)
    np.testing.assert_allclose(to_numpy(array.row_norms()), [7.0])


@pytest.mark.parametrize("format", ["csr", "csc", "coo"])
def test_full_validation_reads_only_existing_index_buffers_once(
    mx, monkeypatch, format
):
    buffers = _buffers(mx, format, [0, 1], [0, 0] if format == "coo" else [0, 2])
    shape = (2, 1) if format in ("csc", "coo") else (1, 2)
    original_read = _validation.to_numpy
    reads = []

    def record_read(array):
        reads.append(array)
        return original_read(array)

    monkeypatch.setattr(_validation, "to_numpy", record_read)
    constructor = getattr(ms, f"{format}_array")
    array = constructor(buffers, shape, validate="full", canonical=True)
    expected = list(buffers[1]) if format == "coo" else list(buffers[1:])
    assert len(reads) == len(expected)
    assert all(
        actual is expected_array for actual, expected_array in zip(reads, expected)
    )
    reads.clear()
    assert constructor(array, shape, validate="full") is array
    assert len(reads) == len(expected)
    assert all(
        actual is expected_array for actual, expected_array in zip(reads, expected)
    )


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
def test_compressed_flag_validation_matches_scipy(
    mx, scipy_sparse, format, index_dtype
):
    rng = np.random.default_rng(61)
    constructor = getattr(ms, f"{format}_array")
    for sample in range(48):
        counts = rng.integers(0, 5, size=7)
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(index_dtype)
        indices = rng.integers(0, 5, size=int(indptr[-1])).astype(index_dtype)
        if sample % 2 == 0:
            for start, end in zip(indptr[:-1], indptr[1:]):
                indices[start:end].sort()
        buffers = _buffers(mx, format, indices, indptr, index_dtype=index_dtype)
        shape = (7, 5) if format == "csr" else (5, 7)
        reference = getattr(scipy_sparse, f"{format}_array")(
            (np.ones(len(indices)), indices, indptr), shape=shape
        )
        for flag, valid in (
            ("sorted_indices", reference.has_sorted_indices),
            ("canonical", reference.has_canonical_format),
        ):
            if valid:
                constructor(buffers, shape, validate="full", **{flag: True})
            else:
                with pytest.raises(ValueError, match=f"{flag}=True"):
                    constructor(buffers, shape, validate="full", **{flag: True})


def test_coo_flag_validation_matches_sorted_unique_coordinate_contract(mx):
    rng = np.random.default_rng(62)
    for sample in range(64):
        count = sample % 6
        coords = list(
            zip(
                rng.integers(0, 3, size=count).tolist(),
                rng.integers(0, 3, size=count).tolist(),
            )
        )
        if sample % 3 == 0:
            coords = sorted(coords)
        elif sample % 3 == 1:
            coords = sorted(set(coords))
        buffers = _buffers(mx, "coo", [r for r, _ in coords], [c for _, c in coords])
        if coords == sorted(set(coords)):
            ms.coo_array(buffers, (3, 3), validate="full", canonical=True)
        else:
            with pytest.raises(ValueError, match="canonical=True"):
                ms.coo_array(buffers, (3, 3), validate="full", canonical=True)


@pytest.mark.native
def test_native_conversion_flags_satisfy_full_validation(mx):
    if not ms.is_available():
        pytest.skip("Native extension is required.")
    coo = ms.coo_array(_buffers(mx, "coo", [2, 0, 2, 1, 0], [1, 2, 1, 0, 0]), (4, 3))
    for compressed in (coo.tocsr(), coo.tocsc()):
        for array in (compressed, compressed.sort_indices(), compressed.canonicalize()):
            constructor = (
                ms.csr_array if isinstance(array, ms.CSRArray) else ms.csc_array
            )
            constructor(array, array.shape, validate="full")
            transposed = array.transpose()
            constructor = (
                ms.csr_array if isinstance(transposed, ms.CSRArray) else ms.csc_array
            )
            constructor(transposed, transposed.shape, validate="full")
            for canonical in (None, False, True):
                coordinates = array.tocoo(canonical=canonical)
                assert (
                    ms.coo_array(coordinates, array.shape, validate="full")
                    is coordinates
                )
