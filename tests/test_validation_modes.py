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

import pytest

import mlx_sparse as ms
from mlx_sparse import _coo, _csc, _csr, _validation

FORMATS = {"csr": _csr, "csc": _csc, "coo": _coo}


def _buffers(mx, format, *, mismatched_lengths=False):
    data = mx.array([2.0], dtype=mx.float32)
    indices = mx.array([0, 0] if mismatched_lengths else [0], dtype=mx.int32)
    if format == "coo":
        return data, (indices, indices)
    return data, indices, mx.array([0, 1], dtype=mx.int32)


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("mode", [False, "none"])
def test_disabled_validation_preserves_buffers_without_evaluation(
    mx, monkeypatch, format, mode
):
    module = FORMATS[format]
    constructor = getattr(ms, f"{format}_array")
    buffers = _buffers(mx, format)

    def unexpected_check(*args, **kwargs):
        pytest.fail("Disabled validation read or checked the buffers.")

    monkeypatch.setattr(module, f"validate_{format}_metadata", unexpected_check)
    monkeypatch.setattr(module, f"validate_{format}_values", unexpected_check)
    monkeypatch.setattr(_validation, "to_numpy", unexpected_check)
    monkeypatch.setattr(mx, "eval", unexpected_check)
    array = constructor(buffers, shape=(1, 1), validate=mode)
    assert array.data is buffers[0]
    if format == "coo":
        assert array.row is buffers[1][0]
        assert array.col is buffers[1][1]
    else:
        assert array.indices is buffers[1]
        assert array.indptr is buffers[2]
    assert constructor(array, shape=(1, 1), validate=mode) is array


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("mode", [False, "none"])
def test_disabled_validation_skips_buffer_length_checks(mx, format, mode):
    constructor = getattr(ms, f"{format}_array")
    buffers = _buffers(mx, format, mismatched_lengths=True)
    array = constructor(buffers, shape=(1, 1), validate=mode)
    assert array.data is buffers[0]
    for checked_mode in ("metadata", "full", True):
        with pytest.raises(ValueError, match="same length"):
            constructor(buffers, shape=(1, 1), validate=checked_mode)


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("mode", [False, "none"])
def test_disabled_validation_still_checks_shape_and_input_structure(mx, format, mode):
    constructor = getattr(ms, f"{format}_array")
    buffers = _buffers(mx, format)
    with pytest.raises(ValueError, match="non-negative"):
        constructor(buffers, shape=(-1, 1), validate=mode)
    with pytest.raises(ValueError, match="rank-2"):
        constructor(buffers, shape=(1,), validate=mode)
    with pytest.raises(TypeError, match=f"{format}_array expects"):
        constructor((), shape=(1, 1), validate=mode)


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("mode", ["invalid", "NONE", None, 0, 1, [], {}])
def test_invalid_validation_mode_has_a_consistent_error(mx, format, mode):
    constructor = getattr(ms, f"{format}_array")
    with pytest.raises(ValueError, match="validate must be one of") as error:
        constructor(_buffers(mx, format), shape=(1, 1), validate=mode)
    assert "'none'" in str(error.value)


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("mode", ["full", True])
def test_full_validation_still_rejects_out_of_bounds_indices(mx, format, mode):
    constructor = getattr(ms, f"{format}_array")
    data = mx.array([2.0], dtype=mx.float32)
    indices = mx.array([1], dtype=mx.int32)
    buffers = (
        (data, (indices, indices))
        if format == "coo"
        else (data, indices, mx.array([0, 1], dtype=mx.int32))
    )
    with pytest.raises(ValueError, match="bounds"):
        constructor(buffers, shape=(1, 1), validate=mode)
