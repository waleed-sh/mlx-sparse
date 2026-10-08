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
from mlx_sparse import linalg
from mlx_sparse._ext_loader import extension
from mlx_sparse.linalg import _eigen

SELECTORS = {
    "eigsh": ("LM", "SM", "LA", "SA"),
    "eigs": ("LM", "SM", "LR", "SR"),
    "svds": ("LM", "SM"),
}
UNSUPPORTED = [
    (routine, which)
    for routine, allowed in SELECTORS.items()
    for which in ("BE", "LI", "SI", "LA", "SA", "LR", "SR")
    if which not in allowed
]
EXPECTED = {
    "eigsh": {"LM": -9.0, "SM": 1.0, "LA": 7.0, "SA": -9.0},
    "eigs": {"LM": -9.0, "SM": 1.0, "LR": 7.0, "SR": -9.0},
    "svds": {"LM": 9.0, "SM": 1.0},
}


def _diagonal(mx, index_dtype):
    return ms.csr_array(
        (
            mx.array([-9.0, -4.0, 1.0, 3.0, 7.0], dtype=mx.float32),
            mx.arange(5, dtype=index_dtype),
            mx.arange(6, dtype=index_dtype),
        ),
        shape=(5, 5),
        canonical=True,
    )


@pytest.mark.parametrize("routine", SELECTORS)
@pytest.mark.parametrize(
    "which", ["invalid", "", " LM", "LM ", None, 0, True, b"LM", ["LM"], {"LM"}]
)
def test_invalid_selector_is_rejected_before_matrix_preparation(
    monkeypatch, routine, which
):
    def unexpected_preparation(*args, **kwargs):
        pytest.fail("An invalid selector reached matrix preparation.")

    monkeypatch.setattr(_eigen, "_as_csr", unexpected_preparation)
    with pytest.raises(ValueError, match=rf"{routine}.*which") as error:
        getattr(linalg, routine)(object(), k=1, which=which)
    for supported in SELECTORS[routine]:
        assert supported in str(error.value)


@pytest.mark.parametrize("routine,which", UNSUPPORTED)
def test_selector_from_another_routine_is_rejected(routine, which):
    with pytest.raises(ValueError, match=rf"{routine}.*which"):
        getattr(linalg, routine)(object(), k=1, which=which)


@pytest.mark.native
@pytest.mark.parametrize(
    "routine,which",
    UNSUPPORTED
    + [(routine, which) for routine in SELECTORS for which in ("invalid", "", "lm")],
)
def test_native_entry_point_rejects_unsupported_selectors(mx, routine, which):
    ext = extension()
    if ext is None:
        pytest.skip("native extension unavailable")
    csr = _diagonal(mx, mx.int32)
    with pytest.raises(ValueError, match=rf"csr_{routine}.*which"):
        getattr(ext, f"csr_{routine}")(
            csr.data, csr.indices, csr.indptr, mx.ones(5), 5, 5, 1, 5, which
        )


@pytest.mark.native
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize(
    "routine,which,expected",
    [
        (routine, which, value)
        for routine, choices in EXPECTED.items()
        for which, value in choices.items()
    ],
)
def test_supported_selectors_choose_the_requested_spectrum(
    mx, to_numpy, index_dtype, lowercase, routine, which, expected
):
    if extension() is None:
        pytest.skip("native extension unavailable")
    csr = _diagonal(mx, getattr(mx, index_dtype))
    kwargs = (
        {"return_singular_vectors": False}
        if routine == "svds"
        else {"return_eigenvectors": False}
    )
    values = getattr(linalg, routine)(
        csr,
        k=1,
        which=which.lower() if lowercase else which,
        v0=mx.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ncv=5,
        **kwargs,
    )
    np.testing.assert_allclose(to_numpy(values), [expected], rtol=2e-4, atol=2e-4)
