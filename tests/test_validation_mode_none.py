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

"""``validate="none"`` is the spelling the documentation gives for no checks."""

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse._validation import normalize_validation_mode

DATA = np.array([1.0, 2.0], dtype=np.float32)
INDICES = np.array([0, 1], dtype=np.int32)
INDPTR = np.array([0, 1, 2], dtype=np.int32)


def _mx(mx, array):
    return mx.array(array.copy())


@pytest.mark.parametrize("mode", [False, "none"])
def test_both_spellings_of_no_validation_are_accepted(mx, mode):
    array = ms.csr_array(
        (_mx(mx, DATA), _mx(mx, INDICES), _mx(mx, INDPTR)),
        shape=(2, 2),
        validate=mode,
    )
    assert array.nnz == 2


@pytest.mark.parametrize("mode", [False, "none"])
def test_no_validation_skips_the_metadata_checks(mx, mode):
    # An indptr of the wrong length is rejected by "metadata"; neither spelling
    # of no validation looks at it.
    bad_indptr = np.array([0, 1, 2, 2, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="indptr"):
        ms.csr_array(
            (_mx(mx, DATA), _mx(mx, INDICES), _mx(mx, bad_indptr)), shape=(2, 2)
        )
    array = ms.csr_array(
        (_mx(mx, DATA), _mx(mx, INDICES), _mx(mx, bad_indptr)),
        shape=(2, 2),
        validate=mode,
    )
    assert array.indptr.shape == (5,)


@pytest.mark.parametrize("mode", [False, "none"])
def test_coo_and_csc_take_both_spellings_too(mx, mode):
    coo = ms.coo_array(
        (_mx(mx, DATA), (_mx(mx, INDICES), _mx(mx, INDICES))),
        shape=(2, 2),
        validate=mode,
    )
    assert coo.nnz == 2
    csc = ms.csc_array(
        (_mx(mx, DATA), _mx(mx, INDICES), _mx(mx, INDPTR)),
        shape=(2, 2),
        validate=mode,
    )
    assert csc.nnz == 2


def test_the_two_spellings_normalize_to_the_same_mode():
    assert normalize_validation_mode("none") == normalize_validation_mode(False)
    assert normalize_validation_mode(True) == normalize_validation_mode("full")
    assert normalize_validation_mode("metadata") == "metadata"


def test_an_unknown_mode_still_raises_and_names_every_accepted_value():
    with pytest.raises(ValueError) as excinfo:
        normalize_validation_mode("deep")
    message = str(excinfo.value)
    for accepted in ("False", "True", "'none'", "'metadata'", "'full'"):
        assert accepted in message
