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

import importlib
import inspect

import numpy as np
import pytest

import mlx_sparse as ms
from mlx_sparse.random._validation import density_to_nnz


def test_random_namespace_is_public_module():
    module = importlib.import_module("mlx_sparse.random")

    assert ms.random is module
    assert "random" in ms.__all__
    assert module.__all__ == ["random_array", "random", "rand"]
    for name in module.__all__:
        assert callable(getattr(ms.random, name))


def test_random_signatures_match_planned_scipy_surface(mx):
    random_array_sig = inspect.signature(ms.random.random_array)
    assert list(random_array_sig.parameters) == [
        "shape",
        "density",
        "format",
        "dtype",
        "rng",
        "data_sampler",
        "random_state",
        "index_dtype",
        "canonical",
    ]
    assert random_array_sig.parameters["density"].kind is inspect.Parameter.KEYWORD_ONLY
    assert random_array_sig.parameters["index_dtype"].default == mx.int32

    random_sig = inspect.signature(ms.random.random)
    assert list(random_sig.parameters) == [
        "m",
        "n",
        "density",
        "format",
        "dtype",
        "rng",
        "data_rvs",
        "random_state",
        "index_dtype",
        "canonical",
    ]
    assert random_sig.parameters["random_state"].kind is inspect.Parameter.KEYWORD_ONLY

    rand_sig = inspect.signature(ms.random.rand)
    assert list(rand_sig.parameters) == [
        "m",
        "n",
        "density",
        "format",
        "dtype",
        "rng",
        "random_state",
        "index_dtype",
        "canonical",
    ]
    assert rand_sig.parameters["random_state"].kind is inspect.Parameter.KEYWORD_ONLY


def test_random_docstrings_name_required_contract_terms():
    required = (
        "shape",
        "density",
        "format",
        "dtype",
        "rng",
        "random_state",
        "index_dtype",
        "canonical",
        "device",
        "reproduc",
    )

    for name in ("random_array", "random", "rand"):
        doc = inspect.getdoc(getattr(ms.random, name))
        assert doc is not None
        lowered = doc.lower()
        for term in required:
            assert term in lowered, f"{name} docstring is missing {term!r}"


@pytest.mark.parametrize(
    ("shape", "density", "expected"),
    [
        ((0, 10), 0.75, 0),
        ((1, 1), 0.5, 0),
        ((1, 3), 0.5, 2),
        ((2, 3), 0.5, 3),
        ((3, 3), 1.0, 9),
    ],
)
def test_density_to_nnz_matches_documented_scipy_rounding(shape, density, expected):
    assert density_to_nnz(shape, density) == expected


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc", None, "COO"])
def test_valid_random_array_arguments_reach_native_boundary(mx, format_name):
    with pytest.raises(NotImplementedError, match="native CPU/Metal"):
        ms.random.random_array(
            (4, 5),
            density=0.25,
            format=format_name,
            dtype=mx.float32,
            rng=mx.random.key(0),
            index_dtype=mx.int32,
        )


def test_integer_seed_and_random_state_alias_are_accepted_until_native_boundary():
    with pytest.raises(NotImplementedError, match="native CPU/Metal"):
        ms.random.rand(4, 5, density=0.25, rng=123)

    with pytest.raises(NotImplementedError, match="native CPU/Metal"):
        ms.random.random(4, 5, density=0.25, random_state=123)


def test_rng_and_random_state_are_mutually_exclusive(mx):
    with pytest.raises(ValueError, match="rng and random_state"):
        ms.random.random_array((4, 5), rng=mx.random.key(0), random_state=1)


def test_numpy_generator_is_rejected_as_host_rng():
    generator = np.random.default_rng(0)

    with pytest.raises(TypeError, match="NumPy Generator"):
        ms.random.random_array((4, 5), rng=generator)


@pytest.mark.parametrize("format_name", ["bsr", "dia", "dok", "lil"])
def test_unsupported_scipy_formats_have_precise_errors(format_name):
    with pytest.raises(ValueError, match="SciPy sparse format"):
        ms.random.random_array((4, 5), format=format_name)


@pytest.mark.parametrize("density", [-0.01, 1.01, float("nan"), float("inf")])
def test_density_must_be_finite_and_in_unit_interval(density):
    with pytest.raises(ValueError, match="density"):
        ms.random.random_array((4, 5), density=density)


@pytest.mark.parametrize("shape", [(-1, 5), (4,), (4, 5, 6)])
def test_shape_validation_fails_before_native_boundary(shape):
    with pytest.raises(ValueError, match="shape|dimensions"):
        ms.random.random_array(shape)


@pytest.mark.parametrize("shape", [(4.5, 5), (True, 5), 4])
def test_shape_dimensions_must_be_true_integers(shape):
    with pytest.raises(TypeError, match="shape"):
        ms.random.random_array(shape)


def test_dimension_validation_for_random_and_rand():
    with pytest.raises(TypeError, match="m must be a non-negative integer"):
        ms.random.random(1.5, 3)

    with pytest.raises(ValueError, match="n must be non-negative"):
        ms.random.rand(3, -1)


def test_dtype_and_index_dtype_follow_current_sparse_container_policy(mx):
    with pytest.raises(TypeError, match="current sparse containers"):
        ms.random.random_array((4, 5), dtype=mx.float64)

    with pytest.raises(TypeError, match="index_dtype"):
        ms.random.random_array((4, 5), index_dtype=mx.uint32)


def test_noncanonical_random_structure_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError, match="noncanonical"):
        ms.random.random_array((4, 5), canonical=False)


def test_sampler_is_validated_but_not_called_before_native_generation():
    calls: list[int] = []

    def sampler(*, size):
        calls.append(size)
        return np.ones(size, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="native CPU/Metal"):
        ms.random.random(4, 5, data_rvs=sampler)

    assert calls == []

    with pytest.raises(TypeError, match="callable"):
        ms.random.random_array((4, 5), data_sampler=np.ones(3))
