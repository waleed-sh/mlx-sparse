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
from mlx_sparse._coo import COOArray
from mlx_sparse._csc import CSCArray
from mlx_sparse._csr import CSRArray
from mlx_sparse.random._validation import density_to_nnz


def _linear_coordinates(array, to_numpy):
    if isinstance(array, COOArray):
        row = to_numpy(array.row)
        col = to_numpy(array.col)
    elif isinstance(array, CSRArray):
        indptr = to_numpy(array.indptr)
        row = np.repeat(np.arange(array.shape[0]), np.diff(indptr))
        col = to_numpy(array.indices)
    elif isinstance(array, CSCArray):
        indptr = to_numpy(array.indptr)
        col = np.repeat(np.arange(array.shape[1]), np.diff(indptr))
        row = to_numpy(array.indices)
    else:
        raise TypeError(type(array))
    return row * array.shape[1] + col


def _assert_duplicate_free(array, to_numpy):
    linear = _linear_coordinates(array, to_numpy)
    assert linear.shape == (array.nnz,)
    assert np.unique(linear).shape[0] == array.nnz


def _assert_same_sparse(left, right, to_numpy):
    assert type(left) is type(right)
    assert left.shape == right.shape
    np.testing.assert_array_equal(to_numpy(left.data), to_numpy(right.data))
    if isinstance(left, COOArray):
        np.testing.assert_array_equal(to_numpy(left.row), to_numpy(right.row))
        np.testing.assert_array_equal(to_numpy(left.col), to_numpy(right.col))
    else:
        np.testing.assert_array_equal(to_numpy(left.indices), to_numpy(right.indices))
        np.testing.assert_array_equal(to_numpy(left.indptr), to_numpy(right.indptr))


def _structure_buffers(array, to_numpy):
    if isinstance(array, COOArray):
        return (to_numpy(array.row), to_numpy(array.col))
    if isinstance(array, (CSRArray, CSCArray)):
        return (to_numpy(array.indices), to_numpy(array.indptr))
    raise TypeError(type(array))


def _same_structure(left, right, to_numpy) -> bool:
    return all(
        np.array_equal(left_part, right_part)
        for left_part, right_part in zip(
            _structure_buffers(left, to_numpy),
            _structure_buffers(right, to_numpy),
        )
    )


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
def test_valid_random_array_arguments_construct_supported_formats(
    mx, to_numpy, format_name
):
    out = ms.random.random_array(
        (4, 5),
        density=0.25,
        format=format_name,
        dtype=mx.float32,
        rng=mx.random.key(0),
        index_dtype=mx.int32,
    )

    expected_nnz = density_to_nnz((4, 5), 0.25)
    expected_format = "coo" if format_name is None else str(format_name).lower()
    assert out.shape == (4, 5)
    assert out.nnz == expected_nnz
    assert out.dtype == mx.float32
    assert out.index_dtype == mx.int32
    if expected_format == "coo":
        assert isinstance(out, COOArray)
        assert out.has_canonical_format
        linear = _linear_coordinates(out, to_numpy)
        assert np.all(linear[1:] >= linear[:-1])
    elif expected_format == "csr":
        assert isinstance(out, CSRArray)
        assert out.sorted_indices
        assert out.has_canonical_format
        indptr = to_numpy(out.indptr)
        indices = to_numpy(out.indices)
        for start, end in zip(indptr[:-1], indptr[1:]):
            if end - start > 1:
                assert np.all(indices[start + 1 : end] >= indices[start : end - 1])
    else:
        assert isinstance(out, CSCArray)
        assert out.sorted_indices
        assert out.has_canonical_format
        indptr = to_numpy(out.indptr)
        indices = to_numpy(out.indices)
        for start, end in zip(indptr[:-1], indptr[1:]):
            if end - start > 1:
                assert np.all(indices[start + 1 : end] >= indices[start : end - 1])
    _assert_duplicate_free(out, to_numpy)
    data = to_numpy(out.data)
    assert np.all(data >= 0.0)
    assert np.all(data < 1.0)


def test_integer_seed_and_random_state_alias_are_reproducible(mx, to_numpy):
    from_seed = ms.random.rand(4, 5, density=0.25, rng=123)
    from_alias = ms.random.random(4, 5, density=0.25, random_state=123)

    _assert_same_sparse(from_seed, from_alias, to_numpy)


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_integer_seed_matches_explicit_mlx_key(mx, to_numpy, format_name):
    from_seed = ms.random.random_array(
        (37, 41),
        density=0.18,
        format=format_name,
        rng=20260604,
    )
    from_key = ms.random.random_array(
        (37, 41),
        density=0.18,
        format=format_name,
        rng=mx.random.key(20260604),
    )

    _assert_same_sparse(from_seed, from_key, to_numpy)


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


def test_random_shape_and_nnz_overflow_are_rejected_before_native_allocation(mx):
    with pytest.raises(OverflowError, match="native MLX shape limit"):
        ms.random.random_array((2**31, 1), density=0.0, index_dtype=mx.int64)

    with pytest.raises(OverflowError, match="random sparse shape is too large"):
        density_to_nnz((2**32, 2**32), 1.0)

    with pytest.raises(OverflowError, match="nnz=.*native MLX output shape"):
        ms.random.random_array(
            (46_341, 46_341),
            density=1.0,
            index_dtype=mx.int32,
        )


def test_noncanonical_random_structure_is_explicitly_not_implemented():
    with pytest.raises(NotImplementedError, match="noncanonical"):
        ms.random.random_array((4, 5), canonical=False)


def test_sampler_is_called_once_with_density_rounded_size(mx, to_numpy):
    calls: list[int] = []

    def sampler(*, size):
        calls.append(size)
        return np.ones(size, dtype=np.float32)

    out = ms.random.random(4, 5, density=0.25, data_rvs=sampler)

    assert calls == [density_to_nnz((4, 5), 0.25)]
    np.testing.assert_array_equal(to_numpy(out.data), np.ones(out.nnz))

    with pytest.raises(TypeError, match="callable"):
        ms.random.random_array((4, 5), data_sampler=np.ones(3))


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16", "complex64"])
@pytest.mark.parametrize("index_dtype_name", ["int32", "int64"])
def test_random_array_dtype_and_index_dtype_metadata(
    mx, to_numpy, dtype_name, index_dtype_name
):
    dtype = getattr(mx, dtype_name)
    index_dtype = getattr(mx, index_dtype_name)

    out = ms.random.random_array(
        (7, 9),
        density=0.31,
        format="coo",
        dtype=dtype,
        rng=mx.random.key(9),
        index_dtype=index_dtype,
    )

    assert out.dtype == dtype
    assert out.index_dtype == index_dtype
    assert out.nnz == density_to_nnz((7, 9), 0.31)
    _assert_duplicate_free(out, to_numpy)
    if dtype == mx.complex64:
        assert np.any(np.abs(to_numpy(out.data).imag) > 0)


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_same_key_reproduces_structure_and_values(mx, to_numpy, format_name):
    key = mx.random.key(17)

    first = ms.random.random_array(
        (53, 59),
        density=0.17,
        rng=key,
        format=format_name,
        index_dtype=mx.int64,
    )
    second = ms.random.random_array(
        (53, 59),
        density=0.17,
        rng=key,
        format=format_name,
        index_dtype=mx.int64,
    )

    _assert_same_sparse(first, second, to_numpy)


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_different_explicit_keys_change_structure_and_values(mx, to_numpy, format_name):
    first = ms.random.random_array(
        (89, 97),
        density=0.13,
        format=format_name,
        rng=mx.random.key(101),
    )
    second = ms.random.random_array(
        (89, 97),
        density=0.13,
        format=format_name,
        rng=mx.random.key(102),
    )

    assert not _same_structure(first, second, to_numpy)
    assert not np.array_equal(to_numpy(first.data), to_numpy(second.data))


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_generated_values_are_non_degenerate_uniform_samples(mx, to_numpy, format_name):
    out = ms.random.random_array(
        (257, 263),
        density=0.07,
        format=format_name,
        rng=mx.random.key(404),
    )
    data = to_numpy(out.data)

    assert out.nnz == density_to_nnz((257, 263), 0.07)
    assert np.unique(data).size > min(512, out.nnz // 2)
    assert data.std() > 0.20
    assert data.min() < 0.05
    assert data.max() > 0.95


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_generated_structure_is_spread_across_rows_and_columns(
    mx, to_numpy, format_name
):
    out = ms.random.random_array(
        (113, 127),
        density=0.11,
        format=format_name,
        rng=mx.random.key(505),
    )

    linear = _linear_coordinates(out, to_numpy)
    rows = linear // out.shape[1]
    cols = linear % out.shape[1]
    gaps = np.diff(np.sort(linear))

    _assert_duplicate_free(out, to_numpy)
    assert np.unique(rows).size > out.shape[0] * 0.80
    assert np.unique(cols).size > out.shape[1] * 0.80
    assert np.any(gaps > 1)


def test_csr_and_csc_random_do_not_route_through_coo_conversions(mx, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("random compressed generation must not use COO conversion")

    monkeypatch.setattr(ms.random._native, "coo_tocsr", fail)
    monkeypatch.setattr(ms.random._native, "coo_tocsc", fail)

    csr = ms.random.random_array(
        (11, 13),
        density=0.2,
        format="csr",
        rng=mx.random.key(4),
    )
    csc = ms.random.random_array(
        (11, 13),
        density=0.2,
        format="csc",
        rng=mx.random.key(4),
    )

    assert isinstance(csr, CSRArray)
    assert isinstance(csc, CSCArray)
    assert csr.has_canonical_format
    assert csc.has_canonical_format


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_split_keys_diverge_for_structure_and_values(mx, to_numpy, format_name):
    key_a, key_b = mx.random.split(mx.random.key(17), 2)

    first = ms.random.random_array(
        (83, 89),
        density=0.17,
        rng=key_a,
        format=format_name,
    )
    second = ms.random.random_array(
        (83, 89),
        density=0.17,
        rng=key_b,
        format=format_name,
    )

    assert not _same_structure(first, second, to_numpy)
    assert not np.array_equal(to_numpy(first.data), to_numpy(second.data))


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
@pytest.mark.parametrize(
    ("shape", "density", "expected_nnz"),
    [
        ((3, 4), 0.0, 0),
        ((3, 4), 1.0, 12),
        ((0, 5), 1.0, 0),
        ((5, 0), 1.0, 0),
    ],
)
def test_random_extreme_densities_and_empty_shapes(
    mx, to_numpy, format_name, shape, density, expected_nnz
):
    out = ms.random.random_array(
        shape,
        density=density,
        format=format_name,
        rng=mx.random.key(21),
        index_dtype=mx.int64,
    )

    assert out.shape == shape
    assert out.nnz == expected_nnz
    assert out.index_dtype == mx.int64
    _assert_duplicate_free(out, to_numpy)
    if density == 1.0 and shape[0] and shape[1]:
        np.testing.assert_array_equal(
            np.sort(_linear_coordinates(out, to_numpy)),
            np.arange(shape[0] * shape[1]),
        )


def test_data_sampler_accepts_lazy_mlx_values(mx, to_numpy):
    calls: list[int] = []

    def sampler(*, size):
        calls.append(size)
        return mx.arange(size, dtype=mx.float32)

    out = ms.random.random_array(
        (6, 7),
        density=0.4,
        format="coo",
        rng=mx.random.key(0),
        data_sampler=sampler,
    )

    assert calls == [out.nnz]
    np.testing.assert_array_equal(to_numpy(out.data), np.arange(out.nnz))


@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_data_sampler_preserves_user_value_sequence_for_each_format(
    mx, to_numpy, format_name
):
    calls: list[int] = []

    def sampler(*, size):
        calls.append(size)
        return np.linspace(-3.0, 7.0, num=size, dtype=np.float32)

    out = ms.random.random_array(
        (19, 23),
        density=0.23,
        format=format_name,
        rng=mx.random.key(12),
        data_sampler=sampler,
    )

    assert calls == [out.nnz]
    np.testing.assert_array_equal(
        to_numpy(out.data),
        np.linspace(-3.0, 7.0, num=out.nnz, dtype=np.float32),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("format_name", ["coo", "csr", "csc"])
def test_random_structure_matches_between_cpu_and_metal(mx, to_numpy, format_name):
    cpu = mx.Device(mx.cpu, 0)
    gpu = mx.Device(mx.gpu, 0)
    if not mx.is_available(cpu) or not mx.is_available(gpu):
        pytest.skip("CPU and GPU devices are both required for parity.")

    outputs = []
    for device in (cpu, gpu):
        mx.set_default_device(device)
        out = ms.random.random_array(
            (31, 37),
            density=0.19,
            format=format_name,
            rng=mx.random.key(1234),
            index_dtype=mx.int32,
        )
        if format_name == "coo":
            outputs.append((to_numpy(out.row), to_numpy(out.col), to_numpy(out.data)))
        else:
            outputs.append(
                (to_numpy(out.indices), to_numpy(out.indptr), to_numpy(out.data))
            )

    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
