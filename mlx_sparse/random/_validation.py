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

import math
import operator
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable

import mlx.core as mx

from mlx_sparse._typing import INDEX_DTYPES, VALUE_DTYPES, Shape2D

SUPPORTED_FORMATS = ("coo", "csr", "csc")
UNSUPPORTED_SCIPY_FORMATS = ("bsr", "dia", "dok", "lil")

_MAX_NATIVE_ELEMENTS = (1 << 63) - 1
_MAX_UINT64_SEED = (1 << 64) - 1
_MAX_MLX_SHAPE_DIMENSION = (1 << 31) - 1


@dataclass(frozen=True)
class RandomSpec:
    shape: Shape2D
    nnz: int
    format: str
    dtype: mx.Dtype
    rng: mx.array | None
    index_dtype: mx.Dtype
    canonical: bool
    sampler: Callable | None


def normalize_random_array_args(
    shape,
    *,
    density,
    format,
    dtype,
    rng,
    random_state,
    index_dtype,
    canonical,
    sampler,
) -> RandomSpec:
    """Normalize public random-constructor arguments without generating data."""
    shape = normalize_random_shape(shape)
    density = normalize_density(density)
    format = normalize_format(format)
    dtype = normalize_value_dtype(dtype)
    rng = normalize_rng(rng, random_state)
    index_dtype = normalize_index_dtype(index_dtype)
    canonical = normalize_canonical(canonical)
    sampler = normalize_sampler(sampler)
    validate_native_shape_capacity(shape)
    validate_index_capacity(shape, index_dtype)
    nnz = density_to_nnz(shape, density)
    validate_nnz_capacity(nnz, index_dtype)
    return RandomSpec(
        shape=shape,
        nnz=nnz,
        format=format,
        dtype=dtype,
        rng=rng,
        index_dtype=index_dtype,
        canonical=canonical,
        sampler=sampler,
    )


def normalize_random_shape(shape) -> Shape2D:
    try:
        rank = len(shape)
    except TypeError as exc:
        raise TypeError(
            f"shape must be a rank-2 sequence, got {type(shape)!r}."
        ) from exc
    if rank != 2:
        raise ValueError(f"sparse arrays must be rank-2, got shape={tuple(shape)!r}.")
    return (
        normalize_dimension("shape[0]", shape[0]),
        normalize_dimension("shape[1]", shape[1]),
    )


def normalize_dimension(name: str, value) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer, got bool.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a non-negative integer, got {type(value)!r}."
        ) from exc
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative, got {normalized}.")
    return normalized


def normalize_density(density) -> float:
    if isinstance(density, bool) or not isinstance(density, Real):
        raise TypeError(f"density must be a real number in [0, 1], got {density!r}.")
    value = float(density)
    if not math.isfinite(value):
        raise ValueError(f"density must be finite, got {density!r}.")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"density must satisfy 0 <= density <= 1, got {value}.")
    return value


def normalize_format(format) -> str:
    if format is None:
        return "coo"
    if not isinstance(format, str):
        raise TypeError(f"format must be a string or None, got {type(format)!r}.")
    normalized = format.lower()
    if normalized in SUPPORTED_FORMATS:
        return normalized
    if normalized in UNSUPPORTED_SCIPY_FORMATS:
        raise ValueError(
            f"format={format!r} is a SciPy sparse format that mlx_sparse.random "
            "does not support; supported formats are 'coo', 'csr', and 'csc'."
        )
    raise ValueError(
        f"unsupported sparse format {format!r}; supported formats are "
        "'coo', 'csr', and 'csc'."
    )


def normalize_value_dtype(dtype) -> mx.Dtype:
    if dtype is None:
        return mx.float32
    if dtype not in VALUE_DTYPES:
        raise TypeError(
            "dtype must be one of mx.float32, mx.float16, mx.bfloat16, "
            f"or mx.complex64 for the current sparse containers, got {dtype}. "
            "Native random support for float64, integer, and bool values is "
            "reserved for a package-wide sparse dtype expansion."
        )
    return dtype


def normalize_index_dtype(index_dtype) -> mx.Dtype:
    if index_dtype not in INDEX_DTYPES:
        raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")
    return index_dtype


def normalize_canonical(canonical) -> bool:
    if not isinstance(canonical, bool):
        raise TypeError(f"canonical must be a bool, got {type(canonical)!r}.")
    if not canonical:
        raise NotImplementedError(
            "noncanonical duplicate-preserving random output is not implemented; "
            "pass canonical=True to request duplicate-free structure."
        )
    return canonical


def normalize_sampler(sampler):
    if sampler is None:
        return None
    if not callable(sampler):
        raise TypeError(f"data sampler must be callable, got {type(sampler)!r}.")
    return sampler


def normalize_rng(rng, random_state) -> mx.array | None:
    if rng is not None and random_state is not None:
        raise ValueError("rng and random_state are aliases; pass at most one of them.")
    value = rng if rng is not None else random_state
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("rng seeds must be integers, not bool values.")
    if isinstance(value, Integral):
        seed = int(value)
        if seed < 0 or seed > _MAX_UINT64_SEED:
            raise ValueError(
                f"integer rng seeds must satisfy 0 <= seed <= {_MAX_UINT64_SEED}, "
                f"got {seed}."
            )
        return mx.random.key(seed)
    if _is_mlx_prng_key(value):
        return value
    if _is_numpy_generator(value):
        raise TypeError(
            "NumPy Generator and RandomState inputs are not accepted by "
            "mlx_sparse.random. Pass an MLX PRNG key from mx.random.key(seed) "
            "or an integer seed through rng instead."
        )
    if isinstance(value, mx.array):
        raise TypeError(
            "rng must be an MLX PRNG key with dtype mx.uint32 and shape (2,), "
            f"got dtype={value.dtype} and shape={value.shape}."
        )
    raise TypeError(
        "rng must be None, an integer seed, or an MLX PRNG key created by "
        f"mx.random.key(seed), got {type(value)!r}."
    )


def density_to_nnz(shape: Shape2D, density: float) -> int:
    total = shape[0] * shape[1]
    if total > _MAX_NATIVE_ELEMENTS:
        raise OverflowError(
            "random sparse shape is too large for native generation: "
            f"m * n = {total} exceeds {_MAX_NATIVE_ELEMENTS}."
        )
    return max(0, min(total, int(round(total * density))))


def validate_index_capacity(shape: Shape2D, index_dtype) -> None:
    limit = _index_max(index_dtype)
    for name, dimension in (("n_rows", shape[0]), ("n_cols", shape[1])):
        if dimension and dimension - 1 > limit:
            raise OverflowError(
                f"{name}={dimension} cannot be represented by {index_dtype} "
                f"indices with maximum coordinate {limit}."
            )


def validate_native_shape_capacity(shape: Shape2D) -> None:
    for name, dimension in (("n_rows", shape[0]), ("n_cols", shape[1])):
        if dimension > _MAX_MLX_SHAPE_DIMENSION:
            raise OverflowError(
                f"{name}={dimension} exceeds the native MLX shape limit "
                f"{_MAX_MLX_SHAPE_DIMENSION}."
            )


def validate_nnz_capacity(nnz: int, index_dtype) -> None:
    if nnz > _MAX_MLX_SHAPE_DIMENSION:
        raise OverflowError(
            f"nnz={nnz} exceeds the native MLX output shape limit "
            f"{_MAX_MLX_SHAPE_DIMENSION}."
        )
    limit = _index_max(index_dtype)
    if nnz > limit:
        raise OverflowError(
            f"nnz={nnz} cannot be represented by {index_dtype} index buffers "
            f"with maximum value {limit}."
        )


def _index_max(index_dtype) -> int:
    if index_dtype == mx.int32:
        return (1 << 31) - 1
    if index_dtype == mx.int64:
        return (1 << 63) - 1
    raise TypeError(f"index_dtype must be mx.int32 or mx.int64, got {index_dtype}.")


def _is_mlx_prng_key(value) -> bool:
    return (
        isinstance(value, mx.array)
        and value.dtype == mx.uint32
        and value.ndim == 1
        and value.shape == (2,)
    )


def _is_numpy_generator(value) -> bool:
    module = type(value).__module__
    name = type(value).__name__
    return module.startswith("numpy.random") and name in {"Generator", "RandomState"}
