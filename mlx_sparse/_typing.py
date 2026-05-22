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

from typing import Callable, Literal, Protocol, TypeAlias

import mlx.core as mx

Shape2D: TypeAlias = tuple[int, int]
ValidationMode: TypeAlias = bool | Literal["metadata", "full"]

INDEX_DTYPES = (mx.int32, mx.int64)
VALUE_DTYPES = (mx.float32, mx.float16, mx.bfloat16, mx.complex64)

Matvec = Callable[[mx.array], mx.array]
Matmat = Callable[[mx.array], mx.array]


class SparseArray(Protocol):
    shape: Shape2D
    data: mx.array

    @property
    def nnz(self) -> int: ...

    def todense(self) -> mx.array: ...


def is_index_dtype(dtype) -> bool:
    return dtype in INDEX_DTYPES


def is_supported_value_dtype(dtype) -> bool:
    return dtype in VALUE_DTYPES


def is_available() -> bool:
    """Return ``True`` if the native C++ extension is loaded.

    The mlx-sparse native extension (``_ext``) provides MLX-primitive
    implementations of sparse operations with CPU and Metal backends. When it
    is absent (e.g. a pure-source checkout without a build step), all
    operations fall back to NumPy-based Python implementations in
    ``mlx_sparse._fallback``.

    Returns:
        ``True`` if ``mlx_sparse._ext`` was successfully imported at package
        load time, ``False`` otherwise.

    Example::

        import mlx_sparse as ms

        if not ms.is_available():
            print("Native extension not found. Using Python fallback.")
    """
    try:
        import mlx_sparse._ext  # noqa: F401
    except Exception:
        return False
    return True
