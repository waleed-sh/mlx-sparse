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

"""Preconditioner-specific validation helpers."""

from __future__ import annotations

import mlx.core as mx


def normalize_identity_dtype(dtype):
    """Normalize the identity preconditioner dtype argument."""

    if dtype is None or dtype == mx.float32:
        return mx.float32
    raise TypeError("identity preconditioners currently use float32 values.")
