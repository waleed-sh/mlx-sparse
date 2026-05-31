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

"""Internal linalg helper utilities.

The public linalg modules should describe algorithms and public objects. Shared
normalization, dtype promotion, residual checking, and solver bookkeeping live
in this subpackage so the algorithm modules do not accumulate unrelated private
helper functions.
"""

from __future__ import annotations

__all__ = [
    "arrays",
    "factorization",
    "iterative",
    "operators",
    "preconditioners",
    "sparse",
    "spectral",
]
