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

"""Internal iterative-solver diagnostics.

This subpackage owns solver status normalization, structured diagnostics, and
exit-callback plumbing. Input normalization for iterative solvers stays in
``mlx_sparse.linalg.utils.iterative``.
"""

from __future__ import annotations

from mlx_sparse.linalg.utils.diagnostics.callbacks import (
    finish_solver_result,
    invoke_callback,
)
from mlx_sparse.linalg.utils.diagnostics.info import SolverInfo
from mlx_sparse.linalg.utils.diagnostics.status import (
    make_solver_info,
    solver_info_to_int,
)

__all__ = [
    "SolverInfo",
    "finish_solver_result",
    "invoke_callback",
    "make_solver_info",
    "solver_info_to_int",
]
