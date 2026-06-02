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

import json
from pathlib import Path

PRECONDITIONER_NOTEBOOKS = (
    "21_preconditioner_identity",
    "22_preconditioner_diagonal",
    "23_preconditioner_jacobi",
    "24_preconditioner_ilu0",
    "25_preconditioner_ichol0",
    "26_preconditioner_chebyshev",
    "27_preconditioner_exact",
    "28_preconditioner_callable",
)


def test_preconditioner_notebooks_are_listed_and_parseable():
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "docs" / "notebooks"
    index_text = (notebook_dir / "index.rst").read_text()

    assert "Sparse preconditioners" in index_text
    for stem in PRECONDITIONER_NOTEBOOKS:
        assert stem in index_text
        notebook = json.loads((notebook_dir / f"{stem}.ipynb").read_text())
        assert notebook["nbformat"] == 4
        assert len(notebook["cells"]) >= 5
        joined_source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert "preconditioners" in joined_source
        assert "benchmarks/" not in joined_source
