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


def test_public_package_imports_without_build_side_effects():
    module = importlib.import_module("mlx_sparse")

    assert "CSRArray" in module.__all__
    assert "COOArray" in module.__all__
    assert "csr_matmul" in module.__all__
    assert "use_gpu" in module.__all__
    assert callable(module.issparse)


def test_native_extension_imports_after_editable_build():
    ext = importlib.import_module("mlx_sparse._ext")

    assert hasattr(ext, "csr_matvec")
    assert hasattr(ext, "csr_matmul")
    assert hasattr(ext, "csr_transpose")
    assert hasattr(ext, "coo_tocsr")
    assert hasattr(ext, "csr_todense")


def test_native_wrappers_do_not_expose_stream_keyword(monkeypatch):
    native = importlib.import_module("mlx_sparse._native")
    seen: list[tuple[str, dict]] = []

    class FakeExt:
        def identity_like(self, *args, **kwargs):
            seen.append(("identity_like", kwargs))
            return "identity"

        def coo_tocsr(self, *args, **kwargs):
            seen.append(("coo_tocsr", kwargs))
            return "coo"

        def csr_todense(self, *args, **kwargs):
            seen.append(("csr_todense", kwargs))
            return "dense"

        def csr_matvec(self, *args, **kwargs):
            seen.append(("csr_matvec", kwargs))
            return "matvec"

        def csr_matmul(self, *args, **kwargs):
            seen.append(("csr_matmul", kwargs))
            return "matmul"

        def csr_transpose(self, *args, **kwargs):
            seen.append(("csr_transpose", kwargs))
            return "transpose"

        def csr_sort_indices(self, *args, **kwargs):
            seen.append(("csr_sort_indices", kwargs))
            return "sort"

    monkeypatch.setattr(native, "extension", lambda: FakeExt())

    assert native.identity_like("x") == "identity"
    assert native.coo_tocsr("data", "row", "col", (1, 2)) == "coo"
    assert native.csr_todense("data", "indices", "indptr", (1, 2)) == "dense"
    assert native.csr_matvec("data", "indices", "indptr", "x", (1, 2)) == "matvec"
    assert native.csr_matmul("data", "indices", "indptr", "x", (1, 2)) == "matmul"
    assert native.csr_transpose("data", "indices", "indptr", (1, 2)) == "transpose"
    assert native.csr_sort_indices("data", "indices", "indptr") == "sort"
    assert all("stream" not in kwargs for _, kwargs in seen)
