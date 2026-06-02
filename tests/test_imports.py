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
from types import SimpleNamespace


def test_public_package_imports_without_build_side_effects():
    module = importlib.import_module("mlx_sparse")

    assert "CSRArray" in module.__all__
    assert "CSCArray" in module.__all__
    assert "COOArray" in module.__all__
    assert "coo_matmat" in module.__all__
    assert "coo_matmul" in module.__all__
    assert "coo_matvec" in module.__all__
    assert "csc_matmat" in module.__all__
    assert "csc_matmul" in module.__all__
    assert "csc_matvec" in module.__all__
    assert "csr_matmul" in module.__all__
    assert "use_gpu" in module.__all__
    assert callable(module.issparse)


def test_native_extension_imports_after_editable_build():
    ext = importlib.import_module("mlx_sparse._ext")

    assert hasattr(ext, "csr_matvec")
    assert hasattr(ext, "csr_batched_matvec")
    assert hasattr(ext, "csr_matmul")
    assert hasattr(ext, "csr_batched_matmul")
    assert hasattr(ext, "csr_transpose")
    assert hasattr(ext, "coo_tocsr")
    assert hasattr(ext, "coo_tocsc")
    assert hasattr(ext, "csr_todense")
    assert hasattr(ext, "csc_todense")
    assert hasattr(ext, "csr_tocsc")
    assert hasattr(ext, "csc_tocsr")
    assert hasattr(ext, "csc_sort_indices")
    assert hasattr(ext, "csc_sum_duplicates")
    assert hasattr(ext, "_accelerate_csc_adapter_summary_for_testing")
    assert hasattr(ext, "_accelerate_csr_adapter_summary_for_testing")
    assert hasattr(ext, "_accelerate_coo_adapter_summary_for_testing")
    assert hasattr(ext, "_accelerate_factorization_wrapper_summary_for_testing")
    assert hasattr(ext, "_accelerate_factorization_failure_for_testing")
    assert hasattr(ext, "coo_matvec")
    assert hasattr(ext, "coo_matmat")
    assert hasattr(ext, "coo_matmul")
    assert hasattr(ext, "coo_batched_matvec")
    assert hasattr(ext, "coo_batched_matmul")
    assert hasattr(ext, "csc_matvec")
    assert hasattr(ext, "csc_matmat")
    assert hasattr(ext, "csc_matmul")
    assert hasattr(ext, "csc_batched_matvec")
    assert hasattr(ext, "csc_batched_matmul")
    assert hasattr(ext, "csc_matvec_transpose")
    assert hasattr(ext, "csc_matmul_transpose")
    assert hasattr(ext, "coo_matmul_data_vjp")
    assert hasattr(ext, "csc_matmul_data_vjp")
    assert hasattr(ext, "csr_normal_lanczos")


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

        def coo_tocsc(self, *args, **kwargs):
            seen.append(("coo_tocsc", kwargs))
            return "coo_csc"

        def csr_todense(self, *args, **kwargs):
            seen.append(("csr_todense", kwargs))
            return "dense"

        def csc_todense(self, *args, **kwargs):
            seen.append(("csc_todense", kwargs))
            return "dense_csc"

        def csr_matvec(self, *args, **kwargs):
            seen.append(("csr_matvec", kwargs))
            return "matvec"

        def csc_matvec(self, *args, **kwargs):
            seen.append(("csc_matvec", kwargs))
            return "csc_matvec"

        def coo_matvec(self, *args, **kwargs):
            seen.append(("coo_matvec", kwargs))
            return "coo_matvec"

        def csr_batched_matvec(self, *args, **kwargs):
            seen.append(("csr_batched_matvec", kwargs))
            return "batched_matvec"

        def coo_batched_matvec(self, *args, **kwargs):
            seen.append(("coo_batched_matvec", kwargs))
            return "coo_batched_matvec"

        def csc_batched_matvec(self, *args, **kwargs):
            seen.append(("csc_batched_matvec", kwargs))
            return "csc_batched_matvec"

        def csr_matmul(self, *args, **kwargs):
            seen.append(("csr_matmul", kwargs))
            return "matmul"

        def coo_matmul(self, *args, **kwargs):
            seen.append(("coo_matmul", kwargs))
            return "coo_matmul"

        def csc_matmul(self, *args, **kwargs):
            seen.append(("csc_matmul", kwargs))
            return "csc_matmul"

        def coo_matmat(self, *args, **kwargs):
            seen.append(("coo_matmat", kwargs))
            return "coo_matmat"

        def csc_matmat(self, *args, **kwargs):
            seen.append(("csc_matmat", kwargs))
            return "csc_matmat"

        def csr_batched_matmul(self, *args, **kwargs):
            seen.append(("csr_batched_matmul", kwargs))
            return "batched_matmul"

        def coo_batched_matmul(self, *args, **kwargs):
            seen.append(("coo_batched_matmul", kwargs))
            return "coo_batched_matmul"

        def csc_batched_matmul(self, *args, **kwargs):
            seen.append(("csc_batched_matmul", kwargs))
            return "csc_batched_matmul"

        def csr_transpose(self, *args, **kwargs):
            seen.append(("csr_transpose", kwargs))
            return "transpose"

        def csr_tocsc(self, *args, **kwargs):
            seen.append(("csr_tocsc", kwargs))
            return "tocsc"

        def csc_tocsr(self, *args, **kwargs):
            seen.append(("csc_tocsr", kwargs))
            return "tocsr"

        def csc_matvec_transpose(self, *args, **kwargs):
            seen.append(("csc_matvec_transpose", kwargs))
            return "csc_matvec_transpose"

        def csc_matmul_transpose(self, *args, **kwargs):
            seen.append(("csc_matmul_transpose", kwargs))
            return "csc_matmul_transpose"

        def csr_sort_indices(self, *args, **kwargs):
            seen.append(("csr_sort_indices", kwargs))
            return "sort"

        def csc_sort_indices(self, *args, **kwargs):
            seen.append(("csc_sort_indices", kwargs))
            return "sort_csc"

        def csc_sum_duplicates(self, *args, **kwargs):
            seen.append(("csc_sum_duplicates", kwargs))
            return "sum_csc"

        def csr_normal_lanczos(self, *args, **kwargs):
            seen.append(("csr_normal_lanczos", kwargs))
            return "normal_lanczos"

    monkeypatch.setattr(native, "extension", lambda: FakeExt())

    assert native.identity_like("x") == "identity"
    assert native.coo_tocsr("data", "row", "col", (1, 2)) == "coo"
    assert native.coo_tocsc("data", "row", "col", (1, 2)) == "coo_csc"
    assert native.csr_todense("data", "indices", "indptr", (1, 2)) == "dense"
    assert native.csc_todense("data", "indices", "indptr", (1, 2)) == "dense_csc"
    assert native.csr_matvec("data", "indices", "indptr", "x", (1, 2)) == "matvec"
    assert native.csc_matvec("data", "indices", "indptr", "x", (1, 2)) == "csc_matvec"
    assert native.coo_matvec("data", "row", "col", "x", (1, 2)) == "coo_matvec"
    assert (
        native.csr_batched_matvec("data", "indices", "indptr", "x", (1, 2))
        == "batched_matvec"
    )
    assert (
        native.coo_batched_matvec("data", "row", "col", "x", (1, 2))
        == "coo_batched_matvec"
    )
    assert (
        native.csc_batched_matvec("data", "indices", "indptr", "x", (1, 2))
        == "csc_batched_matvec"
    )
    assert native.csr_matmul("data", "indices", "indptr", "x", (1, 2)) == "matmul"
    assert native.coo_matmul("data", "row", "col", "x", (1, 2)) == "coo_matmul"
    assert native.csc_matmul("data", "indices", "indptr", "x", (1, 2)) == "csc_matmul"
    coo_lhs = SimpleNamespace(data="ld", row="lr", col="lc", shape=(1, 2))
    coo_rhs = SimpleNamespace(data="rd", row="rr", col="rc", shape=(2, 3))
    csc_lhs = SimpleNamespace(data="ld", indices="li", indptr="lp", shape=(1, 2))
    csc_rhs = SimpleNamespace(data="rd", indices="ri", indptr="rp", shape=(2, 3))
    assert native.coo_matmat(coo_lhs, coo_rhs) == "coo_matmat"
    assert native.csc_matmat(csc_lhs, csc_rhs) == "csc_matmat"
    assert (
        native.csr_batched_matmul("data", "indices", "indptr", "x", (1, 2))
        == "batched_matmul"
    )
    assert (
        native.coo_batched_matmul("data", "row", "col", "x", (1, 2))
        == "coo_batched_matmul"
    )
    assert (
        native.csc_batched_matmul("data", "indices", "indptr", "x", (1, 2))
        == "csc_batched_matmul"
    )
    assert native.csr_transpose("data", "indices", "indptr", (1, 2)) == "transpose"
    assert native.csr_tocsc("data", "indices", "indptr", (1, 2)) == "tocsc"
    assert native.csc_tocsr("data", "indices", "indptr", (1, 2)) == "tocsr"
    assert (
        native.csc_matvec_transpose("data", "indices", "indptr", "x", (1, 2))
        == "csc_matvec_transpose"
    )
    assert (
        native.csc_matmul_transpose("data", "indices", "indptr", "x", (1, 2))
        == "csc_matmul_transpose"
    )
    assert native.csr_sort_indices("data", "indices", "indptr") == "sort"
    assert native.csc_sort_indices("data", "indices", "indptr") == "sort_csc"
    assert native.csc_sum_duplicates("data", "indices", "indptr") == "sum_csc"
    assert (
        native.csr_normal_lanczos("data", "indices", "indptr", "v0", (2, 3), k=2)
        == "normal_lanczos"
    )
    assert all("stream" not in kwargs for _, kwargs in seen)
