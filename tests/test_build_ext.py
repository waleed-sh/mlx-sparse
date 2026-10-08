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

import runpy
from pathlib import Path

from mlx import extension
from setuptools import Distribution
from setuptools.command.build_ext import build_ext


def test_inplace_build_preserves_python_source_and_copies_native_artifacts(
    tmp_path, monkeypatch
):
    build_class = runpy.run_path(str(Path(__file__).parents[1] / "setup.py"))[
        "SparseCMakeBuild"
    ]
    source_dir = tmp_path / "source" / "mlx_sparse"
    build_dir = tmp_path / "build" / "mlx_sparse"
    source_dir.mkdir(parents=True)
    build_dir.mkdir(parents=True)
    source = source_dir / "_csr.py"
    source.write_text("current source\n")
    (build_dir / "_csr.py").write_text("stale build output\n")
    artifacts = {
        "_ext.so": b"extension",
        "libmlx_sparse_native.dylib": b"native library",
        "mlx_sparse.metallib": b"metal library",
    }
    for name, contents in artifacts.items():
        (build_dir / name).write_bytes(contents)

    command = build_class(
        Distribution({"ext_modules": [extension.CMakeExtension("mlx_sparse._ext")]})
    )
    command.ensure_finalized()
    command.inplace = True
    monkeypatch.setattr(build_ext, "run", lambda self: None)
    monkeypatch.setattr(command, "get_finalized_command", lambda name: object())
    monkeypatch.setattr(
        command,
        "_get_inplace_equivalent",
        lambda build_py, ext: (str(source_dir / "_ext.so"), str(build_dir / "_ext.so")),
    )

    command.run()

    assert source.read_text() == "current source\n"
    for name, contents in artifacts.items():
        assert (source_dir / name).read_bytes() == contents
