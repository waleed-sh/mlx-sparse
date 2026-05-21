import os
import subprocess
import sys

from mlx import extension
from setuptools import setup

if __name__ == "__main__":
    cmake_args = os.environ.get("CMAKE_ARGS", "")
    deployment_target = os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")
    mlx_cmake_dir = subprocess.check_output(
        [sys.executable, "-m", "mlx", "--cmake-dir"],
        text=True,
    ).strip()
    nanobind_cmake_dir = subprocess.check_output(
        [sys.executable, "-m", "nanobind", "--cmake_dir"],
        text=True,
    ).strip()
    python_arg = f"-DPython_EXECUTABLE={sys.executable}"
    deployment_arg = f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"
    mlx_arg = f"-DMLX_DIR={mlx_cmake_dir}"
    nanobind_arg = f"-Dnanobind_DIR={nanobind_cmake_dir}"
    os.environ["CMAKE_ARGS"] = " ".join(
        part
        for part in (
            cmake_args,
            "-UMLX_DIR",
            "-UMLX_LIBRARY",
            "-UMLX_INCLUDE_DIRS",
            "-UFIND_PACKAGE_MESSAGE_DETAILS_MLX",
            "-Unanobind_DIR",
            python_arg,
            deployment_arg,
            mlx_arg,
            nanobind_arg,
        )
        if part
    )

    setup(
        ext_modules=[extension.CMakeExtension("mlx_sparse._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        package_data={"mlx_sparse": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
    )
