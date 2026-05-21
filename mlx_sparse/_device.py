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

import mlx.core as mx


def _device(kind, index: int):
    return mx.Device(kind, index)


def use_cpu(index: int = 0, *, require_available: bool = True) -> mx.Device:
    """Set MLX's default device to CPU and return the device object.

    Calling this once at the start of a script or test is the recommended way
    to pin all subsequent MLX and mlx-sparse operations to the CPU. The setting
    persists for the lifetime of the Python process or until overridden by
    another ``use_*`` call.

    Args:
        index: CPU device index. Almost always ``0``. Multiple CPU devices are
            not typical in Apple Silicon runtimes. Default ``0``.
        require_available: If ``True`` (default), probe the device immediately
            with a trivial ``mx.eval`` and raise ``RuntimeError`` if it fails.
            Set to ``False`` to skip the probe (e.g. in environments where
            eager evaluation is not yet possible).

    Returns:
        The ``mlx.core.Device`` object that was set as the default.

    Raises:
        RuntimeError: If ``require_available=True`` and the CPU device cannot
            be probed successfully.

    Example::

        import mlx_sparse as ms

        ms.use_cpu()  # pin to CPU
        y = A @ x  # runs on CPU
    """
    device = _device(mx.cpu, index)
    mx.set_default_device(device)
    if require_available:
        try:
            probe = mx.array([0.0])
            mx.eval(probe)
        except Exception as exc:
            raise RuntimeError(
                f"MLX CPU device {index} is not available to this Python "
                "process. Verify that native MLX can create a CPU array in the "
                "same virtual environment."
            ) from exc
    return device


def use_gpu(index: int = 0, *, require_available: bool = True) -> mx.Device:
    """Set MLX's default device to GPU (Metal) and return the device object.

    On Apple Silicon, this selects the integrated GPU via MLX's Metal backend.
    Fixed-shape sparse primitives dispatch native Metal kernels for supported
    value and index dtypes. Operations with dynamic output structure, such as
    sparse-sparse products and duplicate summation, may synchronize to host for
    structural assembly.

    Args:
        index: GPU device index. ``0`` selects the primary GPU. Default ``0``.
        require_available: If ``True`` (default), verify that ``mx.is_available``
            returns ``True`` for the selected device and that a trivial array
            can be evaluated. Raises ``RuntimeError`` if either check fails.

    Returns:
        The ``mlx.core.Device`` object that was set as the default.

    Raises:
        RuntimeError: If ``require_available=True`` and the GPU is not
            available or cannot be probed.

    Example::

        import mlx_sparse as ms

        ms.use_gpu()  # pin to GPU
        y = A @ x  # dispatches Metal csr_matvec kernel
    """
    device = _device(mx.gpu, index)
    mx.set_default_device(device)
    if require_available:
        try:
            available = mx.is_available(device)
            probe = mx.array([0.0])
            mx.eval(probe)
        except Exception as exc:
            raise RuntimeError(
                f"MLX GPU device {index} is not available to this Python "
                "process. Verify that native MLX can create a GPU array in the "
                "same virtual environment."
            ) from exc
        if not available:
            raise RuntimeError(
                f"MLX GPU device {index} is not available. Native MLX must be "
                "able to create arrays on the GPU before mlx-sparse GPU kernels "
                "can run."
            )
    return device


def use_device(name: str, index: int = 0) -> mx.Device:
    """Set MLX's default device by name string.

    A convenience wrapper around :func:`use_cpu` and :func:`use_gpu` that
    accepts a plain string device name. Useful when the target device is
    provided as a command-line argument.

    Args:
        name: ``"cpu"`` or ``"gpu"`` (case-insensitive).
        index: Device index. Default ``0``.

    Returns:
        The ``mlx.core.Device`` object that was set as the default.

    Raises:
        ValueError: If ``name`` is not ``"cpu"`` or ``"gpu"``.

    Example::

        import argparse
        import mlx_sparse as ms

        parser = argparse.ArgumentParser()
        parser.add_argument("--device", default="gpu")
        args = parser.parse_args()
        ms.use_device(args.device)
    """
    normalized = name.lower()
    if normalized == "cpu":
        return use_cpu(index)
    if normalized == "gpu":
        return use_gpu(index)
    raise ValueError(f"device must be 'cpu' or 'gpu', got {name!r}.")
