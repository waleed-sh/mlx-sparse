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

from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Iterator, Mapping

import mlx.core as mx

from mlx_sparse._ext_loader import extension


class NativeBackend(str, Enum):
    """Native execution backend families known to ``mlx-sparse``."""

    CPU = "cpu"
    METAL = "metal"
    ACCELERATE = "accelerate"
    CUDA = "cuda"
    ROCM = "rocm"


class NativeCapability(str, Enum):
    """Runtime-checkable native capabilities.

    The enum is intentionally backend-oriented. Future releases can add more
    fine-grained operation capabilities without changing the public
    :data:`capabilities` view.
    """

    NATIVE_EXTENSION = "native_extension"
    CPU_KERNELS = "cpu_kernels"
    METAL_KERNELS = "metal_kernels"
    ACCELERATE_SOLVERS = "accelerate_solvers"
    CUDA_KERNELS = "cuda_kernels"
    ROCM_KERNELS = "rocm_kernels"


class NativeCapabilityStatus(str, Enum):
    """Availability state for a native capability."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_BUILT = "not_built"


@dataclass(frozen=True)
class NativeCapabilityRecord:
    """Status record for one :class:`NativeCapability`."""

    capability: NativeCapability
    status: NativeCapabilityStatus
    backend: NativeBackend | None = None
    built: bool = False
    runtime_available: bool = False
    reason: str = ""

    @property
    def available(self) -> bool:
        """Return ``True`` when the capability can be used now."""

        return self.status is NativeCapabilityStatus.AVAILABLE


@dataclass(frozen=True)
class NativeCapabilities:
    """Snapshot of native capabilities for the current Python process."""

    records: tuple[NativeCapabilityRecord, ...]
    platform: str
    architecture: str

    def __iter__(self) -> Iterator[NativeCapabilityRecord]:
        return iter(self.records)

    def __contains__(self, capability: NativeCapability | str) -> bool:
        return self.has(capability)

    def __getitem__(self, capability: NativeCapability | str) -> NativeCapabilityRecord:
        return self.get(capability)

    def get(self, capability: NativeCapability | str) -> NativeCapabilityRecord:
        """Return the status record for ``capability``."""

        key = _coerce_capability(capability)
        for record in self.records:
            if record.capability is key:
                return record
        raise KeyError(key)

    def has(self, capability: NativeCapability | str) -> bool:
        """Return ``True`` if ``capability`` is available now."""

        return self.get(capability).available

    def status(self, capability: NativeCapability | str) -> NativeCapabilityStatus:
        """Return the enum status for ``capability``."""

        return self.get(capability).status

    def by_backend(
        self, backend: NativeBackend | str
    ) -> tuple[NativeCapabilityRecord, ...]:
        """Return all capability records associated with ``backend``."""

        key = _coerce_backend(backend)
        return tuple(record for record in self.records if record.backend is key)

    @property
    def available(self) -> frozenset[NativeCapability]:
        """Capabilities that are available in the current process."""

        return frozenset(
            record.capability for record in self.records if record.available
        )


class _CapabilityView:
    """User-facing native capability view.

    ``mlx_sparse.capabilities`` is intentionally small and string-friendly:

    * capability names: ``"extension"``, ``"cpu"``, ``"metal"``,
      ``"accelerate"``, ``"cuda"``, ``"rocm"``
    * statuses: ``"available"``, ``"unavailable"``, ``"not_built"``

    Example::

        import mlx_sparse as ms

        if ms.capabilities.METAL:
            ms.use_gpu()

        if ms.capabilities.status("accelerate") == "not_built":
            ...
    """

    _PUBLIC_NAMES = ("extension", "cpu", "metal", "accelerate", "cuda", "rocm")

    def __repr__(self) -> str:
        statuses = ", ".join(
            f"{name}={self.status(name)!r}" for name in self._PUBLIC_NAMES
        )
        return f"mlx_sparse.capabilities({statuses})"

    @property
    def extension(self) -> bool:
        """Whether the native extension is loaded."""

        return self.has("extension")

    @property
    def EXTENSION(self) -> bool:
        """Whether the native extension is loaded."""

        return self.extension

    @property
    def cpu(self) -> bool:
        """Whether native CPU kernels are available."""

        return self.has("cpu")

    @property
    def CPU(self) -> bool:
        """Whether native CPU kernels are available."""

        return self.cpu

    @property
    def metal(self) -> bool:
        """Whether native Metal kernels are available."""

        return self.has("metal")

    @property
    def METAL(self) -> bool:
        """Whether native Metal kernels are available."""

        return self.metal

    @property
    def accelerate(self) -> bool:
        """Whether Accelerate solver support is available."""

        return self.has("accelerate")

    @property
    def ACCELERATE(self) -> bool:
        """Whether Accelerate solver support is available."""

        return self.accelerate

    @property
    def cuda(self) -> bool:
        """Whether native CUDA kernels are available."""

        return self.has("cuda")

    @property
    def CUDA(self) -> bool:
        """Whether native CUDA kernels are available."""

        return self.cuda

    @property
    def rocm(self) -> bool:
        """Whether native ROCm/HIP kernels are available."""

        return self.has("rocm")

    @property
    def ROCM(self) -> bool:
        """Whether native ROCm/HIP kernels are available."""

        return self.rocm

    @property
    def names(self) -> tuple[str, ...]:
        """Public capability names accepted by ``has`` and ``status``."""

        return self._PUBLIC_NAMES

    @property
    def platform(self) -> str:
        """Native extension platform reported by the current build."""

        return _native_capabilities().platform

    @property
    def architecture(self) -> str:
        """Native extension architecture reported by the current build."""

        return _native_capabilities().architecture

    def has(self, capability: NativeCapability | str) -> bool:
        """Return ``True`` if ``capability`` is available now."""

        return _native_capabilities().has(capability)

    def status(self, capability: NativeCapability | str) -> str:
        """Return ``"available"``, ``"unavailable"``, or ``"not_built"``."""

        return _native_capabilities().status(capability).value

    def reason(self, capability: NativeCapability | str) -> str:
        """Return a human-readable reason for the current status."""

        return _native_capabilities().get(capability).reason

    def built(self, capability: NativeCapability | str) -> bool:
        """Return ``True`` if ``capability`` was compiled into this build."""

        return _native_capabilities().get(capability).built

    def runtime_available(self, capability: NativeCapability | str) -> bool:
        """Return ``True`` if the runtime can use a compiled capability."""

        return _native_capabilities().get(capability).runtime_available


capabilities = _CapabilityView()


def _native_capabilities() -> NativeCapabilities:
    """Return enum-backed native capability status for this process."""

    facts = _compiled_facts()
    platform = str(facts.get("platform") or _python_platform())
    architecture = str(facts.get("architecture") or _python_architecture())
    ext_loaded = bool(facts.get("extension", False))

    records = [
        _extension_record(ext_loaded),
        _cpu_record(ext_loaded, bool(facts.get("cpu", False))),
        _metal_record(ext_loaded, bool(facts.get("metal", False))),
        _backend_record(
            NativeCapability.ACCELERATE_SOLVERS,
            NativeBackend.ACCELERATE,
            ext_loaded,
            bool(facts.get("accelerate", False)),
            platform,
            "Accelerate sparse solver integration is not compiled into this build.",
        ),
        _backend_record(
            NativeCapability.CUDA_KERNELS,
            NativeBackend.CUDA,
            ext_loaded,
            bool(facts.get("cuda", False)),
            platform,
            "CUDA kernels are not compiled into this build.",
        ),
        _backend_record(
            NativeCapability.ROCM_KERNELS,
            NativeBackend.ROCM,
            ext_loaded,
            bool(facts.get("rocm", False)),
            platform,
            "ROCm/HIP kernels are not compiled into this build.",
        ),
    ]
    return NativeCapabilities(
        records=tuple(records),
        platform=platform,
        architecture=architecture,
    )


def has_capability(capability: NativeCapability | str) -> bool:
    """Return ``True`` if ``capability`` is available in this process."""

    return capabilities.has(capability)


def _coerce_capability(capability: NativeCapability | str) -> NativeCapability:
    if isinstance(capability, NativeCapability):
        return capability
    aliases = {
        "extension": NativeCapability.NATIVE_EXTENSION,
        "native": NativeCapability.NATIVE_EXTENSION,
        "native_extension": NativeCapability.NATIVE_EXTENSION,
        "cpu": NativeCapability.CPU_KERNELS,
        "cpu_kernels": NativeCapability.CPU_KERNELS,
        "metal": NativeCapability.METAL_KERNELS,
        "gpu": NativeCapability.METAL_KERNELS,
        "metal_kernels": NativeCapability.METAL_KERNELS,
        "accelerate": NativeCapability.ACCELERATE_SOLVERS,
        "accelerate_solvers": NativeCapability.ACCELERATE_SOLVERS,
        "cuda": NativeCapability.CUDA_KERNELS,
        "cuda_kernels": NativeCapability.CUDA_KERNELS,
        "rocm": NativeCapability.ROCM_KERNELS,
        "hip": NativeCapability.ROCM_KERNELS,
        "rocm_kernels": NativeCapability.ROCM_KERNELS,
    }
    normalized = capability.lower().replace("-", "_")
    if normalized in aliases:
        return aliases[normalized]
    return NativeCapability(capability)


def _coerce_backend(backend: NativeBackend | str) -> NativeBackend:
    if isinstance(backend, NativeBackend):
        return backend
    return NativeBackend(backend)


def _compiled_facts() -> Mapping[str, object]:
    ext = extension()
    if ext is None:
        return {
            "extension": False,
            "cpu": False,
            "metal": False,
            "accelerate": False,
            "cuda": False,
            "rocm": False,
            "platform": _python_platform(),
            "architecture": _python_architecture(),
        }

    getter = getattr(ext, "_compiled_capabilities", None)
    if getter is not None:
        return getter()

    # Older editable builds may have the extension loaded before this binding
    # exists. Keep capability checks useful until the extension is rebuilt.
    return {
        "extension": True,
        "cpu": True,
        "metal": _metallib_present(),
        "accelerate": False,
        "cuda": False,
        "rocm": False,
        "platform": _python_platform(),
        "architecture": _python_architecture(),
    }


def _extension_record(loaded: bool) -> NativeCapabilityRecord:
    if loaded:
        return NativeCapabilityRecord(
            capability=NativeCapability.NATIVE_EXTENSION,
            status=NativeCapabilityStatus.AVAILABLE,
            built=True,
            runtime_available=True,
            reason="The mlx_sparse native extension is loaded.",
        )
    return NativeCapabilityRecord(
        capability=NativeCapability.NATIVE_EXTENSION,
        status=NativeCapabilityStatus.UNAVAILABLE,
        reason=(
            "The mlx_sparse native extension is not loaded; Python fallback "
            "implementations will be used where available."
        ),
    )


def _cpu_record(extension_loaded: bool, built: bool) -> NativeCapabilityRecord:
    if not extension_loaded:
        return NativeCapabilityRecord(
            capability=NativeCapability.CPU_KERNELS,
            backend=NativeBackend.CPU,
            status=NativeCapabilityStatus.UNAVAILABLE,
            reason="Native CPU kernels require the mlx_sparse extension.",
        )
    if not built:
        return NativeCapabilityRecord(
            capability=NativeCapability.CPU_KERNELS,
            backend=NativeBackend.CPU,
            status=NativeCapabilityStatus.NOT_BUILT,
            reason="Native CPU kernels are not compiled into this build.",
        )

    available, reason = _probe_mlx_device(mx.cpu, "CPU")
    return NativeCapabilityRecord(
        capability=NativeCapability.CPU_KERNELS,
        backend=NativeBackend.CPU,
        status=(
            NativeCapabilityStatus.AVAILABLE
            if available
            else NativeCapabilityStatus.UNAVAILABLE
        ),
        built=True,
        runtime_available=available,
        reason=reason if reason else "Native C++ CPU sparse kernels are available.",
    )


def _metal_record(extension_loaded: bool, built: bool) -> NativeCapabilityRecord:
    if not extension_loaded:
        return NativeCapabilityRecord(
            capability=NativeCapability.METAL_KERNELS,
            backend=NativeBackend.METAL,
            status=NativeCapabilityStatus.UNAVAILABLE,
            reason="Metal kernels require the mlx_sparse extension.",
        )
    if not built:
        return NativeCapabilityRecord(
            capability=NativeCapability.METAL_KERNELS,
            backend=NativeBackend.METAL,
            status=NativeCapabilityStatus.NOT_BUILT,
            reason=(
                "Metal kernels are not compiled into this build or the "
                "mlx_sparse.metallib resource is missing."
            ),
        )

    available, reason = _probe_mlx_device(mx.gpu, "Metal GPU")
    return NativeCapabilityRecord(
        capability=NativeCapability.METAL_KERNELS,
        backend=NativeBackend.METAL,
        status=(
            NativeCapabilityStatus.AVAILABLE
            if available
            else NativeCapabilityStatus.UNAVAILABLE
        ),
        built=True,
        runtime_available=available,
        reason=reason if reason else "Native Metal sparse kernels are available.",
    )


def _backend_record(
    capability: NativeCapability,
    backend: NativeBackend,
    extension_loaded: bool,
    built: bool,
    platform: str,
    not_built_reason: str,
) -> NativeCapabilityRecord:
    if not extension_loaded:
        return NativeCapabilityRecord(
            capability=capability,
            backend=backend,
            status=NativeCapabilityStatus.UNAVAILABLE,
            reason=f"{backend.value} support requires the mlx_sparse extension.",
        )
    if not built:
        return NativeCapabilityRecord(
            capability=capability,
            backend=backend,
            status=NativeCapabilityStatus.NOT_BUILT,
            reason=not_built_reason,
        )

    available, reason = _future_backend_runtime_status(backend, platform)
    return NativeCapabilityRecord(
        capability=capability,
        backend=backend,
        status=(
            NativeCapabilityStatus.AVAILABLE
            if available
            else NativeCapabilityStatus.UNAVAILABLE
        ),
        built=True,
        runtime_available=available,
        reason=reason,
    )


def _future_backend_runtime_status(
    backend: NativeBackend, platform: str
) -> tuple[bool, str]:
    if backend is NativeBackend.ACCELERATE and platform == "darwin":
        return True, "Accelerate support is compiled and running on Darwin."
    return False, f"{backend.value} support is compiled but not available at runtime."


def _probe_mlx_device(kind: mx.DeviceType, label: str) -> tuple[bool, str]:
    try:
        device = mx.Device(kind, 0)
    except Exception as exc:
        return False, f"Could not construct MLX {label} device: {exc}"

    try:
        if not mx.is_available(device):
            return False, f"MLX reports that {label} device 0 is unavailable."
    except Exception as exc:
        return False, f"Could not query MLX {label} availability: {exc}"

    try:
        mx.device_info(device)
    except Exception as exc:
        return False, f"MLX could not initialize {label} device 0: {exc}"

    return True, ""


def _metallib_present() -> bool:
    try:
        return resources.files("mlx_sparse").joinpath("mlx_sparse.metallib").is_file()
    except Exception:
        return False


def _python_platform() -> str:
    import sys

    if sys.platform == "darwin":
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win"):
        return "windows"
    return sys.platform


def _python_architecture() -> str:
    import platform

    return platform.machine()


__all__ = [
    "capabilities",
    "has_capability",
]
