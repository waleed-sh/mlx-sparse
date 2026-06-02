Device selection
================

.. currentmodule:: mlx_sparse

These functions set MLX's default device and return the selected
``mlx.core.Device`` object. They are thin wrappers around
``mx.set_default_device`` with an optional availability probe.

use\_gpu
--------

.. autofunction:: use_gpu

use\_cpu
--------

.. autofunction:: use_cpu

use\_device
-----------

.. autofunction:: use_device

Native capabilities
-----------------------------

``mlx-sparse`` exposes enum-backed runtime capability checks for native
backend dispatch. Use these when application code needs to branch on whether a
backend is actually usable, rather than checking platform strings or package
filenames directly.

.. code-block:: python

   import mlx_sparse as ms

   if ms.capabilities.METAL:
       ms.use_gpu()

   if ms.capabilities.ACCELERATE:
       pass  # optional Apple Accelerate sparse direct solvers are available

The public capability names are ``"extension"``, ``"cpu"``, ``"metal"``,
``"accelerate"``, ``"cuda"``, and ``"rocm"``. The corresponding uppercase
attributes on :data:`capabilities` return booleans, for example
``ms.capabilities.CPU`` and ``ms.capabilities.METAL``. Status strings are
``"available"``, ``"unavailable"``, and ``"not_built"``.

``ms.capabilities.ACCELERATE`` reports Accelerate-backed sparse solver
availability, not just framework presence. Current published macOS wheels
(v0.0.4b0 onwards) are built with Accelerate sparse direct-solver support
enabled. Linux wheels are CPU-only and report ``"not_built"`` for Metal,
Accelerate, CUDA, and ROCm. Editable and source macOS builds can opt into the
same Accelerate path with ``MLX_SPARSE_ENABLE_ACCELERATE=ON``.

.. autodata:: capabilities

.. autofunction:: has_capability

Usage notes
-----------

* Call one of these functions once, early in your script, to pin execution
  to a specific device.
* The selected device persists for the lifetime of the Python process or until
  another ``use_*`` call.
* The Metal GPU path (``use_gpu``) requires macOS with Apple Silicon and
  MLX ≥ 0.31. On unsupported hardware ``use_gpu`` raises ``RuntimeError``
  when ``require_available=True`` (the default).
* Linux builds currently support native CPU execution only. CUDA and ROCm are
  reserved capability names for future releases.
* To check sparse native backend availability without setting the default
  device, prefer ``ms.capabilities``.

See :doc:`../user_guide/device_execution` for a full discussion of the lazy
execution model and which operations run on GPU vs CPU.
