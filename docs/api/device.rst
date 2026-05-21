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

Usage notes
-----------

* Call one of these functions once, early in your script, to pin execution
  to a specific device.
* The selected device persists for the lifetime of the Python process or until
  another ``use_*`` call.
* The Metal GPU path (``use_gpu``) requires macOS with Apple Silicon and
  MLX ≥ 0.31. On unsupported hardware ``use_gpu`` raises ``RuntimeError``
  when ``require_available=True`` (the default).
* To check GPU availability without setting it as the default device, call
  ``mlx.core.is_available(mlx.core.Device(mlx.core.gpu))``.

See :doc:`../user_guide/device_execution` for a full discussion of the lazy
execution model and which operations run on GPU vs CPU.
