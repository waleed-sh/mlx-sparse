Installation
============


.. warning::

   ``mlx-sparse`` supports macOS and Linux. Linux support is CPU-only in this
   release: CUDA and ROCm are not implemented, Metal is Apple-only, and Linux
   builds do not use Accelerate, BLAS, or Sparse BLAS backends.

Installing from PyPI
---------------------

For users, the normal installation path is PyPI:

.. code-block:: bash

   python -m pip install mlx-sparse

This installs ``mlx-sparse`` and its runtime dependencies on MLX and NumPy.
macOS wheels include the native extension and the Metal library needed for CPU
and Apple Silicon GPU sparse kernels. Linux wheels include the native extension
and CPU kernels only.

.. note::

   GPU support in this version is Apple Silicon Metal only. CUDA is not
   currently supported.

Installing from source (editable)
----------------------------------

For contributors, install directly from the repository root in editable mode.
This compiles the C++ extension and Metal library in-place and installs the
development tools.

Requirements
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Requirement
     - Notes
   * - macOS 14+
     - Sequoia (macOS 15) is tested in CI. The Metal GPU path requires a
       device that supports Metal 3. CPU-only usage works on any supported
       macOS.
   * - Apple Silicon
     - M1 or newer for the Metal GPU backend. Intel Macs are not tested and
       the Metal backend will not activate.
   * - Linux x86_64
     - CPU-only wheels are built and tested in CI. CUDA and ROCm are reserved
       for future releases, and Linux builds do not use Accelerate, BLAS, or
       Sparse BLAS backends.
   * - Python ≥ 3.10
     - Tested under Python 3.12 in CI.
   * - MLX ≥ 0.31
     - Installed automatically by ``pip install mlx-sparse``. Source builds use
       ``python -m mlx --cmake-dir`` to locate the MLX package.
   * - NumPy ≥ 1.26
     - Used for host-side conversions, constructors, validation, and fallback
       sparse kernels.
   * - CMake ≥ 3.27
     - Required only when building from source.
   * - nanobind ≥ 2.0
     - Python/C++ binding layer used by source builds.


Local install
~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Clone the repository.
   git clone https://github.com/ml-explore/mlx-sparse
   cd mlx-sparse

   # 2. (Optional) create and activate a virtual environment.
   python -m venv .venv
   source .venv/bin/activate

   # 3. Install in editable mode with development extras.
   python -m pip install -e ".[dev]"

The ``dev`` extras include ``pytest``, ``scipy``, ``black``,
``isort``, and ``pre-commit``.

For documentation builds, install the ``docs`` extras instead:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Optional build feature gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributor builds can opt into native feature-detection gates through
``CMAKE_ARGS``. The Accelerate gate detects and links Apple's Accelerate
framework on supported macOS builds and enables the optional Accelerate sparse
direct-solver integration:

.. code-block:: bash

   CMAKE_ARGS="-DMLX_SPARSE_ENABLE_ACCELERATE=ON" python -m pip install -e .

This gate is Darwin-only. Passing ``MLX_SPARSE_ENABLE_ACCELERATE=ON`` on
non-Apple platforms fails at configure time instead of silently pretending that
Accelerate is available. Linux CPU builds should leave it disabled.
Accelerate-enabled builds validate and normalize
``float32`` CSR, CSC, and COO inputs into owned canonical CSC storage, use
opaque Accelerate factorization objects for supported direct solves, and report
``ms.capabilities.ACCELERATE`` as ``True`` when the runtime can use them.

Verifying the installation
---------------------------

.. code-block:: python

   import mlx.core as mx
   import mlx_sparse as ms

   # Check that the native extension compiled successfully.
   print("Native extension available:", ms.is_available())
   print("CPU kernels:", ms.capabilities.CPU)
   print("Metal kernels:", ms.capabilities.METAL)

   # Quick smoke test: identity_like passes a tensor through the extension.
   x = mx.array([1.0, 2.0, 3.0])
   y = ms.identity_like(x)
   mx.eval(y)
   print(y)  # [1.0, 2.0, 3.0]

If ``ms.is_available()`` returns ``False``, the extension did not compile.
For an editable source install, re-run ``python -m pip install -e .`` and check
that CMake, nanobind, and MLX are all accessible in the same virtual
environment. The fallback Python implementations in ``mlx_sparse._fallback``
remain functional, but they are not graph-safe for large workloads.

For finer-grained native dispatch checks, use the enum-backed capability API:

.. code-block:: python

   ms.capabilities.status("metal")
   ms.capabilities.status("accelerate")

Current wheels report native CPU kernels on macOS and Linux. On supported Apple
Silicon runtimes with an accessible GPU, macOS wheels also report Metal kernels.
Accelerate-enabled builds are macOS-only and opt-in unless a platform-specific
release explicitly states otherwise. CUDA and ROCm remain reserved capabilities
for future builds.

Running the test suite
-----------------------

.. code-block:: bash

   python -m pytest tests --cov=mlx_sparse --cov-report=term-missing --cov-fail-under=88

The test suite runs on whichever device MLX selects by default. To force a
specific device:

.. code-block:: bash

   MLX_SPARSE_TEST_DEVICE=cpu pytest   # CPU only
   MLX_SPARSE_TEST_DEVICE=gpu pytest   # GPU only (skips if unavailable)

Tests marked ``native`` are skipped if the compiled extension is not present.
Tests marked ``performance`` run small deterministic microbenchmarks with
lenient regression thresholds, tune them with ``MLX_SPARSE_PERF_*``
environment variables if you need to calibrate a slower CI host. To run only
the performance regression checks:

.. code-block:: bash

   python -m pytest tests/test_performance_regression.py

Building the documentation
---------------------------

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html docs/ docs/_build/html
   open docs/_build/html/index.html

.. note::

   The Sphinx build imports ``mlx_sparse``, so the package must be installed
   (or on ``sys.path``) before running Sphinx. With an editable install the
   package is already importable.
