Runtime
=======

.. currentmodule:: mlx_sparse.runtime

``mlx_sparse.runtime`` exposes enum-backed CPU runtime controls for code that
needs reproducible performance settings without spelling raw configuration
strings. It is the preferred public interface for thread counts and
operation-family parallel gates.

.. code-block:: python

   import mlx_sparse as ms

   print(ms.runtime.N_THREADS)

   ms.runtime.N_THREADS = 8
   ms.runtime.SPGEMM_PARALLEL = True
   ms.runtime.SPGEMM_THREADS = "inherit"
   ms.runtime.SOLVER_PARALLEL = False
   ms.runtime.SOLVER_THREADS = 2

   with ms.runtime.context(n_threads=1):
       C = A @ B

   report_metadata = ms.runtime.info()

Direct controls
---------------

The common runtime controls are module attributes. Reading them returns the
effective value that kernels should use. Assigning them validates and updates
the underlying package configuration and synchronized ``MLX_SPARSE_*``
environment variable.

For native CPU same-format CSR, COO, and CSC sparse-sparse products,
``SPGEMM_THREADS`` is a fixed worker count. The implementation partitions
independent output rows or columns across that count and does not change the
number of workers based on matrix size, density, or estimated work. Very small
outputs may assign empty ranges to some workers rather than silently reducing
the configured count. Use ``SPGEMM_THREADS = 1`` or
``SPGEMM_PARALLEL = False`` to force the serial Gustavson/SPA path.

.. list-table::
   :header-rows: 1

   * - Attribute
     - Read value
     - Accepted writes
   * - ``N_THREADS``
     - Resolved package-wide CPU worker count.
     - Positive integer or ``"auto"``.
   * - ``SPGEMM_PARALLEL``
     - Whether CPU SpGEMM parallelism is enabled.
     - Boolean-like value.
   * - ``SPGEMM_THREADS``
     - Effective CPU worker count for sparse-sparse products.
     - Positive integer, ``"auto"``, or ``"inherit"``.
   * - ``SOLVER_PARALLEL``
     - Whether CPU solver parallelism is enabled.
     - Boolean-like value.
   * - ``SOLVER_THREADS``
     - Effective CPU worker count for solver routines.
     - Positive integer, ``"auto"``, or ``"inherit"``.

Enum keys
---------

The enum remains available for structured helper calls such as
``context(RuntimeOption.N_THREADS, 1)`` and for code that wants stable option
identifiers without relying on strings.

.. autoclass:: RuntimeOption
   :members:

Scoped overrides
----------------

Use the context manager when a benchmark or experiment needs temporary runtime
settings without permanently changing the process configuration.

.. autofunction:: context

.. autofunction:: patch

Thread resolution
-----------------

.. autofunction:: resolve_n_threads

.. autofunction:: resolve_spgemm_threads

.. autofunction:: resolve_solver_threads

Diagnostics
-----------

.. autofunction:: info
