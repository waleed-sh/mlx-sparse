Configuration
=============

.. currentmodule:: mlx_sparse

``mlx_sparse`` exposes a typed package configuration manager as
:data:`config`. Options may be read and written as attributes, through
:func:`get_config` / :func:`set_config`, or temporarily patched with
:func:`config_context`.

The manager keeps the corresponding ``MLX_SPARSE_*`` environment variables in
sync so native code can read the same values without a Python callback.
For runtime worker controls, prefer the enum-based :mod:`mlx_sparse.runtime`
facade documented in :doc:`runtime`.

.. code-block:: python

   import mlx_sparse as ms

   print(ms.runtime.N_THREADS)
   ms.runtime.N_THREADS = 8

   ms.config.EXPERIMENTAL_METAL_SPGEMM = True
   ms.set_config("EXPERIMENTAL_METAL_SPGEMM", False)

   with ms.config_context(EXPERIMENTAL_METAL_SPGEMM=True):
       C = ms.csr_matmat(A, B)

Runtime worker controls
-----------------------

These options control CPU worker budgets and operation-family parallel gates.
They are exposed through both :data:`config` and :mod:`mlx_sparse.runtime`.

.. list-table::
   :header-rows: 1

   * - Config option
     - Environment variable
     - Default
     - Meaning
   * - ``CPU_THREADS``
     - ``MLX_SPARSE_CPU_THREADS``
     - ``"auto"``
     - Package-wide CPU worker setting. Use a positive integer for an explicit
       worker count, or ``"auto"`` to resolve from thread hints, scheduler
       allocations, process affinity, and hardware concurrency.
   * - ``SPGEMM_PARALLEL``
     - ``MLX_SPARSE_SPGEMM_PARALLEL``
     - ``True``
     - Enables fixed-worker CPU parallel sparse-sparse products for native
       same-format CSR, COO, and CSC SpGEMM.
   * - ``SPGEMM_THREADS``
     - ``MLX_SPARSE_SPGEMM_THREADS``
     - ``"inherit"``
     - CPU worker setting for sparse-sparse products. Use a positive integer,
       ``"auto"``, or ``"inherit"`` to use ``CPU_THREADS``.
   * - ``SOLVER_PARALLEL``
     - ``MLX_SPARSE_SOLVER_PARALLEL``
     - ``False``
     - Enables CPU parallel solver routines when parallel kernels are
       available.
   * - ``SOLVER_THREADS``
     - ``MLX_SPARSE_SOLVER_THREADS``
     - ``"inherit"``
     - CPU worker setting for solver routines. Use a positive integer,
       ``"auto"``, or ``"inherit"`` to use ``CPU_THREADS``.

``CPU_THREADS`` also accepts ``MLX_SPARSE_N_THREADS`` as a compatibility alias
at import time. The canonical synchronized variable is
``MLX_SPARSE_CPU_THREADS``.

The native CPU SpGEMM implementation uses the resolved ``SPGEMM_THREADS``
count directly. It does not vary the worker count by matrix shape, density, or
estimated work; very small outputs may assign empty worker ranges instead of
silently reducing the configured count. Set ``SPGEMM_THREADS=1`` or
``SPGEMM_PARALLEL=False`` for the serial Gustavson/SPA path.

Experimental controls
---------------------

.. list-table::
   :header-rows: 1

   * - Config option
     - Environment variable
     - Default
     - Meaning
   * - ``EXPERIMENTAL_METAL_SPGEMM``
     - ``MLX_SPARSE_EXPERIMENTAL_METAL_SPGEMM``
     - ``False``
     - Enables experimental staged Metal implementations for same-format CSR,
       COO, and CSC sparse-sparse products. The optimized native host
       implementation remains the default.
   * - ``EXPERIMENTAL_METAL_SPGEMM``
     - ``MLX_SPARSE_FORCE_EXPERIMENTAL_METAL_SPGEMM``
     - unset
     - Forced override. When set, Python code cannot change the option.

Environment precedence
----------------------

Effective configuration values are resolved in this order:

#. forced environment variables
#. programmatic overrides
#. default environment variables
#. built-in defaults

For the runtime thread count, ``"auto"`` is resolved dynamically by
:func:`mlx_sparse.runtime.resolve_n_threads`. Explicit
``MLX_SPARSE_CPU_THREADS`` or ``ms.runtime.N_THREADS = ...`` values win first.
Auto mode then checks ``OMP_NUM_THREADS``, common scheduler variables such as
``SLURM_CPUS_PER_TASK``, process affinity where available, and finally
hardware concurrency.

Top-level objects
-----------------

.. autodata:: config

.. autofunction:: get_config

.. autofunction:: set_config

.. autofunction:: config_context
