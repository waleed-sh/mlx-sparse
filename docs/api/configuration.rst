Configuration
=============

.. currentmodule:: mlx_sparse

``mlx_sparse`` exposes a typed package configuration manager as
:data:`config`. Options may be read and written as attributes, through
:func:`get_config` / :func:`set_config`, or temporarily patched with
:func:`config_context`.

The manager keeps the corresponding ``MLX_SPARSE_*`` environment variables in
sync so native code can read the same values without a Python callback.

.. code-block:: python

   import mlx_sparse as ms

   ms.config.EXPERIMENTAL_METAL_SPGEMM = True
   ms.set_config("EXPERIMENTAL_METAL_SPGEMM", False)

   with ms.config_context(EXPERIMENTAL_METAL_SPGEMM=True):
       C = ms.csr_matmat(A, B)

``EXPERIMENTAL_METAL_SPGEMM`` enables the staged Metal implementation for
CSR x CSR. The optimized native host implementation remains the default
because it is faster on current small and medium benchmark cases.

Top-level objects
-----------------

.. autodata:: config

.. autofunction:: get_config

.. autofunction:: set_config

.. autofunction:: config_context
