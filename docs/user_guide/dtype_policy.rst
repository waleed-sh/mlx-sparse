Dtype policy
============

mlx-sparse enforces explicit dtype constraints on both the value arrays
(``data``) and the index arrays (``indices``, ``indptr``, ``row``, ``col``).
Mixed dtypes between the sparse data and the dense operand are rejected at
constructor time rather than silently promoted.

Value dtypes
------------

The following value dtypes are supported:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Dtype
     - Python name
     - Notes
   * - ``float32``
     - ``mx.float32``
     - Fully supported on CPU and Metal GPU. The primary dtype.
   * - ``float16``
     - ``mx.float16``
     - Supported on CPU and Metal GPU. CPU and GPU accumulate in
       ``float32`` and cast back to ``float16`` on output.
   * - ``bfloat16``
     - ``mx.bfloat16``
     - Supported on CPU and Metal GPU. Same accumulation convention
       as ``float16``.
   * - ``complex64``
     - ``mx.complex64``
     - Supported on CPU and Metal GPU for forward operations and autodiff
       through sparse values and dense RHS operands.

Integer (``int32``, ``int64``), boolean, and ``float64`` are not supported.

Index dtypes
------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Dtype
     - Notes
   * - ``int32``
     - Default. All CPU and Metal kernels support it. Sufficient for
       matrices up to roughly 2 billion non-zeros, which covers all
       practical Apple Silicon workloads.
   * - ``int64``
     - Supported on CPU and Metal GPU. Use for matrices that exceed
       the ``int32`` range.

**``indices`` and ``indptr`` must share the same dtype** in a CSRArray.
Similarly, ``row`` and ``col`` must share the same dtype in a COOArray.
Mismatched index dtypes are caught at metadata validation time.

Mixed dtype rejection
---------------------

The Python constructors and native C++ validation both check that
``data.dtype`` matches the dense operand's dtype for all operation calls.
There is no implicit promotion:

.. code-block:: python

   import mlx.core as mx
   import mlx_sparse as ms

   A = ms.coo_array(
       (mx.array([1.0], dtype=mx.float32), (mx.array([0], dtype=mx.int32), mx.array([0], dtype=mx.int32))),
       shape=(1, 1),
   ).tocsr()

   x_fp16 = mx.array([1.0], dtype=mx.float16)
   A @ x_fp16  # TypeError: csr_matvec requires sparse data and RHS to have
               # the same dtype, got float32 and float16.

To use a different dtype, convert before constructing:

.. code-block:: python

   A_fp16 = ms.csr_array(
       (A.data.astype(mx.float16), A.indices, A.indptr),
       shape=A.shape,
   )
   y = A_fp16 @ x_fp16

Accumulation policy
--------------------

For ``float16`` and ``bfloat16``, both CPU and Metal GPU backends use a
``float32`` accumulator to reduce rounding error during the inner-product
loop, then cast back to the storage dtype on output.

For ``complex64``, both real and imaginary components accumulate in
``complex64`` (i.e. ``float32`` component precision). There is no upcasting
to ``complex128``.

For ``float32``, accumulation is in ``float32`` throughout.

Metal dtype coverage
--------------------

Every Metal GPU kernel is compiled for all four value dtypes (``float32``,
``float16``, ``bfloat16``, ``complex64``) and both index dtypes (``int32``,
``int64``). The kernels with Metal implementations are:

* ``csr_matvec``
* ``csr_matmul``
* ``coo_matvec`` / ``coo_matmul`` and batched variants
* ``csc_matvec`` / ``csc_matmul`` and batched variants
* ``coo_tocsr``
* ``coo_tocsc``
* ``csr_todense``
* ``csc_todense``
* ``csr_sort_indices``
* ``csc_sort_indices``
* ``csr_transpose``

Dynamic-output helpers such as ``canonicalize()``, ``fromdense()``, and
``CSR @ CSR`` synchronize to host because their output sizes depend on values.
These are host-side assembly operations and are not Metal kernels.
