Structural Constructors
=======================

``mlx_sparse`` includes SciPy-compatible structural constructors for assembling
larger sparse matrices without materializing dense intermediates.

Block Assembly
--------------

Use :func:`mlx_sparse.block_array` for a rectangular grid of blocks:

.. code-block:: python

   import mlx_sparse as ms

   A = ms.eye(3)
   B = ms.diags([1.0, 2.0], offsets=1, shape=(3, 3))

   K = ms.block_array([[A, B], [None, A]], format="csr")

``None`` entries represent implicit all-zero blocks. Their dimensions are
inferred from the other blocks in the same block row and block column. A block
row or block column containing only ``None`` has size zero. The grid must be
rectangular, all non-``None`` blocks in a block row must have the same height,
and all non-``None`` blocks in a block column must have the same width.

The related helpers are:

* :func:`mlx_sparse.bmat`, a compatibility alias returning mlx-sparse arrays.
* :func:`mlx_sparse.block_diag`, for diagonal block assembly.
* :func:`mlx_sparse.vstack`, for vertical stacking.
* :func:`mlx_sparse.hstack`, for horizontal stacking.

All block and stack constructors accept COO, CSR, CSC, and dense rank-2 inputs.
Dense blocks are first converted with the native :func:`mlx_sparse.fromdense`
path. Sparse blocks are converted to COO with native format-conversion kernels,
then assembled by a native coordinate-offset primitive. Python validates the
grid and block metadata, but it does not loop over stored entries.

Formats and Dtypes
------------------

Supported output formats are ``"coo"``, ``"csr"``, and ``"csc"``. Passing
``format=None`` returns COO for block and stack constructors. CSR and CSC
requests canonicalize through native compressed conversion, which also sums
duplicate coordinates.

Value dtypes follow the package sparse-value policy: ``complex64`` wins over
real dtypes, ``float32`` wins over lower-precision real dtypes, equal
``float16`` or ``bfloat16`` inputs keep that dtype, and mixed low-precision
real inputs promote to ``float32``. Dense integer and boolean blocks are
converted to ``float32`` because public sparse value buffers do not yet store
integer or boolean values.

Triangular Extraction
---------------------

:func:`mlx_sparse.tril` and :func:`mlx_sparse.triu` keep the lower or upper
triangular part of a sparse or dense rank-2 input:

.. code-block:: python

   L = ms.tril(K, k=0, format="csr")
   U = ms.triu(K, k=1, format="coo")

``tril`` keeps entries where ``column - row <= k``. ``triu`` keeps entries
where ``column - row >= k``. Dense inputs use native ``fromdense`` first.
COO, CSR, and CSC sparse inputs use staged native count/fill kernels for their
own storage layout. The default output format is COO, matching SciPy's
triangular extraction default.

Device Support
--------------

The structural constructors are backed by native CPU kernels on every platform.
On Apple platforms with a Metal-enabled build, block assembly and triangular
extraction also include Metal kernels. Linux remains CPU-only in this release.

Autodiff Boundary
-----------------

Fixed-topology block and stack assembly is differentiable with respect to the
stored sparse values: JVP and VJP split or concatenate value tangents according
to the fixed block placement. Gradients with respect to integer coordinate
buffers, block shapes, ``None`` placement, dense-to-sparse extraction, and
canonicalization are not supported.
