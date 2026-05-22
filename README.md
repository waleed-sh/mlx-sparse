# mlx-sparse

[![PyPI](https://img.shields.io/pypi/v/mlx-sparse)](https://pypi.org/project/mlx-sparse/)
[![License](https://img.shields.io/pypi/l/mlx-sparse)](https://github.com/waleed-sh/mlx-sparse/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/mlx-sparse/badge/?version=latest)](https://mlx-sparse.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/waleed-sh/mlx-sparse/graph/badge.svg?token=EV2KVPZTP0)](https://codecov.io/gh/waleed-sh/mlx-sparse)

> **Warning: beta release**
> This is an early beta. APIs may change, bugs are expected, and some features
> are still incomplete. Feedback and issue reports are very welcome!
> 
> A lot of the linalg functionality is new and is currently being tested.
> We welcome any and all feedback! Not all solvers are GPU supported ([see here](https://mlx-sparse.readthedocs.io/en/latest/user_guide/linalg.html#gpu-coverage)).

> **Platform note**
> GPU support in this version is Apple Silicon Metal only. CUDA is not
> currently supported.

`mlx-sparse` is an attempt at an MLX-native sparse array package. The public API is Python,
while performance-critical operations are implemented as MLX primitives in C++
with CPU backends and Metal kernels for fixed-shape sparse operations.

The supported format surface is COO and CSR for 2D sparse arrays. Current
functionality includes construction, validation, COO to CSR, CSR to dense, CSR
canonicalization, CSR matrix-vector multiply, CSR matrix-matrix multiply,
batched dense RHS products, CSR sparse-sparse products, transpose, Hermitian
transpose. ``mlx-sparse`` also supports sparse linalg solvers (`cg`, `gmres`, `minres`), sparse spectral
routines (`eigsh`, `eigs`, `svds`), sparse Cholesky/LU factors, sparse
triangular solves, sparse `dot`/`vdot`, and autodiff through sparse values and
dense RHS operands, including `complex64`. 


Supported value dtypes are `float32`, `float16`, `bfloat16`, and `complex64`.
Supported index dtypes are `int32` and `int64` on CPU and GPU.

## Quick Start

Install from PyPI:

```bash
python -m pip install mlx-sparse
```

```python
import mlx.core as mx
import numpy as np

import mlx_sparse as ms

ms.use_gpu()

data = mx.array(np.array([2.0, -1.0, 4.0], dtype=np.float32))
row = mx.array(np.array([0, 0, 1], dtype=np.int32))
col = mx.array(np.array([0, 2, 1], dtype=np.int32))

a = ms.coo_array((data, (row, col)), shape=(2, 3)).tocsr(canonical=True)
x = mx.array(np.array([3.0, 10.0, 7.0], dtype=np.float32))

y = a @ x
dense = a.todense()
at = a.T

b = mx.array(np.array([1.0, 2.0], dtype=np.float32))
spd = ms.csr_array(
    (
        mx.array([4.0, 1.0, 1.0, 3.0], dtype=mx.float32),
        mx.array([0, 1, 0, 1], dtype=mx.int32),
        mx.array([0, 2, 4], dtype=mx.int32),
    ),
    shape=(2, 2),
    canonical=True,
)
solution, info = ms.linalg.cg(spd, b)
factor = ms.linalg.sparse_cholesky(spd)
```

The package build compiles `src/sparse/*.metal` into
`mlx_sparse/mlx_sparse.metallib` when the macOS Metal toolchain is available,
and the wheel ships that metallib beside the Python package.

## Development

For contributors, use an editable install from the repository root. This builds
the native extension and installs the development tooling.

```bash
python -m pip install -e ".[dev]"
```

## License

This package is licensed under the [Apache License 2.0](https://github.com/waleed-sh/mlx-sparse/blob/main/LICENSE).
