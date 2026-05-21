Physics workloads and sparse neural layers
==========================================

This tutorial covers two applications that demonstrate how mlx-sparse handles
real workloads: building a quantum Hamiltonian and running a block-sparse
neural network layer. Both examples are available as standalone scripts in
``examples/``.

Quantum Hamiltonians: transverse-field Ising model
----------------------------------------------------

The transverse-field Ising model is one of the simplest quantum spin chain
models that shows a quantum phase transition. The Hamiltonian for ``n`` spin-1/2
particles on a chain with periodic boundary conditions is:

.. math::

   H = -\sum_{i=0}^{n-2} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=0}^{n-1} \sigma_i^x

where :math:`\sigma^z` and :math:`\sigma^x` are Pauli matrices and :math:`h` is
the transverse field strength. In the computational basis of :math:`2^n`
states, the diagonal part encodes Ising spin-spin interactions and the
off-diagonal part encodes single-spin flips (from :math:`\sigma^x`).

The Hamiltonian matrix is of size :math:`2^n \times 2^n`. For ``n=8`` qubits,
the matrix is ``256 x 256`` with at most ``n+1`` non-zeros per row (one
diagonal entry plus one entry per spin flip). Sparsity grows rapidly with
``n``, which makes this a natural application for CSR.

Building the Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   def transverse_field_ising(n_qubits: int, field: float) -> ms.CSRArray:
       dim = 1 << n_qubits    # 2^n_qubits
       rows = []
       cols = []
       values = []

       for state in range(dim):
           # Diagonal: Ising ZZ interactions
           energy = 0.0
           for site in range(n_qubits - 1):
               z_i = 1 if state & (1 << site)       else -1
               z_j = 1 if state & (1 << (site + 1)) else -1
               energy += -z_i * z_j
           rows.append(state)
           cols.append(state)
           values.append(energy)

           # Off-diagonal: transverse field X flips
           for site in range(n_qubits):
               rows.append(state)
               cols.append(state ^ (1 << site))   # flip bit at 'site'
               values.append(-field)

       return ms.coo_array(
           (
               mx.array(np.asarray(values, dtype=np.float32)),
               (
                   mx.array(np.asarray(rows, dtype=np.int32)),
                   mx.array(np.asarray(cols, dtype=np.int32)),
               ),
           ),
           shape=(dim, dim),
       ).tocsr(canonical=True)

Each state contributes one diagonal entry and ``n_qubits`` off-diagonal
entries, giving ``nnz = dim * (n_qubits + 1)``. For ``n=8`` qubits:
``256 * 9 = 2304`` stored values in a ``256 x 256`` matrix.

Applying the Hamiltonian to a state vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In quantum simulation, the most common operation is multiplying the Hamiltonian
by a state vector to compute the energy expectation value or to propagate a
state in imaginary time.

.. code-block:: python

   ms.use_gpu()

   n_qubits = 8
   field = 0.7    # below the phase transition (field < 1.0 in units of J)

   H = transverse_field_ising(n_qubits, field)
   dim = 1 << n_qubits

   # Uniform superposition state, normalized
   psi = mx.ones((dim,), dtype=mx.float32)
   psi = psi / mx.sqrt(mx.sum(psi * psi))

   # Apply Hamiltonian
   Hpsi = H @ psi
   mx.eval(Hpsi)

   # Energy expectation value: <psi|H|psi>
   energy = float(mx.sum(psi * Hpsi))
   print(f"<H> = {energy:.6f}")

For ``n=8`` at ``h=0.7``, the ground state energy per site is approximately
``-1.27`` (in units where the coupling ``J=1``). The uniform superposition
expectation value is further from the ground state energy.

Scaling to larger systems
~~~~~~~~~~~~~~~~~~~~~~~~~~

The construction loop above is a Python-level loop over ``dim = 2^n`` states.
For ``n=12``, this is 4096 states and runs quickly. For ``n=16`` (65536 states),
the Python loop dominates and construction takes several seconds. For production
use with large systems, generate the COO triplets with NumPy vectorization
rather than a Python loop.

Running the example script
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/quantum_hamiltonian.py --qubits 8 --field 0.7 --device gpu
   python examples/quantum_hamiltonian.py --qubits 10 --field 1.0 --device cpu

Block-sparse neural network layer
-----------------------------------

A block-sparse weight matrix is a common approximation for large dense linear
layers: the weight matrix is divided into blocks, and only a subset of blocks
are kept as non-zero. This reduces parameter count and can accelerate inference
when the sparsity is high enough.

The mlx-sparse ``csr_matmul`` primitive handles this pattern naturally: the
sparse matrix ``W`` is the weight, and the dense right-hand side is a batch of
activation vectors.

Constructing a block-sparse weight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The construction below builds a simple block-diagonal structure as a baseline:
each block is a bidiagonal matrix (identity plus first superdiagonal scaled
by 0.25).

.. code-block:: python

   import mlx.core as mx
   import numpy as np
   import mlx_sparse as ms

   def block_sparse_weight(features: int, blocks: int) -> ms.CSRArray:
       if features % blocks != 0:
           raise ValueError("features must be divisible by blocks.")
       block = features // blocks
       rows = []
       cols = []
       values = []

       for b in range(blocks):
           start = b * block
           for i in range(block):
               row = start + i
               # Diagonal entry
               rows.append(row)
               cols.append(row)
               values.append(1.0)
               # Superdiagonal entry within the block
               if i + 1 < block:
                   rows.append(row)
                   cols.append(row + 1)
                   values.append(0.25)

       return ms.coo_array(
           (
               mx.array(np.asarray(values, dtype=np.float32)),
               (
                   mx.array(np.asarray(rows, dtype=np.int32)),
                   mx.array(np.asarray(cols, dtype=np.int32)),
               ),
           ),
           shape=(features, features),
       ).tocsr(canonical=True)

For ``features=64`` and ``blocks=8`` (block size 8), each block has
``8 + 7 = 15`` entries (diagonal plus superdiagonal), giving
``8 * 15 = 120`` stored values in a ``64 x 64`` matrix. The density is
``120 / (64*64) ≈ 2.9%``.

Batched forward pass
~~~~~~~~~~~~~~~~~~~~~

The ``@`` operator dispatches to batched matmul when the right-hand side has
``ndim > 2``. A batch of activation matrices with shape ``(batch, features, 1)``
maps naturally to this path.

.. code-block:: python

   ms.use_gpu()

   features = 64
   blocks   = 8
   batch    = 4

   W = block_sparse_weight(features, blocks)
   x = mx.random.normal((batch, features, 1))

   # Sparse weight times batched activations
   y = W @ x   # shape: (batch, features, 1)
   mx.eval(y)
   print(y.shape)   # (4, 64, 1)

The batched path reshapes the ``(batch, features, 1)`` input to
``(features, batch * 1)``, calls one rank-2 ``csr_matmul``, then reshapes
the output back to ``(batch, features, 1)``. No explicit loops over the batch
dimension are needed.

Differentiating through the layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradients with respect to the input activation ``x`` work out of the box
via ``mx.grad``:

.. code-block:: python

   def forward(x):
       y = W @ x    # shape (batch, features, 1)
       return mx.sum(y ** 2)

   grad_x = mx.grad(forward)(x)
   mx.eval(grad_x)
   print(grad_x.shape)   # (4, 64, 1)

The VJP dispatches through the batched matmul and the transpose matmul, both
of which have Metal implementations.

Running the example script
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python examples/sparse_linear_layer.py --features 64 --blocks 8 --batch 4 --device gpu
   python examples/sparse_linear_layer.py --features 128 --blocks 16 --batch 16 --device gpu
