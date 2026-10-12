.. _experimental-apis:

Experimental APIs
=================

Some solver entry points have known numerical limitations that can produce an
incorrect result without a failure status. They emit
:class:`~mlx_sparse.experimental.ExperimentalWarning` at runtime. Each notice
names the API, explains its limitation, suggests a result check, and links to
the :doc:`solver limitations guide <../user_guide/solver_limitations>`.

The warnings do not enable a different implementation. They also do not check
the result or certify inputs. Read the notice before relying on the solver.
An API without this marker can still have undocumented defects.

Warning behavior
----------------

Warnings point to the call in user code. Python's default filter displays a
given notice once per source location. Calls on another line can display it
again. A high-level operation such as ``spsolve`` emits one notice rather than
one for each internal factorization and solve. Calling a reusable factor later
emits that factor's solve notice. Independent solver calls from user callbacks
also retain their own notices.

Under normal filters, execution continues after the warning. An application
can explicitly choose another policy, including treating warnings as errors.
The implementation respects that choice. See the
`Python warning filter documentation
<https://docs.python.org/3.10/library/warnings.html#the-warnings-filter>`_.

After reviewing the limitations, acknowledge them through the configuration:

.. code-block:: python

   import mlx_sparse as ms

   ms.config.EXPERIMENTAL = True

This silences experimental notices for the process. It does not improve
accuracy, enable new solvers, or suppress unrelated warnings. The default is
``False``. Set ``MLX_SPARSE_EXPERIMENTAL=1`` before importing ``mlx_sparse`` for
the same policy at startup.

For a temporary acknowledgment:

.. code-block:: python

   with ms.config_context(EXPERIMENTAL=True):
       x, info = ms.linalg.cg(A, b)

Configuration contexts change the process-wide setting and restore it on exit.
They are not a separate policy for each thread or asynchronous task.
``EXPERIMENTAL`` is independent of ``EXPERIMENTAL_METAL_SPGEMM``, which selects
an experimental sparse product implementation.

Applications can instead use Python's warning filters:

.. code-block:: python

   import warnings

   warnings.simplefilter("always", ms.ExperimentalWarning)

Marked solver entry points
--------------------------

``cg`` is marked because convergence uses a recursive residual that can drift
from the true residual. ``eigsh``, ``eigs``, and ``svds`` are marked because a
bounded projection can return unconverged spectral approximations.

The direct solver family is also marked. This includes
``spsolve_triangular``, ``spsolve``, ``sparse_lu``, ``splu``,
``sparse_cholesky``, ``cholesky``, ``factorized``, and the ``solve`` methods of
``SparseLU``, ``SparseCholesky``, and ``FactorizedSolve``. Their notices cover
the relevant diagonal, scaling, symmetry, right-hand-side, and operator
semantics described in the limitations guide. A warning identifies risks of
the entry point, even when the particular input does not encounter them.

``gmres``, ``minres``, and ``bicgstab`` are not marked solely for the audited
cases of nonconvergence that they report through solver status. Inspect that
status and check the true residual as described in the limitations guide.

Registering an experimental implementation
------------------------------------------

Keep notices in ``mlx_sparse/_experimental_features.py``. The decorator itself
has no message arguments:

.. code-block:: python

   from mlx_sparse.experimental import experimental, registry

   registry.register(
       "my_package.solvers.solve",
       message="The current method has no convergence check. Check the residual.",
       docs_url="https://example.com/solver-limitations",
   )

   @experimental
   def solve(A, b):
       """Solve the system using a bounded iteration."""
       ...

Register the notice before decorating the function. The qualified
implementation name includes the module and, for a method, the class name.
``display_name`` can give the public import path when it differs from the
implementation path. A missing registration raises ``KeyError`` at definition
time. This catches incomplete notices during import.

A string registration without a signature covers all implementations of that
qualified name. To distinguish overloads, supply an ``inspect.Signature`` or
register the implementation function directly. Parameter names, kinds, and
annotations form the key. Defaults and return annotations do not. Postponed
annotations are kept as written, without importing or evaluating their types.
Unhashable annotation metadata is matched by identity. A signature-specific
entry takes precedence over a function-wide entry.

The decorator resolves the entry once and stores it on
``function.__experimental_feature__``. Later registration does not update
existing wrappers. Calls do not inspect signatures, resolve types, evaluate
MLX arrays, or synchronize devices to produce the notice. The decorator also
appends the notice to the function's docstring as a documentation warning.

Multiple dispatch
-----------------

For `Plum <https://github.com/beartype/plum>`_, apply ``@experimental`` to each
experimental implementation below ``@dispatch``:

.. code-block:: python

   import inspect
   from plum import dispatch
   from mlx_sparse.experimental import experimental, registry

   registry.register(
       "my_package.solvers.solve",
       signature=inspect.Signature([
           inspect.Parameter(
               "value", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
           ),
       ]),
       message="The integer implementation has no convergence check.",
       docs_url="https://example.com/integer-solver-limitations",
   )

   @dispatch
   @experimental
   def solve(value: int):
       ...

   @dispatch
   def solve(value: float):
       ...

Only the selected experimental implementation warns. The wrapper preserves
the implementation's inspectable signature, annotations, original globals,
and ``__wrapped__`` link. This also supports Plum's method ownership lookup
and postponed type annotations. Keep the registration's annotation form
consistent with the implementation, including when using
``from __future__ import annotations``. For methods, put ``@experimental``
below ``@staticmethod`` or ``@classmethod`` as well.

Plum is not required by ``mlx_sparse``. The integration tests run when Plum is
installed. Metadata preservation follows
`functools.wraps <https://docs.python.org/3.10/library/functools.html#functools.wraps>`_.

API reference
-------------

.. automodule:: mlx_sparse.experimental
   :members: ExperimentalWarning, ExperimentalFeature, ExperimentalRegistry, experimental

.. autodata:: mlx_sparse.experimental.registry
   :annotation: = ExperimentalRegistry()
