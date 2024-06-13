tlm_adjoint
===========

tlm_adjoint is a high-level algorithmic differentiation tool, principally for
use with `Firedrake <https://firedrakeproject.org>`_.

The primary aim of tlm_adjoint is to enable higher order adjoint calculations
-- and in particular to compute Hessian information -- while also using adjoint
checkpointing schedules, and allowing for caching of assembled finite element
data, and caching of linear solver data.

Features
--------

    - Integrates with the Firedrake automated code generation library, and
      applies a high-level algorithmic differentiation approach to enable
      tangent-linear and adjoint calculations.
    - Applies a *reverse-over-multiple-forward* approach for higher order
      adjoint calculations. For example a Hessian action on some given
      direction is computed by deriving an adjoint associated with a combined
      forward/tangent-linear calculation.
    - Implements adjoint checkpointing schedules to enable large problems to
      be tackled.
    - Can apply automated finite element assembly and linear solver caching.
      Cached data can be reused across the forward, tangent-linear, and adjoint
      calculations.
    - Provides a simple 'escape-hatch' to enable parts of the problem, not
      otherwise supported by tlm_adjoint or the backend library, to be added,
      with adjoint or tangent-linear information defined manually.

Examples
--------

The following Jupyter notebooks introduce derivative calculations using
tlm_adjoint.

- `Getting started with tlm_adjoint <examples/0_getting_started.ipynb>`__:
  Introduces derivative calculations using tlm_adjoint.
- `Time-independent example <examples/1_time_independent.ipynb>`__: An example
  including the solution of a time-independent partial differential equation
  using Firedrake.
- `Verifying derivative calculations <examples/2_verification.ipynb>`__:
  Introduces Taylor remainder convergence testing.
- `Time-dependent example <examples/3_time_dependent.ipynb>`__: An example
  including the solution of a time-dependent partial differential equation
  using Firedrake. Introduces checkpointing.
- `Visualizing derivatives <examples/4_riesz_maps.ipynb>`__: The use of a Riesz
  map to visualize the result of differentiating with respect to a finite
  element discretized function using the adjoint method.
- `Functional minimization <examples/5_optimization.ipynb>`__: The solution of
  partial differential equation constrained optimization problems.
- `Defining custom operations <examples/6_custom_operations.ipynb>`__: An
  example using the tlm_adjoint 'escape-hatch' to define a custom operation.
- `JAX integration <examples/7_jax_integration.ipynb>`__: A finite difference
  example using JAX.

Advanced tutorials
------------------

- `Hessian-based uncertainty quantification <examples/8_hessian_uq.ipynb>`__

Source and license
------------------

The source code is available from the
`tlm_adjoint GitHub repository <https://github.com/tlm-adjoint/tlm_adjoint>`_.
tlm_adjoint is licensed under the GNU LGPL version 3.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`

.. toctree::
    :hidden:

    self

.. toctree::
    :hidden:
    :maxdepth: 0

    dependencies
    acknowledgements
