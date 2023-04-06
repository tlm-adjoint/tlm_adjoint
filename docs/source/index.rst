tlm_adjoint
===========

tlm_adjoint is a high-level algorithmic differentiation tool, principally for
use with `FEniCS <https://fenicsproject.org/>`_ or `Firedrake
<https://firedrakeproject.org/>`_.

The primary aim of tlm_adjoint is to enable higher order adjoint calculations
-- and in particular to compute Hessian information -- while also using adjoint
checkpointing schedules, and allowing for caching of finite element matrices
and linear solvers.

Features
========

    - Integrates with the FEniCS or Firedrake automated code generation
      libraries, and applies a high-level algorithmic differentiation approach
      to enable tangent-linear and adjoint calculations.
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

Example
=======

The following example, using Firedrake, solves a discrete Poisson equation with
:math:`P_1` continuous finite elements and homogeneous Dirichlet boundary
conditions, and then computes a derivative with respect to a control, and also
a Hessian action on a given direction.

.. code-block:: python

    from firedrake import *
    from tlm_adjoint.firedrake import *

    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)

    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    trial = TrialFunction(space)

    # Define the control
    m = Function(space, name="m")
    m.interpolate(X[0] * X[1], annotate=False, tlm=False)

    # Configure a tangent-linear model, computing directional derivatives with
    # respect to m with direction defined by zeta
    zeta = Function(space, name="zeta")
    zeta.assign(Constant(1.0), annotate=False, tlm=False)
    configure_tlm((m, zeta))

    # Solve the Poisson equation with homogeneous Dirichlet boundary conditions
    u = Function(space, name="u")
    solve(inner(grad(trial), grad(test)) * dx == inner(m * m, test) * dx, u,
          DirichletBC(space, 0.0, "on_boundary"))

    # Define a functional
    J = Functional(name="J")
    J.assign(inner(u, u) * dx)

    # Compute the derivative of J with respect to m, and a Hessian action on
    # zeta
    dJ, ddJ = compute_gradient((J, J.tlm_functional((m, zeta))), m)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
