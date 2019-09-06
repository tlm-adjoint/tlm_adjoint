#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from fenics import *
from tlm_adjoint_fenics import *

import numpy as np
# import petsc4py.PETSc as PETSc

# Disable the manager until it is needed
stop_manager()

# Save relevant citation information
# PETSc.Options().setValue("citations", "petsc.bib")

# Seed the random number generator, to ensure reproducibility of the later
# Taylor verification
np.random.seed(1352)

# Configure a simple discrete function space
mesh = UnitSquareMesh(20, 20)
space = FunctionSpace(mesh, "Lagrange", 1)

# Configure a boundary condition. The optional "static" and "homogeneous"
# arguments are flags used for optimization.
bc = DirichletBC(space, 1.0, "on_boundary", static=True, homogeneous=False)


def forward(F, x0=None):
    # Clear caches
    clear_caches()

    # Construct and solve a simple equation (the Poisson equation with
    # inhomogeneous Dirichlet boundary conditions)
    x = Function(space, name="x")
    test, trial = TestFunction(space), TrialFunction(space)
    ls_parameters = {"linear_solver": "cg", "preconditioner": "sor",
                     "krylov_solver": {"absolute_tolerance": 1.0e-16,
                                       "relative_tolerance": 1.0e-14}}
    EquationSolver(inner(grad(test), grad(trial)) * dx
                   == inner(test, F * F) * dx,
                   x, bc, solver_parameters=ls_parameters).solve()

    if x0 is None:
        # If a reference is not provided, create it by copying x
        x0 = function_copy(x, name="x0")

    # The functional
    J = Functional(name="J")
    J.assign(inner(x - x0, x - x0) * dx)

    # Return the reference and the functional
    return x0, J


# Generate a reference solution x0 using F0. The optional "static" flag is used
# for optimization.
F0 = Function(space, name="F0", static=True)
F0.interpolate(Expression(
    "sin(pi * x[0]) * sin(3.0 * pi * x[1]) * exp(x[0] * x[1])",
    element=space.ufl_element()))
x0, _ = forward(F0)

# Set F to one everywhere ...
F = Function(space, name="F", static=True)
function_assign(F, 1.0)
# ... and re-run the forward with this value of F, now obtaining a mis-match
# functional, and processing equations using the manager
start_manager()
_, J = forward(F, x0=x0)
stop_manager()

# Display equation manager information
manager_info()

# Compute a forward model constrained derivative of J with respect to F
dJ = compute_gradient(J, F)
# Taylor verify the forward model constrained derivative
min_order = taylor_test(lambda F: forward(F, x0=x0)[1], F, J_val=J.value(),
                        dJ=dJ, seed=1.0e-3, size=5)
assert(min_order > 2.00)

# Taylor verify forward model constrained Hessian actions for the case of two
# equal perturbation directions. Omitting the dJ argument here includes the
# first order tangent-linear in the verification test.
ddJ = Hessian(lambda F: forward(F, x0=x0)[1])
min_order = taylor_test(lambda F: forward(F, x0=x0)[1], F, J_val=J.value(),
                        ddJ=ddJ, seed=1.0e-2, size=5)
assert(min_order > 3.00)

# Taylor verify the first order tangent-linear
min_order = taylor_test_tlm(lambda F: forward(F, x0=x0)[1], F,
                            tlm_order=1, seed=1.0e-3, size=5)
assert(min_order > 2.00)

# Taylor verify forward model constrained Hessian actions for the case of two
# possibly different perturbation directions. Assumes validity of the first
# order tangent-linear.
min_order = taylor_test_tlm_adjoint(lambda F: forward(F, x0=x0)[1], F,
                                    adjoint_order=2, seed=1.0e-3, size=5)
assert(min_order > 2.00)
