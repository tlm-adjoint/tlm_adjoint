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

# Import FEniCS and tlm_adjoint
from fenics import *
from tlm_adjoint import *
# Disable the manager until it is needed
stop_manager()

import petsc4py.PETSc
petsc4py.PETSc.Options().setValue("citations", "petsc.bib")

# Seed the random number generator, to ensure reproducibility of the later
# Taylor verification
import numpy
numpy.random.seed(1352)

# Configure a simple discrete function space
mesh = UnitSquareMesh(20, 20)
space = FunctionSpace(mesh, "Lagrange", 1)

# Configure a boundary condition. The optional "static" and "homogeneous"
# arguments are flags used for optimization.
bc = DirichletBC(space, 1.0, "on_boundary", static = True, homogeneous = False)

def forward(F, x0 = None):  
  # Clear assembly and linear solver caches
  clear_caches()

  # Construct a simple equation (the Poisson equation with inhomogeneous
  # Dirichlet boundary conditions)
  x = Function(space, name = "x0" if x0 is None else "x")
  test, trial = TestFunction(space), TrialFunction(space)
  eq = EquationSolver(inner(grad(test), grad(trial)) * dx == inner(test, F * F) * dx,
    x, bc, solver_parameters = {"linear_solver":"cg", "preconditioner":"sor",
                                "krylov_solver":{"absolute_tolerance":1.0e-16,
                                                 "relative_tolerance":1.0e-14}})
  
  # Solve the equation
  eq.solve()
  
  # Drop references to Function objects within the equation. eq.solve cannot be
  # used again after this step.
  eq.replace()
  
  if x0 is None:
    # If x0 is not supplied, return a reference solution
    return x
  else:
    # Otherwise, return a mis-match functional
    J = Functional()
    J.assign(inner(x - x0, x - x0) * dx)
    return J

# Generate a reference solution x0 using F0. The optional "static" flag is used
# for optimization.
F0 = Function(space, name = "F0", static = True)
F0.interpolate(Expression("sin(pi * x[0]) * sin(3 * pi * x[1]) * exp(x[0] * x[1])", element = space.ufl_element()))
x0 = forward(F0)

# Set F to one everywhere ...
F = Function(space, name = "F", static = True)
function_assign(F, 1.0)
# ... and re-run the forward with this value of F, now obtaining a mis-match
# functional, and processing equations using the manager
start_manager()
J = forward(F, x0 = x0)
stop_manager()

# Compute a forward model constrained derivative of J with respect to F
dJ = compute_gradient(J, F)
# Taylor verify the forward model constrained derivative
min_order = taylor_test(lambda F : forward(F, x0 = x0), F, J_val = J.value(), dJ = dJ, seed = 1.0e-3, size = 5)
assert(min_order > 2.00)

# An object which can be used to evaluate forward model constrained Hessian
# actions
ddJ = Hessian(lambda F : forward(F, x0 = x0))
# Taylor verify forward model constrained Hessian actions
min_order = taylor_test(lambda F : forward(F, x0 = x0), F, J_val = J.value(), ddJ = ddJ, seed = 1.0e-2, size = 5)
assert(min_order > 3.00)
