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

n_steps = 20
dt = Constant(0.01)
kappa = Constant(1.0)

mesh = UnitIntervalMesh(100)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

T_n = Function(space, name="T_n")
T_np1 = Function(space, name="T_np1")

T_n.interpolate(Expression("sin(pi * x[0]) + sin(10.0 * pi * x[0])",
                element=space.ufl_element()))
for n in range(n_steps):
    solve(inner(test, trial / dt) * dx
          + inner(grad(test), kappa * grad(trial)) * dx
          == inner(test, T_n / dt) * dx,
          T_np1,
          DirichletBC(space, 1.0, "on_boundary"),
          solver_parameters={"linear_solver": "umfpack"})
    T_n, T_np1 = T_np1, T_n

J = assemble(inner(T_n, T_n) * dx)
info("J = %.16e" % J)
