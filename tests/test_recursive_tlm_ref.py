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

nsteps = 20
dt = Constant(0.01)
alpha = Constant(1.0)
T_0 = Constant((1.0, 0.0))

mesh = UnitIntervalMesh(1)
space = FunctionSpace(mesh, "R", 0)
space = FunctionSpace(mesh, space.ufl_element() * space.ufl_element())
test, trial = TestFunction(space), TrialFunction(space)

T_n = Function(space, name = "T_n")
T_np1 = Function(space, name = "T_np1")
T_s = 0.5 * (T_n + T_np1)

T_n.assign(T_0)
for n in range(nsteps):
    solve(inner(test, (T_np1 - T_n) / dt) * dx - inner(test[0], T_s[1]) * dx + inner(test[1], sin(alpha * T_s[0])) * dx == 0,
        T_np1, solver_parameters = {"nonlinear_solver":"newton",
                                                                "newton_solver":{"linear_solver":"umfpack",
                                                                                                 "relative_tolerance":1.0e-13,
                                                                                                 "absolute_tolerance":1.0e-15}})
    T_n, T_np1 = T_np1, T_n

J = T_n.vector().max()
info("J = %.16e" % J)
