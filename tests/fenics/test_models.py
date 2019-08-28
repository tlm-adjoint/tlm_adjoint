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

from test_base import *

import copy
import pytest


def oscillator_ref():
    assert(not manager().annotation_enabled())
    assert(not manager().tlm_enabled())

    nsteps = 20
    dt = Constant(0.01)
    T_0 = Constant((1.0, 0.0))

    import mpi4py.MPI as MPI
    mesh = UnitIntervalMesh(MPI.COMM_WORLD.size)
    space = FunctionSpace(mesh, "R", 0)
    space = FunctionSpace(mesh, space.ufl_element() * space.ufl_element())
    test = TestFunction(space)

    T_n = Function(space, name="T_n")
    T_np1 = Function(space, name="T_np1")
    T_s = 0.5 * (T_n + T_np1)

    T_n.assign(T_0)
    for n in range(nsteps):
        solve(
            inner(test, (T_np1 - T_n) / dt) * dx
            - inner(test[0], T_s[1]) * dx
            + inner(test[1], sin(T_s[0])) * dx == 0,
            T_np1,
            solver_parameters={"nonlinear_solver": "newton",
                               "newton_solver": ns_parameters_newton_gmres})
        T_n, T_np1 = T_np1, T_n

    return T_n.vector().max()


@pytest.mark.fenics
@pytest.mark.parametrize(
    "cp_method, cp_parameters",
    [("memory", {"replace": True}),
     ("periodic_disk", {"period": 3, "format": "pickle"}),
     ("periodic_disk", {"period": 3, "format": "hdf5"}),
     ("multistage", {"format": "pickle", "snaps_on_disk": 1, "snaps_in_ram": 2,
                     "verbose": True}),
     ("multistage", {"format": "hdf5", "snaps_on_disk": 1, "snaps_in_ram": 2,
                     "verbose": True})])
def test_oscillator(setup_test, test_leaks,
                    cp_method, cp_parameters):
    n_steps = 20
    if cp_method == "multistage":
        cp_parameters = copy.copy(cp_parameters)
        cp_parameters["blocks"] = n_steps
    configure_checkpointing(cp_method, cp_parameters)

    mesh = UnitIntervalMesh(20)
    r0 = FiniteElement("R", mesh.ufl_cell(), 0)
    space = FunctionSpace(mesh, r0 * r0)
    test = TestFunction(space)
    T_0 = Function(space, name="T_0", static=True)
    T_0.assign(Constant((1.0, 0.0)))
    dt = Constant(0.01, static=True)

    def forward(T_0):
        T_n = Function(space, name="T_n")
        T_np1 = Function(space, name="T_np1")
        T_s = 0.5 * (T_n + T_np1)

        AssignmentSolver(T_0, T_n).solve()

        solver_parameters = {"nonlinear_solver": "newton",
                             "newton_solver": ns_parameters_newton_gmres}
        eq = EquationSolver(inner(test, (T_np1 - T_n) / dt) * dx
                            - inner(test[0], T_s[1]) * dx
                            + inner(test[1], sin(T_s[0])) * dx == 0,
                            T_np1,
                            solver_parameters=solver_parameters)

        for n in range(n_steps):
            eq.solve()
            T_n.assign(T_np1)
            if n < n_steps - 1:
                new_block()

        J = Functional(name="J")
        J.assign(inner(T_n[0], T_n[0]) * dx)
        return J

    start_manager()
    J = forward(T_0)
    stop_manager()

    J_val = J.value()
    J_val_ref = oscillator_ref() ** 2
    assert(abs(J_val - J_val_ref) < 1.0e-15)

    dJ = compute_gradient(J, T_0)

    dm = Function(space, name="dm", static=True)
    function_assign(dm, 1.0)

    # Usage as in dolfin-adjoint tests
    min_order = taylor_test(forward, T_0, J_val=J_val, dJ=dJ, dM=dm)
    assert(min_order > 2.00)

    ddJ = Hessian(forward)
    # Usage as in dolfin-adjoint tests
    min_order = taylor_test(forward, T_0, J_val=J_val, ddJ=ddJ, dM=dm)
    assert(min_order > 3.00)

    min_order = taylor_test_tlm(forward, T_0, tlm_order=1, dMs=(dm,))
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward, T_0, adjoint_order=1,
                                        dMs=(dm,))
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward, T_0, adjoint_order=2,
                                        dMs=(dm, dm))
    assert(min_order > 2.00)
