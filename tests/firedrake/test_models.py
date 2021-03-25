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

from firedrake import *
from tlm_adjoint.firedrake import *

from test_base import *

import copy
import pytest


def oscillator_ref():
    assert not manager().annotation_enabled()
    assert not manager().tlm_enabled()

    nsteps = 20
    dt = Constant(0.01)
    T_0 = Constant((1.0, 0.0))

    mesh = UnitSquareMesh(5, 5)
    space = VectorFunctionSpace(mesh, "Discontinuous Lagrange", 0)
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
            solver_parameters=ns_parameters_newton_gmres)
        T_n, T_np1 = T_np1, T_n

    return assemble(T_n[0] * dx)


def diffusion_ref():
    assert not manager().annotation_enabled()
    assert not manager().tlm_enabled()

    n_steps = 20
    dt = Constant(0.01)
    kappa = Constant(1.0)

    mesh = UnitIntervalMesh(100)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    T_n = Function(space, name="T_n")
    T_np1 = Function(space, name="T_np1")

    T_n.interpolate(sin(pi * X[0]) + sin(10.0 * pi * X[0]))
    for n in range(n_steps):
        solve(inner(test, trial / dt) * dx
              + inner(grad(test), kappa * grad(trial)) * dx
              == inner(test, T_n / dt) * dx,
              T_np1,
              DirichletBC(space, 1.0, "on_boundary"),
              solver_parameters=ls_parameters_cg)
        T_n, T_np1 = T_np1, T_n

    return assemble(inner(T_n * T_n, T_n * T_n) * dx)


@pytest.mark.firedrake
@pytest.mark.parametrize(
    "cp_method, cp_parameters",
    [("memory", {"drop_references": True}),
     ("periodic_disk", {"period": 3, "format": "pickle"}),
     ("periodic_disk", {"period": 3, "format": "hdf5"}),
     ("multistage", {"format": "pickle", "snaps_on_disk": 1,
                     "snaps_in_ram": 2}),
     ("multistage", {"format": "hdf5", "snaps_on_disk": 1,
                     "snaps_in_ram": 2})])
def test_oscillator(setup_test, test_leaks,
                    cp_method, cp_parameters):
    n_steps = 20
    if cp_method == "multistage":
        cp_parameters = copy.copy(cp_parameters)
        cp_parameters["blocks"] = n_steps
    configure_checkpointing(cp_method, cp_parameters)

    mesh = UnitSquareMesh(5, 5)
    r0 = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
    space = VectorFunctionSpace(mesh, r0)
    test = TestFunction(space)
    T_0 = Function(space, name="T_0", static=True)
    T_0.assign(Constant((1.0, 0.0)))
    dt = Constant(0.01, static=True)

    def forward(T_0):
        clear_caches()

        T_n = Function(space, name="T_n")
        T_np1 = Function(space, name="T_np1")
        T_s = 0.5 * (T_n + T_np1)

        AssignmentSolver(T_0, T_n).solve()

        eq = EquationSolver(inner(test, (T_np1 - T_n) / dt) * dx
                            - inner(test[0], T_s[1]) * dx
                            + inner(test[1], sin(T_s[0])) * dx == 0,
                            T_np1,
                            solver_parameters=ns_parameters_newton_gmres)

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
    assert abs(J_val - J_val_ref) < 1.0e-14

    dJ = compute_gradient(J, T_0)

    dm = Function(space, name="dm", static=True)
    dm.assign(Constant((1.0, 0.0)))

    min_order = taylor_test(forward, T_0, J_val=J_val, dJ=dJ, dM=dm)
    assert min_order > 2.00

    ddJ = Hessian(forward)
    min_order = taylor_test(forward, T_0, J_val=J_val, ddJ=ddJ, dM=dm)
    assert min_order > 2.99

    min_order = taylor_test_tlm(forward, T_0, tlm_order=1, dMs=(dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, T_0, adjoint_order=1,
                                        dMs=(dm,))
    assert min_order > 2.00

    min_order = taylor_test_tlm_adjoint(forward, T_0, adjoint_order=2,
                                        dMs=(dm, dm))
    assert min_order > 2.00


@pytest.mark.firedrake
@pytest.mark.parametrize("n_steps", [1, 2, 5, 20])
def test_diffusion_1d_timestepping(setup_test, test_leaks,
                                   n_steps):
    configure_checkpointing("multistage",
                            {"blocks": n_steps, "snaps_on_disk": 2,
                             "snaps_in_ram": 3})

    mesh = UnitIntervalMesh(100)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    T_0 = Function(space, name="T_0", static=True)
    interpolate_expression(T_0, sin(pi * X[0]) + sin(10.0 * pi * X[0]))
    dt = Constant(0.01, static=True)
    kappa = Constant(1.0, domain=mesh, name="kappa", static=True)

    def forward(T_0, kappa):
        from tlm_adjoint.timestepping import N, TimeFunction, TimeLevels, \
            TimeSystem, n

        levels = TimeLevels([n, n + 1], {n: n + 1})
        T = TimeFunction(levels, space, name="T")
        T[n].rename("T_n", "a Function")
        T[n + 1].rename("T_np1", "a Function")

        system = TimeSystem()

        system.add_solve(T_0, T[0])

        system.add_solve(inner(test, trial) * dx
                         + dt * inner(grad(test), kappa * grad(trial)) * dx
                         == inner(test, T[n]) * dx,
                         T[n + 1],
                         DirichletBC(space, 1.0, "on_boundary"),
                         solver_parameters=ls_parameters_cg)

        for n_step in range(n_steps):
            system.timestep()
            if n_step < n_steps - 1:
                new_block()
        system.finalize()

        J = Functional(name="J")
        J.assign(inner(T[N] * T[N], T[N] * T[N]) * dx)
        return J

    start_manager()
    J = forward(T_0, kappa)
    stop_manager()

    J_val = J.value()
    if n_steps == 20:
        J_val_ref = diffusion_ref()
        assert abs(J_val - J_val_ref) < 1.0e-13

    controls = [T_0, kappa]
    dJs = compute_gradient(J, controls)

    for m, m0, forward_J, dJ, dm in \
            [(controls[0], T_0, lambda T_0: forward(T_0, kappa), dJs[0],
              None),
             (controls[1], kappa, lambda kappa: forward(T_0, kappa), dJs[1],
              Constant(1.0, name="dm", static=True))]:
        min_order = taylor_test(forward_J, m, J_val=J_val, dJ=dJ, dM=dm)
        assert min_order > 1.99

        ddJ = Hessian(forward_J)
        min_order = taylor_test(forward_J, m, J_val=J_val, ddJ=ddJ, dM=dm)
        assert min_order > 2.92

        min_order = taylor_test_tlm(forward_J, m0, tlm_order=1,
                                    dMs=None if dm is None else (dm,))
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(forward_J, m0, adjoint_order=1,
                                            dMs=None if dm is None else (dm,))
        assert min_order > 1.99

        min_order = taylor_test_tlm_adjoint(
            forward_J, m0, adjoint_order=2,
            dMs=None if dm is None else (dm, dm), seed=1.0e-3)
        assert min_order > 1.99


@pytest.mark.firedrake
def test_diffusion_2d(setup_test, test_leaks):
    n_steps = 20
    configure_checkpointing("multistage",
                            {"blocks": n_steps, "snaps_on_disk": 2,
                             "snaps_in_ram": 3})

    mesh = UnitSquareMesh(20, 20)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    T_0 = Function(space, name="T_0", static=True)
    interpolate_expression(T_0, sin(pi * X[0]) * sin(2.0 * pi * X[1]))
    dt = Constant(0.01, static=True)
    kappa = [Constant(1.0, domain=mesh, name="kappa_00", static=True),
             Constant(0.0, domain=mesh, name="kappa_10", static=True),
             Constant(0.0, domain=mesh, name="kappa_01", static=True),
             Constant(1.0, domain=mesh, name="kappa_11", static=True)]
    bc = HomogeneousDirichletBC(space, "on_boundary")

    def forward(kappa_00, kappa_01, kappa_10, kappa_11):
        clear_caches()
        kappa = as_tensor([[kappa_00, kappa_01],
                           [kappa_10, kappa_11]])

        T_n = Function(space, name="T_n")
        T_np1 = Function(space, name="T_np1")

        AssignmentSolver(T_0, T_n).solve()

        eq = (inner(test, trial / dt) * dx
              + inner(grad(test), dot(kappa, grad(trial))) * dx
              == inner(test, T_n / dt) * dx)
        eqs = [EquationSolver(eq, T_np1, bc,
                              solver_parameters=ls_parameters_cg),
               AssignmentSolver(T_np1, T_n)]

        for n in range(n_steps):
            for eq in eqs:
                eq.solve()
            if n < n_steps - 1:
                new_block()

        J = Functional(name="J")
        J.assign(inner(T_np1, T_np1) * dx)

        return J

    for tlm_order in range(1, 4):
        min_order = taylor_test_tlm(forward, kappa, tlm_order=tlm_order,
                                    seed=1.0e-3)
        assert min_order > 1.99

    for adjoint_order in range(1, 5):
        min_order = taylor_test_tlm_adjoint(forward, kappa,
                                            adjoint_order=adjoint_order,
                                            seed=1.0e-3)
        assert min_order > 1.99
