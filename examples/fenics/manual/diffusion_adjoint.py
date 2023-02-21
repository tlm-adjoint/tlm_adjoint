#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *
from tlm_adjoint.fenics import *
stop_manager()

N = 100
configure_checkpointing("multistage", {"blocks": N, "snaps_in_ram": 5})

mesh = UnitSquareMesh(100, 100)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

psi_0 = Function(space, name="psi_0", static=True)
psi_0.interpolate(Expression(
    "exp(x[0] * x[1]) * sin(2.0 * pi * x[0]) * sin(5.0 * pi * x[1])"
    " + sin(pi * x[0]) * sin(2.0 * pi * x[1])",
    element=space.ufl_element()))

kappa = Constant(0.001, static=True)
dt = Constant(0.2, static=True)
bc = HomogeneousDirichletBC(space, "on_boundary")


def forward(psi_0, psi_n_file=None):
    clear_caches()

    psi_n = Function(space, name="psi_n")
    psi_np1 = Function(space, name="psi_np1")

    class InteriorAssignmentSolver(Equation):
        def __init__(self, y, x):
            super().__init__(x, [x, y], nl_deps=[], ic=False, adj_ic=False)
            self._bc = DirichletBC(function_space(x), 0.0, "on_boundary")

        def forward_solve(self, x, deps=None):
            _, y = self.dependencies() if deps is None else deps
            function_assign(x, y)
            self._bc.apply(x.vector())

        def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
            if dep_index == 0:
                return adj_x
            elif dep_index == 1:
                b = function_copy(adj_x)
                self._bc.apply(b.vector())
                return (-1.0, b)
            else:
                raise IndexError("dep_index out of bounds")

        def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
            return b

        def tangent_linear(self, M, dM, tlm_map):
            x, y = self.dependencies()
            tlm_y = get_tangent_linear(y, M, dM, tlm_map)
            if tlm_y is None:
                return NullSolver(tlm_map[x])
            else:
                return InteriorAssignmentSolver(tlm_y, tlm_map[x])

    InteriorAssignmentSolver(psi_0, psi_n).solve()

    eq = EquationSolver(
        inner(trial / dt, test) * dx
        + inner(kappa * grad(trial), grad(test)) * dx
        == inner(psi_n / dt, test) * dx,
        psi_np1, bc,
        solver_parameters={"linear_solver": "direct"})
    cycle = Assignment(psi_n, psi_np1)

    if psi_n_file is not None:
        psi_n_file << (psi_n, 0.0)

    for n in range(N):
        eq.solve()
        cycle.solve()

        if psi_n_file is not None:
            psi_n_file << (psi_n, (n + 1) * float(dt))
        if n < N - 1:
            new_block()

    J = Functional(name="J")
    J.assign(dot(psi_n - Constant(1.0), psi_n - Constant(1.0)) * dx)
    return J


start_manager()
# J = forward(psi_0, psi_n_file=File("psi.pvd", "compressed"))
J = forward(psi_0)
stop_manager()

dJ = compute_gradient(J, psi_0)

import mpi4py.MPI as MPI  # noqa: E402
import numpy as np  # noqa: E402
np.random.seed(65038062 + MPI.COMM_WORLD.rank)

min_order = taylor_test(forward, psi_0, J_val=J.value(), dJ=dJ, seed=1.0e-3)
assert min_order > 1.99
