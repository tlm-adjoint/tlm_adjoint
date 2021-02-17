#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *
from tlm_adjoint.fenics import *

mesh = UnitIntervalMesh(10)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

F = Function(space, name="F")
G = Function(space, name="G")

F.interpolate(Expression("sin(pi * x[0])", element=space.ufl_element()))
solve(inner(test, trial) * dx == inner(test, F * F) * dx,
      G, solver_parameters={"linear_solver": "direct"})

J = Functional(name="J")
J.assign(inner(G, G) * dx)

info(f"G L^2 norm = {sqrt(J.value()):.16e}")

dJ = compute_gradient(J, F)

import mpi4py.MPI as MPI  # noqa: E402
import numpy as np  # noqa: E402
np.random.seed(174632238 + MPI.COMM_WORLD.rank)


def forward(F):
    G = Function(space, name="G")

    solve(inner(test, trial) * dx == inner(test, F * F) * dx,
          G, solver_parameters={"linear_solver": "direct"})

    J = Functional(name="J")
    J.assign(inner(G, G) * dx)
    return J


min_order = taylor_test(forward, F, J_val=J.value(), dJ=dJ)
assert min_order > 2.00
