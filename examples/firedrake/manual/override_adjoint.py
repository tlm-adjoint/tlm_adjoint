#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from tlm_adjoint.firedrake import *

mesh = UnitIntervalMesh(10)
X = SpatialCoordinate(mesh)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

F = Function(space, name="F")
G = Function(space, name="G")

F.interpolate(sin(pi * X[0]), annotate=False, tlm=False)
solve(inner(trial, test) * dx == inner(F * F, test) * dx,
      G, solver_parameters={"ksp_type": "preonly",
                            "pc_type": "lu"})

J = Functional(name="J")
J.assign(inner(G, G) * dx)

info(f"G L^2 norm = {sqrt(J.value()):.16e}")

dJ = compute_gradient(J, F)

import mpi4py.MPI as MPI  # noqa: E402
import numpy as np  # noqa: E402
np.random.seed(79459258 + MPI.COMM_WORLD.rank)


def forward(F):
    G = Function(space, name="G")

    solve(inner(trial, test) * dx == inner(F * F, test) * dx,
          G, solver_parameters={"ksp_type": "preonly",
                                "pc_type": "lu"})

    J = Functional(name="J")
    J.assign(inner(G, G) * dx)
    return J


min_order = taylor_test(forward, F, J_val=J.value(), dJ=dJ)
assert min_order > 2.00
