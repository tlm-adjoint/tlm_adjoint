#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *

mesh = UnitSquareMesh(100, 100)
X = SpatialCoordinate(mesh)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

psi_n = Function(space, name="psi_n")
psi_n.interpolate(
    exp(X[0] * X[1]) * sin(2.0 * pi * X[0]) * sin(5.0 * pi * X[1])
    + sin(pi * X[0]) * sin(2.0 * pi * X[1]))

psi_np1 = Function(space, name="psi_np1")

kappa = Constant(0.001)
dt = Constant(0.2)
bc = DirichletBC(space, 0.0, "on_boundary")
N = 100

solver = LinearVariationalSolver(
    LinearVariationalProblem(inner(trial / dt, test) * dx
                             + inner(kappa * grad(trial), grad(test)) * dx,
                             inner(psi_n / dt, test) * dx,
                             psi_np1, bc),
    solver_parameters={"ksp_type": "preonly",
                       "pc_type": "lu"})

# psi_n_file = File("psi.pvd")
# psi_n_file.write(psi_n, time=0.0)

for n in range(N):
    solver.solve()
    psi_n.assign(psi_np1)

    # psi_n_file.write(psi_n, time=(n + 1) * float(dt))
