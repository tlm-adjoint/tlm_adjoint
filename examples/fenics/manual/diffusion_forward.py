#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *

mesh = UnitSquareMesh(100, 100)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

psi_n = Function(space, name="psi_n")
psi_n.interpolate(Expression(
    "exp(x[0] * x[1]) * sin(2.0 * pi * x[0]) * sin(5.0 * pi * x[1])"
    " + sin(pi * x[0]) * sin(2.0 * pi * x[1])",
    element=space.ufl_element()))

psi_np1 = Function(space, name="psi_np1")

kappa = Constant(0.001)
dt = Constant(0.2)
bc = DirichletBC(space, 0.0, "on_boundary")
N = 100

solver = LinearVariationalSolver(
    LinearVariationalProblem(inner(test, trial / dt) * dx
                             + inner(grad(test), kappa * grad(trial)) * dx,
                             inner(test, psi_n / dt) * dx,
                             psi_np1, bc))
solver.parameters.update({"linear_solver": "direct"})

# psi_n_file = File("psi.pvd", "compressed")
# psi_n_file << (psi_n, 0.0)

for n in range(N):
    solver.solve()
    psi_n.assign(psi_np1)

    # psi_n_file << (psi_n, (n + 1) * float(dt))
