#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import *

mesh = UnitIntervalMesh(10)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

F = Function(space, name="F")
G = Function(space, name="G")

F.interpolate(Expression("sin(pi * x[0])", element=space.ufl_element()))
solve(inner(test, trial) * dx == inner(test, F * F) * dx,
      G, solver_parameters={"linear_solver": "direct"})

J = assemble(inner(G, G) * dx)

info(f"G L^2 norm = {sqrt(J):.16e}")
