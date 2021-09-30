#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *

mesh = UnitIntervalMesh(10)
X = SpatialCoordinate(mesh)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

F = Function(space, name="F")
G = Function(space, name="G")

F.interpolate(sin(pi * X[0]))
solve(inner(trial, test) * dx == inner(F * F, test) * dx,
      G, solver_parameters={"ksp_type": "preonly",
                            "pc_type": "lu"})

J = assemble(inner(G, G) * dx)

info(f"G L^2 norm = {sqrt(J):.16e}")
