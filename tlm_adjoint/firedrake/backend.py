#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import *
from firedrake.utils import complex_mode  # noqa: F401
import firedrake

backend = "Firedrake"

backend_ScalarType = firedrake.utils.ScalarType.type

extract_args = firedrake.solving._extract_args
extract_linear_solver_args = firedrake.solving._extract_linear_solver_args

backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_FunctionSpace = firedrake.functionspaceimpl.WithGeometry
backend_LinearSolver = LinearSolver
backend_LinearVariationalProblem = LinearVariationalProblem
backend_LinearVariationalSolver = LinearVariationalSolver
backend_Matrix = firedrake.matrix.Matrix
backend_NonlinearVariationalSolver = NonlinearVariationalSolver
backend_Vector = Vector
backend_assemble = assemble
backend_interpolate = interpolate
backend_project = project
backend_solve = solve
