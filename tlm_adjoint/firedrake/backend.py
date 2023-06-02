#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from firedrake import Constant, DirichletBC, Function, FunctionSpace, \
    Interpolator, LinearSolver, LinearVariationalProblem, \
    LinearVariationalSolver, NonlinearVariationalSolver, Parameters, \
    Projector, Tensor, TestFunction, TrialFunction, UnitIntervalMesh, Vector, \
    VertexOnlyMesh, adjoint, assemble, homogenize, info, interpolate, \
    parameters, project, solve
from firedrake.utils import complex_mode
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

__all__ = \
    [
        "backend",

        "backend_ScalarType",

        "extract_args",
        "extract_linear_solver_args",

        "backend_Constant",
        "backend_DirichletBC",
        "backend_Function",
        "backend_FunctionSpace",
        "backend_LinearSolver",
        "backend_LinearVariationalProblem",
        "backend_LinearVariationalSolver",
        "backend_Matrix",
        "backend_NonlinearVariationalSolver",
        "backend_Vector",
        "backend_assemble",
        "backend_interpolate",
        "backend_project",
        "backend_solve",

        "FunctionSpace",
        "Interpolator",
        "Parameters",
        "Projector",
        "Tensor",
        "TestFunction",
        "TrialFunction",
        "UnitIntervalMesh",
        "VertexOnlyMesh",
        "adjoint",
        "complex_mode",
        "homogenize",
        "info",
        "parameters"
    ]
