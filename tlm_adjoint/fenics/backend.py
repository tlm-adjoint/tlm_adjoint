#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fenics import Cell, Constant, DirichletBC, Form, Function, \
    FunctionSpace, LocalSolver, Mesh, MeshEditor, KrylovSolver, LUSolver, \
    LinearVariationalSolver, NonlinearVariationalProblem, \
    NonlinearVariationalSolver, Parameters, Point, TensorFunctionSpace, \
    TestFunction, TrialFunction, UnitIntervalMesh, UserExpression, adjoint, \
    as_backend_type, assemble, assemble_system, has_lu_solver_method, info, \
    parameters, project, solve
import fenics
import numpy as np
import petsc4py.PETSc as PETSc

backend = "FEniCS"

backend_ScalarType = PETSc.ScalarType

if not issubclass(backend_ScalarType, (float, np.floating)):
    raise ImportError(f"Invalid backend scalar type: {backend_ScalarType}")

extract_args = fenics.fem.solving._extract_args

backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_FunctionSpace = FunctionSpace
backend_KrylovSolver = KrylovSolver
backend_LUSolver = LUSolver
backend_LinearVariationalSolver = LinearVariationalSolver
backend_Matrix = fenics.cpp.la.GenericMatrix
backend_NonlinearVariationalProblem = NonlinearVariationalProblem
backend_NonlinearVariationalSolver = NonlinearVariationalSolver
backend_Vector = fenics.cpp.la.GenericVector
backend_assemble = assemble
backend_assemble_system = assemble_system
backend_project = project
backend_solve = solve

cpp_LinearVariationalProblem = fenics.cpp.fem.LinearVariationalProblem
cpp_NonlinearVariationalProblem = fenics.cpp.fem.NonlinearVariationalProblem
cpp_PETScVector = fenics.cpp.la.PETScVector

__all__ = \
    [
        "backend",

        "backend_ScalarType",

        "extract_args",

        "backend_Constant",
        "backend_DirichletBC",
        "backend_Function",
        "backend_FunctionSpace",
        "backend_KrylovSolver",
        "backend_LUSolver",
        "backend_LinearVariationalSolver",
        "backend_NonlinearVariationalProblem",
        "backend_NonlinearVariationalSolver",
        "backend_Matrix",
        "backend_Vector",
        "backend_assemble",
        "backend_assemble_system",
        "backend_project",
        "backend_solve",

        "cpp_LinearVariationalProblem",
        "cpp_NonlinearVariationalProblem",
        "cpp_PETScVector",

        "Cell",
        "Form",
        "FunctionSpace",
        "LocalSolver",
        "Mesh",
        "MeshEditor",
        "Parameters",
        "Point",
        "TensorFunctionSpace",
        "TestFunction",
        "TrialFunction",
        "UnitIntervalMesh",
        "UserExpression",
        "adjoint",
        "as_backend_type",
        "has_lu_solver_method",
        "info",
        "parameters"
    ]
