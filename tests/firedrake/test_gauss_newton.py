#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# For tlm_adjoint copyright information see ACKNOWLEDGEMENTS in the tlm_adjoint
# root directory

# This file is part of tlm_adjoint.
#
# tlm_adjoint is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# tlm_adjoint is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tlm_adjoint.  If not, see <https://www.gnu.org/licenses/>.

from firedrake import *
from tlm_adjoint.firedrake import *
from tlm_adjoint.firedrake.backend_code_generator_interface import \
    function_vector

from .test_base import *

import numpy as np
import pytest
import ufl


@pytest.mark.firedrake
@seed_test
def test_GaussNewton(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    eps = Constant(0.1234, static=True)

    def forward(F):
        u = Function(space, name="u")
        solve(inner(grad(trial), grad(test)) * dx == -inner(F, test) * dx,
              u,
              HomogeneousDirichletBC(space, "on_boundary"),
              solver_parameters=ls_parameters_cg)
        return u

    u_ref = Function(space, name="u_ref")
    interpolate_expression(u_ref, X[0] * sin(2.0 * pi * X[0]) * sin(pi * X[1]))

    def forward_J(F):
        u = forward(F)
        J = Functional(name="J")
        J.assign(0.5 * dot(grad(u - u_ref), grad(u - u_ref)) * dx  # Likelihood
                 + 0.5 * eps * dot(F, F) * dx)  # Prior
        return J

    def R_inv_action(x):
        y = function_new(x)
        assemble(inner(grad(ufl.conj(x)), grad(test)) * dx,
                 tensor=function_vector(y))
        return y

    def B_inv_action(x):
        y = function_new(x)
        assemble(eps * inner(ufl.conj(x), test) * dx,
                 tensor=function_vector(y))
        return y

    F = Function(space, name="F")
    interpolate_expression(F, sin(pi * X[0]) * exp(X[1]))

    H = Hessian(forward_J)
    H_GN = GaussNewton(forward, R_inv_action, B_inv_action=B_inv_action)

    for i in range(20):
        dm = Function(space, static=True)
        dm_arr = np.random.random(function_local_size(dm))
        if issubclass(function_dtype(dm), (complex, np.complexfloating)):
            dm_arr = dm_arr + 1.0j * np.random.random(function_local_size(dm))
        function_set_values(dm, dm_arr)
        del dm_arr

        _, _, H_action = H.action(F, function_copy(dm, static=True))
        H_action_GN = H_GN.action(F, dm)

        H_action_error = function_copy(H_action)
        function_axpy(H_action_error, -1.0, H_action_GN)
        assert function_linf_norm(H_action_error) < 1.0e-18


@pytest.mark.firedrake
@seed_test
def test_CachedGaussNewton(setup_test):
    configure_checkpointing("memory", {"drop_references": False})

    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    eps = Constant(0.1234, static=True)

    def forward(F):
        u = Function(space, name="u")
        solve(inner(grad(trial), grad(test)) * dx == -inner(F, test) * dx,
              u,
              HomogeneousDirichletBC(space, "on_boundary"),
              solver_parameters=ls_parameters_cg)
        return u

    u_ref = Function(space, name="u_ref")
    interpolate_expression(u_ref, X[0] * sin(2.0 * pi * X[0]) * sin(pi * X[1]))

    def forward_J(F):
        u = forward(F)
        J = Functional(name="J")
        J.assign(0.5 * dot(grad(u - u_ref), grad(u - u_ref)) * dx  # Likelihood
                 + 0.5 * eps * dot(F, F) * dx)  # Prior
        return J

    def R_inv_action(x):
        y = function_new(x)
        assemble(inner(grad(ufl.conj(x)), grad(test)) * dx,
                 tensor=function_vector(y))
        return y

    def B_inv_action(x):
        y = function_new(x)
        assemble(eps * inner(ufl.conj(x), test) * dx,
                 tensor=function_vector(y))
        return y

    F = Function(space, name="F")
    interpolate_expression(F, sin(pi * X[0]) * exp(X[1]))

    H = Hessian(forward_J)
    start_manager()
    u = forward(F)
    stop_manager()
    H_GN = CachedGaussNewton(u, R_inv_action, B_inv_action=B_inv_action)

    for i in range(20):
        dm = Function(space, static=True)
        dm_arr = np.random.random(function_local_size(dm))
        if issubclass(function_dtype(dm), (complex, np.complexfloating)):
            dm_arr = dm_arr + 1.0j * np.random.random(function_local_size(dm))
        function_set_values(dm, dm_arr)
        del dm_arr

        _, _, H_action = H.action(F, function_copy(dm, static=True))
        H_action_GN = H_GN.action(F, dm)

        H_action_error = function_copy(H_action)
        function_axpy(H_action_error, -1.0, H_action_GN)
        assert function_linf_norm(H_action_error) < 1.0e-18
