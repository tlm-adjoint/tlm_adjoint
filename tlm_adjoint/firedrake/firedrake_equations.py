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

from .backend import Tensor, TestFunction, TrialFunction, backend_Function, \
    backend_assemble
from ..interface import function_assign, function_comm, function_get_values, \
    function_local_size, function_new, function_set_values, function_space, \
    is_function, is_real_function, real_function_value, weakref_method
from .backend_code_generator_interface import assemble, matrix_multiply

from ..caches import Cache
from ..equations import Equation, EquationException, NullSolver, \
    get_tangent_linear

from .caches import form_dependencies, form_key, parameters_key
from .equations import EquationSolver, bind_form, derivative, unbind_form, \
    unbound_form
from .functions import eliminate_zeros

import mpi4py.MPI as MPI
import numpy as np
import ufl

__all__ = \
    [
        "LocalSolverCache",
        "local_solver_cache",
        "set_local_solver_cache",

        "LocalProjectionSolver",
        "PointInterpolationSolver"
    ]


def local_solver_key(form, form_compiler_parameters):
    return (form_key(form),
            parameters_key(form_compiler_parameters))


def LocalSolver(form, form_compiler_parameters={}):
    # Perform zero elimination here, rather than in overridden assemble, as
    # Tensor(form).inv is not a Form
    form = eliminate_zeros(form, force_non_empty_form=True)
    local_solver = backend_assemble(
        Tensor(form).inv,
        form_compiler_parameters=form_compiler_parameters)

    def solve_local(self, x, b):
        matrix_multiply(self, b, tensor=x)
    local_solver.solve_local = weakref_method(solve_local, local_solver)

    return local_solver


class LocalSolverCache(Cache):
    def local_solver(self, form, form_compiler_parameters={},
                     replace_map=None):
        form = eliminate_zeros(form, force_non_empty_form=True)
        key = local_solver_key(form, form_compiler_parameters)

        def value():
            if replace_map is None:
                assemble_form = form
            else:
                assemble_form = ufl.replace(form, replace_map)
            return LocalSolver(
                assemble_form,
                form_compiler_parameters=form_compiler_parameters)

        return self.add(key, value,
                        deps=tuple(form_dependencies(form).values()))


_local_solver_cache = [LocalSolverCache()]


def local_solver_cache():
    return _local_solver_cache[0]


def set_local_solver_cache(local_solver_cache):
    _local_solver_cache[0] = local_solver_cache


class LocalProjectionSolver(EquationSolver):
    def __init__(self, rhs, x, form_compiler_parameters={},
                 cache_jacobian=None, cache_rhs_assembly=None,
                 match_quadrature=None, defer_adjoint_assembly=None):
        space = function_space(x)
        test, trial = TestFunction(space), TrialFunction(space)
        lhs = ufl.inner(test, trial) * ufl.dx
        if not isinstance(rhs, ufl.classes.Form):
            rhs = ufl.inner(test, rhs) * ufl.dx

        super().__init__(
            lhs == rhs, x,
            form_compiler_parameters=form_compiler_parameters,
            solver_parameters={},
            cache_jacobian=cache_jacobian,
            cache_rhs_assembly=cache_rhs_assembly,
            match_quadrature=match_quadrature,
            defer_adjoint_assembly=defer_adjoint_assembly)

    def forward_solve(self, x, deps=None):
        if self._cache_rhs_assembly:
            b = self._cached_rhs(deps)
        elif deps is None:
            b = assemble(
                self._rhs,
                form_compiler_parameters=self._form_compiler_parameters)
        else:
            if self._forward_eq is None:
                self._forward_eq = \
                    (None,
                     None,
                     unbound_form(self._rhs, self.dependencies()))
            _, _, rhs = self._forward_eq
            bind_form(rhs, deps)
            b = assemble(
                rhs, deps,
                form_compiler_parameters=self._form_compiler_parameters)
            unbind_form(rhs)

        if self._cache_jacobian:
            local_solver = self._forward_J_solver()
            if local_solver is None:
                self._forward_J_solver, local_solver = \
                    local_solver_cache().local_solver(
                        self._lhs,
                        form_compiler_parameters=self._form_compiler_parameters)  # noqa: E501
        else:
            local_solver = LocalSolver(
                self._lhs,
                form_compiler_parameters=self._form_compiler_parameters)

        local_solver.solve_local(x, b)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        if self._cache_jacobian:
            local_solver = self._forward_J_solver()
            if local_solver is None:
                self._forward_J_solver, local_solver = \
                    local_solver_cache().local_solver(
                        self._lhs,
                        form_compiler_parameters=self._form_compiler_parameters)  # noqa: E501
        else:
            local_solver = LocalSolver(
                self._lhs,
                form_compiler_parameters=self._form_compiler_parameters)

        adj_x = function_new(b)
        local_solver.solve_local(adj_x, b)
        return adj_x

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = ufl.classes.Form([])
        for dep in self.dependencies():
            if dep != x:
                tau_dep = get_tangent_linear(dep, M, dM, tlm_map)
                if tau_dep is not None:
                    tlm_rhs += derivative(self._rhs, dep, argument=tau_dep)

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if tlm_rhs.empty():
            return NullSolver(tlm_map[x])
        else:
            return LocalProjectionSolver(
                tlm_rhs, tlm_map[x],
                form_compiler_parameters=self._form_compiler_parameters,
                cache_jacobian=self._cache_jacobian,
                cache_rhs_assembly=self._cache_rhs_assembly,
                defer_adjoint_assembly=self._defer_adjoint_assembly)


def interpolation_matrix(x_coords, y, y_nodes):
    N = function_local_size(y)
    lg_map = function_space(y).local_to_global_map([]).indices
    gl_map = {g: l for l, g in enumerate(lg_map)}  # noqa: E741

    from scipy.sparse import dok_matrix
    P = dok_matrix((x_coords.shape[0], N), dtype=np.float64)

    y_v = function_new(y)
    for x_node, x_coord in enumerate(x_coords):
        for j, y_node in enumerate(y_nodes[x_node, :]):
            with y_v.dat.vec as y_v_v:
                y_v_v.setValue(y_node, 1.0)
                y_v_v.assemblyBegin()
                y_v_v.assemblyEnd()
            x_v = y_v(tuple(x_coord))
            if y_node in gl_map:
                y_node_local = gl_map[y_node]
                if y_node_local < N:
                    P[x_node, y_node_local] = x_v
            with y_v.dat.vec as y_v_v:
                y_v_v.setValue(y_node, 0.0)
                y_v_v.assemblyBegin()
                y_v_v.assemblyEnd()

    return P.tocsr()


class PointInterpolationSolver(Equation):
    def __init__(self, y, X, X_coords=None, P=None, P_T=None, tolerance=None):
        """
        Defines an equation which interpolates the continuous scalar Function y
        at the points X_coords.

        Arguments:

        y          A continuous scalar Function. The Function to be
                   interpolated.
        X          A real function, or a list or tuple of real functions. The
                   solution to the equation.
        X_coords   A float NumPy matrix. Points at which to interpolate y.
                   Ignored if P is supplied, required otherwise.
        P          (Optional) Interpolation matrix.
        P_T        (Optional) Interpolation matrix transpose.
        tolerance  (Optional) Cell containment tolerance, passed to the
                   MeshGeometry.locate_cell method. Ignored if P is supplied.
        """

        if is_function(X):
            X = (X,)
        for x in X:
            if not is_real_function(x):
                raise EquationException("Solution must be a real function, "
                                        "or a list or tuple of real functions")
        if X_coords is None:
            if P is None:
                raise EquationException("X_coords required when P is not supplied")  # noqa: E501
        else:
            if len(X) != X_coords.shape[0]:
                raise EquationException("Invalid number of functions")
        if not isinstance(y, backend_Function) or len(y.ufl_shape) > 0:
            raise EquationException("y must be a scalar Function")

        if P is None:
            y_space = function_space(y)
            y_cell_node_graph = y_space.cell_node_map().values
            y_mesh = y_space.mesh()
            lg_map = y_space.local_to_global_map([]).indices

            y_nodes_local = np.full((len(X), y_cell_node_graph.shape[1]),
                                    -1, dtype=np.int64)
            for i, x_coord in enumerate(X_coords):
                y_cell = y_mesh.locate_cell(x_coord, tolerance=tolerance)
                if y_cell is None or y_cell >= y_cell_node_graph.shape[0]:
                    y_nodes_local[i, :] = -1
                else:
                    assert y_cell >= 0
                    for j, y_node in enumerate(y_cell_node_graph[y_cell, :]):
                        y_nodes_local[i, j] = lg_map[y_node]

            y_nodes = np.full(y_nodes_local.shape, -1, dtype=np.int64)
            comm = function_comm(y)
            comm.Allreduce(y_nodes_local, y_nodes, op=MPI.MAX)
            if (y_nodes < 0).any():
                raise EquationException("Unable to locate one or more cells")

            P = interpolation_matrix(X_coords, y, y_nodes)

        if P_T is None:
            P_T = P.T

        super().__init__(X, list(X) + [y], nl_deps=[], ic=False, adj_ic=False)
        self._P = P
        self._P_T = P_T

    def forward_solve(self, X, deps=None):
        if is_function(X):
            X = (X,)
        y = (self.dependencies() if deps is None else deps)[-1]

        y_v = function_get_values(y)
        x_v_local = np.full(len(X), np.NAN, dtype=np.float64)
        for i in range(len(X)):
            x_v_local[i] = self._P.getrow(i).dot(y_v)

        comm = function_comm(y)
        x_v = np.full(len(X), np.NAN, dtype=np.float64)
        comm.Allreduce(x_v_local, x_v, op=MPI.SUM)

        for i, x in enumerate(X):
            function_assign(x, x_v[i])

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_function(adj_X):
            adj_X = (adj_X,)

        if dep_index < len(adj_X):
            return adj_X[dep_index]
        elif dep_index == len(adj_X):
            adj_x_v = np.full(len(adj_X), np.NAN, dtype=np.float64)
            for i, adj_x in enumerate(adj_X):
                adj_x_v[i] = real_function_value(adj_x)
            F = function_new(self.dependencies()[-1])
            function_set_values(F, self._P_T.dot(adj_x_v))
            return (-1.0, F)
        else:
            raise EquationException("dep_index out of bounds")

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def tangent_linear(self, M, dM, tlm_map):
        X = self.X()
        y = self.dependencies()[-1]

        tlm_y = get_tangent_linear(y, M, dM, tlm_map)
        if tlm_y is None:
            return NullSolver([tlm_map[x] for x in X])
        else:
            return PointInterpolationSolver(tlm_y, [tlm_map[x] for x in X],
                                            P=self._P, P_T=self._P_T)
