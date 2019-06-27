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

from .backend import *
from .backend_code_generator_interface import *
from .backend_interface import *

from .base_equations import *
from .caches import Cache, CacheRef, form_dependencies, form_key, parameters_key
from .equations import EquationSolver, alias_assemble, alias_form

import numpy
import types
import ufl

__all__ = \
    [
        "LocalProjectionSolver",
        "PointInterpolationSolver",

        "LocalSolverCache",
        "local_solver_cache",
        "set_local_solver_cache"
    ]

def local_solver_key(form, bcs, form_compiler_parameters):
    return (form_key(form), tuple(bcs), parameters_key(form_compiler_parameters))

def LocalSolver(form, bcs = [], form_compiler_parameters = {}):
    local_solver = assemble(Tensor(form).inv, bcs = bcs,
        form_compiler_parameters = form_compiler_parameters)
    local_solver.force_evaluation()
    def solve_local(self, x, b):
        matrix_multiply(self, b, tensor = x)
    local_solver.solve_local = types.MethodType(solve_local, local_solver)
    return local_solver

class LocalSolverCache(Cache):
    def local_solver(self, form, bcs = [], form_compiler_parameters = {},
        replace_map = None):
        key = local_solver_key(form, bcs, form_compiler_parameters)
        value = self.get(key, None)
        if value is None:
            assemble_form = form if replace_map is None else ufl.replace(form, replace_map)
            local_solver = LocalSolver(form, bcs = bcs,
                form_compiler_parameters = form_compiler_parameters)
            value = self.add(key, local_solver, deps = tuple(form_dependencies(form).values()))
        else:
            local_solver = value()

        return value, local_solver

_local_solver_cache = [LocalSolverCache()]
def local_solver_cache():
    return _local_solver_cache[0]
def set_local_solver_cache(local_solver_cache):
    _local_solver_cache[0] = local_solver_cache

class LocalProjectionSolver(EquationSolver):
    def __init__(self, rhs, x, bcs = [], form_compiler_parameters = {},
        cache_jacobian = None, cache_rhs_assembly = None, match_quadrature = None,
        defer_adjoint_assembly = None):
        space = x.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        lhs = ufl.inner(test, trial) * ufl.dx
        if not isinstance(rhs, ufl.classes.Form):
            rhs = ufl.inner(test, rhs) * ufl.dx

        EquationSolver.__init__(self, lhs == rhs, x, bcs = bcs,
            form_compiler_parameters = form_compiler_parameters,
            solver_parameters = {},
            cache_jacobian = cache_jacobian,
            cache_rhs_assembly = cache_rhs_assembly,
            match_quadrature = match_quadrature,
            defer_adjoint_assembly = defer_adjoint_assembly)

    def forward_solve(self, x, deps = None):
        if self._cache_rhs_assembly:
            b = self._cached_rhs(deps)
        elif deps is None:
            b = assemble(self._rhs,
                form_compiler_parameters = self._form_compiler_parameters)
        else:
            if self._forward_eq is None:
                self._forward_eq = None, None, alias_form(self._rhs, self.dependencies())
            _, _, rhs = self._forward_eq
            b = alias_assemble(rhs, deps,
                form_compiler_parameters = self._form_compiler_parameters)

        if self._cache_jacobian:
            local_solver = self._forward_J_solver()
            if local_solver is None:
                self._forward_J_solver, local_solver = local_solver_cache().local_solver(
                    self._lhs, bcs = self._bcs,
                    form_compiler_parameters = self._form_compiler_parameters)
        else:
            local_solver = LocalSolver(
                self._lhs, bcs = self._bcs,
                form_compiler_parameters = self._form_compiler_parameters)

        local_solver.solve_local(x.vector(), b)

    def adjoint_jacobian_solve(self, nl_deps, b):
        if self._cache_jacobian:
            local_solver = self._forward_J_solver()
            if local_solver is None:
                self._forward_J_solver, local_solver = local_solver_cache().local_solver(
                    self._lhs, bcs = self._hbcs,
                    form_compiler_parameters = self._form_compiler_parameters)
        else:
            local_solver = LocalSolver(
                self._lhs, bcs = self._hbcs,
                form_compiler_parameters = self._form_compiler_parameters)

        adj_x = function_new(b)
        local_solver.solve_local(adj_x.vector(), b.vector())
        return adj_x

    def reset_adjoint_jacobian_solve(self):
        self._forward_J_solver = CacheRef()

    #def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    # A consistent diagonal block adjoint derivative action requires an
    # appropriate quadrature degree to have been selected

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()
        if x in M:
            raise EquationException("Invalid tangent-linear parameter")

        tlm_rhs = ufl.classes.Zero()
        for m, dm in zip(M, dM):
            tlm_rhs += ufl.derivative(self._rhs, m, argument = dm)

        for dep in self.dependencies():
            if dep != x and not dep in M:
                tau_dep = tlm_map[dep]
                if not tau_dep is None:
                    tlm_rhs += ufl.derivative(self._rhs, dep, argument = tau_dep)

        if isinstance(tlm_rhs, ufl.classes.Zero):
            return NullSolver(tlm_map[x])
        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if tlm_rhs.empty():
            return NullSolver(tlm_map[x])
        else:
            return LocalProjectionSolver(tlm_rhs, tlm_map[x],
                form_compiler_parameters = self._form_compiler_parameters,
                cache_jacobian = self._cache_jacobian,
                cache_rhs_assembly = self._cache_rhs_assembly,
                defer_adjoint_assembly = self._defer_adjoint_assembly)

def interpolation_matrix(x_coords, y, y_nodes):
    N = function_local_size(y)
    lg_map = y.function_space().local_to_global_map([]).indices
    gl_map = {g:l for l, g in enumerate(lg_map)}

    import scipy.sparse
    P = scipy.sparse.dok_matrix((x_coords.shape[0], N), dtype = numpy.float64)

    y_v = function_new(y)
    for x_node, x_coord in enumerate(x_coords):
        for j, y_node in enumerate(y_nodes[x_node, :]):
            with y_v.dat.vec as y_v_v:
                y_v_v.setValue(y_node, 1.0)
                y_v_v.assemblyBegin()
                y_v_v.assemblyEnd()
            x_v = y_v(x_coord)
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
    def __init__(self, y, X, X_coords = None, P = None, P_T = None):
        """
        Defines an equation which interpolates the continuous scalar function y at
        the points X_coords.

        Arguments:

        y         A continuous scalar Function. The function to be interpolated.
        X         A real Function, or a list or tuple of real Function objects.
                            The solution to the equation.
        X_coords  A float NumPy matrix. Points at which to interpolate y. Ignored if
                            P is supplied, required otherwise.
        P         (Optional) Interpolation matrix.
        P_T       (Optional) Interpolation matrix transpose.
        """

        if is_function(X): X = (X,)
        for x in X:
            if not is_real_function(x):
                raise EquationException("Solution must be a real Function, or a list or tuple of real Function objects")

        if P is None:
            y_space = y.function_space()
            y_cell_node_graph = y_space.cell_node_map().values
            y_mesh = y_space.mesh()
            lg_map = y_space.local_to_global_map([]).indices

            y_nodes_local = numpy.empty((len(X), y_cell_node_graph.shape[1]), dtype = numpy.int64)
            for i, x_coord in enumerate(X_coords):
                y_cell = y_mesh.locate_cell(x_coord)
                if y_cell is None or y_cell >= y_cell_node_graph.shape[0]:
                    y_nodes_local[i, :] = -1
                else:
                    for j, y_node in enumerate(y_cell_node_graph[y_cell, :]):
                        y_nodes_local[i, j] = lg_map[y_node]

            y_nodes = numpy.empty(y_nodes_local.shape, dtype = numpy.int64)
            import mpi4py.MPI
            comm = function_comm(y)
            comm.Allreduce(y_nodes_local, y_nodes, op = mpi4py.MPI.MAX)

            P = interpolation_matrix(X_coords, y, y_nodes)

        if P_T is None:
            P_T = P.T

        Equation.__init__(self, X, list(X) + [y], nl_deps = [], ic_deps = [])
        self._P = P
        self._P_T = P_T

    def forward_solve(self, X, deps = None):
        if is_function(X): X = (X,)
        y = (self.dependencies() if deps is None else deps)[-1]

        y_v = function_get_values(y)
        x_v_local = numpy.empty(len(X), dtype = numpy.float64)
        for i in range(len(X)):
            x_v_local[i] = self._P.getrow(i).dot(y_v)

        import mpi4py.MPI
        comm = function_comm(y)
        x_v = numpy.empty(len(X), dtype = numpy.float64)
        comm.Allreduce(x_v_local, x_v, op = mpi4py.MPI.SUM)

        for i, x in enumerate(X):
            function_assign(x, x_v[i])

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_function(adj_X): adj_X = (adj_X,)

        if dep_index < len(adj_X):
            return adj_X[dep_index]
        elif dep_index == len(adj_X):
            adj_x_v = numpy.empty(len(adj_X), dtype = numpy.float64)
            for i, adj_x in enumerate(adj_X):
                adj_x_v[i] = function_max_value(adj_x)
            F = function_new(self.dependencies()[-1])
            function_set_values(F, self._P_T.dot(adj_x_v))
            return (-1.0, F)
        else:
            return None

    def adjoint_jacobian_solve(self, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        X = self.X()
        y = self.dependencies()[-1]

        for x in X:
            if x in M:
                raise EquationException("Invalid tangent-linear parameter")

        try:
            tlm_y_index = M.index(y)
        except ValueError:
            tlm_y_index = None
        if tlm_y_index is None:
            tlm_y = tlm_map[y]
        else:
            tlm_y = dM[tlm_y_index]

        if tlm_y is None:
            return NullSolver([tlm_map[x] for x in X])
        else:
            return PointInterpolationSolver(tlm_y, [tlm_map[x] for x in X], P = self._P, P_T = self._P_T)
