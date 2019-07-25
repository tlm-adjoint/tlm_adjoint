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
from .caches import Cache, CacheRef, form_dependencies, form_key
from .equations import EquationSolver, alias_assemble, alias_form

import numpy
import ufl

__all__ = \
    [
        "InterpolationSolver",
        "LocalProjectionSolver",
        "PointInterpolationSolver",

        "LocalSolverCache",
        "local_solver_cache",
        "set_local_solver_cache"
    ]

def greedy_coloring(space):
    """
    A basic greedy coloring of the (process local) node-node graph.
    """

    dofmap = space.dofmap()
    ownership_range = dofmap.ownership_range()
    N = ownership_range[1] - ownership_range[0]

    node_node_graph = [set() for i in range(N)]
    for i in range(space.mesh().num_cells()):
        cell_nodes = dofmap.cell_dofs(i)
        for j in cell_nodes:
            for k in cell_nodes:
                if j != k:
                    node_node_graph[j].add(k)
    node_node_graph = [sorted(nodes, reverse = True) for nodes in node_node_graph]

    seen = numpy.empty(N, dtype = numpy.bool)
    seen[:] = False
    colors = numpy.empty(N, dtype = numpy.int64)
    colors[:] = -1
    i = 0
    while True:
        # Initialise the advancing front
        while i < N and colors[i] >= 0:
            i += 1
        if i == N:
            break  # All nodes have been considered
        front = [i]
        seen[i] = True
        while len(front) > 0:
            # Consider a new node, and the smallest non-negative available color
            j = front.pop()
            neighbouring_colors = set(colors[node_node_graph[j]])
            color = 0
            while color in neighbouring_colors:
                color += 1
            colors[j] = color
            # Advance the front
            for k in node_node_graph[j]:
                if not seen[k]:
                    front.append(k)
                    seen[k] = True
        # If the graph is not connected then we need to restart the front with a new
        # starting node

    return colors

def function_coords(x):
    space = x.function_space()
    coords = numpy.empty((function_local_size(x), space.mesh().geometry().dim()), dtype = numpy.float64)
    for i in range(coords.shape[1]):
        coords[:, i] = function_get_values(interpolate(Expression("x[%i]" % i, element = space.ufl_element()), space))
    return coords

def local_solver_key(form, solver_type):
    return (form_key(form), solver_type)

class LocalSolverCache(Cache):
    def local_solver(self, form, solver_type, replace_map = None):
        key = local_solver_key(form, solver_type)
        value = self.get(key, None)
        if value is None or value() is None:
            assemble_form = form if replace_map is None else ufl.replace(form, replace_map)
            local_solver = LocalSolver(assemble_form, solver_type = solver_type)
            local_solver.factorize()
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
    def __init__(self, rhs, x, form_compiler_parameters = {},
        cache_jacobian = None, cache_rhs_assembly = None, match_quadrature = None,
        defer_adjoint_assembly = None):
        space = x.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        lhs = ufl.inner(test, trial) * ufl.dx
        if not isinstance(rhs, ufl.classes.Form):
            rhs = ufl.inner(test, rhs) * ufl.dx

        EquationSolver.__init__(self, lhs == rhs, x,
            form_compiler_parameters = form_compiler_parameters,
            solver_parameters = {"linear_solver":"direct"},
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
                    self._lhs,
                    LocalSolver.SolverType.Cholesky if hasattr(LocalSolver, "SolverType") else LocalSolver.SolverType_Cholesky)
        else:
            local_solver = LocalSolver(
                self._lhs,
                solver_type = LocalSolver.SolverType.Cholesky if hasattr(LocalSolver, "SolverType") else LocalSolver.SolverType_Cholesky)

        local_solver.solve_local(x.vector(), b, x.function_space().dofmap())

    def adjoint_jacobian_solve(self, nl_deps, b):
        if self._cache_jacobian:
            local_solver = self._forward_J_solver()
            if local_solver is None:
                self._forward_J_solver, local_solver = local_solver_cache().local_solver(
                    self._lhs,
                    LocalSolver.SolverType.Cholesky if hasattr(LocalSolver, "SolverType") else LocalSolver.SolverType_Cholesky)
        else:
            local_solver = LocalSolver(
                self._lhs,
                solver_type = LocalSolver.SolverType.Cholesky if hasattr(LocalSolver, "SolverType") else LocalSolver.SolverType_Cholesky)

        adj_x = function_new(b)
        local_solver.solve_local(adj_x.vector(), b.vector(), adj_x.function_space().dofmap())
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

def interpolation_matrix(x_coords, y, y_cells, y_colors):
    y_space = y.function_space()
    y_mesh = y_space.mesh()
    y_dofmap = y_space.dofmap()

    # Verify process locality assumption
    # The process locality assumption can be avoided by additionally defining P
    # for non-owned nodes, but this requires a parallel graph coloring.
    y_ownership_range = y_dofmap.ownership_range()
    for y_cell in range(y_mesh.num_cells()):
        owned = numpy.array([j >= y_ownership_range[0] and j < y_ownership_range[1]
            for j in [y_dofmap.local_to_global_index(i) for i in y_dofmap.cell_dofs(y_cell)]],
            dtype = numpy.bool)
        if owned.any() and not owned.all():
            raise EquationException("Non-process-local node-node graph")

    y_colors_N = numpy.empty((1,), dtype = y_colors.dtype)
    comm = function_comm(y)
    import mpi4py.MPI
    (comm.tompi4py() if hasattr(comm, "tompi4py") else comm).Allreduce(
        numpy.array([y_colors.max() + 1], dtype = y_colors.dtype),
        y_colors_N, op = mpi4py.MPI.MAX)
    y_colors_N = y_colors_N[0]
    y_nodes = [[] for i in range(y_colors_N)]
    for y_node, color in enumerate(y_colors):
        y_nodes[color].append(y_node)

    import scipy.sparse
    P = scipy.sparse.dok_matrix((x_coords.shape[0], function_local_size(y)), dtype = numpy.float64)

    y_v = function_new(y)
    x_v = numpy.empty((1,), dtype = numpy.float64)
    for color, y_color_nodes in enumerate(y_nodes):
        y_v.vector()[y_color_nodes] = 1.0
        for x_node, y_cell in enumerate(y_cells):
            if y_cell < 0:
                continue
            y_cell_nodes = y_dofmap.cell_dofs(y_cell)
            try:
                i = y_colors[y_cell_nodes].tolist().index(color)
            except ValueError:
                continue
            y_node = y_cell_nodes[i]
            y_v.eval_cell(x_v, x_coords[x_node, :], Cell(y_mesh, y_cell))  # Broken in parallel with FEniCS 2017.2.0
            P[x_node, y_node] = x_v[0]
        if color < len(y_nodes) - 1: y_v.vector()[y_color_nodes] = 0.0

    return P.tocsr()

class InterpolationSolver(LinearEquation):
    def __init__(self, y, x, y_colors = None, P = None, P_T = None):
        """
        Defines an equation which interpolates the scalar function y.

        Internally this builds (or uses a supplied) interpolation matrix for the
        *local process only*. This works correctly in parallel if y is in a
        discontinuous function space (e.g. Discontinuous Lagrange) but may fail in
        parallel otherwise.

        For parallel cases this equation can be combined with LocalProjectionSolver
        to first project the input field onto an appropriate discontinuous space.

        Arguments:

        y         A scalar Function. The function to be interpolated.
        x         A scalar Function. The solution to the equation.
        y_colors  (Optional) An integer NumPy vector. Node-node graph coloring for
                            the space for y. Defaults to a basic greedy graph coloring.
                            Ignored if P is supplied.
        P         (Optional) Interpolation matrix.
        P_T       (Optional) Interpolation matrix transpose.
        """

        if P is None:
            y_space = y.function_space()
            if y_colors is None:
                y_colors = greedy_coloring(y_space)
            y_tree = y_space.mesh().bounding_box_tree()

            x_coords = function_coords(x)
            y_cells = [y_tree.compute_closest_entity(Point(*x_coord))[0] for x_coord in x_coords]

            P = interpolation_matrix(x_coords, y, y_cells, y_colors)

        class InterpolationMatrix(Matrix):
            def __init__(self, P, P_T = None):
                Matrix.__init__(self, nl_deps = [], has_ic_dep = False)
                self._P = P
                self._P_T = P.T if P_T is None else P_T

            def forward_action(self, nl_deps, x, b, method = "assign"):
                if method == "assign":
                    function_set_values(b, self._P.dot(function_get_values(x)))
                elif method == "add":
                    b.vector()[:] += self._P.dot(function_get_values(x))
                elif method == "sub":
                    b.vector()[:] -= self._P.dot(function_get_values(x))
                else:
                    raise EquationException("Invalid method: '%s'" % method)

            def adjoint_action(self, nl_deps, adj_x, b, b_index = 0, method = "assign"):
                if b_index != 0: raise EquationException("Invalid index")
                if method == "assign":
                    function_set_values(b, self._P_T.dot(function_get_values(adj_x)))
                elif method == "add":
                    b.vector()[:] += self._P_T.dot(function_get_values(adj_x))
                elif method == "sub":
                    b.vector()[:] -= self._P_T.dot(function_get_values(adj_x))
                else:
                    raise EquationException("Invalid method: '%s'" % method)

        LinearEquation.__init__(self, MatrixActionRHS(InterpolationMatrix(P, P_T = P_T), y), x)

class PointInterpolationSolver(Equation):
    def __init__(self, y, X, X_coords = None, y_colors = None, y_cells = None,
        P = None, P_T = None):
        """
        Defines an equation which interpolates the scalar function y at the points
        X_coords. It is assumed that the given points are all within the y mesh.

        Internally this builds (or uses a supplied) interpolation matrix for the
        *local process only*. This works correctly in parallel if y is in a
        discontinuous function space (e.g. Discontinuous Lagrange) but may fail in
        parallel otherwise.

        For parallel cases this equation can be combined with LocalProjectionSolver
        to first project the input field onto an appropriate discontinuous space.

        Arguments:

        y         A scalar Function. The function to be interpolated.
        X         A real Function, or a list or tuple of real Function objects.
                            The solution to the equation.
        X_coords  A float NumPy matrix. Points at which to interpolate y. Ignored if
                            P is supplied, required otherwise.
        y_colors  (Optional) An integer NumPy vector. Node-node graph coloring for
                            the space for y. Defaults to a basic greedy graph coloring.
                            Ignored if P is supplied.
        y_cells   (Optional) An integer NumPy vector. The cells in the y mesh
                            containing each point. Ignored if P is supplied.
        P         (Optional) Interpolation matrix.
        P_T       (Optional) Interpolation matrix transpose.
        """

        if is_function(X): X = (X,)
        for x in X:
            if not is_real_function(x):
                raise EquationException("Solution must be a real Function, or a list or tuple of real Function objects")

        if P is None:
            y_space = y.function_space()
            if y_colors is None:
                y_colors = greedy_coloring(y_space)

            if y_cells is None:
                y_tree = y_space.mesh().bounding_box_tree()

                import mpi4py.MPI
                comm = function_comm(y)
                if hasattr(comm, "tompi4py"): comm = comm.tompi4py()
                rank = comm.rank

                y_cells = numpy.empty(len(X), dtype = numpy.int64)
                distances_local = numpy.empty(len(X), dtype = numpy.float64)
                for i, x_coord in enumerate(X_coords):
                    y_cells[i], distances_local[i] = y_tree.compute_closest_entity(Point(*x_coord))
                distances = numpy.empty(len(X), dtype = numpy.float64)
                comm.Allreduce(distances_local, distances, op = mpi4py.MPI.MIN)

                owner_local = numpy.empty(len(X), dtype = numpy.int64)
                owner_local[:] = rank
                for i, (distance_local, distance) in enumerate(zip(distances_local, distances)):
                    if distance_local != distance:
                        y_cells[i] = -1
                        owner_local[i] = -1
                owner = numpy.empty(len(X), dtype = numpy.int64)
                comm.Allreduce(owner_local, owner, op = mpi4py.MPI.MAX)

                for i in range(len(X)):
                    if owner[i] == -1:
                        raise EquationException("Unable to find owning process for point")
                    if owner[i] != rank:
                        y_cells[i] = -1

            P = interpolation_matrix(X_coords, y, y_cells, y_colors)

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
        if hasattr(comm, "tompi4py"): comm = comm.tompi4py()
        x_v = numpy.empty(len(X), dtype = numpy.float64)
        comm.Allreduce(x_v_local, x_v, op = mpi4py.MPI.MAX)

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
