"""Interpolation operations with FEniCS.
"""

from .backend import (
    Cell, Mesh, MeshEditor, Point, UserExpression, backend_Constant,
    backend_Function, backend_ScalarType, parameters)
from ..interface import (
    check_space_type, is_var, space_comm, var_assign, var_comm, var_get_values,
    var_id, var_inner, var_is_scalar, var_local_size, var_new,
    var_new_conjugate_dual, var_replacement, var_scalar_value, var_set_values,
    var_space)

from ..equation import Equation, ZeroAssignment
from ..equations import MatrixActionRHS
from ..linear_equation import LinearEquation, Matrix
from ..manager import manager_disabled

from .expr import (
    ExprEquation, derivative, eliminate_zeros, expr_zero, extract_coefficients,
    extract_dependencies)
from .functions import ReplacementConstant

import functools
import mpi4py.MPI as MPI
import numpy as np
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = \
    [
        "ExprInterpolation",
        "Interpolation",
        "PointInterpolation"
    ]


@manager_disabled()
def interpolate_expression(x, expr, *, adj_x=None):
    if adj_x is None:
        check_space_type(x, "primal")
    else:
        check_space_type(x, "conjugate_dual")
        check_space_type(adj_x, "conjugate_dual")
    for dep in extract_coefficients(expr):
        if is_var(dep):
            check_space_type(dep, "primal")

    expr = eliminate_zeros(expr)

    class Expr(UserExpression):
        def eval(self, value, x):
            value[:] = expr(tuple(x))

        def value_shape(self):
            return x.ufl_shape

    if adj_x is None:
        if isinstance(x, backend_Constant):
            if isinstance(expr, backend_Constant):
                value = expr
            else:
                if len(x.ufl_shape) > 0:
                    raise ValueError("Scalar Constant required")
                value = x.values()
                Expr().eval(value, ())
                value, = value
            var_assign(x, value)
        elif isinstance(x, backend_Function):
            try:
                x.assign(expr)
            except RuntimeError:
                x.interpolate(Expr())
        else:
            raise TypeError(f"Unexpected type: {type(x)}")
    else:
        expr_val = var_new_conjugate_dual(adj_x)
        expr_arguments = ufl.algorithms.extract_arguments(expr)
        if len(expr_arguments) > 0:
            test, = expr_arguments
            if len(test.ufl_shape) > 0:
                raise NotImplementedError("Case not implemented")
            expr = ufl.replace(expr, {test: ufl.classes.IntValue(1)})
        interpolate_expression(expr_val, expr)

        if isinstance(x, backend_Constant):
            if len(x.ufl_shape) > 0:
                raise ValueError("Scalar Constant required")
            var_assign(x, var_inner(adj_x, expr_val))
        elif isinstance(x, backend_Function):
            x_space = var_space(x)
            adj_x_space = var_space(adj_x)
            if x_space.ufl_domains() != adj_x_space.ufl_domains() \
                    or x_space.ufl_element() != adj_x_space.ufl_element():
                raise ValueError("Unable to perform transpose interpolation")
            var_set_values(
                x, var_get_values(expr_val).conjugate() * var_get_values(adj_x))  # noqa: E501
        else:
            raise TypeError(f"Unexpected type: {type(x)}")


class ExprInterpolation(ExprEquation):
    r"""Represents interpolation of `rhs` onto the space for `x`.

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    :arg x: The forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to
        interpolate onto the space for `x`. Should not depend on `x`.
    """

    def __init__(self, x, rhs):
        deps, nl_deps = extract_dependencies(rhs, space_type="primal")
        if var_id(x) in deps:
            raise ValueError("Invalid dependency")
        deps, nl_deps = list(deps.values()), tuple(nl_deps.values())
        deps.insert(0, x)

        super().__init__(x, deps, nl_deps=nl_deps, ic=False, adj_ic=False)
        self._rhs = rhs

    def drop_references(self):
        replace_map = {dep: var_replacement(dep)
                       for dep in self.dependencies()}

        super().drop_references()
        self._rhs = ufl.replace(self._rhs, replace_map)

    def forward_solve(self, x, deps=None):
        interpolate_expression(x, self._replace(self._rhs, deps))

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        eq_deps = self.dependencies()
        if dep_index <= 0 or dep_index >= len(eq_deps):
            raise ValueError("Unexpected dep_index")

        dep = eq_deps[dep_index]

        if isinstance(dep, (backend_Constant, ReplacementConstant)):
            if len(dep.ufl_shape) > 0:
                raise NotImplementedError("Case not implemented")
            dF = derivative(self._rhs, dep, argument=ufl.classes.IntValue(1))
        else:
            dF = derivative(self._rhs, dep)
        dF = ufl.algorithms.expand_derivatives(dF)
        dF = eliminate_zeros(dF)
        dF = self._nonlinear_replace(dF, nl_deps)

        F = var_new_conjugate_dual(dep)
        interpolate_expression(F, dF, adj_x=adj_x)
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x = self.x()

        tlm_rhs = expr_zero(x)
        for dep in self.dependencies():
            if dep != x:
                tau_dep = tlm_map[dep]
                if tau_dep is not None:
                    tlm_rhs = (tlm_rhs
                               + derivative(self._rhs, dep, argument=tau_dep))

        tlm_rhs = ufl.algorithms.expand_derivatives(tlm_rhs)
        if isinstance(tlm_rhs, ufl.classes.Zero):
            return ZeroAssignment(tlm_map[x])
        else:
            return ExprInterpolation(tlm_map[x], tlm_rhs)


def function_coords(x):
    return x.function_space().tabulate_dof_coordinates()


def has_ghost_cells(mesh):
    for cell in range(mesh.num_cells()):
        if Cell(mesh, cell).is_ghost():
            return True
    return False


def reorder_gps_disabled(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        reorder_vertices_gps = parameters["reorder_vertices_gps"]
        reorder_cells_gps = parameters["reorder_cells_gps"]
        parameters["reorder_vertices_gps"] = False
        parameters["reorder_cells_gps"] = False
        try:
            return fn(*args, **kwargs)
        finally:
            parameters["reorder_vertices_gps"] = reorder_vertices_gps
            parameters["reorder_cells_gps"] = reorder_cells_gps
    return wrapped_fn


@reorder_gps_disabled
def local_mesh(mesh):
    coordinates = mesh.coordinates()
    cells = mesh.cells()
    assert coordinates.shape[0] == mesh.num_vertices()
    assert cells.shape[0] == mesh.num_cells()

    local_vertex_map = {}
    full_vertex_map = []
    full_cell_map = []
    for full_cell in range(cells.shape[0]):
        if not Cell(mesh, full_cell).is_ghost():
            for full_vertex in cells[full_cell, :]:
                if full_vertex not in local_vertex_map:
                    local_vertex_map[full_vertex] = len(full_vertex_map)
                    full_vertex_map.append(full_vertex)
            full_cell_map.append(full_cell)

    l_mesh = Mesh(MPI.COMM_SELF)
    ed = MeshEditor()
    ed.open(mesh=l_mesh, type=mesh.cell_name(), tdim=mesh.topology().dim(),
            gdim=mesh.geometry().dim(), degree=mesh.geometry().degree())

    N_local_vertices = len(full_vertex_map)
    ed.init_vertices_global(N_local_vertices, N_local_vertices)

    N_local_cells = len(full_cell_map)
    ed.init_cells_global(N_local_cells, N_local_cells)

    for local_vertex, full_vertex in enumerate(full_vertex_map):
        ed.add_vertex(local_vertex, Point(*coordinates[full_vertex, :]))

    for local_cell, full_cell in enumerate(full_cell_map):
        local_vertices = [local_vertex_map[full_vertex]
                          for full_vertex in cells[full_cell, :]]
        ed.add_cell(local_cell, local_vertices)

    ed.close(order=True)
    l_mesh.init()

    return l_mesh, full_vertex_map, full_cell_map


def point_cells(coords, mesh):
    full_cells = np.full(coords.shape[0], -1, dtype=np.int_)
    distances = np.full(coords.shape[0], np.NAN, dtype=backend_ScalarType)

    if mesh.mpi_comm().size == 1 or not has_ghost_cells(mesh):
        full_tree = mesh.bounding_box_tree()
        for i in range(coords.shape[0]):
            point = Point(*coords[i, :])
            full_cell, distance = full_tree.compute_closest_entity(point)
            full_cells[i] = full_cell
            distances[i] = distance
    else:
        l_mesh, _, full_cell_map = local_mesh(mesh)
        local_tree = l_mesh.bounding_box_tree()
        for i in range(coords.shape[0]):
            point = Point(*coords[i, :])
            local_cell, distance = local_tree.compute_closest_entity(point)
            full_cells[i] = full_cell_map[local_cell]
            distances[i] = distance

    assert (full_cells[i] >= 0).all()
    assert (full_cells[i] < mesh.num_cells()).all()
    assert (distances >= 0.0).all()

    return full_cells, distances


def greedy_coloring(space):
    mesh = space.mesh()
    dofmap = space.dofmap()
    ownership_range = dofmap.ownership_range()
    N = ownership_range[1] - ownership_range[0]

    node_node_graph = tuple(set() for _ in range(N))
    for i in range(mesh.num_cells()):
        if Cell(mesh, i).is_ghost():
            continue

        cell_nodes = dofmap.cell_dofs(i)
        for j in cell_nodes:
            for k in cell_nodes:
                if j < 0 or j >= N:
                    raise ValueError("Node index out of range")
                if j != k:
                    node_node_graph[j].add(k)
    node_node_graph = tuple(sorted(nodes, reverse=True)
                            for nodes in node_node_graph)

    # A basic greedy coloring of the (process local) node-node graph, ordered
    # using an advancing front

    seen = np.full(N, False, dtype=bool)
    colors = np.full(N, -1, dtype=np.int_)
    i = 0
    while True:
        # Initialize the advancing front
        while i < N and colors[i] >= 0:
            i += 1
        if i == N:
            break  # All nodes have been considered
        front = [i]
        seen[i] = True
        while len(front) > 0:
            # Consider a new node, and the smallest non-negative available
            # color
            j = front.pop()
            neighbouring_colors = set(colors[k] for k in node_node_graph[j])
            color = 0
            while color in neighbouring_colors:
                color += 1
            colors[j] = color
            # Advance the front
            for k in node_node_graph[j]:
                if not seen[k]:
                    front.append(k)
                    seen[k] = True
        # If the graph is not connected then we need to restart the front with
        # a new starting node
    if (colors < 0).any():
        raise RuntimeError("Invalid graph coloring")

    return colors


def point_owners(x_coords, y_space, *,
                 tolerance=0.0):
    comm = space_comm(y_space)
    rank = comm.rank

    y_cells, distances_local = point_cells(x_coords, y_space.mesh())
    distances = np.full(x_coords.shape[0], np.NAN, dtype=distances_local.dtype)
    comm.Allreduce(distances_local, distances, op=MPI.MIN)

    owner_local = np.full(x_coords.shape[0], rank, dtype=np.int_)
    assert len(distances_local) == len(distances)
    for i, (distance_local, distance) in enumerate(zip(distances_local,
                                                       distances)):
        if distance_local != distance:
            y_cells[i] = -1
            owner_local[i] = -1
    owner = np.full(x_coords.shape[0], -1, dtype=owner_local.dtype)
    comm.Allreduce(owner_local, owner, op=MPI.MAX)

    for i in range(x_coords.shape[0]):
        if owner[i] == -1:
            raise RuntimeError("Unable to find owning process for point")
        if owner[i] == rank:
            if distances_local[i] > tolerance:
                raise RuntimeError("Unable to find owning process for point")
        else:
            y_cells[i] = -1

    return y_cells


def interpolation_matrix(x_coords, y, y_cells, y_colors):
    y_space = y.function_space()
    y_mesh = y_space.mesh()
    y_dofmap = y_space.dofmap()

    # Verify process locality assumption
    y_ownership_range = y_dofmap.ownership_range()
    for y_cell in range(y_mesh.num_cells()):
        owned = np.array([j >= y_ownership_range[0]
                          and j < y_ownership_range[1]
                          for j in [y_dofmap.local_to_global_index(i)
                                    for i in y_dofmap.cell_dofs(y_cell)]],
                         dtype=bool)
        if owned.any() and not owned.all():
            raise RuntimeError("Non-process-local node-node graph")

    comm = var_comm(y)
    if len(y_colors) == 0 or y_colors.min() != 0:
        raise ValueError("Invalid graph coloring")
    y_colors_N = y_colors.max() + 1
    y_colors_N = comm.allreduce(y_colors_N, op=MPI.MAX)
    assert y_colors_N >= 0
    y_nodes = tuple([] for _ in range(y_colors_N))
    for y_node, color in enumerate(y_colors):
        y_nodes[color].append(y_node)

    from scipy.sparse import dok_matrix
    P = dok_matrix((x_coords.shape[0], var_local_size(y)),
                   dtype=backend_ScalarType)

    y_v = var_new(y)
    for color, y_color_nodes in enumerate(y_nodes):
        y_v.vector()[y_color_nodes] = 1.0
        for x_node, y_cell in enumerate(y_cells):
            if y_cell < 0:
                # Skip -- x_node is owned by a different process
                continue
            if Cell(y_mesh, y_cell).is_ghost():
                raise RuntimeError("Cannot interpolate within a ghost cell")

            y_cell_nodes = y_dofmap.cell_dofs(y_cell)
            y_cell_colors = [y_colors[j] for j in y_cell_nodes]
            if color in y_cell_colors:
                assert y_cell_colors.count(color) == 1
                i = y_cell_colors.index(color)
            else:
                continue
            y_node = y_cell_nodes[i]
            x_v = np.full((1,), np.NAN, dtype=backend_ScalarType)
            y_v.eval_cell(x_v, x_coords[x_node, :], Cell(y_mesh, y_cell))
            P[x_node, y_node] = x_v[0]
        y_v.vector()[y_color_nodes] = 0.0

    return P.tocsr()


class LocalMatrix(Matrix):
    def __init__(self, P):
        super().__init__(nl_deps=[], ic=False, adj_ic=False)
        self._P = P.copy()
        self._P_T = P.T

    def forward_action(self, nl_deps, x, b, *, method="assign"):
        if method == "assign":
            var_set_values(b, self._P.dot(var_get_values(x)))
        elif method == "add":
            b.vector()[:] += self._P.dot(var_get_values(x))
        elif method == "sub":
            b.vector()[:] -= self._P.dot(var_get_values(x))
        else:
            raise ValueError(f"Invalid method: '{method:s}'")

    def adjoint_action(self, nl_deps, adj_x, b, b_index=0, *, method="assign"):
        if b_index != 0:
            raise ValueError("Unexpected b_index")

        if method == "assign":
            var_set_values(
                b, self._P_T.dot(var_get_values(adj_x)))
        elif method == "add":
            b.vector()[:] += self._P_T.dot(var_get_values(adj_x))
        elif method == "sub":
            b.vector()[:] -= self._P_T.dot(var_get_values(adj_x))
        else:
            raise ValueError(f"Invalid method: '{method:s}'")


class Interpolation(LinearEquation):
    r"""Represents interpolation of the scalar-valued function `y` onto the
    space for `x`.

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    Internally this builds (or uses a supplied) interpolation matrix for the
    local process *only*. This behaves correctly if the there are no edges
    between owned and non-owned nodes in the degree of freedom graph associated
    with the discrete function space for `y`.

    :arg x: A scalar-valued DOLFIN `Function` defining the forward solution.
    :arg y: A scalar-valued DOLFIN `Function` to interpolate onto the space for
        `x`.
    :arg X_coords: A :class:`numpy.ndarray` defining the coordinates at which
        to interpolate `y`. Shape is `(n, d)` where `n` is the number of
        process local degrees of freedom for `x` and `d` is the geometric
        dimension. Defaults to the process local degree of freedom locations
        for `x`. Ignored if `P` is supplied.
    :arg P: The interpolation matrix. A :class:`scipy.sparse.spmatrix`.
    :arg tolerance: Maximum permitted distance (as returned by the DOLFIN
        `BoundingBoxTree.compute_closest_entity` method) of an interpolation
        point from a cell in the mesh for `y`. Ignored if `P` is supplied.
    """

    def __init__(self, x, y, *, x_coords=None, P=None,
                 tolerance=0.0):
        check_space_type(x, "primal")
        check_space_type(y, "primal")

        if not isinstance(x, backend_Function):
            raise TypeError("Solution must be a Function")
        if len(x.ufl_shape) > 0:
            raise ValueError("Solution must be a scalar-valued Function")
        if not isinstance(y, backend_Function):
            raise TypeError("y must be a Function")
        if len(y.ufl_shape) > 0:
            raise ValueError("y must be a scalar-valued Function")
        if (x_coords is not None) and (var_comm(x).size > 1):
            raise TypeError("Cannot prescribe x_coords in parallel")

        if P is None:
            y_space = y.function_space()

            if x_coords is None:
                x_coords = function_coords(x)

            y_cells, y_distances = point_cells(x_coords, y_space.mesh())
            if (y_distances > tolerance).any():
                raise RuntimeError("Unable to locate one or more cells")

            y_colors = greedy_coloring(y_space)
            P = interpolation_matrix(x_coords, y, y_cells, y_colors)
        else:
            P = P.copy()

        super().__init__(
            x, MatrixActionRHS(LocalMatrix(P), y))


class PointInterpolation(Equation):
    r"""Represents interpolation of a scalar-valued function at given points.

    The forward residual :math:`\mathcal{F}` is defined so that :math:`\partial
    \mathcal{F} / \partial x` is the identity.

    Internally this builds (or uses a supplied) interpolation matrix for the
    local process *only*. This behaves correctly if the there are no edges
    between owned and non-owned nodes in the degree of freedom graph associated
    with the discrete function space for `y`.

    :arg X: A scalar variable, or a :class:`Sequence` of scalar variables,
        defining the forward solution.
    :arg y: A scalar-valued DOLFIN `Function` to interpolate.
    :arg X_coords: A :class:`numpy.ndarray` defining the coordinates at which
        to interpolate `y`. Shape is `(n, d)` where `n` is the number of
        interpolation points and `d` is the geometric dimension. Ignored if `P`
        is supplied.
    :arg P: The interpolation matrix. A :class:`scipy.sparse.spmatrix`.
    :arg tolerance: Maximum permitted distance (as returned by the DOLFIN
        `BoundingBoxTree.compute_closest_entity` method) of an interpolation
        point from a cell in the mesh for `y`. Ignored if `P` is supplied.
    """

    def __init__(self, X, y, X_coords=None, *,
                 P=None, tolerance=0.0):
        if is_var(X):
            X = (X,)
        for x in X:
            check_space_type(x, "primal")
            if not var_is_scalar(x):
                raise ValueError("Solution must be a scalar variable, or a "
                                 "Sequence of scalar variables")
        check_space_type(y, "primal")

        if X_coords is None:
            if P is None:
                raise TypeError("X_coords required when P is not supplied")
        else:
            if len(X) != X_coords.shape[0]:
                raise ValueError("Invalid number of variables")
        if not isinstance(y, backend_Function):
            raise TypeError("y must be a Function")
        if len(y.ufl_shape) > 0:
            raise ValueError("y must be a scalar-valued Function")

        if P is None:
            y_space = y.function_space()

            y_cells = point_owners(X_coords, y_space, tolerance=tolerance)
            y_colors = greedy_coloring(y_space)
            P = interpolation_matrix(X_coords, y, y_cells, y_colors)
        else:
            P = P.copy()

        super().__init__(X, list(X) + [y], nl_deps=[], ic=False, adj_ic=False)
        self._P = P
        self._P_T = P.T

    def forward_solve(self, X, deps=None):
        if is_var(X):
            X = (X,)
        y = (self.dependencies() if deps is None else deps)[-1]

        check_space_type(y, "primal")
        y_v = var_get_values(y)
        x_v_local = np.full(len(X), np.NAN, dtype=backend_ScalarType)
        for i in range(len(X)):
            x_v_local[i] = self._P.getrow(i).dot(y_v)

        comm = var_comm(y)
        x_v = np.full(len(X), np.NAN, dtype=x_v_local.dtype)
        comm.Allreduce(x_v_local, x_v, op=MPI.SUM)

        for i, x in enumerate(X):
            var_assign(x, x_v[i])

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_var(adj_X):
            adj_X = (adj_X,)
        if dep_index != len(self.X()):
            raise ValueError("Unexpected dep_index")

        adj_x_v = np.full(len(adj_X), np.NAN, dtype=backend_ScalarType)
        for i, adj_x in enumerate(adj_X):
            adj_x_v[i] = var_scalar_value(adj_x)

        F = var_new_conjugate_dual(self.dependencies()[-1])
        var_set_values(F, self._P_T.dot(adj_x_v))
        return (-1.0, F)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B

    def tangent_linear(self, M, dM, tlm_map):
        X = self.X()
        y = self.dependencies()[-1]

        tlm_y = tlm_map[y]
        if tlm_y is None:
            return ZeroAssignment([tlm_map[x] for x in X])
        else:
            return PointInterpolation([tlm_map[x] for x in X], tlm_y,
                                      P=self._P)
