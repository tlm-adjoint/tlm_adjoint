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
from .equations import EquationSolver, alias_assemble, alias_form

import numpy
import ufl

__all__ = \
  [
    "InterpolationSolver",
    "LocalProjectionSolver"
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
  node_node_graph = [sorted(list(nodes), reverse = True) for nodes in node_node_graph]
    
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
  
class LocalProjectionSolver(EquationSolver):
  def __init__(self, rhs, x, solver = None, form_compiler_parameters = {},
    cache_rhs_assembly = None, match_quadrature = None,
    defer_adjoint_assembly = None):
    space = x.function_space()
    test, trial = TestFunction(space), TrialFunction(space)
    
    lhs = ufl.inner(test, trial) * ufl.dx
    if not isinstance(rhs, ufl.classes.Form):
      rhs = ufl.inner(test, rhs) * ufl.dx
    if solver is None:
      solver = LocalSolver(lhs,
        solver_type = LocalSolver.SolverType.Cholesky if hasattr(LocalSolver, "SolverType") else LocalSolver.SolverType_Cholesky)
      solver.factorize()
      solver.solve = lambda x, b : solver.solve_local(x, b, space.dofmap())
    
    EquationSolver.__init__(self, lhs == rhs, x,
      form_compiler_parameters = form_compiler_parameters,
      solver_parameters = {"linear_solver":"direct"},
      cache_rhs_assembly = cache_rhs_assembly,
      match_quadrature = match_quadrature,
      defer_adjoint_assembly = defer_adjoint_assembly)
    self._local_solver = solver
  
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
    self._local_solver.solve(x.vector(), b)
    
  def adjoint_jacobian_solve(self, nl_deps, b):
    adj_x = function_new(b)
    self._local_solver.solve(adj_x.vector(), b.vector())
    return adj_x
  
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
        solver = self._local_solver,
        form_compiler_parameters = self._form_compiler_parameters,
        cache_rhs_assembly = self._cache_rhs_assembly,
        defer_adjoint_assembly = self._defer_adjoint_assembly)
  
class InterpolationSolver(Equation):
  def __init__(self, y, x, y_colors = None, x_coords = None,
    P = None, P_T = None):
    """
    Defines an equation which interpolates y.
    
    Internally this builds (or uses a supplied) interpolation matrix for the
    *local process only*. This works correctly in parallel if y is in a
    discontinuous function space (e.g. Discontinuous Lagrange) but may fail in
    parallel otherwise.
    
    For parallel cases this equation can be combined with LocalProjectionSolver
    to first project the input field onto an appropriate discontinuous space.
    
    Arguments:
    
    y         A Function. The function to be interpolated.
    x         A Function. The solution to the equation.
    y_colors  (Optional) An integer NumPy vector. Node-node graph coloring for
              the space for y. Defaults to a basic greedy graph coloring.
              Ignored if P is supplied.
    x_coords  (Optional) Coordinates of nodes in the space for x. Defaults to
              the coordinates as defined using the mesh for x. Ignored if P is
              supplied.
    P         (Optional) Interpolation matrix.
    P_T       (Optional) Interpolation matrix transpose.
    """
    
    # The process locality assumption can be avoided by additionally defining P
    # for non-owned nodes, but this requires a parallel graph coloring.
  
    if P is None:
      y_space = y.function_space()
      if y_colors is None:
        y_colors = greedy_coloring(y_space)
        
      y_mesh = y_space.mesh()
      y_dofmap = y_space.dofmap()
      
      # Verify process locality assumption
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
      P = scipy.sparse.dok_matrix((function_local_size(x), function_local_size(y)), dtype = numpy.float64)
      if x_coords is None and x.function_space().mesh() == y_mesh:
        if x_coords is None:
          x_coords = function_coords(x)
        x_dofmap = x.function_space().dofmap()
        y_cells = numpy.empty(function_local_size(x), dtype = numpy.int64)
        for y_cell in range(y_mesh.num_cells()):
          for y_node in x_dofmap.cell_dofs(y_cell):
            if y_node < y_cells.shape[0]:
              y_cells[y_node] = y_cell
      else:
        if x_coords is None:
          x_coords = function_coords(x)
        y_tree = y_mesh.bounding_box_tree()
        y_cells = [y_tree.compute_closest_entity(Point(*x_coord))[0] for x_coord in x_coords]
      
      y_v = function_new(y)
      x_v = numpy.empty((1,), dtype = numpy.float64)
      for color, y_color_nodes in enumerate(y_nodes):
        y_v.vector()[y_color_nodes] = 1.0
        for x_node, y_cell in enumerate(y_cells):
          y_cell_nodes = y_dofmap.cell_dofs(y_cell)
          try:
            i = y_colors[y_cell_nodes].tolist().index(color)
          except ValueError:
            continue
          y_node = y_cell_nodes[i]
          y_v.eval_cell(x_v, x_coords[x_node, :], Cell(y_mesh, y_cell))  # Broken in parallel with FEniCS 2017.2.0
          P[x_node, y_node] = x_v[0]
        if color < len(y_nodes) - 1: y_v.vector()[y_color_nodes] = 0.0
      del(y_v)
      P = P.tocsr()
    
    if P_T is None:
      P_T = P.T
    
    Equation.__init__(self, x, [x, y], nl_deps = [], ic_deps = [])
    self._P = P
    self._P_T = P_T
    
  def forward_solve(self, x, deps = None):
    _, y = self.dependencies() if deps is None else deps
    function_set_values(x, self._P.dot(function_get_values(y)))
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      return adj_x
    elif dep_index == 1:
      F = function_new(self.dependencies()[1])
      function_set_values(F, self._P_T.dot(function_get_values(adj_x)))
      return (-1.0, F)
    else:
      return None
    
  def adjoint_jacobian_solve(self, nl_deps, b):
    return b
    
  def tangent_linear(self, M, dM, tlm_map):
    x, y = self.dependencies()
    
    tlm_y = None
    for m, dm in zip(M, dM):
      if m == x:
        raise EquationException("Invalid tangent-linear parameter")
      elif m == y:
        tlm_y = dm
    if tlm_y is None:
      tlm_y = tlm_map[y]
      
    if tlm_y is None:
      return NullSolver(tlm_map[x])
    else:
      return InterpolationSolver(tlm_y, tlm_map[x], P = self._P, P_T = self._P_T)
