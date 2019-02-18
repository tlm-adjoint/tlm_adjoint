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

from .backend_interface import *

from .manager import manager as _manager

from collections import OrderedDict
import copy
import numpy

__all__ = \
  [
    "AssignmentSolver",
    "AxpySolver",
    "Equation",
    "EquationException",
    "FixedPointSolver",
    "LinearCombinationSolver",
    "NullSolver",
    "ScaleSolver",
    
    "LinearEquation",
    "Matrix",
    "MatrixActionRHS",
    "RHS"
  ]

class EquationException(Exception):
  pass

class Equation:
  def __init__(self, X, deps, nl_deps = None, ic_deps = None):
    """
    An equation. The equation is expressed in the form:
      F ( X, y_0, y_1, ... ) = 0,
    where X is the equation solution and F is a residual function. Information
    regarding F is provided by the methods which should be overridden as
    required by derived classes.
    
    Arguments:
    
    X        A Function, or a list or tuple of Function objects. The solution to
             the equation.
    deps     A list or tuple of Function dependencies, which must include x.
    nl_deps  (Optional) A list or tuple of Function non-linear dependencies.
             Must be a subset of deps. Defaults to deps.
    ic_deps  (Optional) A list or tuple of Function dependencies whose initial
             value should be available prior to solving the forward equation.
             Defaults to the elements of X which are in nl_deps.
    """

    if is_function(X):
      X = (X,)
    for x in X:
      if not is_function(x):
        raise EquationException("Solution must be a Function")
      if function_is_static(x):
        raise EquationException("Solution cannot be static")
      if not x in deps:
        raise EquationException("Solution must be a dependency")
    dep_ids = {dep.id():i for i, dep in enumerate(deps)}
    if len(dep_ids) != len(deps):
      raise EquationException("Duplicate dependency")
    if nl_deps is None:
      nl_deps_map = tuple(range(len(deps)))
    else:
      if len(set(map(lambda dep : dep.id(), nl_deps))) != len(nl_deps):
        raise EquationException("Duplicate non-linear dependency")
      nl_deps_map = tuple(dep_ids[dep.id()] for dep in nl_deps)
    if ic_deps is None:
      ic_deps = []
      for x in X:
        if x in (deps if nl_deps is None else nl_deps):
          ic_deps.append(x)
    elif len(set(map(lambda dep : dep.id(), ic_deps))) != len(ic_deps):
      raise EquationException("Duplicate initial condition dependency")

    self._X = X
    self._deps = tuple(deps)
    self._nl_deps = None if nl_deps is None else tuple(nl_deps)
    self._nl_deps_map = nl_deps_map
    self._ic_deps = tuple(ic_deps)
    
  def replace(self, manager = None):
    """
    Replace all internal Function objects.
    """
  
    if manager is None:
      manager = _manager()
    manager.replace(self)
  
  def _replace(self, replace_map):
    """
    Replace all internal Function objects using the supplied replace map. Must
    call the base class _replace method.
    """
    
    self._X = tuple(replace_map.get(x, x) for x in self._X)
    self._deps = tuple(replace_map.get(dep, dep) for dep in self._deps)
    if not self._nl_deps is None:
      self._nl_deps = tuple(replace_map.get(dep, dep) for dep in self._nl_deps)
    
  def x(self):
    """
    If the equation solves for exactly one Function, return it. Otherwise raise
    an error.
    """
    
    if len(self._X) != 1:
      raise EquationException("Equation does not solve for exactly one Function")
    return self._X[0]
  
  def X(self):
    """
    A tuple of Function objects. The solution to the equation.
    """
    
    return self._X
  
  def dependencies(self):
    return self._deps
  
  def nonlinear_dependencies(self):
    if self._nl_deps is None:
      return self.dependencies()
    else:
      return self._nl_deps
  
  def nonlinear_dependencies_map(self):
    return self._nl_deps_map
  
  def initial_condition_dependencies(self):
    return self._ic_deps
    
  def _pre_process(self, manager = None, annotate = None):
    if manager is None:
      manager = _manager()
    for dep in self.initial_condition_dependencies():
      manager.add_initial_condition(dep, annotate = annotate)
      
  def _post_process(self, manager = None, annotate = None, replace = False, tlm = None, tlm_skip = None):    
    if manager is None:
      manager = _manager()
    manager.add_equation(self, annotate = annotate, replace = replace, tlm = tlm, tlm_skip = tlm_skip)
    
  def solve(self, manager = None, annotate = None, replace = False, tlm = None, _tlm_skip = None):
    """
    Solve the equation.
    
    Arguments:
    
    manager   (Optional) The equation manager.
    annotate  (Optional) Whether the equation should be annotated.
    replace   (Optional) Whether, after the equation has been solved, its
              internal Function objects should be replaced with Coefficients.
              Can be used to save memory.
    tlm       (Optional) Whether to derive (and solve) an associated
              tangent-linear equation.
    """
    
    if manager is None:
      manager = _manager()
  
    self._pre_process(manager = manager, annotate = annotate)

    annotation_enabled, tlm_enabled = manager.stop()
    X = self.X()
    self.forward_solve(X[0] if len(X) == 1 else X)
    manager.start(annotation = annotation_enabled, tlm = tlm_enabled)

    self._post_process(manager = manager, annotate = annotate, replace = replace, tlm = tlm, tlm_skip = _tlm_skip)

  def forward_solve(self, X, deps = None):
    """
    Solve the equation. The manager is stopped when this method is called.
    
    The form:
      forward_solve(self, x, deps = None)
    should be used for equations which solve for a single Function object.
    
    Arguments:
    
    x/X   The solution, which should be set by this method.
    deps  (Optional) A list or tuple of Function objects defining the values of
          dependencies. self.dependencies() should be used if this is not
          supplied.
    """
  
    raise EquationException("Method not overridden")
    
  def reset_forward_solve(self):
    """
    Reset the forward solver. Can be used to clear caches used by forward_solve.
    """
    
    pass
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
    """
    Return the action of the adjoint of a derivative of the RHS.
    
    Boundary conditions need not be applied in the returned result. The return
    value should never be modified by calling code.
    
    The form:
      adjoint_derivative_action(self, nl_deps, dep_index, adj_x)
    should be used for equations which solve for a single Function object.
    
    Arguments:
    
    nl_deps      A list or tuple of Function objects defining the values of
                 non-linear dependencies.
    dep_index    The index of the dependency in self.dependencies() with respect
                 to which a derivative should be taken.
    adj_x/adj_X  The direction of the adjoint derivative action.
    """
  
    raise EquationException("Method not overridden")
    
  def reset_adjoint_derivative_action(self):
    """
    Can be used to clear caches used by adjoint_derivative_action.
    """
    
    pass

  def adjoint_jacobian_solve(self, nl_deps, B):
    """
    Solve an adjoint equation, returning the result. The result must have
    relevant boundary conditions applied, and should never be modified by
    calling code.
    
    The form:
      adjoint_jacobian_solve(self, nl_deps, b)
    should be used for equations which solve for a single Function object.
    
    Arguments:
    
    nl_deps    A list or tuple of Function objects defining the values of
               non-linear dependencies.
    b/B        The right-hand-side. May be modified by this method. May not
               have previously have had boundary conditions applied.
    """
  
    raise EquationException("Method not overridden")
    
  def reset_adjoint_jacobian_solve(self):
    """
    Can be used to clear caches used by adjoint_jacobian_solve.
    """
    
    pass
  
  def tangent_linear(self, M, dM, tlm_map):
    """
    Return an Equation corresponding to a tangent linear equation. It is
    guaranteed that self.replace has not been called when this method is called.
    
    Arguments:
    
    M        A list or tuple of Function objects defining the tangent-linear
             equation.
    dM       A list or tuple of Function objects. The tangent linear is defined
             via the parameter c, where M = M_0 + c dM, c is equal to zero, and
             dM defines a direction.
    tlm_map  The TangentLinearMap.
    """
  
    raise EquationException("Method not overridden")

class NullSolver(Equation):
  def __init__(self, X):
    if is_function(X):
      X = (X,)
    Equation.__init__(self, X, X, nl_deps = [], ic_deps = [])
    
  def forward_solve(self, X, deps = None):
    if is_function(X):
      X = (X,)
    for x in X:
      function_zero(x)
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
    if is_function(adj_X):
      adj_X = (adj_X,)
    if dep_index < len(adj_X):
      return adj_X[dep_index]
    else:
      return None
      
  def adjoint_jacobian_solve(self, nl_deps, B):
    return B
    
  def tangent_linear(self, M, dM, tlm_map):
    return NullSolver([tlm_map[x] for x in self.X()])

class AssignmentSolver(Equation):
  def __init__(self, y, x):
    if x == y:
      raise EquationException("Invalid dependency in linear equation")
    Equation.__init__(self, x, [x, y], nl_deps = [], ic_deps = [])
    
  def forward_solve(self, x, deps = None):
    _, y = self.dependencies() if deps is None else deps
    function_assign(x, y)
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      return adj_x
    elif dep_index == 1:
      return (-1.0, adj_x)
    else:
      return None
      
  def adjoint_jacobian_solve(self, nl_deps, b):
    return b
    
  def tangent_linear(self, M, dM, tlm_map):
    x, y = self.dependencies()
    
    tau_y = None
    for i, m in enumerate(M):
      if m == x:
        raise EquationException("Invalid tangent-linear parameter")
      elif m == y:
        tau_y = dM[i]
    if tau_y is None:
      tau_y = tlm_map[y]
    
    if tau_y is None:
      return NullSolver(tlm_map[x])
    else:
      return AssignmentSolver(tau_y, tlm_map[x])

class LinearCombinationSolver(Equation):
  def __init__(self, x, *args):
    alpha = [float(arg[0]) for arg in args]
    y = [arg[1] for arg in args]
    
    if x in y:
      raise EquationException("Invalid dependency in linear equation")
    
    Equation.__init__(self, x, [x] + y, nl_deps = [], ic_deps = [])
    self._alpha = alpha
  
  def forward_solve(self, x, deps = None):
    deps = self.dependencies() if deps is None else list(deps)
    function_zero(x)
    for alpha, y in zip(self._alpha, deps[1:]):
      function_axpy(x, alpha, y)
  
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      return adj_x
    elif dep_index <= len(self._alpha):
      return (-self._alpha[dep_index - 1], adj_x)
    else:
      return None
      
  def adjoint_jacobian_solve(self, nl_deps, b):
    return b
  
  def tangent_linear(self, M, dM, tlm_map):
    deps = self.dependencies()
    x, ys = deps[0], deps[1:]
    tau_ys = [None for y in ys]
    for m, dm in zip(M, dM):
      if m == x:
        raise EquationException("Invalid tangent-linear parameter")
      else:
        try:
          i = ys.index(m)
        except ValueError:
          continue
        tau_ys[i] = dm
    for i, (y, tau_y) in enumerate(zip(ys, tau_ys)):
      if tau_y is None:
        tau_ys[i] = tlm_map[y]

    args = []
    for alpha, tau_y in zip(self._alpha, tau_ys):
      if not tau_y is None:
        args.append((alpha, tau_y))
    if len(args) == 0:
      return NullSolver(tlm_map[x])
    else:
      return LinearCombinationSolver(tlm_map[x], *args)

#class AssignmentSolver(LinearCombinationSolver):
#  def __init__(self, y, x):
#    LinearCombinationSolver.__init__(self, x, (1.0, y))

class ScaleSolver(LinearCombinationSolver):
  def __init__(self, alpha, y, x_new):
    LinearCombinationSolver.__init__(self, x_new, (alpha, y))

class AxpySolver(LinearCombinationSolver):
  def __init__(self, x_old, alpha, y, x_new):
    LinearCombinationSolver.__init__(self, x_new, (1.0, x_old), (alpha, y))
    
# Derives tangent-linear and adjoint information using the approach described in
#  J. G. Gilbert, "Automatic differentiation and iterative processes",
#    Optimization Methods and Software, 1(1), pp. 13--21, 1992
#  B. Christianson, "Reverse accumulation and attractive fixed points",
#    Optimization Methods and Software, 3(4), pp. 311--326, 1994
class FixedPointSolver(Equation):
  def __init__(self, eqs, solver_parameters, initial_guess = None):
    """
    A fixed point solver.
    
    Arguments:
    
    eqs
      A list or tuple of Equation objects. The last equation defines the
      solution of the fixed point iteration. All equations must solve for single
      and distinct Function objects.
    solver_parameters
      Solver parameters dictionary. Parameters (based on KrylovSolver parameters
      in FEniCS 2017.2.0):
        absolute_tolerance     Absolute tolerance for the solution change
                               2-norm. Float, required.
        relative_tolerance     Relative tolerance for the solution change
                               2-norm. Float, required.
        maximum_iterations     Maximum permitted iterations. Positive integer,
                               optional, default 1000.
        nonzero_initial_guess  Whether to use a non-zero initial guess for the
                               forward solve. Logical, optional, default True.
        report                 Whether to display output during fixed point
                               iteration. Optional, default False.
    initial_guess
      (Optional) Initial guess for the forward equation solution (the solution
      to the final equation in eqs).
    """
  
    x_ids = set()
    for eq in eqs:
      eq_x_id = eq.x().id()  # Raises an error here if eq solves for more than one Function
      if eq_x_id in x_ids:
        raise EquationException("Duplicate solve")
      x_ids.add(eq_x_id)
    x = eqs[-1].x()
    
    solver_parameters = copy_parameters_dict(solver_parameters)
    # Based on KrylovSolver parameters in FEniCS 2017.2.0
    for key, default_value in [("maximum_iterations", 1000),
                               ("nonzero_initial_guess", True),
                               ("report", False)]:
      if not key in solver_parameters:
        solver_parameters[key] = default_value
  
    deps = []
    dep_ids = {}
    nl_deps = []
    nl_dep_ids = {}
    ic_deps = []
    ic_dep_ids = set()
    
    if solver_parameters["nonzero_initial_guess"]:
      if initial_guess is None:
        initial_guess_index = None
        initial_guess = x
      elif initial_guess == x:
        initial_guess_index = None
      else:
        initial_guess_index = 0
        deps.append(initial_guess)
        dep_ids[initial_guess.id()] = 0
    elif not initial_guess is None:
      raise EquationException("Initial guess provided, but nonzero_initial_guess parameter is False")
    
    eq_dep_indices = [[] for eq in eqs]
    eq_nl_dep_indices = [[] for eq in eqs]
    
    for i, eq in enumerate(eqs):
      x_ids.remove(eq.x().id())
      for dep in eq.dependencies():
        dep_id = dep.id()
        if not dep_id in dep_ids:
          deps.append(dep)
          dep_ids[dep_id] = len(deps) - 1
        eq_dep_indices[i].append(dep_ids[dep_id])
        if dep_id in x_ids and not dep_id in ic_dep_ids:
          ic_deps.append(dep)
          ic_dep_ids.add(dep_id)
      for dep in eq.nonlinear_dependencies():
        dep_id = dep.id()
        if not dep_id in nl_dep_ids:
          nl_deps.append(dep)
          nl_dep_ids[dep_id] = len(nl_deps) - 1
        eq_nl_dep_indices[i].append(nl_dep_ids[dep_id])
      if i < len(eqs) - 1 or (solver_parameters["nonzero_initial_guess"] and initial_guess is None):
        for dep in eq.initial_condition_dependencies():
          dep_id = dep.id()
          if not dep_id in ic_dep_ids:
            ic_deps.append(dep)
            ic_dep_ids.add(dep_id)
    
    del(x_ids, dep_ids, nl_dep_ids, ic_dep_ids)
    
    Equation.__init__(self, [eq.x() for eq in eqs], deps, nl_deps = nl_deps, ic_deps = ic_deps)
    self._eqs = copy.copy(eqs)
    self._initial_guess_index = None if initial_guess is None else initial_guess_index
    self._eq_dep_indices = eq_dep_indices
    self._eq_nl_dep_indices = eq_nl_dep_indices
    self._solver_parameters = solver_parameters
    
    self._tdeps = None
    
  def _replace(self, replace_map):
    Equation._replace(self, replace_map)
    for eq in self._eqs:
      eq._replace(replace_map)
    
  def forward_solve(self, X, deps = None):
    # Based on KrylovSolver parameters in FEniCS 2017.2.0
    absolute_tolerance = self._solver_parameters["absolute_tolerance"]
    relative_tolerance = self._solver_parameters["relative_tolerance"]
    maximum_iterations = self._solver_parameters["maximum_iterations"]
    nonzero_initial_guess = self._solver_parameters["nonzero_initial_guess"]
    report = self._solver_parameters["report"]
    
    x = X[-1]
    if nonzero_initial_guess:
      if not self._initial_guess_index is None:
        function_assign(x, (self.dependencies() if deps is None else deps)[self._initial_guess_index])
    else:
      function_zero(x)
      
    if deps is None:
      eq_deps = [None for i in range(len(self._eqs))]
    else:
      eq_deps = [[deps[j] for j in self._eq_dep_indices[i]] for i in range(len(self._eqs))]
    
    it = 0
    x_0 = function_new(x)
    tolerance_sq = absolute_tolerance ** 2
    while True:
      it += 1
    
      for i, eq in enumerate(self._eqs):
        eq.forward_solve(X[i], deps = eq_deps[i])
      
      r = x_0;  del(x_0)
      function_axpy(r, -1.0, x)
      r_norm_sq = function_inner(r, r)
      if report:
        info("Fixed point iteration (forward equation for %s): iteration %i, change norm %.16e (tolerance %.16e)" % (self.X()[-1].name(), it, numpy.sqrt(r_norm_sq), numpy.sqrt(tolerance_sq)))
      if numpy.isnan(r_norm_sq):
        raise EquationException("Fixed point iteration (forward equation for %s): NaN encountered after %i iteration(s)" % (self.X()[-1].name(), it))
      if r_norm_sq < tolerance_sq or r_norm_sq == 0.0:
        break
      if it >= maximum_iterations:
        raise EquationException("Fixed point iteration (forward equation for %s): did not converge after %i iteration(s)" % (self.X()[-1].name(), it))
      if it == 1:
        tolerance_sq = max(tolerance_sq, r_norm_sq * (relative_tolerance ** 2))
      
      x_0 = r;  del(r)
      function_assign(x_0, x)
      
  def reset_forward_solve(self):
    for eq in self._eqs:
      eq.reset_forward_solve()
  
  def adjoint_jacobian_solve(self, nl_deps, B):
    # Based on KrylovSolver parameters in FEniCS 2017.2.0
    absolute_tolerance = self._solver_parameters["absolute_tolerance"]
    relative_tolerance = self._solver_parameters["relative_tolerance"]
    maximum_iterations = self._solver_parameters["maximum_iterations"]
    report = self._solver_parameters["report"]
    
    self._init_adjoint_jacobian_solve()
    adj_X = [function_copy(b) for b in B]
    x = adj_X[-1]
    
    eq_nl_deps = [[nl_deps[j] for j in self._eq_nl_dep_indices[i]] for i in range(len(self._eqs))]
              
    it = 0
    x_0 = function_new(x)
    tolerance_sq = absolute_tolerance ** 2
    while True:
      it += 1
    
      for i in range(len(self._eqs) - 1, - 1, -1):
        i = (i - 1) % len(self._eqs)
        function_assign(adj_X[i], B[i])
          
        for j, k in self._tdeps[i]:
          sb = self._eqs[k].adjoint_derivative_action(eq_nl_deps[k], j, adj_X[k])
          subtract_adjoint_derivative_action(adj_X[i], sb)
          del(sb)
        finalise_adjoint_derivative_action(adj_X[i])
        adj_X[i] = self._eqs[i].adjoint_jacobian_solve(eq_nl_deps[i], adj_X[i])
      x = adj_X[-1]

      r = x_0;  del(x_0)
      function_axpy(r, -1.0, x)
      r_norm_sq = function_inner(r, r)
      if report:
        info("Fixed point iteration (adjoint equation for %s): iteration %i, change norm %.16e (tolerance %.16e)" % (self.X()[-1].name(), it, numpy.sqrt(r_norm_sq), numpy.sqrt(tolerance_sq)))
      if numpy.isnan(r_norm_sq):
        raise EquationException("Fixed point iteration (adjoint equation for %s): NaN encountered after %i iteration(s)" % (self.X()[-1].name(), it))
      if r_norm_sq < tolerance_sq or r_norm_sq == 0.0:
        break
      if it >= maximum_iterations:
        raise EquationException("Fixed point iteration (adjoint equation for %s): did not converge after %i iteration(s)" % (self.X()[-1].name(), it))
      if it == 1:
        tolerance_sq = max(tolerance_sq, r_norm_sq * (relative_tolerance ** 2))
  
      x_0 = r;  del(r)
      function_assign(x_0, x)
      
    return adj_X
          
  def _init_adjoint_jacobian_solve(self):  
    if self._tdeps is None:
      eq_x_ids = {eq.x().id():i for i, eq in enumerate(self._eqs)}
      self._tdeps = [[] for eq in self._eqs]
      for i, eq in enumerate(self._eqs):
        for j, dep in enumerate(eq.dependencies()):
          dep_id = dep.id()
          if dep_id in eq_x_ids:
            k = eq_x_ids[dep_id]
            if k != i:
              self._tdeps[k].append((j, i))
  
  def reset_adjoint_jacobian_solve(self):
    self._tdeps = None
    for eq in self._eqs:
      eq.reset_adjoint_jacobian_solve()
    
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
    dep = self.dependencies()[dep_index]
    F = function_new(dep)
    for i in range(len(self._eqs)):
      eq_deps = self._eqs[i].dependencies()
      if dep in eq_deps:
        sb = self._eqs[i].adjoint_derivative_action([nl_deps[j] for j in self._eq_nl_dep_indices[i]], eq_deps.index(dep), adj_X[i])
        subtract_adjoint_derivative_action(F, sb)
        del(sb)
    finalise_adjoint_derivative_action(F)
    
    return (-1.0, F)
    
  def reset_adjoint_derivative_action(self):
    for eq in self._eqs:
      eq.reset_adjoint_derivative_action()
    
  def tangent_linear(self, M, dM, tlm_map):
    return FixedPointSolver([NullSolver(tlm_map[eq.x()]) if tlm_eq is None else tlm_eq
        for eq, tlm_eq in zip(self._eqs, [eq.tangent_linear(M, dM, tlm_map) for eq in self._eqs])],
      solver_parameters = self._solver_parameters,
      initial_guess = None if self._initial_guess_index is None else tlm_map[self.dependencies()[self._initial_guess_index]])

class LinearEquation(Equation):
  def __init__(self, B, X, A = None):
    if isinstance(B, RHS):
      B = (B,)
    if is_function(X):
      X = (X,)
    if len(X) != len(B):
      raise EquationException("Invalid Equation")
  
    deps = []
    dep_ids = {}
    nl_deps = []
    nl_dep_ids = {}
    
    x_ids = set()
    for x in X:
      x_id = x.id()
      x_ids.add(x_id)
      if x_id in dep_ids:
        raise EquationException("Duplicate solve")
      deps.append(x)
      dep_ids[x_id] = len(deps) - 1
    
    b_dep_indices = [[] for b in B]
    b_nl_dep_indices = [[] for b in B]
    
    for i, b in enumerate(B):
      for dep in b.dependencies():
        dep_id = dep.id()
        if dep_id in x_ids:
          raise EquationException("Invalid dependency in linear Equation")
        if not dep_id in dep_ids:
          deps.append(dep)
          dep_ids[dep_id] = len(deps) - 1
        b_dep_indices[i].append(dep_ids[dep_id])
      for dep in b.nonlinear_dependencies():
        dep_id = dep.id()
        if dep_id in x_ids:
          raise EquationException("Invalid dependency in linear Equation")
        if not dep_id in nl_dep_ids:
          nl_deps.append(dep)
          nl_dep_ids[dep_id] = len(nl_deps) - 1
        b_nl_dep_indices[i].append(nl_dep_ids[dep_id])
    
    if A is None:
      ic_deps = []
    else:
      A_dep_indices = []
      A_nl_dep_indices = []
      for dep in A.nonlinear_dependencies():
        dep_id = dep.id()
        if not dep_id in dep_ids:
          deps.append(dep)
          dep_ids[dep_id] = len(deps) - 1
        A_dep_indices.append(dep_ids[dep_id])
        if not dep_id in nl_dep_ids:
          nl_deps.append(dep)
          nl_dep_ids[dep_id] = len(nl_deps) - 1
        A_nl_dep_indices.append(nl_dep_ids[dep_id])
      if len(A.nonlinear_dependencies()) > 0:
        A_x_indices = []
        for x in X:
          x_id = x.id()
          if not x_id in nl_dep_ids:
            nl_deps.append(x)
            nl_dep_ids[x_id] = len(nl_deps) - 1
          A_x_indices.append(nl_dep_ids[x_id])
      ic_deps = X if A.has_initial_condition_dependency() else []
    
    del(x_ids, dep_ids, nl_dep_ids)
    
    Equation.__init__(self, X, deps, nl_deps = nl_deps, ic_deps = ic_deps)
    self._B = tuple(B)
    self._b_dep_indices = b_dep_indices
    self._b_nl_dep_indices = b_nl_dep_indices
    self._A = A
    if not A is None:
      self._A_dep_indices = A_dep_indices
      self._A_nl_dep_indices = A_nl_dep_indices
      if len(A.nonlinear_dependencies()) > 0:
        self._A_x_indices = A_x_indices
    
  def _replace(self, replace_map):
    Equation._replace(self, replace_map)
    for b in self._B:
      b.replace(replace_map)
    if not self._A is None:
      self._A.replace(replace_map)
    
  def forward_solve(self, X, deps = None):
    if is_function(X):
      X = (X,)      
    if deps is None:
      deps = self.dependencies()
      
    for x in X:
      function_zero(x)    
    for i, b in enumerate(self._B):
      b.add_forward(X[0] if len(X) == 1 else X, [deps[j] for j in self._b_dep_indices[i]])
    if not self._A is None:
      if len(X) == 1:
        X_new = (self._A.forward_solve(X[0], [deps[j] for j in self._A_dep_indices]),)
      else:
        X_new = self._A.forward_solve(X, [deps[j] for j in self._A_dep_indices])
      for x, x_new in zip(X, X_new):
        function_assign(x, x_new)
  
  def reset_forward_solve(self):
    for b in self._B:
      b.reset_add_forward()
    if not self._A is None:
      self._A.reset_forward_solve()
      
  def adjoint_jacobian_solve(self, nl_deps, B):
    if self._A is None:
      return B
    else:
      return self._A.adjoint_solve(B, [nl_deps[j] for j in self._A_nl_dep_indices])
      
  def reset_adjoint_jacobian_solve(self):
    if not self._A is None:
      self._A.reset_adjoint_solve()
  
  def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
    if is_function(adj_X):
      adj_X = (adj_X,)      
    if dep_index < len(self.X()):
      if self._A is None:
        return adj_X[dep_index]
      else:
        return self._A.adjoint_action([nl_deps[j] for j in self._A_nl_dep_indices], adj_X[0] if len(adj_X) == 1 else adj_X, x_index = dep_index)
    else:
      dep = self.dependencies()[dep_index]
      F = function_new(dep)
      for i, b in enumerate(self._B):
        try:
          b_dep_index = b.dependencies().index(dep)
        except ValueError:
          b_dep_index = None
        if not b_dep_index is None:
          b.subtract_adjoint_derivative_action(F,
            [nl_deps[j] for j in self._b_nl_dep_indices[i]],
            b_dep_index,
            adj_X[0] if len(adj_X) == 1 else adj_X)
      if not self._A is None:
        try:
          A_nl_dep_index = self._A.nonlinear_dependencies().index(dep)
        except ValueError:
          A_nl_dep_index = None
        if not A_nl_dep_index is None:
          X = [nl_deps[j] for j in self._A_x_indices]
          self._A.add_adjoint_derivative_action(F,
            [nl_deps[j] for j in self._A_nl_dep_indices],
            A_nl_dep_index,
            X[0] if len(X) == 1 else X,
            adj_X[0] if len(adj_X) == 1 else adj_X)
      return F
  
  def reset_adjoint_derivative_action(self):
    for b in self._B:
      b.reset_subtract_adjoint_derivative_action()
    if not self._A is None:
      self._A.reset_adjoint_action()
      self._A.reset_add_adjoint_derivative_action()

  def tangent_linear(self, M, dM, tlm_map):
    X = self.X()
    for x in X:
      if x in M:
        raise EquationException("Invalid tangent-linear parameter")
    
    if self._A is None:
      tlm_B = []
    else:
      tlm_B = self._A.tangent_linear_rhs(M, dM, tlm_map, X[0] if len(X) == 1 else X)
      if tlm_B is None:
        tlm_B = []
      elif isinstance(tlm_B, RHS):
        tlm_B = [tlm_B]
    for b in self._B:
      tlm_b = b.tangent_linear_rhs(M, dM, tlm_map)
      if tlm_b is None:
        pass
      elif isinstance(tlm_b, RHS):
        tlm_B.append(tlm_b)
      else:
        tlm_B += list(tlm_b)
          
    if len(tlm_B) == 0:
      return NullSolver([tlm_map[x] for x in self.X()])
    else:
      return LinearEquation(tlm_B, [tlm_map[x] for x in self.X()], A = self._A)

class Matrix:
  def __init__(self, nl_deps = None, has_ic_dep = False):
    if not nl_deps is None:
      if len(set(dep.id() for dep in nl_deps)) != len(nl_deps):
        raise EquationException("Duplicate non-linear dependency")
      
    self._nl_deps = None if nl_deps is None else tuple(nl_deps)
    self._has_ic_dep = has_ic_dep
  
  def replace(self, replace_map):
    if not self._nl_deps is None:
      self._nl_deps = tuple(replace_map.get(dep, dep) for dep in self._nl_deps)
  
  def nonlinear_dependencies(self):
    return tuple() if self._nl_deps is None else self._nl_deps
  
  def has_initial_condition_dependency(self):
    return self._has_ic_dep
  
  def add_forward_action(self, B, nl_deps, X):
    raise EquationException("Method not overridden")
  
  def reset_add_forward_action(self):
    pass
  
  def add_adjoint_action(self, b, nl_deps, X, x_index = 0):
    raise EquationException("Method not overridden")
  
  def reset_add_adjoint_action(self):
    pass
  
  def forward_solve(self, B, nl_deps):
    raise EquationException("Method not overridden")
  
  def reset_forward_solve(self):
    pass
  
  def adjoint_action(self, nl_deps, adj_X, x_index = 0):
    raise EquationException("Method not overridden")
  
  def reset_adjoint_action(self):
    pass
  
  def add_adjoint_derivative_action(self, b, nl_deps, nl_dep_index, X, adj_X):
    raise EquationException("Method not overridden")
  
  def reset_add_adjoint_derivative_action(self):
    pass
  
  def adjoint_solve(self, B, nl_deps):
    raise EquationException("Method not overridden")
  
  def reset_adjoint_solve(self):
    pass
  
  def tangent_linear_rhs(self, M, dM, tlm_map, X):
    raise EquationException("Method not overridden")
  
class RHS:
  def __init__(self, deps, nl_deps = None):
    if len(set(dep.id() for dep in deps)) != len(deps):
      raise EquationException("Duplicate dependency")
    if not nl_deps is None:
      if len(set(dep.id() for dep in nl_deps)) != len(nl_deps):
        raise EquationException("Duplicate non-linear dependency")
    
    self._deps = tuple(deps)
    self._nl_deps = None if nl_deps is None else tuple(nl_deps)
    
  def replace(self, replace_map):
    self._deps = tuple(replace_map.get(dep, dep) for dep in self._deps)
    if not self._nl_deps is None:
      self._nl_deps = tuple(replace_map.get(dep, dep) for dep in self._nl_deps)
  
  def dependencies(self):
    return self._deps
  
  def nonlinear_dependencies(self):
    if self._nl_deps is None:
      return self.dependencies()
    else:
      return self._nl_deps

  def add_forward(self, B, deps):
    raise EquationException("Method not overridden")
  
  def reset_add_forward(self):
    pass

  def subtract_adjoint_derivative_action(self, b, nl_deps, dep_index, adj_X):
    raise EquationException("Method not overridden")
  
  def reset_subtract_adjoint_derivative_action(self):
    pass

  def tangent_linear_rhs(self, M, dM, tlm_map):
    raise EquationException("Method not overridden")

class MatrixActionRHS(RHS):
  def __init__(self, A, X):
    if is_function(X):
      X = (X,)
    if len(set(x.id() for x in X)) != len(X):
      raise EquationException("Invalid dependency")
    
    A_nl_deps = A.nonlinear_dependencies()
    if len(A_nl_deps) == 0:
      x_indices = list(range(len(X)))
      RHS.__init__(self, X, nl_deps = [])
    else:
      nl_deps = list(A_nl_deps)
      nl_dep_ids = {dep.id():i for i, dep in enumerate(nl_deps)}
      x_indices = []
      for x in X:
        x_id = x.id()
        if not x_id in nl_dep_ids:
          nl_deps.append(x)
          nl_dep_ids[x_id] = len(nl_deps) - 1
        x_indices.append(nl_dep_ids[x_id])
      RHS.__init__(self, nl_deps, nl_deps = nl_deps)
      
    self._A = A
    self._x_indices = x_indices
    
  def replace(self, replace_map):
    RHS.replace(self, replace_map)
    self._A.replace(replace_map)
  
  def add_forward(self, B, deps):
    if is_function(B):
      B = (B,)
    X = [deps[j] for j in self._x_indices]
    self._A.add_forward_action(B[0] if len(B) == 1 else B,
      deps[:len(self._A.nonlinear_dependencies())],
      X[0] if len(X) == 1 else X)

  def reset_add_forward(self):
    self._A.reset_add_forward_action()

  def subtract_adjoint_derivative_action(self, b, nl_deps, dep_index, adj_X):
    if is_function(adj_X):
      adj_X = (adj_X,)
    N_A_nl_deps = len(self._A.nonlinear_dependencies())
    if dep_index < N_A_nl_deps:
      X = [nl_deps[j] for j in self._x_indices]
      sb = function_new(b)
      self._A.add_adjoint_derivative_action(sb,
        nl_deps[:N_A_nl_deps], dep_index,
        X[0] if len(X) == 1 else X,
        adj_X[0] if len(adj_X) == 1 else adj_X)
      function_axpy(b, -1.0, sb)
      del(sb)
    elif dep_index < len(self.dependencies()):
      sb = function_new(b)
      self._A.add_adjoint_action(sb,
        nl_deps[:N_A_nl_deps],
        adj_X[0] if len(adj_X) == 1 else adj_X,
        x_index = self._x_indices.index(dep_index))
      function_axpy(b, -1.0, sb)
      del(sb)
  
  def reset_subtract_adjoint_derivative_action(self):
    self._A.reset_add_adjoint_derivative_action()
    self._A.reset_add_adjoint_action()
    
  def tangent_linear_rhs(self, M, dM, tlm_map):
    deps = self.dependencies()
    N_A_nl_deps = len(self._A.nonlinear_dependencies())
    
    X = [deps[j] for j in self._x_indices]
    tlm_X = []
    for i, x in enumerate(X):
      if x in M:
        tlm_X.append(dM[M.index(x)])
      else:
        tlm_X.append(tlm_map[x])
    tlm_B = [MatrixActionRHS(self._A, tlm_X)]
    
    if N_A_nl_deps > 0:
      tlm_b = self._A.tangent_linear_rhs(M, dM, tlm_map, X)
      if tlm_b is None:
        pass
      elif isinstance(tlm_b, RHS):
        tlm_B.append(tlm_b)
      else:
        tlm_B += list(tlm_b)
    
    return tlm_B
