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

from fenics import *
from tlm_adjoint import *
from tlm_adjoint import manager as _manager
from tlm_adjoint.backend import backend_Function

import gc
import numpy
import unittest
import weakref

Function_ids = {}
_orig_Function_init = backend_Function.__init__
def _Function__init__(self, *args, **kwargs):
  _orig_Function_init(self, *args, **kwargs)
  Function_ids[self.id()] = weakref.ref(self)
backend_Function.__init__ = _Function__init__

def leak_check(test):
  def wrapped_test(self, *args, **kwargs):    
    Function_ids.clear()
    
    test(self, *args, **kwargs)
    
    # Clear some internal storage that is allowed to keep references
    manager = _manager()
    manager._cp.clear(clear_refs = True)
    tlm_values = manager._tlm.values()
    manager._tlm.clear()
    tlm_eqs_values = manager._tlm_eqs.values()
    manager._tlm_eqs.clear()
    
    gc.collect()
  
    refs = 0
    for F in Function_ids.values():
      F = F()
      if not F is None:
        info("%s referenced" % F.name())
        refs += 1
    if refs == 0:
      info("No references")

    Function_ids.clear()
    self.assertEqual(refs, 0)
  return wrapped_test    
  
class tests(unittest.TestCase):
  @leak_check
  def test_clear_caches(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test = TestFunction(space)
    F = Function(space, name = "F", cache = True)
    self.assertEqual(len(assembly_cache()), 0)
    
    cached_form, _ = assembly_cache().assemble(inner(test, F) * dx)
    self.assertEqual(len(assembly_cache()), 1)
    self.assertIsNotNone(cached_form())
    
    clear_caches(F)
    self.assertEqual(len(assembly_cache()), 0)
    self.assertIsNone(cached_form())

  @leak_check
  def test_LocalProjectionSolver(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitSquareMesh(10, 10)
    space_1 = FunctionSpace(mesh, "Discontinuous Lagrange", 1)
    space_2 = FunctionSpace(mesh, "Lagrange", 2)
    test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)
    
    def forward(G):
      F = Function(space_1, name = "F")
      LocalProjectionSolver(G, F).solve()
      J = Functional(name = "J")
      J.assign(F * F * F * dx)
      return F, J
    
    G = Function(space_2, name = "G", static = True)
    G.interpolate(Expression("sin(pi * x[0]) * sin(2.0 * pi * x[1])", element = space_2.ufl_element()))
    
    start_manager()
    F, J = forward(G)
    stop_manager()
    
    F_ref = Function(space_1, name = "F_ref")
    solve(inner(test_1, trial_1) * dx == inner(test_1, G) * dx, F_ref,
      solver_parameters = {"linear_solver":"cg",
                           "preconditioner":"sor",
                           "krylov_solver":{"relative_tolerance":1.0e-14,
                                            "absolute_tolerance":1.0e-16}})
    F_error = Function(space_1, name = "F_error")
    function_assign(F_error, F_ref)
    function_axpy(F_error, -1.0, F)

    F_error_norm = function_linf_norm(F_error)
    info("Error norm = %.16e" % F_error_norm)
    self.assertLess(F_error_norm, 1.0e-15)
    
    dJ = compute_gradient(J, G)
    min_order = taylor_test(lambda G : forward(G)[1], G, J_val = J.value(), dJ = dJ, seed = 1.0e-4)
    self.assertGreater(min_order, 1.99)
    
    ddJ = Hessian(lambda G : forward(G)[1])
    min_order = taylor_test(lambda G : forward(G)[1], G, J_val = J.value(), dJ = dJ, ddJ = ddJ)
    self.assertGreater(min_order, 2.99)
  
  @leak_check
  def test_LongRange(self):
    n_steps = 200
    reset("multistage", {"blocks":n_steps, "snaps_on_disk":0, "snaps_in_ram":2, "verbose":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    
    def forward(F, x_ref = None):
      G = Function(space, name = "G")
      AssignmentSolver(F, G).solve()
      
      x_old = Function(space, name = "x_old")
      x = Function(space, name = "x")
      AssignmentSolver(G, x_old).solve()
      J = Functional(name = "J")
      gather_ref = x_ref is None
      if gather_ref:
        x_ref = {}
      for n in range(n_steps):
        terms = [(1.0, x_old)]
        if n % 11 == 0:
          terms.append((1.0, G))
        LinearCombinationSolver(x, *terms).solve()
        if n % 17 == 0:
          if gather_ref:
            x_ref[n] = function_copy(x, name = "x_ref_%i" % n)
          J.addto(inner(x * x * x, x_ref[n]) * dx)
        AssignmentSolver(x, x_old).solve()
        if n < n_steps - 1:
          new_block()
          
      return x_ref, J
    
    F = Function(space, name = "F", static = True)
    F.interpolate(Expression("sin(pi * x[0])", element = space.ufl_element()))
    zeta = Function(space, name = "zeta", static = True)
    zeta.interpolate(Expression("exp(x[0])", element = space.ufl_element()))
    add_tlm(F, zeta)
    start_manager()
    x_ref, J = forward(F)
    stop_manager()
    
    dJ = compute_gradient(J, F)
    min_order = taylor_test(lambda F : forward(F, x_ref = x_ref)[1], F, J_val = J.value(), dJ = dJ)
    self.assertGreater(min_order, 2.00)
    
    dJ_tlm = J.tlm(F, zeta).value()
    dJ_adj = function_inner(dJ, zeta)
    error = abs(dJ_tlm - dJ_adj)
    info("dJ/dF zeta, TLM     = %.16e" % dJ_tlm)
    info("dJ/dF zeta, adjoint = %.16e" % dJ_adj)
    info("Error               = %.16e" % error)
    self.assertEqual(error, 0.0)
    
    ddJ = Hessian(lambda F : forward(F, x_ref = x_ref)[1])
    min_order = taylor_test(lambda F : forward(F, x_ref = x_ref)[1], F, J_val = J.value(), dJ = dJ, ddJ = ddJ)
    self.assertGreater(min_order, 2.99)

  @leak_check
  def test_ExprEvaluationSolver(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    
    def test_expression(y, y_int):
      return y_int * y * (sin if is_function(y) else numpy.sin)(y) + 2.0 + (y ** 2) + y / (1.0 + (y ** 2))
    
    def forward(y):
      x = Function(space, name = "x")
      y_int = Function(RealFunctionSpace(), name = "y_int")
      AssembleSolver(y * dx, y_int).solve()
      ExprEvaluationSolver(test_expression(y, y_int), x).solve()
      
      J = Functional(name = "J")
      J.assign(x * x * x * dx)
      return x, J
    
    y = Function(space, name = "y", static = True)
    y.interpolate(Expression("cos(3.0 * pi * x[0])", element = space.ufl_element()))
    start_manager()
    x, J = forward(y)
    stop_manager()
    
    error_norm = abs(function_get_values(x) - test_expression(function_get_values(y), assemble(y * dx))).max()
    info("Error norm = %.16e" % error_norm)
    self.assertEqual(error_norm, 0.0)
    
    dJ = compute_gradient(J, y)
    min_order = taylor_test(lambda y : forward(y)[1], y, J_val = J.value(), dJ = dJ)
    self.assertGreater(min_order, 2.00)
    
    ddJ = Hessian(lambda y : forward(y)[1])
    min_order = taylor_test(lambda y : forward(y)[1], y, J_val = J.value(), dJ = dJ, ddJ = ddJ, seed = 1.0e-3)
    self.assertGreater(min_order, 2.99)
    
  @leak_check
  def test_PointInterpolationSolver(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitCubeMesh(5, 5, 5)
    z_space = FunctionSpace(mesh, "Lagrange", 3)
    if default_comm().size > 1:
      y_space = FunctionSpace(mesh, "Discontinuous Lagrange", 3)
    space_0 = RealFunctionSpace()
    X_coords = numpy.array([[0.1, 0.1, 0.1], [0.2, 0.3, 0.4], [0.9, 0.8, 0.7], [0.4, 0.2, 0.3]], dtype = numpy.float64)

    def forward(z):
      if default_comm().size > 1:
        y = Function(y_space, name = "y")
        LocalProjectionSolver(z, y).solve()
      else:
        y = z
      
      X = [Function(space_0, name = "x_%i" % i) for i in range(X_coords.shape[0])]
      PointInterpolationSolver(y, X, X_coords).solve()
      
      J = Functional(name = "J")
      for x in X:
        J.addto(x * x * x * dx)
      
      return X, J
      
    z = Function(z_space, name = "z", static = True)
    z.interpolate(Expression("pow(x[0], 3) - 1.5 * x[0] * x[1] + 1.5", element = z_space.ufl_element()))

    start_manager()
    X, J = forward(z)
    stop_manager()

    def x_ref(x):
      return x[0] ** 3 - 1.5 * x[0] * x[1] + 1.5

    x_error_norm = 0.0
    for x, x_coord in zip(X, X_coords):
      x_error_norm = max(x_error_norm, abs(function_max_value(x) - x_ref(x_coord)))
    info("Error norm = %.16e" % x_error_norm)
    self.assertLess(x_error_norm, 1.0e-14)

    dJ = compute_gradient(J, z)
    min_order = taylor_test(lambda z : forward(z)[1], z, J_val = J.value(), dJ = dJ, seed = 1.0e-4)
    self.assertGreater(min_order, 1.99)

    ddJ = Hessian(lambda z : forward(z)[1])
    min_order = taylor_test(lambda z : forward(z)[1], z, J_val = J.value(), dJ = dJ, ddJ = ddJ)
    self.assertGreater(min_order, 2.99)
    
  @leak_check
  def test_InterpolationSolver(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitCubeMesh(5, 5, 5)
    z_space = FunctionSpace(mesh, "Lagrange", 3)
    if default_comm().size > 1:
      y_space = FunctionSpace(mesh, "Discontinuous Lagrange", 3)
    x_space = FunctionSpace(mesh, "Lagrange", 2)

    def forward(z):
      if default_comm().size > 1:
        y = Function(y_space, name = "y")
        LocalProjectionSolver(z, y).solve()
      else:
        y = z
      
      x = Function(x_space, name = "x")
      InterpolationSolver(y, x).solve()
      
      J = Functional(name = "J")
      J.assign(x * x * x * dx)
      
      return x, J
      
    z = Function(z_space, name = "z", static = True)
    z.interpolate(Expression("sin(pi * x[0]) * sin(2.0 * pi * x[1]) * exp(x[2])", element = z_space.ufl_element()))

    start_manager()
    x, J = forward(z)
    stop_manager()

    x_ref = Function(x_space, name = "x_ref")
    x_ref.interpolate(z)

    x_error = Function(x_space, name = "x_error")
    function_assign(x_error, x_ref)
    function_axpy(x_error, -1.0, x)

    x_error_norm = function_linf_norm(x_error)
    info("Error norm = %.16e" % x_error_norm)
    self.assertLess(x_error_norm, 1.0e-13)

    dJ = compute_gradient(J, z)
    min_order = taylor_test(lambda z : forward(z)[1], z, J_val = J.value(), dJ = dJ, seed = 1.0e-4)
    self.assertGreater(min_order, 1.99)

    ddJ = Hessian(lambda z : forward(z)[1])
    min_order = taylor_test(lambda z : forward(z)[1], z, J_val = J.value(), dJ = dJ, ddJ = ddJ)
    self.assertGreater(min_order, 2.99)

  @leak_check
  def test_FixedPointSolver(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
  
    space = RealFunctionSpace()
    
    x = Function(space, name = "x")
    z = Function(space, name = "z")
    
    a = Function(space, name = "a", static = True)
    function_assign(a, 2.0)
    b = Function(space, name = "b", static = True)
    function_assign(b, 3.0)

    def forward(a, b):    
      eqs = [LinearCombinationSolver(z, (1.0, x), (1.0, b)),
             AssembleSolver((a / sqrt(z)) * dx, x)]

      eq = FixedPointSolver(eqs, solver_parameters = {"absolute_tolerance":0.0,
                                                      "relative_tolerance":1.0e-14})
                                                      
      eq.solve()
      
      J = Functional(name = "J")
      J.assign(x)
      
      return J
    
    start_manager()
    J = forward(a, b)
    stop_manager()
    
    x_val = function_max_value(x)
    a_val = function_max_value(a)
    b_val = function_max_value(b)
    self.assertAlmostEqual(x_val * numpy.sqrt(x_val + b_val) - a_val, 0.0, places = 14)

    dJda, dJdb = compute_gradient(J, [a, b])
    dm = Function(space, name = "dm", static = True)
    function_assign(dm, 1.0)
    min_order = taylor_test(lambda a : forward(a, b), a, J_val = J.value(), dJ = dJda, dM = dm)
    self.assertGreater(min_order, 1.99)
    min_order = taylor_test(lambda b : forward(a, b), b, J_val = J.value(), dJ = dJdb, dM = dm)
    self.assertGreater(min_order, 1.99)
    
    ddJ = Hessian(lambda a : forward(a, b))
    min_order = taylor_test(lambda a : forward(a, b), a, J_val = J.value(), ddJ = ddJ, dM = dm)
    self.assertGreater(min_order, 2.99)
    
    ddJ = Hessian(lambda b : forward(a, b))
    min_order = taylor_test(lambda b : forward(a, b), b, J_val = J.value(), ddJ = ddJ, dM = dm)
    self.assertGreater(min_order, 2.99)

  @leak_check
  def test_higher_order_adjoint(self):
    n_steps = 20
    reset("multistage", {"blocks":n_steps, "snaps_on_disk":2, "snaps_in_ram":2, "verbose":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitSquareMesh(20, 20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    
    def forward(kappa):
      clear_caches()
    
      x_n = Function(space, name = "x_n")
      x_n.interpolate(Expression("sin(pi * x[0]) * sin(2.0 * pi * x[1])", element = space.ufl_element()))
      x_np1 = Function(space, name = "x_np1")
      dt = Constant(0.01, static = True)
      bc = DirichletBC(space, 0.0, "on_boundary", static = True, homogeneous = True)
      
      eqs = [EquationSolver(inner(test, trial / dt) * dx + inner(grad(test), kappa * grad(trial)) * dx == inner(test, x_n / dt) * dx,
               x_np1, bc, solver_parameters = {"linear_solver":"cg",
                                               "preconditioner":"sor",
                                               "krylov_solver":{"relative_tolerance":1.0e-14,
                                                                "absolute_tolerance":1.0e-16}}),
             AssignmentSolver(x_np1, x_n)]
      
      for n in range(n_steps):
        for eq in eqs:
          eq.solve()
        if n < n_steps - 1:
          new_block()
      
      J = Functional(name = "J")
      J.assign(inner(x_np1, x_np1) * dx)
      
      return J
    
    kappa = Function(space, name = "kappa", static = True)
    function_assign(kappa, 1.0)
    
    perturb = Function(space, name = "perturb", static = True)
    function_set_values(perturb, 2.0 * (numpy.random.random(function_local_size(perturb)) - 1.0))
    
    add_tlm(kappa, perturb, max_depth = 3)
    start_manager()
    J = forward(kappa)
    dJ_tlm = J.tlm(kappa, perturb)
    ddJ_tlm = dJ_tlm.tlm(kappa, perturb)
    dddJ_tlm = ddJ_tlm.tlm(kappa, perturb)
    stop_manager()
    
    J_val, dJ_tlm_val, ddJ_tlm_val, dddJ_tlm_val = J.value(), dJ_tlm.value(), ddJ_tlm.value(), dddJ_tlm.value()
    dJ_adj, ddJ_adj, dddJ_adj, ddddJ_adj = compute_gradient([J, dJ_tlm, ddJ_tlm, dddJ_tlm], kappa)
    
    eps_vals = 2.0e-2 * numpy.array([2 ** -p for p in range(5)], dtype = numpy.float64)
    J_vals = []
    for eps in eps_vals:
      kappa_perturb = Function(space, name = "kappa_perturb", static = True)
      function_assign(kappa_perturb, kappa)
      function_axpy(kappa_perturb, eps, perturb)
      J_vals.append(forward(kappa_perturb).value())
    J_vals = numpy.array(J_vals, dtype = numpy.float64)
    errors_0 = abs(J_vals - J_val)
    orders_0 = numpy.log(errors_0[1:] / errors_0[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 0 derivative information = %s" % errors_0)
    info("Orders, maximal degree 0 derivative information = %s" % orders_0)
    errors_1_adj = abs(J_vals - J_val - eps_vals * function_inner(dJ_adj, perturb))
    orders_1_adj = numpy.log(errors_1_adj[1:] / errors_1_adj[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 1 derivative information, adjoint = %s" % errors_1_adj)
    info("Orders, maximal degree 1 derivative information, adjoint = %s" % orders_1_adj)
    errors_1_tlm = abs(J_vals - J_val - eps_vals * dJ_tlm_val)
    orders_1_tlm = numpy.log(errors_1_tlm[1:] / errors_1_tlm[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 1 derivative information, TLM = %s" % errors_1_tlm)
    info("Orders, maximal degree 1 derivative information, TLM = %s" % orders_1_tlm)
    errors_2_adj = abs(J_vals - J_val - eps_vals * dJ_tlm_val - 0.5 * eps_vals * eps_vals * function_inner(ddJ_adj, perturb))
    orders_2_adj = numpy.log(errors_2_adj[1:] / errors_2_adj[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 2 derivative information, adjoint(TLM) = %s" % errors_2_adj)
    info("Orders, maximal degree 2 derivative information, adjoint(TLM) = %s" % orders_2_adj)
    errors_2_tlm = abs(J_vals - J_val - eps_vals * dJ_tlm_val - 0.5 * eps_vals * eps_vals * ddJ_tlm_val)
    orders_2_tlm = numpy.log(errors_2_tlm[1:] / errors_2_tlm[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 2 derivative information, TLM(TLM) = %s" % errors_2_tlm)
    info("Orders, maximal degree 2 derivative information, TLM(TLM) = %s" % orders_2_tlm)
    errors_3_adj = abs(J_vals - J_val - eps_vals * dJ_tlm_val - 0.5 * eps_vals * eps_vals * ddJ_tlm_val
      - (1.0 / 6.0) * numpy.power(eps_vals, 3.0) * function_inner(dddJ_adj, perturb))
    orders_3_adj = numpy.log(errors_3_adj[1:] / errors_3_adj[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 3 derivative information, adjoint(TLM(TLM)) = %s" % errors_3_adj)
    info("Orders, maximal degree 3 derivative information, adjoint(TLM(TLM)) = %s" % orders_3_adj)
    errors_3_tlm = abs(J_vals - J_val - eps_vals * dJ_tlm_val - 0.5 * eps_vals * eps_vals * ddJ_tlm_val
      - (1.0 / 6.0) * numpy.power(eps_vals, 3.0) * dddJ_tlm_val)
    orders_3_tlm = numpy.log(errors_3_tlm[1:] / errors_3_tlm[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 3 derivative information, TLM(TLM(TLM)) = %s" % errors_3_tlm)
    info("Orders, maximal degree 3 derivative information, TLM(TLM(TLM)) = %s" % orders_3_tlm)
    errors_4_adj = abs(J_vals - J_val - eps_vals * dJ_tlm_val - 0.5 * eps_vals * eps_vals * ddJ_tlm_val
      - (1.0 / 6.0) * numpy.power(eps_vals, 3.0) * dddJ_tlm_val
      - (1.0 / 24.0) * numpy.power(eps_vals, 4.0) * function_inner(ddddJ_adj, perturb))
    orders_4_adj = numpy.log(errors_4_adj[1:] / errors_4_adj[:-1]) / numpy.log(0.5)
    info("Errors, maximal degree 4 derivative information, adjoint(TLM(TLM(TLM))) = %s" % errors_4_adj)
    info("Orders, maximal degree 4 derivative information, adjoint(TLM(TLM(TLM))) = %s" % orders_4_adj)
    
    self.assertGreater(orders_0[-1], 1.00)
    self.assertGreater(orders_1_adj.min(), 2.00)
    self.assertGreater(orders_1_tlm.min(), 2.00)
    self.assertGreater(orders_2_adj.min(), 3.00)
    self.assertGreater(orders_2_tlm.min(), 3.00)
    self.assertGreater(orders_3_adj.min(), 4.00)
    self.assertGreater(orders_3_tlm.min(), 4.00)
    self.assertGreater(orders_4_adj.min(), 5.00)
  
  @leak_check
  def test_minimize_scipy_multiple(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitSquareMesh(20, 20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    
    def forward(alpha, beta, x_ref = None, y_ref = None):
      clear_caches()
    
      x = Function(space, name = "x")
      solve(inner(test, trial) * dx == inner(test, alpha) * dx,
        x, solver_parameters = {"linear_solver":"cg",
                                "preconditioner":"sor",
                                "krylov_solver":{"relative_tolerance":1.0e-14,
                                                 "absolute_tolerance":1.0e-16}})
      y = Function(space, name = "y")
      solve(inner(test, trial) * dx == inner(test, beta) * dx,
        y, solver_parameters = {"linear_solver":"cg",
                                "preconditioner":"sor",
                                "krylov_solver":{"relative_tolerance":1.0e-14,
                                                 "absolute_tolerance":1.0e-16}})
      if x_ref is None:
        x_ref = Function(space, name = "x_ref", static = True)
        function_assign(x_ref, x)
      if y_ref is None:
        y_ref = Function(space, name = "y_ref", static = True)
        function_assign(y_ref, y)
      
      J = Functional(name = "J")
      J.assign(inner(x - x_ref, x - x_ref) * dx)
      J.addto(inner(y - y_ref, y - y_ref) * dx)
      return x_ref, y_ref, J
    
    alpha_ref = Function(space, name = "alpha_ref", static = True)
    alpha_ref.interpolate(Expression("exp(x[0] + x[1])", element = space.ufl_element()))
    beta_ref = Function(space, name = "beta_ref", static = True)
    beta_ref.interpolate(Expression("sin(pi * x[0]) * sin(2.0 * pi * x[1])", element = space.ufl_element()))
    x_ref, y_ref, _ = forward(alpha_ref, beta_ref)
    
    alpha0 = Function(space, name = "alpha0", static = True)
    beta0 = Function(space, name = "beta0", static = True)
    start_manager()
    _, _, J = forward(alpha0, beta0, x_ref = x_ref, y_ref = y_ref)
    stop_manager()
    
    (alpha, beta), result = minimize_scipy(lambda alpha, beta : forward(alpha, beta, x_ref = x_ref, y_ref = y_ref)[2], (alpha0, beta0), J0 = J,
      method = "L-BFGS-B", options = {"ftol":0.0, "gtol":1.0e-11})
    self.assertTrue(result.success)
    
    error = Function(space, name = "error")
    function_assign(error, alpha_ref)
    function_axpy(error, -1.0, alpha)
    self.assertLess(function_linf_norm(error), 1.0e-8)
    function_assign(error, beta_ref)
    function_axpy(error, -1.0, beta)
    self.assertLess(function_linf_norm(error), 1.0e-9)

  @leak_check
  def test_minimize_scipy(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitSquareMesh(20, 20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    
    def forward(alpha, x_ref = None):
      clear_caches()
    
      x = Function(space, name = "x")
      solve(inner(test, trial) * dx == inner(test, alpha) * dx,
        x, solver_parameters = {"linear_solver":"cg",
                                "preconditioner":"sor",
                                "krylov_solver":{"relative_tolerance":1.0e-14,
                                                 "absolute_tolerance":1.0e-16}})

      if x_ref is None:
        x_ref = Function(space, name = "x_ref", static = True)
        function_assign(x_ref, x)
      
      J = Functional(name = "J")
      J.assign(inner(x - x_ref, x - x_ref) * dx)
      return x_ref, J
    
    alpha_ref = Function(space, name = "alpha_ref", static = True)
    alpha_ref.interpolate(Expression("exp(x[0] + x[1])", element = space.ufl_element()))
    x_ref, _ = forward(alpha_ref)
    
    alpha0 = Function(space, name = "alpha0", static = True)
    start_manager()
    _, J = forward(alpha0, x_ref = x_ref)
    stop_manager()
    
    alpha, result = minimize_scipy(lambda alpha : forward(alpha, x_ref = x_ref)[1], alpha0, J0 = J,
      method = "L-BFGS-B", options = {"ftol":0.0, "gtol":1.0e-10})
    self.assertTrue(result.success)
    
    error = Function(space, name = "error")
    function_assign(error, alpha_ref)
    function_axpy(error, -1.0, alpha)
    self.assertLess(function_linf_norm(error), 1.0e-7)

  @leak_check
  def test_overrides(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitSquareMesh(20, 20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    F = Function(space, name = "F", static = True)
    F.interpolate(Expression("1.0 + sin(pi * x[0]) * sin(3.0 * pi * x[1])", element = space.ufl_element()))
    
    bc = DirichletBC(space, 1.0, "on_boundary", static = True, homogeneous = False)
    
    def forward(F):
      G = [Function(space, name = "G_%i" % i) for i in range(7)]
      
      G[0] = project(F, space, function = G[0])
      
      A, b = assemble_system(Constant(0.2, static = True) * inner(test, trial) * dx, Constant(0.3, static = True) * inner(test, G[0]) * dx)
      A, b = assemble_system(Constant(0.8, static = True) * inner(test, trial) * dx, Constant(0.7, static = True) * inner(test, G[0]) * dx,
        A_tensor = A, b_tensor = b, add_values = True)      
      bc.apply(A, b)      
      solver = KrylovSolver("cg")
      solver.parameters.update({"absolute_tolerance":1.0e-16, "relative_tolerance":1.0e-14})
      solver.set_operator(A)
      solver.solve(G[1].vector(), b)
      
      b = assemble(inner(test, G[1]) * dx)
      bc.apply(b)
      solver.solve(G[2].vector(), b)
      
      A = assemble(inner(test, trial) * dx)
      b = A * G[2].vector()
      bc.apply(A, b)
      solver = KrylovSolver("cg")
      solver.parameters.update({"absolute_tolerance":1.0e-16, "relative_tolerance":1.0e-14})
      solver.solve(A, G[3].vector(), b)
      
      A, b = assemble_system(inner(test, trial) * dx, inner(test, G[3]) * dx, bcs = bc)
      solver.solve(A, G[4].vector(), b)
      
      solver = LinearVariationalSolver(LinearVariationalProblem(
        inner(test, trial) * dx, inner(test, G[4]) * dx, G[5]))
      solver.parameters.update({"linear_solver":"cg", "krylov_solver":{"absolute_tolerance":1.0e-16, "relative_tolerance":1.0e-14}})
      solver.solve()
      
      solver = NonlinearVariationalSolver(NonlinearVariationalProblem(
        inner(test, G[6]) * dx - inner(test, G[5]) * dx, G[6],
        J = inner(test, trial) * dx))
      solver.parameters.update({"nonlinear_solver":"newton",
                                "newton_solver":{"linear_solver":"cg",
                                                 "preconditioner":"sor",
                                                 "krylov_solver":{"relative_tolerance":1.0e-14,
                                                                  "absolute_tolerance":1.0e-16},
                                                 "relative_tolerance":1.0e-13,
                                                 "absolute_tolerance":1.0e-15,
                                                 "report":False}})
      solver.solve()
      
      J = Functional(name = "J")
      J.assign(inner(G[-1], G[-1]) * dx)
      
      return J, G[-1]
    
    start_manager()
    J, G = forward(F)
    stop_manager()
    
    self.assertAlmostEqual(assemble(inner(F - G, F - G) * dx), 0.0, places = 27)

    J_val = J.value()    
    dJ = compute_gradient(J, F)    
    min_order = taylor_test(lambda F : forward(F)[0], F, J_val = J_val, dJ = dJ)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 1.99)

  @leak_check
  def test_bc(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitSquareMesh(20, 20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    F = Function(space, name = "F", static = True)
    F.interpolate(Expression("sin(pi * x[0]) * sin(3.0 * pi * x[1])", element = space.ufl_element()))
    
    def forward(bc):
      x_0 = Function(space, name = "x_0")
      x_1 = Function(space, name = "x_1")
      x = Function(space, name = "x")
      
      DirichletBCSolver(bc, x_1, "on_boundary").solve()
      
      solve(inner(grad(test), grad(trial)) * dx == inner(test, F) * dx - inner(grad(test), grad(x_1)) * dx,
        x_0, DirichletBC(space, 0.0, "on_boundary", static = True, homogeneous = True),
        solver_parameters = {"linear_solver":"cg",
                             "preconditioner":"sor",
                             "krylov_solver":{"relative_tolerance":1.0e-14,
                                              "absolute_tolerance":1.0e-16}})
      
      AxpySolver(x_0, 1.0, x_1, x).solve()
      
      J = Functional(name = "J")
      J.assign(inner(x, x) * dx)
      return x, J
    
    bc = Function(space, name = "bc", static = True)
    function_assign(bc, 1.0)
    
    start_manager()
    x, J = forward(bc)
    stop_manager()
    
    x_ref = Function(space, name = "x_ref")
    solve(inner(grad(test), grad(trial)) * dx == inner(test, F) * dx,
      x_ref, DirichletBC(space, 1.0, "on_boundary", static = True, homogeneous = False),
      solver_parameters = {"linear_solver":"cg",
                           "preconditioner":"sor",
                           "krylov_solver":{"relative_tolerance":1.0e-14,
                                            "absolute_tolerance":1.0e-16}})
    error = Function(space, name = "error")
    function_assign(error, x_ref)
    function_axpy(error, -1.0, x)
    self.assertLess(function_linf_norm(error), 1.0e-13)

    J_val = J.value()    
    dJ = compute_gradient(J, bc)    
    min_order = taylor_test(lambda bc : forward(bc)[1], bc, J_val = J_val, dJ = dJ)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 1.99)

  @leak_check
  def test_recursive_tlm(self):
    n_steps = 20
    reset("multistage", {"blocks":n_steps, "snaps_on_disk":4, "snaps_in_ram":2, "verbose":True})
    clear_caches()
    stop_manager()
    
    # Use a mesh of non-unit size to test that volume factors are handled
    # correctly
    mesh = IntervalMesh(20, 0.0, 2.0)
    r0 = FiniteElement("R", mesh.ufl_cell(), 0)
    space = FunctionSpace(mesh, r0 * r0)
    test = TestFunction(space)
    dt = Constant(0.01, static = True)
    control_space = FunctionSpace(mesh, r0)
    alpha = Function(control_space, name = "alpha", static = True)
    function_assign(alpha, 1.0)
    inv_V = Constant(1.0 / assemble(Constant(1.0) * dx(mesh)), static = True)
    dalpha = Function(control_space, name = "dalpha", static = True)
    function_assign(dalpha, 1.0)
    
    def forward(alpha, dalpha = None):
      clear_caches()

      T_n = Function(space, name = "T_n")
      T_np1 = Function(space, name = "T_np1")
      
      # Forward model initialisation and definition
      T_n.assign(Constant((1.0, 0.0)))
      eq = EquationSolver(inner(test[0], (T_np1[0] - T_n[0]) / dt - Constant(0.5, static = True) * T_n[1] - Constant(0.5, static = True) * T_np1[1]) * dx
                        + inner(test[1], (T_np1[1] - T_n[1]) / dt + sin(alpha * (Constant(0.5, static = True) * T_n[0] + Constant(0.5, static = True) * T_np1[0]))) * dx == 0,
             T_np1, solver_parameters = {"nonlinear_solver":"newton",
                                         "newton_solver":{"linear_solver":"gmres",
                                                          "preconditioner":"sor",
                                                          "krylov_solver":{"relative_tolerance":1.0e-14,
                                                                           "absolute_tolerance":1.0e-16},
                                                          "relative_tolerance":1.0e-13,
                                                          "absolute_tolerance":1.0e-15,
                                                          "report":False}})
      cycle = AssignmentSolver(T_np1, T_n)
      J = Functional(name = "J")
     
      for n in range(n_steps):
        eq.solve()
        cycle.solve()
        if n == n_steps - 1:
          J.addto(inv_V * T_n[0] * dx)
        if n < n_steps - 1:
          new_block()

      if dalpha is None:
        J_tlm, K = None, None
      else:
        K = J.tlm(alpha, dalpha)
      
      return J, K
      
    add_tlm(alpha, dalpha)
    start_manager()
    J, K = forward(alpha, dalpha = dalpha)
    stop_manager()

    J_val = J.value()
    info("J = %.16e" % J_val)
    self.assertAlmostEqual(J_val, 9.8320117858590805e-01, places = 15)
    
    dJ = K.value()
    info("TLM sensitivity = %.16e" % dJ)
    
    # Run the adjoint of the forward+TLM system to compute the Hessian action
    ddJ = compute_gradient(K, alpha)
    ddJ_val = function_max_value(ddJ)
    info("ddJ = %.16e" % ddJ_val)
    
    # Taylor verify the Hessian (and gradient)
    eps_vals = numpy.array([numpy.power(4.0, -p) for p in range(1, 5)], dtype = numpy.float64)
    def control_value(value):
      alpha = Function(control_space, static = True)
      function_assign(alpha, value)
      return alpha
    J_vals = numpy.array([forward(control_value(1.0 + eps))[0].value() for eps in eps_vals], dtype = numpy.float64)
    errors_0 = abs(J_vals - J_val)
    errors_1 = abs(J_vals - J_val - dJ * eps_vals)
    errors_2 = abs(J_vals - J_val - dJ * eps_vals - 0.5 * ddJ_val * numpy.power(eps_vals, 2))
    orders_0 = numpy.log(errors_0[1:] / errors_0[:-1]) / numpy.log(eps_vals[1:] / eps_vals[:-1])
    orders_1 = numpy.log(errors_1[1:] / errors_1[:-1]) / numpy.log(eps_vals[1:] / eps_vals[:-1])
    orders_2 = numpy.log(errors_2[1:] / errors_2[:-1]) / numpy.log(eps_vals[1:] / eps_vals[:-1])
    info("dJ errors, first order  = %s" % errors_0)
    info("dJ orders, first order  = %s" % orders_0)
    info("dJ errors, second order = %s" % errors_1)
    info("dJ orders, second order = %s" % orders_1)
    info("dJ errors, third order  = %s" % errors_2)
    info("dJ orders, third order  = %s" % orders_2)
    self.assertGreater(orders_0[-1], 0.99)
    self.assertGreater(orders_1[-1], 2.00)
    self.assertGreater(orders_2[-1], 2.99)
    self.assertGreater(orders_0.min(), 0.87)
    self.assertGreater(orders_1.min(), 2.00)
    self.assertGreater(orders_2.min(), 2.94)

  @leak_check
  def test_timestepping(self):    
    for n_steps in [1, 2, 5, 20]:
      reset("multistage", {"blocks":n_steps, "snaps_on_disk":4, "snaps_in_ram":2, "verbose":True})
      clear_caches()
      stop_manager()
      
      mesh = UnitIntervalMesh(100)
      space = FunctionSpace(mesh, "Lagrange", 1)
      test, trial = TestFunction(space), TrialFunction(space)
      T_0 = Function(space, name = "T_0", static = True)
      T_0.interpolate(Expression("sin(pi * x[0]) + sin(10.0 * pi * x[0])", element = space.ufl_element()))
      dt = Constant(0.01, static = True)
      space_r0 = FunctionSpace(mesh, "R", 0)
      kappa = Function(space_r0, name = "kappa", static = True)
      function_assign(kappa, 1.0)
      
      def forward(T_0, kappa):
        from tlm_adjoint.timestepping import N, TimeFunction, TimeLevels, \
          TimeSystem, n

        levels = TimeLevels([n, n + 1], {n:n + 1})
        T = TimeFunction(levels, space, name = "T")
        T[n].rename("T_n", "a Function")
        T[n + 1].rename("T_np1", "a Function")
        
        system = TimeSystem()
        
        system.add_solve(T_0, T[0])
        
        system.add_solve(inner(test, trial) * dx + dt * inner(grad(test), kappa * grad(trial)) * dx == inner(test, T[n]) * dx,
          T[n + 1], DirichletBC(space, 1.0, "on_boundary", static = True, homogeneous = False),
          solver_parameters = {"linear_solver":"cg",
                               "preconditioner":"sor",
                               "krylov_solver":{"relative_tolerance":1.0e-14,
                                                "absolute_tolerance":1.0e-16}})
                                                     
        for n_step in range(n_steps):
          system.timestep()
          if n_step < n_steps - 1:
            new_block()
        system.finalise()
          
        J = Functional(name = "J")
        J.assign(inner(T[N], T[N]) * dx)
        return J
      
      start_manager()  
      J = forward(T_0, kappa)
      stop_manager()

      J_val = J.value()
      if n_steps == 20:
        self.assertAlmostEqual(J_val, 9.4790204396919131e-01, places = 12)

      controls = [Control("T_0"), Control(kappa)]
      dJ = compute_gradient(J, controls)      
      min_order = taylor_test(lambda T_0 : forward(T_0, kappa), controls[0], J_val = J_val, dJ = dJ[0])  # Usage as in dolfin-adjoint tests
      self.assertGreater(min_order, 1.99)
      dm = Function(space_r0, name = "dm", static = True)
      function_assign(dm, 1.0)
      min_order = taylor_test(lambda kappa : forward(T_0, kappa), controls[1], J_val = J_val, dJ = dJ[1], dM = dm)  # Usage as in dolfin-adjoint tests
      self.assertGreater(min_order, 1.99)

  @leak_check
  def test_second_order_adjoint(self):    
    n_steps = 20
    reset("multistage", {"blocks":n_steps, "snaps_on_disk":4, "snaps_in_ram":2, "verbose":True})
    clear_caches()
    stop_manager()
  
    mesh = UnitIntervalMesh(20)
    r0 = FiniteElement("R", mesh.ufl_cell(), 0)
    space = FunctionSpace(mesh, r0 * r0)
    test = TestFunction(space)
    T_0 = Function(space, name = "T_0", static = True)
    T_0.assign(Constant((1.0, 0.0)))
    dt = Constant(0.01, static = True)
    
    def forward(T_0):
      T_n = Function(space, name = "T_n")
      T_np1 = Function(space, name = "T_np1")
      
      AssignmentSolver(T_0, T_n).solve()
      eq = EquationSolver(inner(test[0], (T_np1[0] - T_n[0]) / dt - Constant(0.5, static = True) * T_n[1] - Constant(0.5, static = True) * T_np1[1]) * dx
                        + inner(test[1], (T_np1[1] - T_n[1]) / dt + sin(Constant(0.5, static = True) * T_n[0] + Constant(0.5, static = True) * T_np1[0])) * dx == 0,
             T_np1, solver_parameters = {"nonlinear_solver":"newton",
                                         "newton_solver":{"linear_solver":"gmres",
                                                          "preconditioner":"sor",
                                                          "krylov_solver":{"relative_tolerance":1.0e-14,
                                                                           "absolute_tolerance":1.0e-16},
                                                          "relative_tolerance":1.0e-13,
                                                          "absolute_tolerance":1.0e-15}})
      for n in range(n_steps):
        eq.solve()
        T_n.assign(T_np1)
        if n < n_steps - 1:
          new_block()
    
      J = Functional(name = "J")
      J.assign(T_n[0] * T_n[0] * dx)
    
      return J
      
    start_manager()
    J = forward(T_0)
    stop_manager()

    J_val = J.value()
    self.assertAlmostEqual(J_val, 9.8320117858590805e-01 ** 2, places = 15)

    dJ = compute_gradient(J, T_0)    
    dm = Function(space, name = "dm", static = True)
    function_assign(dm, 1.0)
    min_order = taylor_test(forward, T_0, J_val = J_val, dJ = dJ, dM = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)
    
    ddJ = Hessian(forward)
    min_order = taylor_test(forward, T_0, J_val = J_val, ddJ = ddJ, dM = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.99)

  @leak_check
  def test_AxpySolver(self):    
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()

    space = RealFunctionSpace()
    x = Function(space, name = "x", static = True)
    function_assign(x, 1.0)  
    
    def forward(x):
      y = [Function(space, name = "y_%i" % i) for i in range(5)]
      z = [Function(space, name = "z_%i" % i) for i in range(2)]
      function_assign(z[0], 7.0)
    
      AssignmentSolver(x, y[0]).solve()
      for i in range(len(y) - 1):
        AxpySolver(y[i], i + 1, z[0], y[i + 1]).solve()
      AssembleSolver(y[-1] * y[-1] * dx, z[1]).solve()
      
      J = Functional(name = "J")
      J.assign(inner(z[1], z[1]) * dx)
      
      return J
    
    start_manager()
    J = forward(x)
    stop_manager()
    
    J_val = J.value()
    self.assertAlmostEqual(J_val, 25411681.0, places = 7)
    
    dJ = compute_gradient(J, x)    
    dm = Function(space, name = "dm", static = True)
    function_assign(dm, 1.0)
    min_order = taylor_test(forward, x, J_val = J_val, dJ = dJ, dM = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)

  @leak_check
  def test_AssignmentSolver(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
  
    space = RealFunctionSpace()
    x = Function(space, name = "x", static = True)
    function_assign(x, 16.0)  
    
    def forward(x):
      y = [Function(space, name = "y_%i" % i) for i in range(9)]
      z = Function(space, name = "z")
    
      AssignmentSolver(x, y[0]).solve()
      for i in range(len(y) - 1):
        AssignmentSolver(y[i], y[i + 1]).solve()
      AssembleSolver(y[-1] * y[-1] * dx, z).solve()

      J = Functional(name = "J")
      J.assign(inner(z, z) * dx)
      J.addto(2 * inner(x, x) * dx)
      
      K = Functional(name = "K")
      K.assign(inner(z, z) * dx)

      return J, K
    
    start_manager()
    J, K = forward(x)
    stop_manager()
    
    J_val = J.value()
    K_val = K.value()
    self.assertAlmostEqual(J_val, 66048.0, places = 9)
    self.assertAlmostEqual(K_val, 65536.0, places = 9)
    
    dJs = compute_gradient([J, K], x)    
    dm = Function(space, name = "dm", static = True)
    function_assign(dm, 1.0)
    min_order = taylor_test(lambda x : forward(x)[0], x, J_val = J_val, dJ = dJs[0], dM = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)
    min_order = taylor_test(lambda x : forward(x)[1], x, J_val = K_val, dJ = dJs[1], dM = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)

    ddJ = Hessian(lambda m : forward(m)[0])
    min_order = taylor_test(lambda x : forward(x)[0], x, J_val = J_val, ddJ = ddJ, dM = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 3.00)
  
  @leak_check
  def test_HEP(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    M = assemble(inner(test, trial) * dx)
    def M_action(F):
      return (M * F.vector()).get_local()
    
    import slepc4py.SLEPc
    lam, V_r = eigendecompose(space, M_action, problem_type = slepc4py.SLEPc.EPS.ProblemType.HEP)
    diff = Function(space)
    for lam_val, v_r in zip(lam, V_r):
      function_set_values(diff, M_action(v_r))
      function_axpy(diff, -lam_val, v_r)
      self.assertAlmostEqual(function_linf_norm(diff), 0.0, places = 16)

  @leak_check
  def test_NHEP(self):
    reset("memory", {"replace":True})
    clear_caches()
    stop_manager()
    
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    N = assemble(inner(test, trial.dx(0)) * dx)
    def N_action(F):
      return (N * F.vector()).get_local()
    
    lam, (V_r, V_i) = eigendecompose(space, N_action)
    diff = Function(space)
    for lam_val, v_r, v_i in zip(lam, V_r, V_i):
      function_set_values(diff, N_action(v_r))
      function_axpy(diff, -float(lam_val.real), v_r)
      function_axpy(diff, +float(lam_val.imag), v_i)
      self.assertAlmostEqual(function_linf_norm(diff), 0.0, places = 7)
      function_set_values(diff, N_action(v_i))
      function_axpy(diff, -float(lam_val.real), v_i)
      function_axpy(diff, -float(lam_val.imag), v_r)
      self.assertAlmostEqual(function_linf_norm(diff), 0.0, places = 7)
    
if __name__ == "__main__":
  numpy.random.seed(1201)
  unittest.main()

#  tests().test_AssignmentSolver()
#  tests().test_AxpySolver()
#  tests().test_second_order_adjoint()
#  tests().test_timestepping()
#  tests().test_recursive_tlm()
#  tests().test_bc()
#  tests().test_overrides()
#  tests().test_minimize_scipy()
#  tests().test_minimize_scipy_multiple()
#  tests().test_higher_order_adjoint()
#  tests().test_FixedPointSolver()
#  tests().test_InterpolationSolver()
#  tests().test_PointInterpolationSolver()
#  tests().test_ExprEvaluationSolver()
#  tests().test_LongRange()
#  tests().test_LocalProjectionSolver()
#  tests().test_clear_caches()

#  tests().test_HEP()
#  tests().test_NHEP()
