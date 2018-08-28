#!/usr/bin/env python3
# -*- coding: utf-8 -*

# Copyright(c) 2018 The University of Edinburgh
#
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
from tlm_adjoint import *

import numpy
import unittest
  
class tests(unittest.TestCase):
  def test_FixedPointSolver(self):
    reset("memory")
    clear_caches()
    stop_manager()
  
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "R", 0)
    test, trial = TestFunction(space), TrialFunction(space)
    
    x = Function(space, name = "x")
    z = Function(space, name = "z")
    
    a = Function(space, name = "a", static = True)
    function_assign(a, 2.0)
    b = Function(space, name = "b", static = True)
    function_assign(b, 3.0)

    def forward(a, b):    
      eqs = [LinearCombinationSolver(z, (1.0, x), (1.0, b)),
             EquationSolver(inner(test, trial) * dx == inner(test, a / sqrt(z)) * dx, x,
               solver_parameters = {"ksp_type":"cg",
                                    "pc_type":"jacobi",
                                    "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16})]

      eq = FixedPointSolver(eqs, solver_parameters = {"absolute_tolerance":0.0,
                                                      "relative_tolerance":1.0e-14})
                                                      
      eq.solve(replace = True)
      
      J = Functional(name = "J")
      J.assign(x * dx)
      
      return J
    
    start_manager()
    J = forward(a, b)
    stop_manager()
    
    x_val = function_max_value(x)
    a_val = function_max_value(a)
    b_val = function_max_value(b)
    self.assertAlmostEqual(x_val * numpy.sqrt(x_val + b_val) - a_val, 0.0, places = 14)

    dJda, dJdb = compute_gradient(J, [a, b])
    min_order = taylor_test(lambda a : forward(a, b), a, J_val = J.value(), dJ = dJda)
    self.assertGreater(min_order, 1.99)
    min_order = taylor_test(lambda b : forward(a, b), b, J_val = J.value(), dJ = dJdb)
    self.assertGreater(min_order, 1.99)
    
    ddJ = Hessian(lambda a : forward(a, b))
    min_order = taylor_test(lambda a : forward(a, b), a, J_val = J.value(), ddJ = ddJ)
    self.assertGreater(min_order, 2.99)
    
    ddJ = Hessian(lambda b : forward(a, b))
    min_order = taylor_test(lambda b : forward(a, b), b, J_val = J.value(), ddJ = ddJ)
    self.assertGreater(min_order, 2.99)

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
      dt = Constant(0.01)
      bc = DirichletBC(space, 0.0, "on_boundary")
      
      eqs = [EquationSolver(inner(test, trial / dt) * dx + inner(grad(test), kappa * grad(trial)) * dx == inner(test, x_n / dt) * dx,
               x_np1, bc, solver_parameters = {"ksp_type":"cg",
                                               "pc_type":"jacobi",
                                               "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16}),
             AssignmentSolver(x_np1, x_n)]
      
      for n in range(n_steps):
        for eq in eqs:
          eq.solve()
        if n < n_steps - 1:
          new_block()
      for eq in eqs:
        eq.replace()
      
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
    
  def test_replace(self):
    reset("memory")
    clear_caches()
    stop_manager()
    
    mesh = UnitSquareMesh(20, 20)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    
    def forward(alpha):
      x = Function(space, name = "x")
      y = Function(space, name = "y")
      EquationSolver(inner(test, trial) * dx == inner(test, alpha) * dx,
        x, solver_parameters = {"ksp_type":"cg",
                                "pc_type":"jacobi",
                                "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16}).solve(replace = False)
      EquationSolver(inner(test, trial) * dx == inner(test, x) * dx + inner(test, alpha) * dx,
        y, solver_parameters = {"ksp_type":"cg",
                                "pc_type":"jacobi",
                                "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16}).solve(replace = True)
      
      J = Functional(name = "J")
      J.assign(inner(y, y) * dx)
      return J
    
    alpha = Function(space, name = "alpha", static = True)
    function_assign(alpha, 1.0)
    start_manager()
    J = forward(alpha)
    stop_manager()
    
    dJ = compute_gradient(J, alpha)
    min_order = taylor_test(forward, alpha, J_val = J.value(), dJ = dJ)
    self.assertGreater(min_order, 1.99)

  def test_bc(self):
    reset("memory")
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
      
      DirichletBCSolver(bc, x_1, "on_boundary").solve(replace = True)
      
      EquationSolver(inner(grad(test), grad(trial)) * dx == inner(test, F) * dx - inner(grad(test), grad(x_1)) * dx,
        x_0, DirichletBC(space, 0.0, "on_boundary"),
        solver_parameters = {"ksp_type":"cg",
                             "pc_type":"jacobi",
                             "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16}).solve(replace = True)
      
      AxpySolver(x_0, 1.0, x_1, x).solve(replace = True)
      
      J = Functional(name = "J")
      J.assign(inner(x, x) * dx)
      return x, J
    
    bc_mesh = mesh
    bc_space = space
    bc = Function(bc_space, name = "bc", static = True)
    function_assign(bc, 1.0)
    
    start_manager()
    x, J = forward(bc)
    stop_manager()
    
    x_ref = Function(space, name = "x_ref")
    solve(inner(grad(test), grad(trial)) * dx == inner(test, F) * dx,
      x_ref, DirichletBC(space, 1.0, "on_boundary"),
      solver_parameters = {"ksp_type":"cg",
                           "pc_type":"jacobi",
                           "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16})
    error = Function(space, name = "error")
    function_assign(error, x_ref)
    function_axpy(error, -1.0, x)
    self.assertEqual(function_linf_norm(error), 0.0)

    J_val = J.value()    
    dJ = compute_gradient(J, bc)    
    min_order = taylor_test(lambda bc : forward(bc)[1], bc, J_val = J_val, dJ = dJ)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 1.99)

  def test_recursive_tlm(self):
    n_steps = 20
    reset("multistage", {"blocks":n_steps, "snaps_on_disk":4, "snaps_in_ram":2, "verbose":True})
    clear_caches()
    stop_manager()
    
    # Use an interval of non-unit size to test that volume factors are handled
    # correctly
    mesh = IntervalMesh(1, 0.0, 2.0)
    r0 = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
    space = FunctionSpace(mesh, r0 * r0)
    test = TestFunction(space)
    dt = Constant(0.01)
    control_space = FunctionSpace(mesh, r0)
    alpha = Function(control_space, name = "alpha", static = True)
    function_assign(alpha, 1.0)
    inv_V = Constant(1.0 / assemble(Constant(1.0) * dx(mesh)))
    dalpha = Function(control_space, name = "dalpha", static = True)
    function_assign(dalpha, 1.0)
    
    def forward(alpha, dalpha = None):
      clear_caches()

      T_n = Function(space, name = "T_n")
      T_np1 = Function(space, name = "T_np1")
      
      # Forward model initialisation and definition
      T_n.assign(Constant((1.0, 0.0)))
      eq = EquationSolver(inner(test[0], (T_np1[0] - T_n[0]) / dt - Constant(0.5) * T_n[1] - Constant(0.5) * T_np1[1]) * dx
                        + inner(test[1], (T_np1[1] - T_n[1]) / dt + sin(alpha * (Constant(0.5) * T_n[0] + Constant(0.5) * T_np1[0]))) * dx == 0,
             T_np1, solver_parameters = {"snes_type":"newtonls",
                                         "ksp_type":"gmres",
                                         "pc_type":"jacobi",
                                         "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16,
                                         "snes_rtol":1.0e-13, "snes_atol":1.0e-15})
      cycle = AssignmentSolver(T_np1, T_n)
      J = Functional(name = "J")
     
      for n in range(n_steps):
        eq.solve()
        cycle.solve()
        if n == n_steps - 1:
          J.addto(inv_V * T_n[0] * dx)
        if n < n_steps - 1:
          new_block()
      eq.replace()
      cycle.replace()

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
    self.assertEqual(J_val, 9.8320117858590805e-01)
    
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
    
  def test_second_order_adjoint(self):    
    n_steps = 20
    reset("multistage", {"blocks":n_steps, "snaps_on_disk":4, "snaps_in_ram":2, "verbose":True})
    clear_caches()
    stop_manager()
  
    mesh = UnitIntervalMesh(1)
    r0 = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
    space = FunctionSpace(mesh, r0 * r0)
    test = TestFunction(space)
    T_0 = Function(space, name = "T_0", static = True)
    T_0.assign(Constant((1.0, 0.0)))
    dt = Constant(0.01)
    
    def forward(T_0):
      T_n = Function(space, name = "T_n")
      T_np1 = Function(space, name = "T_np1")
      
      AssignmentSolver(T_0, T_n).solve(replace = True)
      eq = EquationSolver(inner(test[0], (T_np1[0] - T_n[0]) / dt - Constant(0.5) * T_n[1] - Constant(0.5) * T_np1[1]) * dx
                        + inner(test[1], (T_np1[1] - T_n[1]) / dt + sin(Constant(0.5) * T_n[0] + Constant(0.5) * T_np1[0])) * dx == 0,
             T_np1, solver_parameters = {"snes_type":"newtonls",
                                         "ksp_type":"gmres",
                                         "pc_type":"jacobi",
                                         "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16,
                                         "snes_rtol":1.0e-13, "snes_atol":1.0e-15})
      cycle = AssignmentSolver(T_np1, T_n)
      for n in range(n_steps):
        eq.solve()
        cycle.solve()
        if n < n_steps - 1:
          new_block()
      eq.replace()
      cycle.replace()
    
      J = Functional(name = "J")
      J.assign(T_n[0] * T_n[0] * dx)
    
      return J
      
    start_manager()
    J = forward(T_0)
    stop_manager()

    J_val = J.value()
    self.assertEqual(J_val, 9.8320117858590805e-01 ** 2)

    dJ = compute_gradient(J, T_0)    
    min_order = taylor_test(forward, T_0, J_val = J_val, dJ = dJ)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)
    
    ddJ = Hessian(forward)
    min_order = taylor_test(forward, T_0, J_val = J_val, ddJ = ddJ)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.99)

  def test_AxpySolver(self):    
    reset("memory")
    clear_caches()
    stop_manager()

    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "R", 0)
    test, trial = TestFunction(space), TrialFunction(space)
    x = Function(space, name = "x", static = True)
    function_assign(x, 1.0)  
    
    def forward(x):
      y = [Function(space, name = "y_%i" % i) for i in range(5)]
      z = [Function(space, name = "z_%i" % i) for i in range(2)]
      function_assign(z[0], 7.0)
    
      AssignmentSolver(x, y[0]).solve(replace = True)
      for i in range(len(y) - 1):
        AxpySolver(y[i], i + 1, z[0], y[i + 1]).solve(replace = True)
      EquationSolver(inner(test, trial) * dx == inner(test, y[-1] * y[-1]) * dx, z[1],
        solver_parameters = {"ksp_type":"cg",
                             "pc_type":"jacobi",
                             "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16}).solve(replace = True)
      
      J = Functional(name = "J")
      J.assign(inner(z[1], z[1]) * dx)
      
      return J
    
    start_manager()
    J = forward(x)
    stop_manager()
    
    J_val = J.value()
    self.assertAlmostEqual(J_val, 25411681.0, places = 7)
    
    dJ = compute_gradient(J, x)    
    dm = Function(space, name = "dm")
    function_assign(dm, 1.0)
    min_order = taylor_test(forward, x, J_val = J_val, dJ = dJ, dm = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)

  def test_AssignmentSolver(self):
    reset("memory")
    clear_caches()
    stop_manager()
  
    mesh = UnitIntervalMesh(20)
    space = FunctionSpace(mesh, "R", 0)
    test, trial = TestFunction(space), TrialFunction(space)
    x = Function(space, name = "x", static = True)
    function_assign(x, 16.0)  
    
    def forward(x):
      y = [Function(space, name = "y_%i" % i) for i in range(9)]
      z = Function(space, name = "z")
    
      AssignmentSolver(x, y[0]).solve(replace = True)
      for i in range(len(y) - 1):
        AssignmentSolver(y[i], y[i + 1]).solve(replace = True)
      EquationSolver(inner(test, trial) * dx == inner(test, y[-1] * y[-1]) * dx, z,
        solver_parameters = {"ksp_type":"cg",
                             "pc_type":"jacobi",
                             "ksp_rtol":1.0e-14, "ksp_atol":1.0e-16}).solve(replace = True)

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
    self.assertEqual(J_val, 66048.0)
    self.assertEqual(K_val, 65536.0)
    
    dJs = compute_gradient([J, K], x)    
    dm = Function(space, name = "dm")
    function_assign(dm, 1.0)
    min_order = taylor_test(lambda x : forward(x)[0], x, J_val = J_val, dJ = dJs[0], dm = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)
    min_order = taylor_test(lambda x : forward(x)[1], x, J_val = K_val, dJ = dJs[1], dm = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 2.00)

    ddJ = Hessian(lambda m : forward(m)[0])
    min_order = taylor_test(lambda x : forward(x)[0], x, J_val = J_val, ddJ = ddJ, dm = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreater(min_order, 3.00)
    
if __name__ == "__main__":
  numpy.random.seed(1201)
  unittest.main()

#  tests().test_AssignmentSolver()
#  tests().test_AxpySolver()
#  tests().test_second_order_adjoint()
#  tests().test_recursive_tlm()
#  tests().test_bc()
#  tests().test_replace()
#  tests().test_higher_order_adjoint()
#  tests().test_FixedPointSolver()
