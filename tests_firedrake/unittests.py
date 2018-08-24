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
  def test_second_order_adjoint(self):    
    n_steps = 20
    reset("multistage", {"blocks":n_steps, "snaps_on_disk":4, "snaps_in_ram":2, "verbose":True})
    clear_caches()
    stop_manager()
  
    mesh = UnitIntervalMesh(20)
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
    self.assertAlmostEqual(J_val, 9.8320117858590805e-01 ** 2, places = 15)

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
    self.assertAlmostEqual(J_val, 66048.0, places = 16)
    self.assertAlmostEqual(K_val, 65536.0, places = 16)
    
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
