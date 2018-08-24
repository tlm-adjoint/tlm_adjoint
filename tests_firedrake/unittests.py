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
  def test_AssignmentSolver(self):
    reset("memory")
    clear_caches()
    stop_manager()
  
    mesh = UnitIntervalMesh(1)
    space = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    test, trial = TestFunction(space), TrialFunction(space)
    x = Function(space, name = "x", static = True)
    function_assign(x, 16.0)  
    
    def forward(x):
      y = [Function(space, name = "y_%i" % i) for i in range(9)]
      z = Function(space, name = "z")
    
      AssignmentSolver(x, y[0]).solve(replace = True)
      for i in range(len(y) - 1):
        AssignmentSolver(y[i], y[i + 1]).solve(replace = True)
      EquationSolver(inner(test, trial) * dx == inner(test, y[-1] * y[-1]) * dx, z).solve(replace = True)

      J = Functional(name = "J")
      J_0 = Function(space)
      J_1 = Function(space)
      EquationSolver(inner(test, trial) * dx == inner(test, z * z) * dx,
        J_0).solve(replace = True)
      EquationSolver(inner(test, trial) * dx == inner(test, 2 * x * x) * dx,
        J_1).solve(replace = True)
      J.assign(J_0)
      J.addto(J_1)
#      J.assign(inner(z, z) * dx)
#      J.addto(2 * inner(x, x) * dx)
      
      K = Functional(name = "K")
      K_0 = Function(space)
      EquationSolver(inner(test, trial) * dx == inner(test, z * z) * dx,
        K_0).solve(replace = True)
      K.assign(K_0)
#      K.assign(inner(z, z) * dx)

      return J, K
    
    start_manager()
    J, K = forward(x)
    stop_manager()
    
    J_val = J.value()
    K_val = K.value()
    self.assertAlmostEqual(J_val, 66048.0, places = 10)
    self.assertAlmostEqual(K_val, 65536.0, places = 10)
    
    dJs = compute_gradient([J, K], x)    
    dm = Function(space, name = "dm")
    function_assign(dm, 1.0)
    min_order = taylor_test(lambda x : forward(x)[0], x, J_val = J_val, dJ = dJs[0], dm = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreaterEqual(min_order, 2.00)
    min_order = taylor_test(lambda x : forward(x)[1], x, J_val = K_val, dJ = dJs[1], dm = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreaterEqual(min_order, 2.00)

    ddJ = Hessian(lambda m : forward(m)[0])
    min_order = taylor_test(lambda x : forward(x)[0], x, J_val = J_val, ddJ = ddJ, dm = dm)  # Usage as in dolfin-adjoint tests
    self.assertGreaterEqual(min_order, 3.00)
    
if __name__ == "__main__":
  numpy.random.seed(1201)
  unittest.main()

#  tests().test_AssignmentSolver()
