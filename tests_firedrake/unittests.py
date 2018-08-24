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
      AssignmentSolver(y[-1], z).solve(replace = True)
      J = Functional(name = "J")
      J.assign(z)

      return J
    
    start_manager()
    J = forward(x)
    stop_manager()
    
    J_val = J.value()
    
    dJs = compute_gradient(J, x)    
    dm = Function(space, name = "dm")
    function_assign(dm, 1.0)
    min_order = taylor_test(lambda x : forward(x), x, J_val = J_val, dJ = dJs, dm = dm)  # Usage as in dolfin-adjoint tests
#    self.assertGreaterEqual(min_order, 2.00)

    ddJ = Hessian(lambda m : forward(m))
    min_order = taylor_test(lambda x : forward(x), x, J_val = J_val, ddJ = ddJ, dm = dm)  # Usage as in dolfin-adjoint tests
#    self.assertGreaterEqual(min_order, 3.00)
    
if __name__ == "__main__":
  numpy.random.seed(1201)
  unittest.main()

  tests().test_AssignmentSolver()
