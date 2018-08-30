#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from tlm_adjoint import *

import numpy
import unittest
  
class tests(unittest.TestCase):
  def test_ContractionEquation(self):
    reset("memory")
    clear_caches()
    stop_manager()
    
    space_0 = FunctionSpace(1)
    space = FunctionSpace(3)
    A = numpy.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]], dtype = numpy.float64)
    
    def forward(m):
      x = Function(space, name = "x")
      ContractionEquation(A, (1,), (m,), x).solve(replace = True)
      
      norm_sq = Function(space_0, name = "norm_sq")
      NormSqEquation(x, norm_sq).solve(replace = True)
      
      J = Functional(name = "J")
      NormSqEquation(norm_sq, J.fn()).solve(replace = True)
      
      return x, J
    
    m = Function(space, name = "m", static = True)
    function_set_values(m, numpy.array([7.0, 8.0, 9.0], dtype = numpy.float64))
    
    start_manager()
    x, J = forward(m)
    stop_manager()
    
    self.assertEqual(abs(A.dot(m.vector()) - x.vector()).max(), 0.0)

    dJ = compute_gradient(J, m)
    min_order = taylor_test(lambda m : forward(m)[1], m, J_val = J.value(), dJ = dJ)
    self.assertGreater(min_order, 2.00)
    
    ddJ = Hessian(lambda m : forward(m)[1])
    min_order = taylor_test(lambda m : forward(m)[1], m, J_val = J.value(), dJ = dJ, ddJ = ddJ)
    self.assertGreater(min_order, 3.00)

  def test_InnerProductEquation(self):
    reset("memory")
    clear_caches()
    stop_manager()
    
    space = FunctionSpace(10)
    
    def forward(F):
      G = Function(space, name = "G")
      AssignmentSolver(F, G).solve(replace = True)

      J = Functional(name = "J")
      InnerProductEquation(F, G, J.fn()).solve(replace = True)
      
      return J

    F = Function(space, name = "F", static = True)
    function_set_values(F, numpy.random.random(function_local_size(F)))

    start_manager()
    J = forward(F)
    stop_manager()

    dJ = compute_gradient(J, F)
    min_order = taylor_test(forward, F, J_val = J.value(), dJ = dJ)
    self.assertGreater(min_order, 1.99)

  def test_SumEquation(self):
    reset("memory")
    clear_caches()
    stop_manager()
    
    space = FunctionSpace(10)
    
    def forward(F):
      G = Function(space, name = "G")
      AssignmentSolver(F, G).solve(replace = True)

      J = Functional(name = "J")
      SumEquation(G, J.fn()).solve(replace = True)
      
      return J

    F = Function(space, name = "F", static = True)
    function_set_values(F, numpy.random.random(function_local_size(F)))

    start_manager()
    J = forward(F)
    stop_manager()
    
    self.assertEqual(J.value(), F.vector().sum())

    dJ = compute_gradient(J, F)
    self.assertEqual(abs(function_get_values(dJ) - 1.0).max(), 0.0)
    
if __name__ == "__main__":
  numpy.random.seed(1201)
  unittest.main()

#  tests().test_ContractionEquation()
#  tests().test_InnerProductEquation()
#  tests().test_SumEquation()
