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

from .base_equations import *

from collections import OrderedDict
import copy
import numpy

__all__ = \
  [     
    "ConstantMatrix",
    
    "ContractionEquation",
    "ContractionRHS",
    "IdentityRHS",
    "InnerProductEquation",
    "InnerProductRHS",
    "NormSqEquation",
    "NormSqRHS",
    "SumRHS",
    
    "SumEquation"
  ]

class IdentityRHS(RHS):
  def __init__(self, x):
    RHS.__init__(self, [x], nl_deps = [])

  def add_forward(self, b, deps):
    x, = deps
    b.vector()[:] += x.vector()

  def subtract_adjoint_derivative_action(self, b, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      b.vector()[:] -= adj_x.vector()

  def tangent_linear_rhs(self, M, dM, tlm_map):
    x, = self.dependencies()
    tau_x = None
    for m, dm in zip(M, dM):
      if m == x:
        tau_x = dm
    if tau_x is None:
      tau_x = tlm_map[x]
      
    if tau_x is None:
      return None
    else: 
      return IdentityRHS(tau_x)

class ConstantMatrix(Matrix):
  def __init__(self, A, A_T = None, has_ic_dep = False):
    Matrix.__init__(self, nl_deps = [], has_ic_dep = has_ic_dep)
    self._A = A
    self._A_T = A_T
  
  def A(self):
    return self._A
  
  def A_T(self):
    return self._A.T if self._A_T is None else self._A_T
  
  def forward_action(self, nl_deps, x, b, method = "assign"):
    getattr(b.vector()[:], {"assign":"__assign__", "add":"__iadd__", "sub":"__isub__"}[method])(self._A.dot(x.vector()))
  
  def adjoint_action(self, nl_deps, adj_x, b, b_index = 0, method = "assign"):
    if b_index != 0: raise EquationException("Invalid index")
    getattr(b.vector()[:], {"assign":"__assign__", "add":"__iadd__", "sub":"__isub__"}[method])(self._A_T().dot(x.vector()))
    
  def forward_solve(self, nl_deps, b):
    return Function(b.function_space(), _data = numpy.linalg.solve(self._A, b.vector()))
  
  def adjoint_derivative_action(self, nl_deps, nl_dep_index, x, adj_x, b, method = "assign"):
    return
  
  def adjoint_solve(self, nl_deps, b):
    return Function(b.function_space(), _data = numpy.linalg.solve(self.A_T(), b.vector()))
    
  def tangent_linear_rhs(self, M, dM, tlm_map, x):
    return None

class SumRHS(RHS):
  def __init__(self, x):
    RHS.__init__(self, [x], nl_deps = [])
    
  def add_forward(self, b, deps):
    y, = deps
    b.vector()[:] += y.vector().sum()
    
  def subtract_adjoint_derivative_action(self, b, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      b.vector()[:] -= adj_x.vector().sum()
      
  def tangent_linear_rhs(self, M, dM, tlm_map):
    y, = self.dependencies()
    
    tau_y = None
    for i, m in enumerate(M):
      if m == y:
        tau_y = dM[i]
    if tau_y is None:
      tau_y = tlm_map[y]
    
    if tau_y is None:
      return None
    else:
      return SumRHS(tau_y)

class SumEquation(LinearEquation):
  def __init__(self, y, x):
    LinearEquation.__init__(self, SumRHS(y), x)

class InnerProductRHS(RHS):
  def __init__(self, y, z, alpha = 1.0, M = None):
    RHS.__init__(self, [y, z], nl_deps = [y, z])
    self._alpha = alpha = float(alpha)
    self._M = M
    if M is None:
      if alpha == 1.0:
        self._dot = lambda x : x
      else:
        self._dot = lambda x : alpha * x
    elif alpha == 1.0:
      self._dot = lambda x : M.dot(x)
    else:
      self._dot = lambda x : alpha * M.dot(x)
    
  def add_forward(self, b, deps):
    y, z = deps
    b.vector()[:] += y.vector().dot(self._dot(z.vector()))
    
  def subtract_adjoint_derivative_action(self, b, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      b.vector()[:] -= adj_x.vector().sum() * self._dot(nl_deps[1].vector())
    elif dep_index == 1:
      b.vector()[:] -= adj_x.vector().sum() * self._dot(nl_deps[0].vector())
      
  def tangent_linear_rhs(self, M, dM, tlm_map):
    y, z = self.dependencies()
    
    tau_y = None
    tau_z = None
    for i, m in enumerate(M):
      if m == y:
        tau_y = dM[i]
      elif m == z:
        tau_z = dM[i]
    if tau_y is None:
      tau_y = tlm_map[y]
    if tau_z is None:
      tau_z = tlm_map[z]
    
    tlm_B = []
    if not tau_y is None:
      if tau_y == z:
        tlm_B.append(NormSqRHS(tau_y, alpha = self._alpha, M = self._M))
      else:
        tlm_B.append(InnerProductRHS(tau_y, z, alpha = self._alpha, M = self._M))
    if not tau_z is None:
      if y == tau_z:
        tlm_B.append(NormSqRHS(tau_z, alpha = self._alpha, M = self._M))
      else:
        tlm_B.append(InnerProductRHS(y, tau_z, alpha = self._alpha, M = self._M))
    return tlm_B

class InnerProductEquation(LinearEquation):
  def __init__(self, y, z, x, alpha = 1.0, M = None):
    LinearEquation.__init__(self, InnerProductRHS(y, z, alpha = alpha, M = M), x)

class NormSqRHS(RHS):
  def __init__(self, y, alpha = 1.0, M = None):
    RHS.__init__(self, [y], nl_deps = [y])
    self._alpha = alpha = float(alpha)
    self._M = M
    if M is None:
      if alpha == 1.0:
        self._dot = lambda x : x
      else:
        self._dot = lambda x : alpha * x
    elif alpha == 1.0:
      self._dot = lambda x : M.dot(x)
    else:
      self._dot = lambda x : alpha * M.dot(x)
    
  def add_forward(self, b, deps):
    y, = deps
    b.vector()[:] += y.vector().dot(self._dot(y.vector()))
    
  def subtract_adjoint_derivative_action(self, b, nl_deps, dep_index, adj_x):
    if dep_index == 0:
      b.vector()[:] -= 2.0 * adj_x.vector().sum() * self._dot(nl_deps[0].vector())
      
  def tangent_linear_rhs(self, M, dM, tlm_map):
    y, = self.dependencies()
    
    tau_y = None
    for i, m in enumerate(M):
      if m == y:
        tau_y = dM[i]
    if tau_y is None:
      tau_y = tlm_map[y]
    
    if tau_y is None:
      return None
    else:
      return InnerProductRHS(tau_y, y, alpha = 2.0 * self._alpha, M = self._M)

class NormSqEquation(LinearEquation):
  def __init__(self, y, x, alpha = 1.0, M = None):
    LinearEquation.__init__(self, NormSqRHS(y, alpha = alpha, M = M), x)

class ContractionArray:
  def __init__(self, A, I, A_T = None, alpha = 1.0):  
    for i in range(len(I) - 1):
      if I[i + 1] <= I[i]:
        raise EquationException("Axes must be in ascending order")
    self._A = A
    self._A_T = A_T
    self._I = tuple(I)
    self._alpha = float(alpha)
  
  def A(self):
    return self._A
  
  def A_T(self):
    return self._A.T if self._A_T is None else self._A_T
  
  def I(self):
    return self._I
    
  def alpha(self):
    return self._alpha
  
  def value(self, X):
    if len(self._A.shape) == 2 and self._I == (0,):
      v = self.A_T().dot(X[0].vector())
    elif len(self._A.shape) == 2 and self._I == (1,):
      v = self._A.dot(X[0].vector())
    else:
      v = self._A
      for i, (j, x) in enumerate(zip(self._I, X)):
        v = numpy.tensordot(v, x.vector(), axes = ((j - i,), (0,)))
    if self._alpha != 1.0:
      if len(X) > 0:
        v *= self._alpha
      else:
        v = v * self._alpha
    return v
    
class ContractionRHS(RHS):
  def __init__(self, A, I, Y, A_T = None, alpha = 1.0):
    if len(Y) != len(A.shape) - 1:
      raise EquationException("Contraction does not result in a vector")
    j = set(range(len(A.shape)))
    for i in I:
      j.remove(i)  # Raises an error if there is a duplicate element in I
    assert(len(j) == 1)
    j = j.pop()
    
    RHS.__init__(self, Y, nl_deps = [] if len(Y) == 1 else Y)
    self._A_T = A_T
    self._c = ContractionArray(A, I, A_T = A_T, alpha = alpha)
    self._j = j

  def add_forward(self, b, deps):
    b.vector()[:] += self._c.value(deps)

  def subtract_adjoint_derivative_action(self, b, nl_deps, dep_index, adj_x):
    if dep_index < len(self._c.I()):
      A, I, alpha = self._c.A(), self._c.I(), self._c.alpha()
      Y = [None for i in range(len(A.shape))]
      k = I[dep_index]
      for j, (i, nl_dep) in enumerate(zip(I, nl_deps)):
        if j != dep_index:
          Y[i] = nl_dep
      Y[self._j] = adj_x
      
      b.vector()[:] -= ContractionArray(A, list(range(k)) + list(range(k + 1, len(A.shape))), A_T = self._A_T, alpha = alpha).value(Y[:k] + Y[k + 1:])

  def tangent_linear_rhs(self, M, dM, tlm_map):
    Y = list(self.dependencies())
    A, I, alpha = self._c.A(), self._c.I(), self._c.alpha()
    m_map = OrderedDict([(m, dm) for m, dm in zip(M, dM)])
    
    J = []
    for j, y in enumerate(Y):
      if y in m_map:
        Y[j] = m_map[y]
      else:
        J.append(j)

    if len(J) == 0:
      return ContractionRHS(A, I, Y, A_T = self._A_T, alpha = alpha)
    else:
      tlm_B = []
      for j in J:
        tau_y = tlm_map[Y[j]]
        if not tau_y is None:
          tlm_B.append(ContractionRHS(A, I, Y[:j] + [tau_y] + Y[j + 1:], A_T = self._A_T, alpha = alpha))
      return tlm_B

class ContractionEquation(LinearEquation):
  def __init__(self, A, I, Y, x, A_T = None, alpha = 1.0):
    LinearEquation.__init__(self, ContractionRHS(A, I, Y, A_T = A_T, alpha = alpha), x)
