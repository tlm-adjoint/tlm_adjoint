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

from .backend_interface import *

from .base_equations import *
from .manager import manager as _manager

__all__ = \
  [
    "Functional"
  ]

class Functional:
  def __init__(self, fn = None, space = None, name = None):
    """
    A functional.
    
    Arguments:
    
    fn:  (Optional) The Function storing the functional value. Replaced by a new
         Function by subsequent assign or addto calls.
    If fn is supplied:
      space:  (Optional) The FunctionSpace for the functional. Default
              fn.function_space().
      name:   (Optional) The name of the functional. Default fn.name().
    If fn is not supplied:
      space:  (Optional) The FunctionSpace for the functional. Default
              RealFunctionSpace().
      name:   (Optional) The name of the functional. Default "Functional".
    """
    
    if fn is None:
      if space is None:
        space = RealFunctionSpace()
      if name is None:
        name = "Functional"
    else:
      if space is None:
        space = fn.function_space()
      if name is None:
        name = fn.name()
    
    self._space = space
    self._name = name
    self._fn = fn
  
  def assign(self, term, manager = None, annotate = None, tlm = None):
    """
    Assign the functional.
    
    Arguments:
    
    term      A form or Function, to which the functional is assigned.
    manager   (Optional) The equation manager.
    annotate  (Optional) Whether the equation should be annotated.
    tlm       (Optional) Whether to derive (and solve) an associated
              tangent-linear equation.
    """
  
    if manager is None:
      manager = _manager()

    if self._fn is None:
      new_fn = function_new(term, name = self._name) if is_function(term) else Function(self._space, name = self._name)
    else:
      new_fn = function_new(self._fn, name = self._name)
    if is_function(term):
      AssignmentSolver(term, new_fn).solve(manager = manager, annotate = annotate, replace = True, tlm = tlm)
    else:
      from .equations import AssembleSolver
      AssembleSolver(term, new_fn).solve(manager = manager, annotate = annotate, replace = True, tlm = tlm)
    self._fn = new_fn
   
  def addto(self, term = None, manager = None, annotate = None, tlm = None):    
    """
    Add to the functional.
    
    Arguments:
    
    term      (Optional) A form or Function, which is added to the functional.
              If not supplied then the functional is copied into a new
              function (useful for stepping over blocks with no contribution to
              the functional).
    manager   (Optional) The equation manager.
    annotate  (Optional) Whether the equations should be annotated.
    tlm       (Optional) Whether to derive (and solve) associated tangent-linear
              equations.
    """
      
    if self._fn is None:
      if not term is None:
        self.assign(term, manager = manager, annotate = annotate, tlm = tlm)
      return
    
    if manager is None:
      manager = _manager()
      
    new_fn = function_new(self._fn, name = self._name)
    if term is None:
      AssignmentSolver(self._fn, new_fn).solve(manager = manager, annotate = annotate, replace = True, tlm = tlm)
    else:
      if is_function(term):
        term_fn = term
      else:
        term_fn = function_new(self._fn, name = "%s_term" % self._name)
        from .equations import AssembleSolver
        AssembleSolver(term, term_fn).solve(manager = manager, annotate = annotate, replace = True, tlm = tlm)
      AxpySolver(self._fn, 1.0, term_fn, new_fn).solve(manager = manager, annotate = annotate, replace = True, tlm = tlm)
    self._fn = new_fn
  
  def fn(self):
    """
    Return the Function storing the functional value.
    """
  
    if self._fn is None:
      self._fn = Function(self._space, name = self._name)
    return self._fn
  
  def function_space(self):
    """
    Return the FunctionSpace for the functional.
    """
    
    return self._space
  
  def value(self):
    """
    Return the value of the functional.
    """
  
    return 0.0 if self._fn is None else function_max_value(self._fn)
    
  def tlm(self, M, dM, manager = None): 
    """
    Return a Functional associated with evaluation of the tangent-linear of the
    functional.
    """
   
    if manager is None:
      manager = _manager()      
    return Functional(fn = manager.tlm(M, dM, self.fn()))
