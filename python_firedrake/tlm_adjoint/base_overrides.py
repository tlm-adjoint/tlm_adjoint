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

from .base import *
from .base_interface import warning

from .equations import AssignmentSolver, EquationSolver
from .tlm_adjoint import annotation_enabled, tlm_enabled

import copy

__all__ = \
  [
    "solve"
  ]

# Following Firedrake API, git master revision
# efea95b923b9f3319204c6e22f7aaabb792c9abd

def solve(*args, **kwargs):
  kwargs = copy.copy(kwargs)
  annotate = kwargs.pop("annotate", None)
  tlm = kwargs.pop("tlm", None)
  
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    eq, x, bcs, J, Jp, M, form_compiler_parameters, solver_parameters, \
      nullspace, transpose_nullspace, near_nullspace, options_prefix = \
      firedrake.solving._extract_args(*args, **kwargs)
    if not J is None or not Jp is None:
      raise ManagerException("Custom Jacobians not supported")
    if not M is None:
      raise ManagerException("Adaptive solves not supported")
    if not nullspace is None or not transpose_nullspace is None or not near_nullspace is None:
      raise ManagerException("Nullspaces not supported")
    if not options_prefix is None:
      raise ManagerException("Options prefixes not supported")
    eq = EquationSolver(eq, x, bcs,
      form_compiler_parameters = form_compiler_parameters,
      solver_parameters = solver_parameters)
    eq._pre_annotate(annotate = annotate)
    return_value = base_solve(*args, **kwargs)
    eq._post_annotate(annotate = annotate, replace = True, tlm = tlm)
    return return_value
  else:
    return base_solve(*args, **kwargs)

_orig_Function_assign = base_Function.assign
def _Function_assign(self, expr, subset = None, annotate = None, tlm = None):
  return_value = _orig_Function_assign(self, expr, subset = subset)
  if not isinstance(expr, base_Function) or not subset is None:
    # Only assignment to a Function annotated
    return
  
  if annotate is None:
    annotate = annotation_enabled()
  if tlm is None:
    tlm = tlm_enabled()
  if annotate or tlm:
    eq = AssignmentSolver(expr, self)
    eq._post_annotate(annotate = annotate, replace = True, tlm = tlm)
  return return_value
base_Function.assign = _Function_assign
