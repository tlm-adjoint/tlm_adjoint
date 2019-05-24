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

from .backend import *
from .backend_code_generator_interface import *

import copy
import ufl
import weakref

__all__ = \
  [
    "AssemblyCache",
    "Cache",
    "CacheException",
    "CacheRef",
    "Constant",
    "DirichletBC",
    "Function",
    "LinearSolverCache",
    "ReplacementFunction",
    "assembly_cache",
    "bcs_is_cached",
    "bcs_is_static",
    "form_dependencies",
    "form_neg",
    "function_caches",
    "function_is_cached",
    "function_is_checkpointed",
    "function_is_static",
    "function_tlm_depth",
    "is_cached",
    "is_function",
    "linear_solver",
    "linear_solver_cache",
    "new_count",
    "replaced_form",
    "replaced_function",
    "set_assembly_cache",
    "set_linear_solver_cache",
    "split_action",
    "split_form"
  ]

class CacheException(Exception):
  pass
  
class Constant(backend_Constant):
  def __init__(self, *args, **kwargs):
    kwargs = copy.copy(kwargs)
    static = kwargs.pop("static", False)
    cache = kwargs.pop("cache", static)
    
    backend_Constant.__init__(self, *args, **kwargs)
    self.__static = static
    self.__cache = cache
  
  def is_static(self):
    return self.__static
  
  def is_cached(self):
    return self.__cache
  
class Function(backend_Function):
  def __init__(self, *args, **kwargs):
    kwargs = copy.copy(kwargs)
    static = kwargs.pop("static", False)
    cache = kwargs.pop("cache", static)
    checkpoint = kwargs.pop("checkpoint", not static)
    tlm_depth = kwargs.pop("tlm_depth", 0)
    
    self.__static = static
    self.__cache = cache
    self.__checkpoint = checkpoint
    self.__tlm_depth = tlm_depth
    backend_Function.__init__(self, *args, **kwargs)
    self.__caches = FunctionCaches(self)
  
  def is_static(self):
    return self.__static
  
  def is_cached(self):
    return self.__cache
  
  def is_checkpointed(self):
    return self.__checkpoint
  
  def tlm_depth(self):
    return self.__tlm_depth
  
  def new_tlm(self, name = None):
    if self.is_static():
      return None
    else:
      return Function(self.function_space(), name = name, static = False,
        cache = self.is_cached(), checkpoint = self.is_checkpointed(),
        tlm_depth = self.tlm_depth() + 1)
  
  def caches(self):
    return self.__caches

class DirichletBC(backend_DirichletBC):
  def __init__(self, *args, **kwargs):      
    kwargs = copy.copy(kwargs)
    static = kwargs.pop("static", False)
    cache = kwargs.pop("cache", static)
    homogeneous = kwargs.pop("homogeneous", False)
    
    backend_DirichletBC.__init__(self, *args, **kwargs)    
    self.__static = static
    self.__cache = cache
    self.__homogeneous = homogeneous
  
  def is_static(self):
    return self.__static
  
  def is_cached(self):
    return self.__cache
  
  def is_homogeneous(self):
    return self.__homogeneous
  
  def homogenize(self):
    if not self.__homogeneous:
      backend_DirichletBC.homogenize(self)
      self.__homogeneous = True

def is_cached(e):
  for c in ufl.algorithms.extract_coefficients(e):
    if not hasattr(c, "is_cached") or not c.is_cached():
      return False
  return True

def function_is_static(x):
  return x.is_static() if hasattr(x, "is_static") else False

def function_is_cached(x):
  return x.is_cached() if hasattr(x, "is_cached") else False

def function_is_checkpointed(x):
  return x.is_checkpointed() if hasattr(x, "is_checkpointed") else True

def function_tlm_depth(x):
  return x.tlm_depth() if hasattr(x, "tlm_depth") else 0

def bcs_is_static(bcs):
  for bc in bcs:
    if not hasattr(bc, "is_static") or not bc.is_static():
      return False
  return True

def bcs_is_cached(bcs):
  for bc in bcs:
    if not hasattr(bc, "is_cached") or not bc.is_cached():
      return False
  return True

def split_form(form):
  def sum_terms(*terms):
    sum = ufl.classes.Zero()
    for term in terms:
      sum += term
    return sum
  def expand(terms):
    new_terms = []
    for term in terms:
      if isinstance(term, ufl.classes.Sum):
        new_terms += expand(term.ufl_operands)
      else:
        new_terms.append(term)
    return new_terms

  cached_integrals, non_cached_integrals = [], []
  
  for integral in form.integrals():
    cached_operands, non_cached_operands = [], []
    for operand in expand([integral.integrand()]):
      if is_cached(operand):
        cached_operands.append(operand)
      else:
        non_cached_operands.append(operand)
    if len(cached_operands) > 0:
      cached_integrals.append(integral.reconstruct(integrand = sum_terms(*cached_operands)))
    if len(non_cached_operands) > 0:
      non_cached_integrals.append(integral.reconstruct(integrand = sum_terms(*non_cached_operands)))
  
  cached_form = ufl.classes.Form(cached_integrals)
  non_cached_form = ufl.classes.Form(non_cached_integrals)
  
  return cached_form, non_cached_form

def form_simplify_sign(form, sign = None):
  integrals = []
  
  for integral in form.integrals():
    integrand = integral.integrand()
    
    integral_sign = sign
    while isinstance(integrand, ufl.classes.Product):
      a, b = integrand.ufl_operands
      if isinstance(a, ufl.classes.IntValue) and a == -1:
        integral_sign = -1 if integral_sign is None else -integral_sign
        integrand = b
      elif isinstance(b, ufl.classes.IntValue) and b == -1:
        integral_sign = -1 if integral_sign is None else -integral_sign
        integrand = a
      else:
        break
    if not integral_sign is None:
      integral = integral.reconstruct(integrand = -integrand if integral_sign < 0 else integrand)
      
    integrals.append(integral)

  return ufl.classes.Form(integrals)

def form_neg(form):
  return form_simplify_sign(form, sign = -1)
  
def split_action(form, x):
  if len(form.arguments()) != 1:
    # Not a linear form
    return ufl.classes.Form([]), form
  
  if not x in form.coefficients():
    # No dependence on x
    return ufl.classes.Form([]), form
  
  trial = TrialFunction(x.function_space())
  if x in ufl.algorithms.expand_derivatives(ufl.derivative(form, x, argument = trial)).coefficients():
    # Non-linear
    return ufl.classes.Form([]), form
  
  try:
    lhs, rhs = ufl.system(ufl.replace(form, {x:trial}))
  except ufl.UFLException:
    # UFL error encountered
    return ufl.classes.Form([]), form
  
  if not is_cached(lhs):
    # Non-cached bi-linear form
    return ufl.classes.Form([]), form
  
  # Success
  return form_simplify_sign(lhs), form_neg(rhs)
  
def parameters_key(parameters):
  key = []
  for name in sorted(parameters.keys()):
    sub_parameters = parameters[name]
    if isinstance(sub_parameters, (Parameters, dict)):
      key.append((name, parameters_key(sub_parameters)))
    elif isinstance(sub_parameters, list):
      key.append((name, tuple(sub_parameters)))
    else:
      key.append((name, sub_parameters))
  return tuple(key)
  
class CacheRef:
  def __init__(self, value = None):
    self._value = value
    
  def __call__(self):
    return self._value
  
  def _clear(self):
    self._value = None

class FunctionCaches:
  def __init__(self, x):
    self._caches = {}
    self._id = x.id()
    
  def __len__(self):
    return len(self._caches)

  def clear(self):
    for cache in tuple(self._caches.values()):
      cache = cache()
      cache.clear(self._id)
      assert(not cache.id() in self._caches)
  
  def add(self, cache):
    cache_id = cache.id()
    if not cache_id in self._caches:
      self._caches[cache_id] = weakref.ref(cache)
    
  def remove(self, cache):
    del(self._caches[cache.id()])

def function_caches(x):
  if hasattr(x, "caches"):
    return x.caches()
  else:
    if not hasattr(x, "_tlm_adjoint__caches"):
      x._tlm_adjoint__caches = FunctionCaches(x)
    return x._tlm_adjoint__caches

Cache_id_counter = [0]
class Cache:
  def __init__(self):
    self._cache = {}
    self._deps_map = {}
    self._dep_caches = {}
    
    self._id = Cache_id_counter[0]
    Cache_id_counter[0] += 1
  
  def __del__(self):
    for value in self._cache.values():
      value._clear()
    for dep_caches in self._dep_caches.values():
      dep_caches = dep_caches()
      if not dep_caches is None:
        dep_caches.remove(self)
  
  def __len__(self):
    return len(self._cache)
  
  def id(self):
    return self._id
  
  def clear(self, *deps):
    if len(deps) == 0:
      for value in self._cache.values():
        value._clear()
      self._cache.clear()
      self._deps_map.clear()
      for dep_caches in self._dep_caches.values():
        dep_caches = dep_caches()
        if not dep_caches is None:
          dep_caches.remove(self)
      self._dep_caches.clear()
    else:
      for dep in deps:
        dep_id = dep if isinstance(dep, int) else dep.id()
        del(dep)
        if dep_id in self._deps_map:
          for key, dep_ids in self._deps_map[dep_id].items():
            self._cache[key]._clear()
            del(self._cache[key])
            for dep_id2 in dep_ids:
              if dep_id2 != dep_id:
                del(self._deps_map[dep_id2][key])
                if len(self._deps_map[dep_id2]) == 0:
                  del(self._deps_map[dep_id2])
                  dep_caches = self._dep_caches[dep_id2]()
                  if not dep_caches is None:
                    dep_caches.remove(self)
                  del(self._dep_caches[dep_id2])
          del(self._deps_map[dep_id])
          dep_caches = self._dep_caches[dep_id]()
          if not dep_caches is None:
            dep_caches.remove(self)
          del(self._dep_caches[dep_id])
  
  def add(self, key, value, deps = []):
    if key in self._cache:
      raise CacheException("Duplicate key")
    value = CacheRef(value)
    dep_ids = tuple(dep.id() for dep in deps)
    
    self._cache[key] = value
    
    for dep, dep_id in zip(deps, dep_ids):      
      if dep_id in self._deps_map:
        self._deps_map[dep_id][key] = dep_ids
      else:
        self._deps_map[dep_id] = {key:dep_ids}
        
      dep_caches = function_caches(dep)
      dep_caches.add(self)
      self._dep_caches[dep_id] = weakref.ref(dep_caches)
    
    return value
  
  def get(self, key, default = None):
    return self._cache.get(key, default)

def new_count():
  return Constant(0).count()
  
class ReplacementFunction(ufl.classes.Coefficient):
  def __init__(self, x):
    ufl.classes.Coefficient.__init__(self, x.function_space(), count = new_count())
    self.__space = x.function_space()
    self.__id = x.id()
    self.__name = x.name()
    self.__static = function_is_static(x)
    self.__cache = function_is_cached(x)
    self.__checkpoint = function_is_checkpointed(x)
    self.__tlm_depth = function_tlm_depth(x)
    self.__caches = function_caches(x)
  
  def function_space(self):
    return self.__space
  
  def id(self):
    return self.__id
  
  def name(self):
    return self.__name
  
  def is_static(self):
    return self.__static
  
  def is_cached(self):
    return self.__cache
  
  def is_checkpointed(self):
    return self.__checkpoint
  
  def tlm_depth(self):
    return self.__tlm_depth
  
  def caches(self):
    return self.__caches

def replaced_function(x):
  if isinstance(x, ReplacementFunction):
    return x
  if not hasattr(x, "_tlm_adjoint__ReplacementFunction"):
    x._tlm_adjoint__ReplacementFunction = ReplacementFunction(x)
  return x._tlm_adjoint__ReplacementFunction

def replaced_form(form):
  replace_map = {}
  for c in form.coefficients():
    if isinstance(c, backend_Function):
      replace_map[c] = replaced_function(c)
  return ufl.replace(form, replace_map)

def is_function(x):
  return isinstance(x, backend_Function)

def form_dependencies(form):
  deps = []
  dep_ids = set()
  for dep in form.coefficients():
    if is_function(dep):
      dep_id = dep.id()
      if not dep_id in dep_ids:
        deps.append(dep)
        dep_ids.add(dep_id)
  return deps

def form_key(form):
  return ufl.algorithms.expand_indices(ufl.algorithms.expand_compounds(ufl.algorithms.expand_derivatives(replaced_form(form))))

def assemble_key(form, bcs, form_compiler_parameters):  
  return (form_key(form), tuple(bcs), parameters_key(form_compiler_parameters))

class AssemblyCache(Cache):
  def assemble(self, form, bcs = [], form_compiler_parameters = {}, replace_map = None):  
    key = assemble_key(form, bcs, form_compiler_parameters)
    value = self.get(key, None)
    if value is None:
      assemble_form = form if replace_map is None else ufl.replace(form, replace_map)
      rank = len(form.arguments())
      if rank == 0:
        if len(bcs) > 0:
          raise CacheException("Unexpected boundary conditions for rank 0 form")
        b = assemble(assemble_form, form_compiler_parameters = form_compiler_parameters)
      elif rank == 1:
        b = assemble(assemble_form, form_compiler_parameters = form_compiler_parameters)
        for bc in bcs:
          bc.apply(b)
      elif rank == 2:
        b = assemble_matrix(assemble_form, bcs, form_compiler_parameters, force_evaluation = True)
      else:
        raise CacheException("Unexpected form rank %i" % rank)
      value = self.add(key, b, deps = form_dependencies(form))
    else:
      b = value()
      
    return value, b

def linear_solver_key(form, bcs, linear_solver_parameters, form_compiler_parameters):
  return (form_key(form), tuple(bcs), parameters_key(linear_solver_parameters), parameters_key(form_compiler_parameters))

class LinearSolverCache(Cache):
  def linear_solver(self, form, A, bcs = [], linear_solver_parameters = {}, form_compiler_parameters = {}):
    key = linear_solver_key(form, bcs, linear_solver_parameters, form_compiler_parameters)
    value = self.get(key, None)
    if value is None:
      solver = linear_solver(A, linear_solver_parameters)
      value = self.add(key, solver, deps = form_dependencies(form))
    else:
      solver = value()

    return value, solver

_caches = [AssemblyCache(), LinearSolverCache()]
def assembly_cache():
  return _caches[0]
def set_assembly_cache(assembly_cache):
  _caches[0] = assembly_cache
def linear_solver_cache():
  return _caches[1]
def set_linear_solver_cache(linear_solver_cache):
  _caches[1] = linear_solver_cache
