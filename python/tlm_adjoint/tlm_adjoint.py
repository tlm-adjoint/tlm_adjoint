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
from .manager import manager as _manager, set_manager

from collections import OrderedDict, defaultdict
import copy
import numpy
import pickle
import os
import zlib

__all__ = \
  [
    "Checkpoint",
    "Control",
    "EquationManager",
    "ManagerException",
    "add_tlm",
    "annotation_enabled",
    "compute_gradient",
    "configure_checkpointing",
    "manager_info",
    "minimize_scipy",
    "new_block",
    "reset",
    "start_manager",
    "stop_manager",
    "taylor_test",
    "tlm",
    "tlm_depth",
    "tlm_enabled"
  ]

class ManagerException(Exception):
  pass

class Control:
  def __init__(self, m, manager = None):
    if manager is None:
      manager = _manager()
  
    if isinstance(m, str):
      m = manager.find_initial_condition(m)

    self._m = m
  
  def m(self):
    return self._m
  
class Checkpoint:
  def __init__(self,
               ics_keys = [], ics_values = [], deps_keys = [], deps_values = [], data_keys = [], data_values = [],
               checkpoint_ics = True, checkpoint_data = True):
    assert(len(ics_keys) == len(ics_values))
    assert(len(deps_keys) == len(deps_values))
    assert(len(data_keys) == len(data_values))

    self._seen_ics = set()
    self._ics = OrderedDict()
    self._deps = {}
    self._data = OrderedDict()
    self._seen_ics.update(ics_keys)
    self._ics.update(OrderedDict(zip(ics_keys, ics_values)))
    self._deps.update(dict(zip(deps_keys, deps_values)))
    self._data.update(OrderedDict(zip(data_keys, data_values)))
  
    self._indices = defaultdict(lambda : 0)
    self._indices.update(deps_keys)
    
    self.configure(checkpoint_ics, checkpoint_data)
  
  def configure(self, checkpoint_ics = None, checkpoint_data = None):
    """
    Configure storage.
    
    Arguments:
    
    checkpoint_ics   Store initial condition data, used by checkpointing
    checkpoint_data  Store equation non-linear dependency data, used in reverse
                     mode
    """
  
    if not checkpoint_ics is None:
      self._checkpoint_ics = checkpoint_ics
    if not checkpoint_data is None:
      self._checkpoint_data = checkpoint_data
  
  def checkpoint_ics(self):
    return self._checkpoint_ics
  
  def checkpoint_data(self):
    return self._checkpoint_data
  
  def clear(self, clear_ics = True, clear_data = True):
    if clear_ics:
      self._seen_ics.clear()
      self._ics.clear()
    if clear_data:
      self._deps.clear()
      self._data.clear()
      self._indices.clear()

  def __getitem__(self, key):
    return [self._data[dep_key] for dep_key in self._deps[key]]
  
  def has_initial_condition(self, x):
    return x.id() in self._ics
  
  def initial_condition(self, x, copy = True):
    ic = self._ics[x.id()]
    if copy:
      ic = function_copy(ic, static = function_is_static(ic))
    return ic
  
  def initial_conditions(self, copy = True):
    return tuple(self._ics.keys()), tuple((function_copy(ic, static = function_is_static(ic)) for ic in self._ics.values()) if copy else self._ics.values())
  
  def data(self, copy = True):
    return tuple(self._deps.keys()), tuple(self._deps.values()), \
           tuple(self._data.keys()), tuple((function_copy(F, static = function_is_static(F)) for F in self._data.values()) if copy else self._data.values())
  
  def _data_key(self, x):
    x_id = x.id()
    return (x_id, self._indices[x_id])
  
  def add_initial_condition(self, x, value = None, copy = True):  
    if value is None:
      value = x
      
    x_id = x.id()
    if self._checkpoint_ics and not x_id in self._seen_ics:
      self._ics[x_id] = function_copy(value, static = function_is_static(x)) if copy else value
      self._seen_ics.add(x_id)
      if self._checkpoint_data:
        # Optimization: Reference the ic in the data
        x_key = self._data_key(x)
        # It is not expected that x_key is in self._data here
        self._data[x_key] = self._ics[x_id]
  
  def add_equation(self, key, eq, deps = None, nl_deps = None, copy = True):
    eq_X = eq.X()
    eq_deps = eq.dependencies()
    if deps is None:
      deps = eq_deps
      
    for eq_x in eq_X:
      self._indices[eq_x.id()] += 1
    
    if self._checkpoint_ics:
      # Optimization: Since we have solved for x, unless an initial condition
      # for x has already been added, we need not add an initial condition for
      # x, and so we mark x as "seen"
      for eq_x in eq_X:
        self._seen_ics.add(eq_x.id())
      for eq_dep, dep in zip(eq_deps, deps):
        self.add_initial_condition(eq_dep, value = dep, copy = copy)
  
    if self._checkpoint_data:
      dep_keys = []
      for eq_dep, dep in zip(eq.nonlinear_dependencies(), [deps[i] for i in eq.nonlinear_dependencies_map()] if nl_deps is None else nl_deps):
        dep_key = self._data_key(eq_dep)
        if not dep_key in self._data:
          self._data[dep_key] = function_copy(dep, static = function_is_static(eq_dep)) if copy else dep
        dep_keys.append(dep_key)
      self._deps[key] = dep_keys

def tlm_depth(x):
  return getattr(x, "_tlm_adjoint__tlm_depth", 0)
      
class TangentLinearMap:
  """
  A map from forward to tangent-linear variables.
  """

  def __init__(self):
    self.clear()
  
  def __contains__(self, x):
    return x.id() in self._map
  
  def __getitem__(self, x):
    if not x.id() in self._map:
      if function_is_static(x):
        self._map[x.id()] = None
      else:
        basename = getattr(x, "_tlm_adjoint__tlm_basename", x.name())
        depth = tlm_depth(x) + 1
        tlm_x = self._map[x.id()] = function_new(x, name = "%s_tlm_%i" % (basename, depth))
        tlm_x._tlm_adjoint__tlm_basename = basename
        tlm_x._tlm_adjoint__tlm_depth = depth
    return self._map[x.id()]
  
  def __setitem__(self, x, tau):
    self._map[x.id()] = tau
    return tau
  
  def clear(self):
    self._map = OrderedDict()
    
class Ids:
  def __init__(self):
    self._ids = []
    self._next = 0
  
  def next(self):
    try:
      next = self._ids.pop()
    except IndexError:
      next = self._next
      self._next += 1
    return next
  
  def free(self, id):
    assert(id >= 0 and id < self._next)
    if id == self._next - 1:
      self._next -= 1
      while True:
        try:
          self._ids.remove(self._next - 1)
        except ValueError:
          break
        self._next -= 1
    elif not id in self._ids:
      self._ids.append(id)
    
class EquationManager:
  _ids = Ids()

  def __init__(self, comm = None, cp_method = "memory", cp_parameters = {}):
    """
    Manager for tangent-linear and adjoint models.

    Divides the equations into 'blocks', where each block must be of the form
      F_i ( x_{i - 1}, x_i ) = 0
    and where F_i defines the equation for the i th block, and x_i is the
    solution variable associated with the i th block. There is no checking of
    invalid cross-block dependencies.
    
    Arguments:
    comm  (Optional) PETSc communicator. Default petsc4py.PETSc.COMM_WORLD.

    cp_method  (Optional) Checkpointing method. Default "memory".      
      Possible methods
        none           Do not store anything.
        memory         Store everything in RAM.
        periodic_disk  Periodically store initial condition data on disk.
        multistage     Binomial checkpointing using the approach described in
          GW2000  A. Griewank and A. Walther, "Algorithm 799: Revolve: An
                  implementation of checkpointing for the reverse or adjoint
                  mode of computational differentiation", ACM Transactions on
                  Mathematical Software, 26(1), pp. 19--45, 2000
                       with a brute force search used to obtain behaviour
                       described in
          SW2009  P. Stumm and A. Walther, "MultiStage approaches for optimal
                  offline checkpointing", SIAM Journal on Scientific Computing,
                  31(3), pp. 1946--1967, 2009
     
    cp_parameters  (Optional) Checkpointing parameters dictionary.           
      Parameters for "none" method
        None
                 
      Parameters for "memory" method
        None
     
      Parameters for "periodic_disk" method
        path                      Directory in which disk checkpoint data should
                                  be stored.
                                  String, optional, default "checkpoints~".
        format                    Disk checkpointing format.
                                  One of {"pickle", "hdf5"}, optional,
                                  default "hdf5".
        period                    Interval between checkpoints.
                                  Positive integer, required. 
     
      Parameters for "multistage" method
        path                      Directory in which disk checkpoint data should
                                  be stored.
                                  String, optional, default "checkpoints~".
        format                    Disk checkpointing format.
                                  One of {"pickle", "hdf5"}, optional,
                                  default "hdf5".
        blocks                    Total number of blocks.
                                  Positive integer, required.
        snaps_in_ram              Number of "snaps" to store in RAM.
                                  Non-negative integer, optional, default 0.
        snaps_on_disk             Number of "snaps" to store on disk.
                                  Non-negative integer, optional, default 0.
        verbose                   Whether to enable increased verbosity.
                                  Logical, optional, default False.
    """
    # "multistage" name, and "snaps_in_ram", "snaps_on_disk" and "verbose" in
    # "multistage" method, are similar to adj_checkpointing arguments in
    # dolfin-adjoint 2017.1.0
  
    if comm is None:
      import petsc4py.PETSc
      comm = petsc4py.PETSc.COMM_WORLD
  
    self._comm = comm
    self._id = self._ids.next()
    self.reset(cp_method = cp_method, cp_parameters = cp_parameters)
  
  def __del__(self):
    self._ids.free(self._id)
  
  def comm(self):
    """
    Return the PETSc communicator.
    """
  
    return self._comm
  
  def info(self, info = info):
    """
    Display information about the equation manager state.
    
    Arguments:
    
    info  A callable which displays a provided string.
    """
  
    info("Equation manager status:")
    info("Annotation state: %s" % self._annotation_state)
    info("Tangent-linear state: %s" % self._tlm_state)
    info("Equations:")
    for i, block in enumerate(self._blocks + ([self._block] if len(self._block) > 0 else [])):
      info("  Block %i" % i)
      for j, eq in enumerate(block):
        eq_X = eq.X()
        info("    Equation %i, %s (Python id %i) solving for (%s) (ids (%s))" % (j, type(eq).__name__, id(eq), ", ".join(eq_x.name() for eq_x in eq_X), ", ".join(map(lambda i : "%i" % i, [eq_x.id() for eq_x in eq_X]))))
        nl_dep_ids = set([dep.id() for dep in eq.nonlinear_dependencies()])
        for k, dep in enumerate(eq.dependencies()):
          info("      Dependency %i, %s (id %i)%s, %s" % (k, dep.name(), dep.id(), ", replaced" if isinstance(dep, ReplacementFunction) else "", "non-linear" if dep.id() in nl_dep_ids else "linear"))
    info("Storage:")
    info("  Recording initial conditions: %s" % ("yes" if self._cp.checkpoint_ics() else "no"))
    info("  Recording equation non-linear dependencies: %s" % ("yes" if self._cp.checkpoint_data() else "no"))
    ics_keys, ics_values = self._cp.initial_conditions(copy = False)
    deps_keys, deps_values, data_keys, data_values = self._cp.data(copy = False)
    info("  Initial conditions stored: %i" % len(ics_keys))
    info("  Equations with stored non-linear dependencies: %i" % len(deps_keys))
    info("  Non-linear dependencies stored: %i" % len(data_keys))
    info("Checkpointing:")
    info("  Method: %s" % self._cp_method)
    if self._cp_method in ["none", "memory"]:
      pass
    elif self._cp_method == "periodic_disk":
      info("  Function spaces referenced: %i" % len(self._cp_disk_spaces))
      info("  Blocks in RAM: %i" % (len(self._cp_manager) + (1 if len(self._block) > 0 else 0)))
    elif self._cp_method == "multistage":
      info("  Function spaces referenced: %i" % len(self._cp_disk_spaces))
      info("  Snapshots in RAM: %i" % self._cp_manager.snapshots_in_ram())
      info("  Snapshots on disk: %i" % self._cp_manager.snapshots_on_disk())
    else:
      raise ManagerException("Unrecognised checkpointing method: %s" % self._cp_method)
  
  def new(self):
    """
    Return a new equation manager sharing the communicator and checkpointing
    configuration of this equation manager.
    """
  
    return EquationManager(comm = self._comm, cp_method = self._cp_method, cp_parameters = self._cp_parameters)
  
  def reset(self, cp_method = None, cp_parameters = None):
    """
    Reset the equation manager. Optionally a new checkpointing configuration can
    be provided.
    """
  
    if cp_method is None:
      cp_method = self._cp_method
    if cp_parameters is None:
      cp_parameters = self._cp_parameters
  
    self._annotation_state = "initial"
    self._tlm_state = "initial"
    self._block = []
    self._replace_map = OrderedDict()
    self._blocks = []
    
    self._tlm = OrderedDict()
    self._tlm_eqs = OrderedDict()
    
    self.configure_checkpointing(cp_method, cp_parameters)
    
  def configure_checkpointing(self, cp_method, cp_parameters = {}):
    """
    Provide a new checkpointing configuration.
    """
  
    if not self._annotation_state in ["initial", "stopped_initial"]:
      raise ManagerException("Cannot configure checkpointing after annotation has started, or after finalisation")
    
    cp_parameters = copy_parameters_dict(cp_parameters)

    if cp_method == "periodic_disk" or (cp_method == "multistage" and cp_parameters.get("snaps_on_disk", 0) > 0):
      cp_path = cp_parameters["path"] = cp_parameters.get("path", "checkpoints~")
      cp_format = cp_parameters["format"] = cp_parameters.get("format", "hdf5")
      
      def create_cp_path():
        if self._comm.rank == 0:
          if not os.path.exists(cp_path):
            os.makedirs(cp_path)
        self._comm.barrier()
      
      if cp_format == "pickle":
        create_cp_path()
      elif cp_format == "hdf5":
        if hasattr(self, "_cp_hdf5_file"):
          for name in self._cp_hdf5_file:
            del(self._cp_hdf5_file[name])
          self._cp_hdf5_file.attrs.clear()
        else:
          create_cp_path()
          cp_filename = os.path.join(cp_path, "%i.hdf5" % self._id)
          import h5py
          self._cp_hdf5_file = h5py.File(cp_filename, "w", driver = "mpio", comm = self._comm.tompi4py())
      else:
        raise ManagerException("Unrecognised checkpointing format: %s" % cp_format)
    
    if cp_method in ["none", "memory"]:
      cp_manager = None
    elif cp_method == "periodic_disk":
      cp_manager = set()
    elif cp_method == "multistage":
      cp_blocks = cp_parameters["blocks"]
      cp_parameters["snaps_in_ram"] = cp_snaps_in_ram = cp_parameters.get("snaps_in_ram", 0)
      cp_parameters["snaps_on_disk"] = cp_snaps_on_disk = cp_parameters.get("snaps_on_disk", 0)
      cp_parameters["verbose"] = cp_verbose = cp_parameters.get("verbose", False)
      
      from .binomial_checkpointing import MultistageManager        
      cp_manager = MultistageManager(cp_blocks, cp_snaps_in_ram, cp_snaps_on_disk)
    else:
      raise ManagerException("Unrecognised checkpointing method: %s" % cp_method)

    self._cp_method = cp_method
    self._cp_parameters = cp_parameters
    self._cp_manager = cp_manager
    self._cp_disk_spaces = []  # FunctionSpace objects are currently stored in RAM
    self._cp_disk_memory = OrderedDict()
    
    if cp_method == "multistage":
      if self._cp_manager.max_n() == 1:
        if cp_verbose: info("forward: configuring storage for reverse")
        self._cp = Checkpoint(checkpoint_ics = True,
                              checkpoint_data = True)
      else:
        if cp_verbose: info("forward: configuring storage for snapshot")
        self._cp = Checkpoint(checkpoint_ics = True,
                              checkpoint_data = False)
        if cp_verbose: info("forward: deferred snapshot at %i" % self._cp_manager.n())
        self._cp_manager.snapshot()
      self._cp_manager.forward()
      if cp_verbose: info("forward: forward advance to %i" % self._cp_manager.n())
    else:
      self._cp = Checkpoint(checkpoint_ics = cp_method != "none",
                            checkpoint_data = cp_method != "none")
  
  def add_tlm(self, M, dM, max_depth = 1):
    """
    Add a tangent-linear model defined by the parameter c, where
    M = M_0 + c dM, M is a Function or a list or tuple of Function objects,
    c = 0, and dM defines a direction.
    """
  
    if self._tlm_state == "final":
      raise ManagerException("Cannot add a tangent-linear model after finalisation")
  
    if is_function(M):
      M = (M,)
    else:
      M = tuple(M)
    if is_function(dM):
      dM = (dM,)
    else:
      dM = tuple(dM)
  
    if (M, dM) in self._tlm:
      raise ManagerException("Duplicate tangent-linear model")
    if self._tlm_state == "initial":
      self._tlm_state = "deriving"
    elif self._tlm_state == "stopped_initial":
      self._tlm_state = "stopped_deriving"
    self._tlm[(M, dM)] = [TangentLinearMap(), TangentLinearMap(), max_depth]
  
  def tlm_enabled(self):
    """
    Return whether addition of tangent-linear models is enabled.
    """
    
    return self._tlm_state == "deriving"
  
  def tlm(self, M, dM, x):
    """
    Return a tangent-linear Function associated with the forward Function x,
    for the tangent-linear model defined by M and dM, or None if x is not
    associated with a tangent-linear Function (or is static).
    """
  
    if is_function(M):
      M = (M,)
    else:
      M = tuple(M)
    if is_function(dM):
      dM = (dM,)
    else:
      dM = tuple(dM)
  
    if (M, dM) in self._tlm:
      if x in self._tlm[(M, dM)][1]:
        return self._tlm[(M, dM)][1][x]
      elif x in self._tlm[(M, dM)][0]:
        return self._tlm[(M, dM)][0][x]
      else:
        raise ManagerException("Tangent-linear not found")
    else:
      raise ManagerException("Tangent-linear not found")
  
  def annotation_enabled(self):
    """
    Return whether the equation manager currently has annotation enabled.
    """
    
    return self._annotation_state in ["initial", "annotating"]
  
  def start(self, annotation = True, tlm = True):
    """
    Start annotation or tangent-linear derivation.
    """
    
    if annotation:
      if self._annotation_state == "stopped_initial":
        self._annotation_state = "initial"
      elif self._annotation_state == "stopped_annotating":
        self._annotation_state = "annotating"
    
    if tlm:
      if self._tlm_state == "stopped_initial":
        self._tlm_state = "initial"
      elif self._tlm_state == "stopped_deriving":
        self._tlm_state = "deriving"
      
  def stop(self, annotation = True, tlm = True):
    """
    Pause annotation or tangent-linear derivation. Returns a tuple containing:
      (annotation_state, tlm_state)
    where annotation_state is True if the annotation is in state "initial" or
    "annotating" and False otherwise, and tlm_state is True if the
    tangent-linear state is "initial" or "deriving" and False otherwise, each
    evaluated before changing the state.
    """
  
    state = (self._annotation_state in ["initial", "annotating"], self._tlm_state in ["initial", "deriving"])
  
    if annotation:
      if self._annotation_state == "initial":
        self._annotation_state = "stopped_initial"
      elif self._annotation_state == "annotating":
        self._annotation_state = "stopped_annotating"
        
    if tlm:
      if self._tlm_state == "initial":
        self._tlm_state = "stopped_initial"
      elif self._tlm_state == "deriving":
        self._tlm_state = "stopped_deriving"
    
    return state
  
  def add_initial_condition(self, x, annotate = None):
    """
    Add an initial condition associated with the Function x on the adjoint
    tape.
    
    annotate (default self.annotation_enabled()):
      Whether to annotate the initial condition on the adjoint tape, storing
      data for checkpointing as required.
    """
    
    if annotate is None:
      annotate = self.annotation_enabled()
    if annotate:
      if self._annotation_state == "final":
        raise ManagerException("Cannot add initial conditions after finalisation")
        
      if self._annotation_state == "initial":
        self._annotation_state = "annotating"
      elif self._annotation_state == "stopped_initial":
        self._annotation_state = "stopped_annotating"
      self._cp.add_initial_condition(x)
  
  def initial_condition(self, x):
    """
    Return the value of the initial condition for x on the tape. Finalises the
    manager.
    """
  
    self.finalise()
  
    x = self.map(x)
    for n, block in enumerate(self._blocks):
      for eq in block:
        if x in self.map(eq.dependencies()):
          self._restore_checkpoint(n)
          return self._cp.initial_condition(x)
    raise ManagerException("Initial condition not found")
  
  def add_equation(self, eq, annotate = None, replace = False, tlm = None, annotate_tlm = None, tlm_skip = None):  
    """
    Process the provided equation, annotating and / or deriving (and solving)
    tangent-linear models as required. Assumes that the equation has already
    been solved, and that the initial condition for eq.X() have been recorded on
    the adjoint tape if necessary.
    
    annotate (default self.annotation_enabled()):
      Whether to annotate the equation on the adjoint tape, storing data for
      checkpointing as required.
    replace (default False):
      Whether to replace internal Function objects in eq with
      ReplacementFunction objects.
    tlm (default self.tlm_enabled()):
      Whether to derive (and solve) an associated tangent-linear equation.
    annotate_tlm (default annotate):
      Whether the tangent-linear equation itself should be annotated on the
      adjoint tape, with data stored for checkpointing as required.
    tlm_skip (default None):
      Used for the derivation of higher order tangent-linear equations.
    """
  
    if annotate is None:
      annotate = self.annotation_enabled()
    if annotate:
      if self._annotation_state == "final":
        raise ManagerException("Cannot add equations after finalisation")
    
      if self._annotation_state == "initial":
        self._annotation_state = "annotating"
      elif self._annotation_state == "stopped_initial":
        self._annotation_state = "stopped_annotating"
      self._block.append(eq)
      self._cp.add_equation((len(self._blocks), len(self._block) - 1), eq)
      
    if tlm is None:
      tlm = self.tlm_enabled()
    if tlm:
      if self._tlm_state == "final":
        raise ManagerException("Cannot add tangent-linear equations after finalisation")

      X = eq.X()
      eq_tlm_depth = tlm_depth(X[0])
      depth = 0 if tlm_skip is None else tlm_skip[1]
      if annotate_tlm is None:
        annotate_tlm = annotate
      for i, ((M, dM), (tlm_map, tlm_map_next, max_depth)) in enumerate(reversed(self._tlm.items())):
        if not tlm_skip is None and i >= tlm_skip[0]:
          break
        eq_tlm_eqs = self._tlm_eqs.get(eq, None)
        if eq_tlm_eqs is None:
          eq_tlm_eqs = self._tlm_eqs[eq] = OrderedDict()
        tlm_eq = eq_tlm_eqs.get((M, dM), None)
        if tlm_eq is None:
          for dep in eq.dependencies():
            if dep in M or dep in tlm_map:
              tlm_eq = eq_tlm_eqs[(M, dM)] = eq.tangent_linear(M, dM, tlm_map)
              break
        if not tlm_eq is None:
          tlm_eq.solve(manager = self, annotate = annotate_tlm,
            _tlm_skip = [i + 1, depth + 1] if max_depth - depth > 1 else [i, 0])
          for x in X:
            tlm_map_next[x] = tlm_map[x]
    
    if replace:
      self.replace(eq)
  
  def replace(self, eq):
    """
    Replace internal Function objects in the provided equation with
    ReplacementFunction objects.
    """
  
    deps = eq.dependencies()
    for dep in deps:
      if not dep.id() in self._replace_map:
        self._replace_map[dep.id()] = replaced_function(dep)
    eq._replace(OrderedDict([(dep, self._replace_map[dep.id()]) for dep in deps]))
    if eq in self._tlm_eqs:
      for tlm_eq in self._tlm_eqs[eq].values():
        if not tlm_eq is None:
          self.replace(tlm_eq)
        
  def map(self, x):
    if is_function(x) or isinstance(x, ReplacementFunction):
      return self._replace_map.get(x.id(), x)
    else:
      return [self.map(y) for y in x]
  
  def _checkpoint_space_index(self, fn):
    try:
      index = self._cp_disk_spaces.index(fn.function_space())
    except ValueError:
      self._cp_disk_spaces.append(fn.function_space())
      index = len(self._cp_disk_spaces) - 1
    return index
  
  def _save_memory_checkpoint(self, cp, n):
    self._cp_disk_memory[n] = cp
  
  def _load_memory_checkpoint(self, n, delete = False):
    return getattr(self._cp_disk_memory, "pop" if delete else "__getitem__")(n)
  
  def _save_disk_checkpoint(self, cp, n):
    cp_path = self._cp_parameters["path"]
    cp_format = self._cp_parameters["format"]
      
    ics_keys, ics_values = cp.initial_conditions(copy = False)
    
    if cp_format == "pickle":
      ics_values = [(fn.name(), self._checkpoint_space_index(fn), function_get_values(fn), function_is_static(fn)) for fn in ics_values]
      data = (ics_keys, ics_values)
      cp_filename = os.path.join(cp_path, "%i_%i_%i" % (self._id, n, self._comm.rank))
      h = open(cp_filename, "wb")
      pickle.dump(data, h)
      h.close()
    elif cp_format == "hdf5":
      self._cp_hdf5_file.create_group("/%i/ics" % n)
      for i, (ics_key, ics_value) in enumerate(zip(ics_keys, ics_values)):
        g = self._cp_hdf5_file.create_group("/%i/ics/%i" % (n, i))
      
        values = function_get_values(ics_value)
        d = g.create_dataset("value", shape = (function_global_size(ics_value),), dtype = values.dtype)
        d[function_local_indices(ics_value)] = values
        d.attrs["name"] = ics_value.name()
        d.attrs["space_index"] = self._checkpoint_space_index(ics_value)
        d.attrs["static"] = function_is_static(ics_value)
        
        d = g.create_dataset("key", shape = (self._comm.size,), dtype = numpy.int64)
        d[self._comm.rank] = ics_key
    else:
      raise ManagerException("Unrecognised checkpointing format: %s" % cp_format)
  
  def _load_disk_checkpoint(self, n, delete = False):
    cp_path = self._cp_parameters["path"]
    cp_format = self._cp_parameters["format"]
      
    if cp_format == "pickle":
      cp_filename = os.path.join(cp_path, "%i_%i_%i" % (self._id, n, self._comm.rank))
      h = open(cp_filename, "rb")
      ics_keys, ics_fns = pickle.load(h)
      h.close()
      if delete:
        os.remove(cp_filename)
      
      ics_values = []
      ics_fns.reverse()
      while len(ics_fns) > 0:
        name, i, values, static = ics_fns.pop()
        F = Function(self._cp_disk_spaces[i], name = name, static = static)
        function_set_values(F, values)
        ics_values.append(F)
        del(name, i, values, static)
        
      cp = Checkpoint(ics_keys, ics_values)
    elif cp_format == "hdf5":
      ics_keys = []
      ics_values = []
      hdf5_name = "/%i/ics" % n
      for name, g in self._cp_hdf5_file[hdf5_name].items():
        d = g["value"]
        F = Function(self._cp_disk_spaces[d.attrs["space_index"]], name = d.attrs["name"], static = d.attrs["static"])
        function_set_values(F, d[function_local_indices(F)])
        ics_values.append(F)
        
        d = g["key"]
        ics_keys.append(d[self._comm.rank])
        
        del(g, d)
        if delete:
          del(self._cp_hdf5_file["/%s/%s" % (hdf5_name, name)])
      if delete:
        del(self._cp_hdf5_file["/%i" % n])
        
      cp = Checkpoint(ics_keys, ics_values)
    else:
      raise ManagerException("Unrecognised checkpointing format: %s" % cp_format)
    
    return cp

  def _checkpoint(self, final = False):
    if self._cp_method in ["none", "memory"]:
      pass
    elif self._cp_method == "periodic_disk":   
      self._periodic_disk_checkpoint(final = final)
    elif self._cp_method == "multistage":
      self._multistage_checkpoint()
    else:
      raise ManagerException("Unrecognised checkpointing method: %s" % self._cp_method)
  
  def _periodic_disk_checkpoint(self, final = False):
    cp_period = self._cp_parameters["period"]

    n = len(self._blocks) - 1  
    self._cp_manager.add(n)
    if final or n % cp_period == cp_period - 1:
      self._save_disk_checkpoint(self._cp, n = (n // cp_period) * cp_period)
      if not final:
        self._cp = Checkpoint(checkpoint_ics = True,
                              checkpoint_data = True)
        self._cp_manager.clear()

  def _save_multistage_checkpoint(self):
    cp_verbose = self._cp_parameters["verbose"]
    
    deferred_snapshot = self._cp_manager.deferred_snapshot()
    if not deferred_snapshot is None:
      snapshot_n, storage = deferred_snapshot
      if storage == "disk":
        if cp_verbose: info("%s: save snapshot at %i on disk" % ("forward" if self._cp_manager.r() == 0 else "reverse", snapshot_n))
        self._save_disk_checkpoint(self._cp, snapshot_n)
      else:
        if cp_verbose: info("%s: save snapshot at %i in RAM" % ("forward" if self._cp_manager.r() == 0 else "reverse", snapshot_n))
        self._save_memory_checkpoint(self._cp, snapshot_n)

  def _multistage_checkpoint(self):    
    cp_verbose = self._cp_parameters["verbose"]
    
    n = len(self._blocks)
    if n < self._cp_manager.n():
      return
    elif n == self._cp_manager.max_n():
      return
    elif n > self._cp_manager.max_n():
      raise ManagerException("Unexpected number of blocks")
    
    self._save_multistage_checkpoint()
    if n == self._cp_manager.max_n() - 1:
      if cp_verbose: info("forward: configuring storage for reverse")
      self._cp = Checkpoint(checkpoint_ics = True,
                            checkpoint_data = True)
    else:
      if cp_verbose: info("forward: configuring storage for snapshot")
      self._cp = Checkpoint(checkpoint_ics = True,
                            checkpoint_data = False)
      if cp_verbose: info("forward: deferred snapshot at %i" % self._cp_manager.n())
      self._cp_manager.snapshot()      
    self._cp_manager.forward()
    if cp_verbose: info("forward: forward advance to %i" % self._cp_manager.n())

  def _restore_checkpoint(self, n):
    if self._cp_method == "none":
      raise ManagerException("Cannot restore data when using 'none' checkpointing method")
    elif self._cp_method == "memory":
      pass
    elif self._cp_method == "periodic_disk":      
      if not n in self._cp_manager:
        cp_period = self._cp_parameters["period"]
        
        N0 = (n // cp_period) * cp_period
        N1 = min(((n // cp_period) + 1) * cp_period, len(self._blocks))
        
        del(self._cp)
        self._cp = self._load_disk_checkpoint(N0, delete = False)
        self._cp.configure(checkpoint_ics = True,
                           checkpoint_data = True)
        self._cp_manager.clear()
        
        replace_map = OrderedDict()
        for n1 in range(N0, N1):
          for i, eq in enumerate(self._blocks[n1]):
            eq_deps = eq.dependencies()

            X = []
            for eq_x in eq.X():
              if eq_x.id() in replace_map:
                X.append(replace_map[eq_x.id()])
              elif self._cp.has_initial_condition(eq_x):
                X.append(self._cp.initial_condition(eq_x))
                replace_map[eq_x.id()] = X[-1]
              else:
                X.append(function_new(eq_x))
                replace_map[eq_x.id()] = X[-1]
            for dep in eq_deps:
              if not dep.id() in replace_map:
                if self._cp.has_initial_condition(dep):
                  replace_map[dep.id()] = self._cp.initial_condition(dep)
                else:
                  replace_map[dep.id()] = function_new(dep, static = function_is_static(dep))
            
            deps = [replace_map[dep.id()] for dep in eq_deps]
            eq.forward_solve(X[0] if len(X) == 1 else X, deps)
            
            self._cp.add_equation((n1, i), eq, deps = deps)
          self._cp_manager.add(n1)
    elif self._cp_method == "multistage":
      cp_verbose = self._cp_parameters["verbose"]
      
      if n == 0 and self._cp_manager.max_n() - self._cp_manager.r() == 0:
        return
      elif n == self._cp_manager.max_n() - 1:
        if cp_verbose: info("reverse: adjoint step back to %i" % n)
        self._cp_manager.reverse()
        return
        
      replace_map = OrderedDict()
      def advance(N0, N1):
        for n in range(N0, N1):
          for i, eq in enumerate(self._blocks[n]):
            eq_deps = eq.dependencies()

            X = []
            for eq_x, checkpoint_ic in zip(eq.X(), eq._checkpoint_ic):
              if eq_x.id() in replace_map:
                X.append(replace_map[eq_x.id()])
                if checkpoint_ic:
                  self._cp.add_initial_condition(eq_x, value = X[-1])
              else:
                X.append(function_new(eq_x))
                replace_map[eq_x.id()] = X[-1]
            for dep in eq_deps:
              if not dep.id() in replace_map:
                replace_map[dep.id()] = function_new(dep, static = function_is_static(dep))
            
            deps = [replace_map[dep.id()] for dep in eq_deps]
            eq.forward_solve(X[0] if len(X) == 1 else X, deps)

            self._cp.add_equation((n, i), eq, deps = deps)

      snapshot_n, storage, delete = self._cp_manager.load_snapshot()
      del(self._cp)
      if storage == "disk":
        if cp_verbose: info("reverse: load snapshot at %i from disk and %s" % (snapshot_n, "delete" if delete else "keep"))
        self._cp = self._load_disk_checkpoint(snapshot_n, delete = delete)
      else:
        if cp_verbose: info("reverse: load snapshot at %i from RAM and %s" % (snapshot_n, "delete" if delete else "keep"))
        self._cp = self._load_memory_checkpoint(snapshot_n, delete = delete)
      replace_map.update(OrderedDict(zip(*self._cp.initial_conditions(copy = storage != "disk"))))  
      del(self._cp)
      if snapshot_n < n:
        if cp_verbose: info("reverse: no storage")
        self._cp = Checkpoint(checkpoint_ics = False,
                              checkpoint_data = False)
      
      snapshot_n_0 = snapshot_n
      while snapshot_n <= n:
        if snapshot_n == n:
          if cp_verbose: info("reverse: configuring storage for reverse")
          self._cp = Checkpoint(checkpoint_ics = True,
                                checkpoint_data = True)
        elif snapshot_n > snapshot_n_0:
          if cp_verbose: info("reverse: configuring storage for snapshot")
          self._cp = Checkpoint(checkpoint_ics = True,
                                checkpoint_data = False)
          if cp_verbose: info("reverse: deferred snapshot at %i" % self._cp_manager.n())
          self._cp_manager.snapshot()
        self._cp_manager.forward()
        if cp_verbose: info("reverse: forward advance to %i" % self._cp_manager.n())
        advance(snapshot_n, self._cp_manager.n())
        self._save_multistage_checkpoint()
        snapshot_n = self._cp_manager.n()
      
      if cp_verbose: info("reverse: adjoint step back to %i" % n)
      self._cp_manager.reverse()
    else:
      raise ManagerException("Unrecognised checkpointing method: %s" % self._cp_method)
  
  def new_block(self):
    """
    End the current block equation and begin a new block. Ignored if
    "multistage" checkpointing is used and the final block has been reached.
    """
  
    if self._annotation_state in ["stopped_initial", "stopped_annotating", "final"]:
      return
    elif self._cp_method == "multistage" and len(self._blocks) == self._cp_parameters["blocks"] - 1:
      # Wait for the finalise
      warning("Attempting to end the final block without finalising -- ignored")
      return
    
    self._blocks.append(self._block)
    self._block = []
    for (M, dM), (tlm_map, tlm_map_next, max_depth) in self._tlm.items():
      self._tlm[(M, dM)] = [tlm_map_next, TangentLinearMap(), max_depth]
    self._checkpoint(final = False)
  
  def finalise(self):
    """
    End the final block equation.
    """
  
    if self._annotation_state == "final":
      return
    self._annotation_state = "final"
    self._tlm_state = "final"
    
    self._blocks.append(self._block)
    self._block = []
    for (M, dM), (tlm_map, tlm_map_next, max_depth) in self._tlm.items():
      self._tlm[(M, dM)] = [tlm_map_next, TangentLinearMap(), max_depth]
    self._checkpoint(final = True)
  
  def dependency_graph_png(self, divider = [255, 127, 127], p = 5):
    P = 2 ** p
  
    blocks = self._blocks
    if len(self._block) > 0:
      blocks += [self._block]
    
    M = 0
    for block in blocks:
      M += len(block) * P
    M += len(blocks) + 1
    pixels = numpy.empty((M, M, 3), dtype = numpy.uint8)
    pixels[:] = 255
    
    pixels[0, :, :] = divider
    pixels[:, 0, :] = divider
    i0 = 1
    for block in blocks:
      pixels[i0 + len(block) * P, :, :] = divider
      pixels[:, i0 + len(block) * P, :] = divider
      i0 += len(block) * P + 1
    
    i0 = 1
    last_block = None
    last_i0 = None
    for block in blocks:
      if not last_block is None:
        x_map = {}
        for i, eq in enumerate(last_block):
          for x in eq.X():
            x_map[x.id()] = i
        for i, eq in enumerate(block):
          for x in eq.X():
            x_id = x.id()
            if x_id in x_map:
              del(x_map[x_id])
            for dep in eq.dependencies():
              dep_id = dep.id()
              if dep_id in x_map:
                i1 = i * P + i0
                j1 = x_map[dep_id] * P + last_i0
                pixels[i1:i1 + P, j1:j1 + P, :] = 0

      x_map = defaultdict(lambda : [])
      for i, eq in enumerate(block):
        for x in eq.X():
          x_map[x.id()].append(i)
        
      for i in range(len(block) - 1, -1, -1):
        eq = block[i]
        for dep in eq.dependencies():
          dep_id = dep.id()
          if dep_id in x_map and len(x_map[dep_id]) > 0:
            i1 = i * P + i0
            j1 = x_map[dep_id][-1] * P + i0
            pixels[i1:i1 + P, j1:j1 + P, :] = 0
        for x in eq.X():
          x_map[x.id()].pop()
        
      last_block = block
      last_i0 = i0
      i0 += len(block) * P + 1
    
    import png
    return png.from_array(pixels, "RGB")
  
  def compute_gradient(self, Js, M, rhs_cache = True):
    """
    Compute the derivative of one or more functionals with respect to one or
    more control parameters by running adjoint models. Finalises the manager.
    
    Arguments:
    
    Js        A Functional or Function, or a list or tuple of these, defining
              the functionals.
    M         A Control or Function, or a list or tuple of these, defining the
              control parameters.
    rhs_cache (Optional) Whether to cache the last RHS Function encountered for
              each function space.
    """
  
    if not isinstance(M, (list, tuple)):
      if not isinstance(Js, (list, tuple)):
        return self.compute_gradient([Js], [M], rhs_cache = rhs_cache)[0][0]
      else:
        return tuple(dJ[0] for dJ in self.compute_gradient(Js, [M], rhs_cache = rhs_cache))
    elif not isinstance(Js, (list, tuple)):
      return self.compute_gradient([Js], M, rhs_cache = rhs_cache)[0]
    
    self.finalise()

    Js = list(Js)
    for J_i, J in enumerate(Js):
      if not is_function(J):
        Js[J_i] = J.fn()

    M = [self.map(m if is_function(m) else m.m()) for m in M]
    dJ = [[function_new(m) for m in M] for J in Js]

    def transpose_dependencies(m, n):      
      if m == n:
        block = self._blocks[m]
      
        x_map = defaultdict(lambda : [])
        for k, eq in enumerate(block):
          for l, x in enumerate(eq.X()):
            x_map[x.id()].append((k, l))
        
        tdeps = [[] for eq in block]
        for i in range(len(block) - 1, -1, -1):
          eq = block[i]
          x_ids = set(x.id() for x in eq.X())
          for x_id in x_ids:
            x_map[x_id].pop()
          for j, dep in enumerate(eq.dependencies()):
            dep_id = dep.id()
            if not dep_id in x_ids and dep_id in x_map and len(x_map[dep_id]) > 0:
              k, l = x_map[dep_id][-1]
              tdeps[i].append((j, k, l))
        
        return tdeps
      else:
        block = self._blocks[m]
      
        x_map = {}
        for k, eq in enumerate(block):
          for l, x in enumerate(eq.X()):
            x_map[x.id()] = (k, l)
          
        block = self._blocks[n]
        
        tdeps = [[] for eq in block]
        x_ids = set()
        for i, eq in enumerate(block):
          for x in eq.X():
            x_ids.add(x.id())
          for j, dep in enumerate(eq.dependencies()):
            dep_id = dep.id()
            if not dep_id in x_ids and dep_id in x_map:
              k, l = x_map[dep_id]
              tdeps[i].append((j, k, l))
        
        return tdeps
    
    self._restore_checkpoint(len(self._blocks) - 1)
    tdeps = transpose_dependencies(-1, -1)        

    if rhs_cache:    
      rhs_cache = OrderedDict()
      def rhs_cache_new(x, static = False):
        space = x.function_space()
        if is_function(x):
          b = rhs_cache[space.id()] = function_new(x, static = static)
        elif space.id() in rhs_cache:
          b = rhs_cache[space.id()] = function_new(rhs_cache[space.id()], static = static)
        else:
          b = rhs_cache[space.id()] = Function(space, static = static)
        return b
    else:
      rhs_cache_new = lambda x, static = False : function_new(x, static = static)

    B = [[None for eq in self._blocks[-1]] for J in Js]
    for n in range(len(self._blocks) - 1, -1, -1):
      if n > 0:
        B_transfer = [[None for eq in self._blocks[n - 1]] for J in Js]
        tdeps_transfer = transpose_dependencies(n - 1, n)

      for i in range(len(self._blocks[n]) - 1, -1, -1):
        eq = self._blocks[n][i]
        X = self.map(eq.X())
        deps = self.map(eq.dependencies())

        for J_i, J in enumerate(Js):
          if len(X) == 1 and X[0].id() == J.id():
            adj_x = rhs_cache_new(X[0], static = True)
            function_assign(adj_x, -1.0)
            j = deps.index(X[0])
            sb = eq.adjoint_derivative_action(self._cp[(n, i)], j, adj_x)
            if not sb is None:
              if B[J_i][i] is None:
                B[J_i][i] = (rhs_cache_new(X[0]),)
              subtract_adjoint_derivative_action(B[J_i][i][0], sb)
            del(adj_x, sb)
          
          if B[J_i][i] is None:
            continue
          for l, x in enumerate(X):
            if B[J_i][i][l] is None:
              B[J_i][i][l] = rhs_cache_new(x)
            else:
              finalise_adjoint_derivative_action(B[J_i][i][l])
          adj_X = eq.adjoint_jacobian_solve(self._cp[(n, i)], B[J_i][i][0] if len(B[J_i][i]) == 1 else B[J_i][i])
          if is_function(adj_X):
            adj_X = (adj_X,)
          B[J_i][i] = None
          
          for j, k, l in tdeps[i]:
            sb = eq.adjoint_derivative_action(self._cp[(n, i)], j, adj_X[0] if len(adj_X) == 1 else adj_X)
            if not sb is None:
              if B[J_i][k] is None:
                B[J_i][k] = [None for x in self._blocks[n][k].X()]
              if B[J_i][k][l] is None:
                B[J_i][k][l] = rhs_cache_new(self._blocks[n][k].X()[l])
              subtract_adjoint_derivative_action(B[J_i][k][l], sb)
            del(sb)
          
          if n > 0:
            for j, k, l in tdeps_transfer[i]:
              sb = eq.adjoint_derivative_action(self._cp[(n, i)], j, adj_X[0] if len(adj_X) == 1 else adj_X)
              if not sb is None:
                if B_transfer[J_i][k] is None:
                  B_transfer[J_i][k] = [None for x in self._blocks[n - 1][k].X()]
                if B_transfer[J_i][k][l] is None:
                  B_transfer[J_i][k][l] = rhs_cache_new(self._blocks[n - 1][k].X()[l])
                subtract_adjoint_derivative_action(B_transfer[J_i][k][l], sb)
              del(sb)
          
          for j, m in enumerate(M):
            if m in deps:
              sdJ = eq.adjoint_derivative_action(self._cp[(n, i)], deps.index(m), adj_X[0] if len(adj_X) == 1 else adj_X)
              subtract_adjoint_derivative_action(dJ[J_i][j], sdJ)
              del(sdJ)
          
          del(adj_X)

      for J_i, J in enumerate(Js):
        for i, m in enumerate(M):
          finalise_adjoint_derivative_action(dJ[J_i][i])          

      if n > 0:
        self._restore_checkpoint(n - 1)        
        B = B_transfer
        for J_i, J in enumerate(Js):
          for i, eq in enumerate(self._blocks[n - 1]):
            if not B[J_i][i] is None:
              for l in range(len(B[J_i][i])):
                if not B[J_i][i][l] is None:
                  finalise_adjoint_derivative_action(B[J_i][i][l])
        tdeps = transpose_dependencies(n - 1, n - 1)
        del(B_transfer, tdeps_transfer)
            
    if self._cp_method == "multistage":
      self._cp.clear(clear_ics = False, clear_data = True)
            
    return tuple(tuple(dJ[J_i][j] for j in range(len(M))) for J_i in range(len(Js)))
  
  def find_initial_condition(self, x):    
    """
    Find the initial condition Function or ReplacementFunction associated with
    the given Function or name.
    """
    
    if is_function(x):
      return self.map(x)
    else:
      for sblock in self._blocks + [self._block]:
        for eq in sblock:
          for dep in eq.dependencies():
            if dep.name() == x:
              return dep
      raise ManagerException("Initial condition not found")

set_manager(EquationManager())

def configure_checkpointing(cp_method, cp_parameters = {}, manager = None):
  (_manager() if manager is None else manager).configure_checkpointing(cp_method, cp_parameters)

def manager_info(info = info, manager = None):
  (_manager() if manager is None else manager).info(info = info)

def reset(cp_method = None, cp_parameters = None, manager = None):
  (_manager() if manager is None else manager).reset(cp_method = cp_method, cp_parameters = cp_parameters)

def annotation_enabled(manager = None):
  return (_manager() if manager is None else manager).annotation_enabled()

def start_manager(manager = None, annotation = True, tlm = True):
  (_manager() if manager is None else manager).start(annotation = annotation, tlm = tlm)

def stop_manager(manager = None, annotation = True, tlm = True):
  (_manager() if manager is None else manager).stop(annotation = annotation, tlm = tlm)

def add_tlm(M, dM, max_depth = 1, manager = None):
  (_manager() if manager is None else manager).add_tlm(M, dM, max_depth = max_depth)

def tlm_enabled(manager = None):
  return (_manager() if manager is None else manager).tlm_enabled()

def tlm(self, M, dM, x):
  return (_manager() if manager is None else manager).tlm(M, dM, x)

def compute_gradient(Js, M, rhs_cache = True, manager = None):
  return (_manager() if manager is None else manager).compute_gradient(Js, M, rhs_cache = rhs_cache)

def new_block(manager = None):
  (_manager() if manager is None else manager).new_block()

def minimize_scipy(forward, M0, J0 = None, manager = None, **kwargs):
  """
  Gradient-based minimization using scipy.optimize.minimize.
  
  Arguments:
  
  forward  A callable which takes as input the control parameters and returns
           the Functional to be minimized.
  M0       Control parameters initial guess.
  J0       (Optional) Initial functional. If supplied assumes that the forward
           has already been run, and annotated on the equation manager, using
           the control parameters given by M0.
  manager  (Optional) The equation manager.
  
  Any remaining keyword arguments are passed directly to
  scipy.optimize.minimize.
  
  Returns a tuple
    (M, return_value)
  return M is the value of the control parameters obtained, and return_value is
  the return value of scipy.optimize.minimize.
  """

  if not isinstance(M0, (list, tuple)):
    (M,), return_value = minimize_scipy(lambda M : forward(M[0]), [M0], J0 = J0, manager = manager, **kwargs)
    return M, return_value

  M0 = [m0 if is_function(m0) else m0.m() for m0 in M0]
  if manager is None:
    manager = _manager()
  comm = manager.comm().tompi4py()  

  N = [0]
  for m in M0:
    N.append(N[-1] + function_local_size(m))
  size_global = comm.allgather(numpy.array(N[-1], dtype = numpy.int64))
  N_global = [0]
  for size in size_global:
    N_global.append(N_global[-1] + size)
  
  def get(F):
    x = numpy.empty(N[-1], dtype = numpy.float64)
    for i, f in enumerate(F):
      x[N[i]:N[i + 1]] = function_get_values(f)
    
    x_global = comm.allgather(x)
    X = numpy.empty(N_global[-1], dtype = numpy.float64)
    for i, x_p in enumerate(x_global):
      X[N_global[i]:N_global[i + 1]] = x_p
    return X
  
  def set(F, x):
    # Basic cross-process synchonisation check
    check1 = numpy.array(zlib.adler32(x.data), dtype = numpy.uint32)
    check_global = comm.allgather(check1)
    for check2 in check_global:
      if check1 != check2:
        raise ManagerException("Parallel desynchronisation detected")
    
    x = x[N_global[comm.rank]:N_global[comm.rank + 1]]
    for i, f in enumerate(F):
      function_set_values(f, x[N[i]:N[i + 1]])

  M = [function_new(m0, static = function_is_static(m0)) for m0 in M0]
  J = [J0]
  J_M = [M0]
  
  def fun(x):
    if not J[0] is None:
      return J[0].value()

    set(M, x)      
    old_manager = _manager()
    set_manager(manager)
    manager.reset()
    clear_caches()  # Could use new caches here
    manager.start()
    J[0] = forward(M)
    manager.stop()    
    set_manager(old_manager)
    J_M[0] = M
    return J[0].value()
  
  def jac(x):
    fun(x)
    dJ = manager.compute_gradient(J[0], J_M[0])
    J[0] = None
    return get(dJ)
  
  import scipy.optimize
  return_value = scipy.optimize.minimize(fun, get(M0), jac = jac, **kwargs)
  set(M, return_value.x)
  
  return M, return_value

# Aims for similar behaviour, and largely API compatible with, the
# dolfin-adjoint taylor_test function in dolfin-adjoint 2017.1.0. Arguments
# based on dolfin-adjoint taylor_test arguments
#   forward (renamed from J)
#   m
#   J_val (renamed from Jm)
#   dJ (renamed from dJdm)
#   ddJ (renamed from HJm)
#   seed
#   dm (renamed from perturbation_direction)
#   m0 (renamed from value)
#   size
def taylor_test(forward, m, J_val, dJ = None, ddJ = None, seed = 1.0e-2,
  dm = None, m0 = None, size = 5, manager = None):
  """
  Perform a Taylor verification test.
  
  Arguments:
  
  forward  A callable which takes as input a Function defining the value of the
           control, and returns the Functional.
  m        A Control or Function. The control.
  J_val    The reference functional value.
  dJ       (Optional if ddJ is not supplied) A Function storing the derivative
           of J with respect to m.
  ddJ      (Optional) A Hessian used to compute Hessian actions associated with
           the second derivative of J with respect to m.
  seed     (Optional) The maximum scaling for the perturbation is seed
           multiplied by the inf norm of the reference value (coefficients
           vector) of the control (or 1 if this is less than 1).
  dm       A perturbation direction. A Function with values generated using
           numpy.random.random is used if not supplied.
  size     (Optional) The number of perturbed forward runs used in the test.
  m0       (Optional) The reference value of the control.
  manager  (Optional) The equation manager.
  """

  if manager is None:
    manager = _manager()

  if not is_function(m):
    m = m.m()
  if m0 is None:
    m0 = manager.initial_condition(m)
  m1 = function_new(m, static = function_is_static(m))
  
  # This combination seems to reproduce dolfin-adjoint behaviour
  eps = numpy.array([2 ** -p for p in range(size)], dtype = numpy.float64)
  eps = seed * eps * max(1.0, function_linf_norm(m0))
  if dm is None:
    dm = function_new(m1, static = True)
    function_set_values(dm, numpy.random.random(function_local_size(dm)))
  
  J_vals = numpy.empty(eps.shape, dtype = numpy.float64)
  for i in range(eps.shape[0]):
    function_assign(m1, m0)
    function_axpy(m1, eps[i], dm)
    clear_caches()  # Could use new caches here
    annotation_enabled, tlm_enabled = manager.stop()
    J_vals[i] = forward(m1).value()
    manager.start(annotation = annotation_enabled, tlm = tlm_enabled)
  
  errors_0 = abs(J_vals - J_val)
  orders_0 = numpy.log(errors_0[1:] / errors_0[:-1]) / numpy.log(0.5)
  info("Errors, no adjoint   = %s" % errors_0)
  info("Orders, no adjoint   = %s" % orders_0)

  if ddJ is None:
    errors_1 = abs(J_vals - J_val - eps * function_inner(dJ, dm))
    orders_1 = numpy.log(errors_1[1:] / errors_1[:-1]) / numpy.log(0.5)  
    info("Errors, with adjoint = %s" % errors_1)
    info("Orders, with adjoint = %s" % orders_1)
    return orders_1.min()
  else:
    if dJ is None:
      _, dJ, ddJ = ddJ.action(m, dm)
    else:
      dJ = function_inner(dJ, dm)
      _, _, ddJ = ddJ.action(m, dm)
    errors_2 = abs(J_vals - J_val - eps * dJ - 0.5 * eps * eps * function_inner(ddJ, dm))
    orders_2 = numpy.log(errors_2[1:] / errors_2[:-1]) / numpy.log(0.5)  
    info("Errors, with adjoint = %s" % errors_2)
    info("Orders, with adjoint = %s" % orders_2)  
    return orders_2.min()
