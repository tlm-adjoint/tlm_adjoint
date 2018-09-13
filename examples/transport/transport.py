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

# Import FEniCS and tlm_adjoint
from fenics import *
from tlm_adjoint import *
# Import an optimization module, used for Hessian actions with single block
# forward models
from tlm_adjoint.hessian_optimization import *
# Disable the manager until it is needed
stop_manager()

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -march=native"
parameters["form_compiler"]["optimize"] = True

import petsc4py.PETSc
petsc4py.PETSc.Options().setValue("citations", "petsc.bib")

# Seed the random number generator, to ensure reproducibility of the later
# Taylor verification
import numpy
numpy.random.seed(1709465)

import h5py
import sys
import time

# Enable Taylor verification
verify = True

# Configure the mesh
N_x, N_y = 100, 50
mesh = RectangleMesh(Point(0.0, 0.0), Point(2.0, 1.0), N_x, N_y)

# Configure the interior domain discrete function space
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

# Approximate Courant number, relative to background uniform flow
C = 0.5
# Number of timesteps
N_t = 2 * N_x
# Approximate grid Péclet number, relative to background uniform flow
Pe = 10.0
# Time step size
dt = Constant(C / float(N_x), static = True)  # Using L_x / U = 1 here
info("dt = %.16e" % float(dt))
# Diffusivity
kappa_space = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
kappa = Function(kappa_space, name = "kappa", static = True)
kappa.assign(Constant(1.0 / (Pe * float(N_x))))  # Using L_x U = 1 here
info("kappa = %.16e" % function_max_value(kappa))
# Regularisation parameter
alpha = Constant(1.0e-15, static = True)

# Stream function
psi = Function(space, name = "psi", static = True)
psi.interpolate(Expression("(1.0 - exp(x[1])) * sin(k * pi * x[0]) * sin(l * pi * x[1]) - x[1]",
  element = psi.function_space().ufl_element(), k = 1.0, l = 1.0))

# Mark the outflow boundary
boundary_markers = MeshFunction("size_t", mesh, 1)
boundary_markers.set_all(0)
class OutflowBoundary(SubDomain):
  # Following the FEniCS 2017.1.0 API
  def inside(self, x, on_boundary):
    return abs(x[0] - 2.0) < DOLFIN_EPS
OutflowBoundary().mark(boundary_markers, 1)

# Extract the inflow mesh
boundary_mesh = BoundaryMesh(mesh, "exterior")
boundary_mesh_markers = MeshFunction("size_t", boundary_mesh, 1)
boundary_mesh_markers.set_all(0)
class InflowBoundary(SubDomain):
  # Following the FEniCS 2017.1.0 API
  def inside(self, x, on_boundary):
    return abs(x[0] - 0.0) < DOLFIN_EPS
InflowBoundary().mark(boundary_mesh_markers, 1)
inflow_mesh = SubMesh(boundary_mesh, boundary_mesh_markers, 1)
# Configure the inflow domain discrete function space
inflow_space = FunctionSpace(inflow_mesh, "Lagrange", 1)
# Inflow boundary condition
T_inflow = Function(inflow_space, name = "T_inflow", static = True)
T_inflow.interpolate(Expression("sin(pi * x[1]) + 0.4 * sin(3.0 * pi * x[1])", element = T_inflow.function_space().ufl_element()))

forward_calls = [0]
def forward(T_inflow_bc, kappa, T_N_ref = None, output_filename = None):
  t0 = time.time()
  # Clear assembly and linear solver caches
  clear_caches()    
  
  # An equation which sets T = T_bc on the boundary at x = 0, and T = 0
  # elsewhere
  class InflowBCSolver(Equation):
    def __init__(self, T_bc, T):
      bc = DirichletBC(T.function_space(), Expression("x[1]", degree = 1), "fabs(x[0]) < DOLFIN_EPS")
      bc = bc.get_boundary_values()
      nodes = list(bc.keys())
      y = list(bc.values())
      
      bc_adj = DirichletBC(T_bc.function_space(), Expression("x[1]", degree = 1), "fabs(x[0]) < DOLFIN_EPS")
      bc_adj = bc_adj.get_boundary_values()
      nodes_adj = list(bc_adj.keys())
      y_adj = list(bc_adj.values())
    
      Equation.__init__(self, T, [T, T_bc], nl_deps = [])
      self._nodes = nodes
      self._y = y
      self._nodes_adj = nodes_adj
      self._y_adj = y_adj
      
      self.reset_forward_solve()
      
    def forward_solve(self, x, deps = None):
      if self._x_arr is None:
        _, T_bc = self.dependencies() if deps is None else deps      
        self._x_arr = numpy.zeros(function_local_size(x), dtype = numpy.float64)
        for node, y in zip(self._nodes, self._y):
          self._x_arr[node] = T_bc(numpy.array([0.0, y], dtype = numpy.float64))
      function_set_values(x, self._x_arr)
    
    def reset_forward_solve(self):
      self._x_arr = None
    
    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
      if dep_index == 0:
        return adj_x
      elif dep_index == 1:
        F = function_new(self.dependencies()[1])
        F_arr = numpy.zeros(function_local_size(F), dtype = numpy.float64)
        for node, y in zip(self._nodes_adj, self._y_adj):
          F_arr[node] = -adj_x(numpy.array([0.0, y], dtype = numpy.float64))
        function_set_values(F, F_arr)
        return F
      else:
        return None

    def adjoint_jacobian_solve(self, nl_deps, b):
      return b
        
    def tangent_linear(self, M, dM, tlm_map):
      T, T_bc = self.dependencies()
    
      tau_T_bc = None
      for i, m in enumerate(M):
        if m == T:
          raise EquationException("Invalid tangent-linear parameter")
        elif m == T_bc:
          tau_T_bc = dM[i]
      if tau_T_bc is None:
        tau_T_bc = tlm_map[T_bc]
        
      if tau_T_bc is None:
        return None
      else:
        return InflowBCSolver(tau_T_bc, tlm_map[T])
  
  # A function equal to the inflow boundary condition value on the inflow, and
  # equal to zero elsewhere
  T_inflow = Function(space, name = "T_inflow")
  # Boundary condition application equation
  bc_eq = InflowBCSolver(T_inflow_bc, T_inflow)
  
  # Solution on the previous time level
  T_n = Function(space, name = "T_n")
  # Solution on the next time level, subject to homogenised boundary conditions
  T_np1_0 = Function(space, name = "T_np1_0")
  
  T_np1 = T_inflow + trial
  # Timestep equation, subject to homogenised boundary conditions
  T_nph = Constant(0.5, static = True) * (T_n + T_np1)
  perp = lambda v : as_vector([-v[1], v[0]])
  F = (inner(test, (T_np1 - T_n) / dt) * dx
     + inner(test, dot(perp(grad(psi)), grad(T_nph))) * dx
     + inner(grad(test), kappa * grad(T_nph)) * dx)  
  timestep_eq = EquationSolver(lhs(F) == rhs(F),
                               T_np1_0,
                               DirichletBC(space, 0.0, "fabs(x[0]) < DOLFIN_EPS", static = True, homogeneous = True),
                               solver_parameters = {"linear_solver":"umfpack"})
  # Equation which constructs the complete solution on the next time level
  update_eq = AxpySolver(T_np1_0, 1.0, T_inflow, T_n)
  # All equations
  eqs = [bc_eq, timestep_eq, update_eq]
  
  if not output_filename is None:
    # Output the forward solution
    T_output = File(output_filename, "compressed")
    T_output << (T_n, 0.0)
  
  for n in range(N_t):
    # Timestep
    for eq in eqs:
      eq.solve()
    if not output_filename is None:
      # Output the forward solution
      T_output << (T_n, (n + 1) * float(dt))
  # Drop references to Function objects within the equations
  for eq in eqs:
    eq.replace()
  
  if T_N_ref is None:
    # Store the solution of the equation in a "reference" function
    T_N_ref = Function(space, name = "T_N_ref", static = True)
    function_assign(T_N_ref, T_n)
  
  # First functional
  J = Functional(name = "J")
  # Mis-match functional
  J.assign(inner(T_n - T_N_ref, T_n - T_N_ref) * ds(subdomain_data = boundary_markers)(1))
  # Regularisation
  J.addto(alpha * inner(grad(T_inflow_bc), grad(T_inflow_bc)) * dx)

  # Second functional
  K = Functional(name = "K")
  K.assign(inner(T_n, T_n) * dx)

  forward_calls[0] += 1
  info("Forward call %i, %.3fs, J = %.16e, K = %.16e" % (forward_calls[0], time.time() - t0, J.value(), K.value()))
  sys.stdout.flush()
  return T_N_ref, J, K

# Generate a reference solution
File("T_inflow.pvd", "compressed") << T_inflow
T_N_ref, _, _ = forward(T_inflow, kappa, output_filename = "forward.pvd")
File("T_N_ref.pvd", "compressed") << T_N_ref

# Delete the original input
T_inflow = Function(inflow_space, name = "T_inflow", static = True)

# Build the Hessian via brute-force
start_manager()
_, J, K = forward(T_inflow, kappa, T_N_ref = T_N_ref)
stop_manager()
ddJ = SingleBlockHessian(J)
if verify:
  # Verify the forward model constrained Hessian
  min_order = taylor_test(lambda T_inflow : forward(T_inflow, kappa, T_N_ref = T_N_ref)[1],
    T_inflow, J_val = J.value(), ddJ = ddJ, seed = 1.0e-6)
H = numpy.empty((function_local_size(T_inflow), function_local_size(T_inflow)), dtype = numpy.float64)
for i in range(H.shape[0]):
  info("Building Hessian row %i of %i" % (i + 1, H.shape[0]))
  dm = Function(inflow_space, static = True)
  dm.vector()[i] = 1.0
  H[i, :] = function_get_values(ddJ.action(T_inflow, dm)[2])
  del(dm)

# Solve the optimization problem
_, dJ = ddJ.compute_gradient(T_inflow)
function_set_values(T_inflow, numpy.linalg.solve(H, -function_get_values(dJ)))
File("T_inflow_inv.pvd", "compressed") << T_inflow
del(ddJ)

# Re-run the forward at the inverted state
reset()
start_manager()
_, J, K = forward(T_inflow, kappa, T_N_ref = T_N_ref, output_filename = "inversion.pvd")
stop_manager()

# Forward model constrained derivatives
(dJ_dinflow, dJ_dkappa), (dK_dinflow, dK_dkappa) = compute_gradient([J, K], [T_inflow, kappa])
if verify:
  # Verify the forward model constrained derivatives
  min_order = taylor_test(lambda T_inflow : forward(T_inflow, kappa, T_N_ref = T_N_ref)[1],
    T_inflow, J_val = J.value(), dJ = dJ_dinflow, seed = 1.0e-4)
  min_order = taylor_test(lambda kappa : forward(T_inflow, kappa, T_N_ref = T_N_ref)[1],
   kappa, J_val = J.value(), dJ = dJ_dkappa, seed = 1.0e-4)
  min_order = taylor_test(lambda T_inflow : forward(T_inflow, kappa, T_N_ref = T_N_ref)[2],
    T_inflow, J_val = K.value(), dJ = dK_dinflow, seed = 1.0e-4)
  min_order = taylor_test(lambda kappa : forward(T_inflow, kappa, T_N_ref = T_N_ref)[2],
   kappa, J_val = K.value(), dJ = dK_dkappa, seed = 1.0e-4)

def project(b, space, name):
  x = Function(space, name = name)
  test, trial = TestFunction(space), TrialFunction(space)
  M = assemble(inner(test, trial) * dx)
  LUSolver(M, "umfpack").solve(x.vector(), b.vector())
  return x
File("dJ_dinflow.pvd", "compressed") << project(dJ_dinflow, inflow_space, name = "dJ_dinflow")
File("dJ_dkappa.pvd",  "compressed") << project(dJ_dkappa,   kappa_space, name = "dJ_dkappa")
File("dK_dinflow.pvd", "compressed") << project(dK_dinflow, inflow_space, name = "dK_dinflow")
File("dK_dkappa.pvd",  "compressed") << project(dK_dkappa,   kappa_space, name = "dK_dkappa")

# Optimality constrained derivative
dJs1 = Function(inflow_space, static = True)
function_set_values(dJs1, numpy.linalg.solve(H, function_get_values(dK_dinflow)))
reset()
add_tlm(T_inflow, dJs1)
start_manager()
_, J, K = forward(T_inflow, kappa, T_N_ref = T_N_ref)
dJ = J.tlm(T_inflow, dJs1)
stop_manager()
dJs2 = compute_gradient(dJ, kappa)
function_axpy(dK_dkappa, -1.0, dJs2)

File("dK_dkappa_2.pvd", "compressed") << project(dK_dkappa, kappa_space, name = "dK_dkappa_2")

if verify:
  # Verify the optimality constrained derivative

  def inversion(kappa, T_N_ref):
    T_inflow = Function(inflow_space, name = "T_inflow", static = True)
    
    reset()
    start_manager()
    _, J, K = forward(T_inflow, kappa, T_N_ref = T_N_ref)
    stop_manager()
    
    ddJ = SingleBlockHessian(J)
    H = numpy.empty((function_local_size(T_inflow), function_local_size(T_inflow)), dtype = numpy.float64)
    for i in range(H.shape[0]):
      info("Building Hessian row %i of %i" % (i + 1, H.shape[0]))
      dm = Function(inflow_space, static = True)
      dm.vector()[i] = 1.0
      H[i, :] = function_get_values(ddJ.action(T_inflow, dm)[2])
      del(dm)
    del(ddJ)

    dJ = compute_gradient(J, T_inflow)
    function_set_values(T_inflow, -numpy.linalg.solve(H, function_get_values(dJ)))
    
    return T_inflow
  
  perturb = function_new(kappa, name = "perturb")
  function_set_values(perturb, 2.0 * numpy.random.random(function_local_size(perturb)) - 1.0)
  File("taylor_perturb.pvd", "compressed") << perturb
  
  K_val = K.value()
  K_vals = []
  errors_0 = []
  errors_1 = []
  eps_values = numpy.array([1.0e-6 * (2 ** -p) for p in range(0, 6)], dtype = numpy.float64)
  for eps in eps_values:
    kappa_perturb = function_copy(kappa, static = True)
    function_axpy(kappa_perturb, eps, perturb)
    T_inflow_perturb = inversion(kappa_perturb, T_N_ref)
    _, J_perturb, K_perturb = forward(T_inflow_perturb, kappa_perturb, T_N_ref = T_N_ref)
    K_vals.append(K_perturb.value())
    errors_0.append(abs(K_vals[-1] - K_val))
    errors_1.append(abs(K_vals[-1] - K_val - eps * function_inner(dK_dkappa, perturb)))
  K_vals = numpy.array(K_vals, dtype = numpy.float64)
  info("Functional values: %s" % K_vals)
  errors_0 = numpy.array(errors_0, dtype = numpy.float64)
  orders_0 = numpy.log(errors_0[1:] / errors_0[:-1]) / numpy.log(0.5)
  info("Errors 0: %s" % errors_0)
  info("Orders 0: %s" % orders_0)
  errors_1 = numpy.array(errors_1, dtype = numpy.float64)
  orders_1 = numpy.log(errors_1[1:] / errors_1[:-1]) / numpy.log(0.5)
  info("Errors 1: %s" % errors_1)
  info("Orders 1: %s" % orders_1)
  
  h = h5py.File("taylor.hdf5", "w")
  h.create_dataset("eps_values", data = eps_values, compression = True, fletcher32 = True, shuffle = True)
  h.create_dataset("K_vals", data = K_vals, compression = True, fletcher32 = True, shuffle = True)
  h.create_dataset("errors_0", data = errors_0, compression = True, fletcher32 = True, shuffle = True)
  h.create_dataset("errors_1", data = errors_1, compression = True, fletcher32 = True, shuffle = True)
  h.close()
