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

from firedrake import *
from tlm_adjoint import *
from tlm_adjoint import manager as _manager
stop_manager()

import petsc4py.PETSc
petsc4py.PETSc.Options().setValue("citations", "petsc.bib")

import numpy
numpy.random.seed(2212983)

import h5py

mesh = UnitSquareMesh(50, 50)
X = SpatialCoordinate(mesh)
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)
bc = DirichletBC(space, 0.0, "on_boundary", static = True, homogeneous = True)

dt = Constant(0.01, static = True)
N = 10
kappa = Function(space, name = "kappa", static = True)
function_assign(kappa, 1.0)
Psi_0 = Function(space, name = "Psi_0", static = True)
Psi_0.interpolate(exp(X[0]) * sin(pi * X[0])
                * sin(10.0 * pi * X[0])
                * sin(2.0 * pi * X[1]))

zeta_1 = Function(space, name = "zeta_1", static = True)
zeta_2 = Function(space, name = "zeta_2", static = True)
function_set_values(zeta_1, 2.0 * numpy.random.random(function_local_size(zeta_1)) - 1.0)
function_set_values(zeta_2, 2.0 * numpy.random.random(function_local_size(zeta_2)) - 1.0)
File("zeta_1.pvd", "compressed").write(zeta_1)
File("zeta_2.pvd", "compressed").write(zeta_2)

def forward(kappa, manager = None, output_filename = None):
  Psi_n = Function(space, name = "Psi_n")                             
  Psi_np1 = Function(space, name = "Psi_np1")

  eq = EquationSolver(inner(test, trial / dt) * dx
                    + inner(grad(test), kappa * grad(trial)) * dx
                    == inner(test, Psi_n / dt) * dx, Psi_np1, bc, solver_parameters = {"ksp_type":"preonly", "pc_type":"lu"})
  cycle = AssignmentSolver(Psi_np1, Psi_n)

  if not output_filename is None:
    f = File(output_filename, "compressed")

  AssignmentSolver(Psi_0, Psi_n).solve(manager = manager) 
  if not output_filename is None:
    f.write(Psi_n)
  for n in range(N):
    eq.solve(manager = manager)
    if n < N - 1:
      cycle.solve(manager = manager)
      (_manager() if manager is None else manager).new_block()
    else:
      Psi_n = Psi_np1
      Psi_n.rename("Psi_n", "a Function")
      del(Psi_np1)
    if not output_filename is None:
      f.write(Psi_n)

  J = Functional(name = "J")
  J.assign(inner(Psi_n, Psi_n) * dx, manager = manager)
  
  return J

def tlm(kappa, zeta):
  manager = _manager().new()
  manager.start()
  manager.add_tlm(kappa, zeta)
  J = forward(kappa, manager = manager)
  manager.stop()
  return J.tlm(kappa, zeta, manager = manager).value()

start_manager()
add_tlm(kappa, zeta_1)
add_tlm(kappa, zeta_2)
J = forward(kappa, output_filename = "forward.pvd")
dJ_tlm_1 = J.tlm(kappa, zeta_1)
dJ_tlm_2 = J.tlm(kappa, zeta_2)
ddJ_tlm = dJ_tlm_1.tlm(kappa, zeta_2)
stop_manager()

dJ_adj, ddJ_adj, dddJ_adj = compute_gradient([J, dJ_tlm_1, ddJ_tlm], kappa)

def info_compare(x, y):
  info("%.16e %.16e %.16e" % (x, y, abs(x - y)))

info("TLM/adjoint consistency, zeta_1")
info_compare(dJ_tlm_1.value(), function_inner(dJ_adj, zeta_1))

info("TLM/adjoint consistency, zeta_2")
info_compare(dJ_tlm_2.value(), function_inner(dJ_adj, zeta_2))

info("Second order TLM/adjoint consistency")
info_compare(ddJ_tlm.value(), function_inner(ddJ_adj, zeta_2))

kappa_perturb = Function(space, name = "kappa_perturb", static = True)

eps_values = 1.0e-2 * numpy.array([2 ** -p for p in range(20)], dtype = numpy.float64)
errors_0 = []
errors_1 = []
errors_2 = []
for eps in eps_values:
  function_assign(kappa_perturb, kappa)
  function_axpy(kappa_perturb, eps, zeta_1)
  clear_caches()
  J_perturb = forward(kappa_perturb)
  errors_0.append(abs(J_perturb.value() - J.value()))
  errors_1.append(abs(J_perturb.value() - J.value()
                        - eps * dJ_tlm_1.value()))
  errors_2.append(abs(J_perturb.value() - J.value()
                        - eps * dJ_tlm_1.value()
                        - 0.5 * eps * eps * function_inner(ddJ_adj, zeta_1)))
errors_0 = numpy.array(errors_0, dtype = numpy.float64)
orders_0 = numpy.log(errors_0[1:] / errors_0[:-1]) / numpy.log(0.5)
errors_1 = numpy.array(errors_1, dtype = numpy.float64)
orders_1 = numpy.log(errors_1[1:] / errors_1[:-1]) / numpy.log(0.5)
errors_2 = numpy.array(errors_2, dtype = numpy.float64)
orders_2 = numpy.log(errors_2[1:] / errors_2[:-1]) / numpy.log(0.5)

h = h5py.File("taylor_0.hdf5", "w")
h.create_dataset("eps_values", data = eps_values, compression = True, fletcher32 = True, shuffle = True)
h.create_dataset("errors_0", data = errors_0, compression = True, fletcher32 = True, shuffle = True)
h.create_dataset("errors_1", data = errors_1, compression = True, fletcher32 = True, shuffle = True)
h.create_dataset("errors_2", data = errors_2, compression = True, fletcher32 = True, shuffle = True)
h.close()

eps_values = 1.0e-2 * numpy.array([2 ** -p for p in range(20)], dtype = numpy.float64)
errors_0 = []
errors_1 = []
errors_2 = []
for eps in eps_values:
  function_assign(kappa_perturb, kappa)
  function_axpy(kappa_perturb, eps, zeta_1)
  clear_caches()
  dJ_tlm_2_perturb = tlm(kappa_perturb, zeta_2)
  errors_0.append(abs(dJ_tlm_2_perturb - dJ_tlm_2.value()))
  errors_1.append(abs(dJ_tlm_2_perturb - dJ_tlm_2.value()
                        - eps * ddJ_tlm.value()))
  errors_2.append(abs(dJ_tlm_2_perturb - dJ_tlm_2.value()
                        - eps * ddJ_tlm.value() - 0.5 * eps * eps * function_inner(dddJ_adj, zeta_1)))
errors_0 = numpy.array(errors_0, dtype = numpy.float64)
orders_0 = numpy.log(errors_0[1:] / errors_0[:-1]) / numpy.log(0.5)
errors_1 = numpy.array(errors_1, dtype = numpy.float64)
orders_1 = numpy.log(errors_1[1:] / errors_1[:-1]) / numpy.log(0.5)
errors_2 = numpy.array(errors_2, dtype = numpy.float64)
orders_2 = numpy.log(errors_2[1:] / errors_2[:-1]) / numpy.log(0.5)

h = h5py.File("taylor_1.hdf5", "w")
h.create_dataset("eps_values", data = eps_values, compression = True, fletcher32 = True, shuffle = True)
h.create_dataset("errors_0", data = errors_0, compression = True, fletcher32 = True, shuffle = True)
h.create_dataset("errors_1", data = errors_1, compression = True, fletcher32 = True, shuffle = True)
h.create_dataset("errors_2", data = errors_2, compression = True, fletcher32 = True, shuffle = True)
h.close()
