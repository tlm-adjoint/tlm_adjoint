#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint import *

import numpy;  numpy.random.seed(16143324)
import scipy.sparse
import scipy.sparse.linalg

kappa_0 = 1.0
dt = 0.01
L = 1.0
N = 50
N_t = 10

x = numpy.linspace(0.0, L, N + 1)
y = x
dx = 1.0 / N

index = lambda i, j : (N + 1) * i + j

def K(kappa):
  kappa = kappa.vector()
  K = scipy.sparse.lil_matrix(((N + 1) * (N + 1), (N + 1) * (N + 1)), dtype = numpy.float64)
  for i in range(1, N):
    for j in range(1, N):
      if i > 1: K[index(i, j), index(i - 1, j)] = -0.5 * (kappa[index(i, j)] + kappa[index(i - 1, j)]) * dt / (dx * dx)
      if j > 1: K[index(i, j), index(i, j - 1)] = -0.5 * (kappa[index(i, j)] + kappa[index(i, j - 1)]) * dt / (dx * dx)
      if i < N - 1: K[index(i, j), index(i + 1, j)] = -0.5 * (kappa[index(i, j)] + kappa[index(i + 1, j)]) * dt / (dx * dx)
      if j < N - 1: K[index(i, j), index(i, j + 1)] = -0.5 * (kappa[index(i, j)] + kappa[index(i, j + 1)]) * dt / (dx * dx)
      K[index(i, j), index(i, j)] = (2.0 * kappa[index(i, j)]
                                   + 0.5 * kappa[index(i - 1, j)]
                                   + 0.5 * kappa[index(i, j - 1)]
                                   + 0.5 * kappa[index(i + 1, j)]
                                   + 0.5 * kappa[index(i, j + 1)]) * dt / (dx * dx)
  K = K.tocsr()
  return K
  
def K_kappa(psi):
  psi = psi.vector()
  K = scipy.sparse.lil_matrix(((N + 1) * (N + 1), (N + 1) * (N + 1)), dtype = numpy.float64)
  for i in range(1, N):
    for j in range(1, N):
      if i > 1:
        K[index(i, j), index(i, j)] += -0.5 * psi[index(i - 1, j)] * dt / (dx * dx)
        K[index(i - 1, j), index(i, j)] += -0.5 * psi[index(i - 1, j)] * dt / (dx * dx)
      if j > 1:
        K[index(i, j), index(i, j)] += -0.5 * psi[index(i, j - 1)] * dt / (dx * dx)
        K[index(i, j - 1), index(i, j)] += -0.5 * psi[index(i, j - 1)] * dt / (dx * dx)
      if i < N - 1:
        K[index(i, j), index(i, j)] += -0.5 * psi[index(i + 1, j)] * dt / (dx * dx)
        K[index(i + 1, j), index(i, j)] += -0.5 * psi[index(i + 1, j)] * dt / (dx * dx)
      if j < N - 1:
        K[index(i, j), index(i, j)] += -0.5 * psi[index(i, j + 1)] * dt / (dx * dx)
        K[index(i, j + 1), index(i, j)] += -0.5 * psi[index(i, j + 1)] * dt / (dx * dx)
      K[index(i, j), index(i, j)] += 2.0 * psi[index(i, j)] * dt / (dx * dx)
      K[index(i - 1, j), index(i, j)] += 0.5 * psi[index(i, j)] * dt / (dx * dx)
      K[index(i, j - 1), index(i, j)] += 0.5 * psi[index(i, j)] * dt / (dx * dx)
      K[index(i + 1, j), index(i, j)] += 0.5 * psi[index(i, j)] * dt / (dx * dx)
      K[index(i, j + 1), index(i, j)] += 0.5 * psi[index(i, j)] * dt / (dx * dx)
  K = K.tocsr()
  return K

def A(kappa, alpha = 1.0, beta = 1.0):
  return (alpha * scipy.sparse.identity((N + 1) * (N + 1), dtype = numpy.float64) + beta * K(kappa)).tocsr()

B = scipy.sparse.lil_matrix(((N + 1) * (N + 1), (N + 1) * (N + 1)), dtype = numpy.float64)
for i in range(1, N):
  for j in range(1, N):
    B[index(i, j), index(i, j)] = 1.0
B = B.tocsr()

mass = scipy.sparse.lil_matrix(((N + 1) * (N + 1), (N + 1) * (N + 1)), dtype = numpy.float64)
mass[index(0, 0), index(0, 0)] = 0.25 * dx * dx
mass[index(0, N), index(0, N)] = 0.25 * dx * dx
mass[index(N, 0), index(N, 0)] = 0.25 * dx * dx
mass[index(N, N), index(N, N)] = 0.25 * dx * dx
for i in range(1, N):
  mass[index(i, 0), index(i, 0)] = 0.5 * dx * dx
  mass[index(i, N), index(i, N)] = 0.5 * dx * dx
  mass[index(0, i), index(0, i)] = 0.5 * dx * dx
  mass[index(N, i), index(N, i)] = 0.5 * dx * dx
  for j in range(1, N):
    mass[index(i, j), index(i, j)] = dx * dx

def forward_reference(psi_0, kappa):
  lA = A(kappa)
  psi = psi_0.vector().copy()
  for n in range(N_t):
    psi, fail = scipy.sparse.linalg.cg(lA, B.dot(psi), x0 = psi, tol = 1.0e-10, atol = 1.0e-14)
    assert(fail == 0)
  J = Functional(name = "J")
  J.fn().vector()[:] = psi.dot(mass.dot(psi))
  return J

def forward(psi_0, kappa):
  class DiffusionMatrix(Matrix):
    def __init__(self, kappa, alpha = 1.0, beta = 1.0):
      Matrix.__init__(self, nl_deps = [kappa])
      self._x_0_forward = Function(space)
      self._x_0_adjoint = Function(space)
      self._alpha = alpha
      self._beta = beta
      self.reset_forward_solve()
      
    def add_forward_action(self, b, nl_deps, x):
      kappa, = nl_deps
      self._assemble_A(kappa)
      b.vector()[:] += self._A.dot(x.vector())
    
    def reset_add_forward_action(self):
      self.reset_forward_solve()
  
    def add_adjoint_action(self, b, nl_deps, x):
      self.add_forward_action(b, nl_deps, x)
    
    def reset_add_adjoint_action(self):
      self.reset_forward_solve()
  
    def forward_solve(self, b, nl_deps):
      self._assemble_A(kappa)
      x = function_new(self._x_0_forward)
      x.vector()[:], fail = scipy.sparse.linalg.cg(self._A, b.vector(), x0 = self._x_0_forward.vector(), tol = 1.0e-10, atol = 1.0e-14)
      assert(fail == 0)
      function_assign(self._x_0_forward, x)
      return x
    
    def reset_forward_solve(self):
      self._A = None
      self._A_kappa = None
      
    def _assemble_A(self, kappa):
      if self._A_kappa is None or kappa != self._A_kappa:
        self._A = A(kappa, alpha = self._alpha, beta = self._beta)
        self._A_kappa = kappa
    
    def add_adjoint_derivative_action(self, b, nl_deps, nl_dep_index, adj_x, x):
      if nl_dep_index == 0:
        b.vector()[:] += self._beta * K_kappa(x).dot(adj_x.vector())
    
    def adjoint_jacobian_solve(self, b, nl_deps):
      self._assemble_A(kappa)
      x = function_new(self._x_0_forward)
      x.vector()[:], fail = scipy.sparse.linalg.cg(self._A, b.vector(), x0 = self._x_0_adjoint.vector(), tol = 1.0e-10, atol = 1.0e-14)
      assert(fail == 0)
      function_assign(self._x_0_adjoint, x)
      return x
    
    def reset_adjoint_jacobian_solve(self):
      self.reset_forward_solve()
    
    def tangent_linear_rhs(self, M, dM, tlm_map, x):
      kappa, = self.nonlinear_dependencies()
      try:
        tau_kappa = dM[M.index(kappa)]
      except ValueError:
        tau_kappa = tlm_map[kappa]
      if tau_kappa is None:
        return None
      else:
        return MatrixActionRHS(DiffusionMatrix(tau_kappa, alpha = 0.0, beta = -self._beta), x)
    
  psi_n = Function(space, name = "psi_n")
  psi_np1 = Function(space, name = "psi_np1")
  AssignmentSolver(psi_0, psi_n).solve(replace = True)
  
  eqs = [LinearEquation(ContractionRHS(B, (1,), (psi_n,)), psi_np1, A = DiffusionMatrix(kappa)),
         AssignmentSolver(psi_np1, psi_n)]
  for n in range(N_t):
    for eq in eqs:
      eq.solve()
    if n < N_t - 1:
      new_block()
  for eq in eqs:
    eq.replace()
  
  J = Functional(name = "J")
  NormSqEquation(psi_n, J.fn(), M = mass).solve(replace = True)
  
  return J

space = FunctionSpace((N + 1) * (N + 1))

psi_0 = Function(space, name = "psi_0", static = True)
psi_0_a = psi_0.vector().reshape((N + 1, N + 1))
for i in range(N + 1):
  for j in range(N + 1):
    psi_0_a[i, j] = numpy.exp(x[0]) * numpy.sin(numpy.pi * x[i]) * numpy.sin(10.0 * numpy.pi * x[i]) * numpy.sin(2.0 * numpy.pi * y[j])

kappa = Function(space, name = "kappa", static = True)
kappa.vector()[:] = kappa_0

J_ref = forward_reference(psi_0, kappa)

start_manager()
J = forward(psi_0, kappa)
stop_manager()

info("J = %.16e" % J.value())
info("Reference J = %.16e" % J_ref.value())
info("Error = %.16e" % abs(J_ref.value() - J.value()))
assert(abs(J_ref.value() - J.value()) < 1.0e-24)

dJ_dpsi_0, dJ_dkappa = compute_gradient(J, [psi_0, kappa])
del(J)

min_order = taylor_test(lambda psi_0 : forward_reference(psi_0, kappa), psi_0, J_val = J_ref.value(), dJ = dJ_dpsi_0)
assert(min_order > 2.00)

min_order = taylor_test(lambda kappa : forward_reference(psi_0, kappa), kappa, J_val = J_ref.value(), dJ = dJ_dkappa)
assert(min_order > 1.98)

reset()
clear_caches()
stop_manager(annotation = True, tlm = False)
zeta = Function(space, name = "zeta", static = True)
zeta.vector()[:] = numpy.random.random(zeta.vector().shape)
add_tlm(psi_0, zeta)
J = forward(psi_0, kappa)
stop_manager()
info("dJ_dpsi_0_adj = %.16e" % dJ_dpsi_0.vector().dot(zeta.vector()))
info("dJ_dpsi_0_tlm = %.16e" % J.tlm(psi_0, zeta).fn().vector()[0])
info("Error = %.16e" % abs(J.tlm(psi_0, zeta).fn().vector()[0] - dJ_dpsi_0.vector().dot(zeta.vector())))
assert(abs(J.tlm(psi_0, zeta).fn().vector()[0] - dJ_dpsi_0.vector().dot(zeta.vector())) < 1.0e-15)
reset()
clear_caches()
stop_manager()

reset()
clear_caches()
stop_manager(annotation = True, tlm = False)
zeta = Function(space, name = "zeta", static = True)
zeta.vector()[:] = 2.0 * numpy.random.random(zeta.vector().shape) - 1.0
add_tlm(kappa, zeta)
J = forward(psi_0, kappa)
stop_manager()
info("dJ_dkappa_adj = %.16e" % dJ_dkappa.vector()[0])
info("dJ_dkappa_tlm = %.16e" % J.tlm(kappa, zeta).fn().vector()[0])
info("Error = %.16e" % abs(dJ_dkappa.vector()[0] - J.tlm(kappa, zeta).fn().vector()[0]))
assert(abs(dJ_dkappa.vector()[0] - J.tlm(kappa, zeta).fn().vector()[0]) < 1.0e-10)
reset()
clear_caches()
stop_manager()

ddJ = Hessian(lambda kappa : forward(psi_0, kappa))
min_order = taylor_test(lambda kappa : forward_reference(psi_0, kappa), kappa, m0 = kappa, J_val = J_ref.value(), dJ = dJ_dkappa, ddJ = ddJ, seed = 0.05)
assert(min_order > 2.95)
