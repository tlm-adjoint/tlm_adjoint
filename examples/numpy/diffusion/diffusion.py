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

from tlm_adjoint.numpy import *

import numpy as np
from scipy.sparse import identity, lil_matrix
import scipy.sparse.linalg

# SciPy backwards compatibility
import inspect
cg_atol = "atol" in inspect.signature(scipy.sparse.linalg.cg).parameters


def cg(A, b, x0):
    # SciPy backwards compatibility
    if cg_atol:
        return scipy.sparse.linalg.cg(A, b, x0=x0, tol=1.0e-10, atol=1.0e-14)
    else:
        return scipy.sparse.linalg.cg(A, b, x0=x0, tol=1.0e-10)


N_t = 10
reset_manager("multistage", {"blocks": N_t, "snaps_on_disk": 1,
                             "snaps_in_ram": 1})
stop_manager()
np.random.seed(48377229)

dtype = default_dtype()
assert issubclass(dtype, (float, np.floating))

kappa_0 = 0.2
dt = 0.01
L = 1.0
N = 50

x = np.linspace(0.0, L, N + 1, dtype=dtype)
y = x.copy()
dx = 1.0 / N

space = FunctionSpace((N + 1) * (N + 1))


def index(i, j):
    return (N + 1) * i + j


def K(kappa):
    kappa = kappa.vector()
    K = lil_matrix(((N + 1) * (N + 1), (N + 1) * (N + 1)), dtype=dtype)
    for i in range(1, N):
        for j in range(1, N):
            if i > 1:
                K[index(i, j), index(i - 1, j)] = \
                    -0.5 * (kappa[index(i, j)] + kappa[index(i - 1, j)])
            if j > 1:
                K[index(i, j), index(i, j - 1)] = \
                    -0.5 * (kappa[index(i, j)] + kappa[index(i, j - 1)])
            if i < N - 1:
                K[index(i, j), index(i + 1, j)] = \
                    -0.5 * (kappa[index(i, j)] + kappa[index(i + 1, j)])
            if j < N - 1:
                K[index(i, j), index(i, j + 1)] = \
                    -0.5 * (kappa[index(i, j)] + kappa[index(i, j + 1)])
            K[index(i, j), index(i, j)] = (2.0 * kappa[index(i, j)]
                                           + 0.5 * kappa[index(i - 1, j)]
                                           + 0.5 * kappa[index(i, j - 1)]
                                           + 0.5 * kappa[index(i + 1, j)]
                                           + 0.5 * kappa[index(i, j + 1)])
    return (K * dt / (dx * dx)).tocsr()


def dK_dkappa_adjoint_action(psi, adj_psi):
    psi = psi.vector()
    adj_psi = adj_psi.vector()
    b = np.zeros((N + 1) * (N + 1), dtype=dtype)
    for i in range(1, N):
        for j in range(1, N):
            if i > 1:
                b[index(i, j)] += -0.5 * psi[index(i - 1, j)] * adj_psi[index(i, j)]  # noqa: E501
                b[index(i - 1, j)] += -0.5 * psi[index(i - 1, j)] * adj_psi[index(i, j)]  # noqa: E501
            if j > 1:
                b[index(i, j)] += -0.5 * psi[index(i, j - 1)] * adj_psi[index(i, j)]  # noqa: E501
                b[index(i, j - 1)] += -0.5 * psi[index(i, j - 1)] * adj_psi[index(i, j)]  # noqa: E501
            if i < N - 1:
                b[index(i, j)] += -0.5 * psi[index(i + 1, j)] * adj_psi[index(i, j)]  # noqa: E501
                b[index(i + 1, j)] += -0.5 * psi[index(i + 1, j)] * adj_psi[index(i, j)]  # noqa: E501
            if j < N - 1:
                b[index(i, j)] += -0.5 * psi[index(i, j + 1)] * adj_psi[index(i, j)]  # noqa: E501
                b[index(i, j + 1)] += -0.5 * psi[index(i, j + 1)] * adj_psi[index(i, j)]  # noqa: E501
            b[index(i, j)] += 2.0 * psi[index(i, j)] * adj_psi[index(i, j)]
            b[index(i - 1, j)] += 0.5 * psi[index(i, j)] * adj_psi[index(i, j)]
            b[index(i, j - 1)] += 0.5 * psi[index(i, j)] * adj_psi[index(i, j)]
            b[index(i + 1, j)] += 0.5 * psi[index(i, j)] * adj_psi[index(i, j)]
            b[index(i, j + 1)] += 0.5 * psi[index(i, j)] * adj_psi[index(i, j)]
    return b * dt / (dx * dx)


def A(kappa, alpha=1.0, beta=1.0):
    return (alpha * identity((N + 1) * (N + 1), dtype=dtype)
            + beta * K(kappa)).tocsr()


B = identity((N + 1) * (N + 1), dtype=dtype).tocsr()

mass = lil_matrix(((N + 1) * (N + 1), (N + 1) * (N + 1)), dtype=dtype)
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
        psi, fail = cg(lA, B.dot(psi), x0=psi)
        assert fail == 0
    J = Functional(name="J")
    function_assign(J.function(), (psi + 1.0).dot(mass.dot(psi + 1.0)))
    return J


def forward(psi_0, kappa):
    class DiffusionMatrix(Matrix):
        def __init__(self, kappa, alpha=1.0, beta=1.0):
            super().__init__(nl_deps=[kappa], ic=True, adj_ic=True)
            self._alpha = alpha
            self._beta = beta

            self._A = None
            self._A_kappa = None

        def forward_action(self, nl_deps, x, b, *, method="assign"):
            check_space_type(x, "primal")
            check_space_type(b, "conjugate_dual")

            kappa, = nl_deps
            self._assemble_A(kappa)
            sb = self._A.dot(x.vector())
            if method == "assign":
                b.vector()[:] = sb
            elif method == "add":
                b.vector()[:] += sb
            elif method == "sub":
                b.vector()[:] -= sb
            else:
                raise ValueError(f"Invalid method: '{method:s}'")

        def adjoint_action(self, nl_deps, adj_x, b, b_index=0, *,
                           method="assign"):
            if b_index != 0:
                raise IndexError("Invalid index")
            self.forward_action(nl_deps, adj_x, b, method=method)

        def forward_solve(self, x, nl_deps, b):
            check_space_type(x, "primal")
            check_space_type(b, "conjugate_dual")

            kappa, = nl_deps
            self._assemble_A(kappa)
            x.vector()[:], fail = cg(self._A, b.vector(), x0=x.vector())
            assert fail == 0

        def _assemble_A(self, kappa):
            if self._A_kappa is None \
               or abs(self._A_kappa - kappa.vector()).max() > 0.0:
                self._A = A(kappa, alpha=self._alpha, beta=self._beta)
                self._A_kappa = kappa.vector().copy()

        def adjoint_derivative_action(self, nl_deps, nl_dep_index, x, adj_x, b, *,  # noqa: E501
                                      method="assign"):
            if nl_dep_index == 0:
                sb = self._beta * dK_dkappa_adjoint_action(x, adj_x)
                if method == "assign":
                    b.vector()[:] = sb
                elif method == "add":
                    b.vector()[:] += sb
                elif method == "sub":
                    b.vector()[:] -= sb
                else:
                    raise ValueError(f"Invalid method: '{method:s}'")
            else:
                raise IndexError("nl_dep_index out of bounds")

        def adjoint_solve(self, adj_x, nl_deps, b):
            kappa, = nl_deps
            self._assemble_A(kappa)
            adj_x.vector()[:], fail = cg(self._A, b.vector(),
                                         x0=adj_x.vector())
            assert fail == 0
            return adj_x

        def tangent_linear_rhs(self, M, dM, tlm_map, x):
            kappa, = self.nonlinear_dependencies()
            tau_kappa = get_tangent_linear(kappa, M, dM, tlm_map)
            if tau_kappa is None:
                return None
            else:
                return MatrixActionRHS(DiffusionMatrix(tau_kappa, alpha=0.0,
                                                       beta=-self._beta),
                                       x)

    psi_n = Function(space, name="psi_n")
    psi_np1 = Function(space, name="psi_np1")
    Assignment(psi_n, psi_0).solve()

    eqs = [LinearEquation(psi_np1, ContractionRHS(B, (1,), (psi_n,)),
                          A=DiffusionMatrix(kappa)),
           Assignment(psi_n, psi_np1)]

    for n in range(N_t):
        for eq in eqs:
            eq.solve()
        if n < N_t - 1:
            new_block()

    one = Function(space, name="one", static=True)
    function_assign(one, 1.0)
    phi = Function(space, name="phi")
    Axpy(phi, psi_n, 1.0, one).solve()

    J = Functional(name="J")
    InnerProduct(J.function(), phi, phi, M=ConstantMatrix(mass)).solve()

    return J


psi_0 = Function(space, name="psi_0", static=True)
psi_0_a = psi_0.vector().reshape((N + 1, N + 1))
for i in range(N + 1):
    for j in range(N + 1):
        psi_0_a[i, j] = (np.exp(x[i]) * np.sin(np.pi * x[i])
                         * np.sin(10.0 * np.pi * x[i])
                         * np.sin(2.0 * np.pi * y[j]))

kappa = Function(space, name="kappa", static=True)
kappa.vector()[:] = kappa_0

J_ref = forward_reference(psi_0, kappa)

start_manager()
J = forward(psi_0, kappa)
stop_manager()

print(f"J = {J.value():.16e}")
print(f"Reference J = {J_ref.value():.16e}")
print(f"Error norm = {abs(J_ref.value() - J.value()):.16e}")
assert abs(J_ref.value() - J.value()) < 1.0e-14

dJ_dpsi_0, dJ_dkappa = compute_gradient(J, [psi_0, kappa])
del J


def forward_J(psi_0):
    return forward(psi_0, kappa)


def forward_reference_J(psi_0):
    return forward_reference(psi_0, kappa)


min_order = taylor_test(forward_reference_J, psi_0, J_val=J_ref.value(),
                        dJ=dJ_dpsi_0, seed=1.0e-3)
assert min_order > 1.99

min_order = taylor_test_tlm(forward_J, psi_0, tlm_order=1, seed=1.0e-3)
assert min_order > 1.99

min_order = taylor_test_tlm_adjoint(forward_J, psi_0, adjoint_order=1,
                                    seed=1.0e-3)
assert min_order > 1.99


def forward_J(kappa):
    return forward(psi_0, kappa)


def forward_reference_J(kappa):
    return forward_reference(psi_0, kappa)


min_order = taylor_test(forward_reference_J, kappa, J_val=J_ref.value(),
                        dJ=dJ_dkappa, seed=1.0e-3)
assert min_order > 1.99

ddJ = Hessian(forward_J)
min_order = taylor_test(forward_reference_J, kappa, J_val=J_ref.value(),
                        ddJ=ddJ, seed=1.0e-2)
assert min_order > 2.97

min_order = taylor_test_tlm(forward_J, kappa, tlm_order=1, seed=1.0e-3)
assert min_order > 1.99

min_order = taylor_test_tlm_adjoint(forward_J, kappa, adjoint_order=1,
                                    seed=1.0e-3)
assert min_order > 2.00

min_order = taylor_test_tlm_adjoint(forward_J, kappa, adjoint_order=2,
                                    seed=1.0e-3)
assert min_order > 1.99
