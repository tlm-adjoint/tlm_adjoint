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

# Import FEniCS and tlm_adjoint
from fenics import *
from tlm_adjoint.fenics import *
# Import an optimization module, used for Hessian actions with single block
# forward models
from tlm_adjoint.hessian_optimization import *

# import h5py
import numpy as np
# import petsc4py.PETSc as PETSc
import time

# Disable the manager until it is needed
stop_manager()

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -march=native"
parameters["form_compiler"]["optimize"] = True

# PETSc.Options().setValue("citations", "petsc.bib")

# Seed the random number generator, to ensure reproducibility of the later
# Taylor verification
np.random.seed(1709465)

# Enable Taylor verification
verify = True

# Configure the mesh
L_x, L_y = 2.0, 1.0
N_x, N_y = 20, 10
mesh = RectangleMesh(Point(0.0, 0.0), Point(L_x, L_y), N_x, N_y)

# Configure the interior domain discrete function space
space = FunctionSpace(mesh, "Lagrange", 1)
test, trial = TestFunction(space), TrialFunction(space)

# Approximate Courant number, relative to background uniform flow
C = 0.05
# Number of timesteps
N_t = 10 * N_x
# Approximate grid Péclet number, relative to background uniform flow
Pe = 20.0
# Time step size
dt = Constant(L_x * C / float(N_x), static=True)
info(f"dt = {float(dt):.16e}")
# Diffusivity
kappa_space = FunctionSpace(mesh, "Discontinuous Lagrange", 0)
kappa = Function(kappa_space, name="kappa", static=True)
kappa.assign(Constant(L_x / (Pe * float(N_x))))
info(f"kappa = {function_max_value(kappa):.16e}")
# Regularization parameter
alpha = Constant(1.0e-8, static=True)

# Stream function
psi = Function(space, name="psi", static=True)
psi.interpolate(Expression(
    "(1.0 - exp(x[1])) * sin(k * pi * x[0]) * sin(m * pi * x[1]) - x[1]",
    element=psi.function_space().ufl_element(),
    k=1.0, m=1.0))


class InflowBoundary(SubDomain):
    # Following the FEniCS 2019.1.0 API
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS


class OutflowBoundary(SubDomain):
    # Following the FEniCS 2019.1.0 API
    def inside(self, x, on_boundary):
        return abs(x[0] - L_x) < DOLFIN_EPS


# Mark the outflow boundary
boundary_markers = MeshFunction("size_t", mesh, 1)
boundary_markers.set_all(0)
OutflowBoundary().mark(boundary_markers, 1)

# Extract the inflow mesh
boundary_mesh = BoundaryMesh(mesh, "exterior")
boundary_mesh_markers = MeshFunction("size_t", boundary_mesh, 1)
boundary_mesh_markers.set_all(0)
InflowBoundary().mark(boundary_mesh_markers, 1)
inflow_mesh = SubMesh(boundary_mesh, boundary_mesh_markers, 1)
# Configure the inflow domain discrete function space
inflow_space = FunctionSpace(inflow_mesh, "Lagrange", 1)
# Inflow boundary condition
T_inflow = Function(inflow_space, name="T_inflow", static=True)
T_inflow.interpolate(Expression(
    "sin(pi * x[1]) + 0.4 * sin(3.0 * pi * x[1])",
    element=T_inflow.function_space().ufl_element()))

forward_calls = [0]


def forward(T_inflow_bc, kappa, T_N_ref=None, output_filename=None):
    t0 = time.time()
    # Clear assembly and linear solver caches
    clear_caches()

    # An equation which sets T = T_bc on the boundary at x = 0, and T = 0
    # elsewhere
    class InflowBCSolver(Equation):
        def __init__(self, T_bc, T):
            bc = DirichletBC(T.function_space(),
                             Expression("x[1]", degree=1),
                             "fabs(x[0]) < DOLFIN_EPS")
            bc = bc.get_boundary_values()
            nodes = tuple(bc.keys())
            y = tuple(bc.values())

            bc_adj = DirichletBC(T_bc.function_space(),
                                 Expression("x[1]", degree=1),
                                 "fabs(x[0]) < DOLFIN_EPS")
            bc_adj = bc_adj.get_boundary_values()
            nodes_adj = tuple(bc_adj.keys())
            y_adj = tuple(bc_adj.values())

            super().__init__(T, [T, T_bc], nl_deps=[], ic=False, adj_ic=False)
            self._nodes = nodes
            self._y = y
            self._nodes_adj = nodes_adj
            self._y_adj = y_adj

        def forward_solve(self, x, deps=None):
            _, T_bc = self.dependencies() if deps is None else deps
            x_arr = np.zeros(function_local_size(x),
                             dtype=np.float64)
            for node, y in zip(self._nodes, self._y):
                x_arr[node] = T_bc(np.array([0.0, y],
                                   dtype=np.float64))
            function_set_values(x, x_arr)

        def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
            if dep_index == 0:
                return adj_x
            elif dep_index == 1:
                F = function_new(self.dependencies()[1])
                F_arr = np.zeros(function_local_size(F), dtype=np.float64)
                for node, y in zip(self._nodes_adj, self._y_adj):
                    F_arr[node] = -adj_x(np.array([0.0, y], dtype=np.float64))
                function_set_values(F, F_arr)
                return F
            else:
                raise EquationException("dep_index out of bounds")

        def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
            return b

        def tangent_linear(self, M, dM, tlm_map):
            T, T_bc = self.dependencies()

            tau_T_bc = get_tangent_linear(T_bc, M, dM, tlm_map)
            if tau_T_bc is None:
                return NullSolver(tlm_map[T])
            else:
                return InflowBCSolver(tau_T_bc, tlm_map[T])

    # A function equal to the inflow boundary condition value on the inflow,
    # and equal to zero elsewhere
    T_inflow = Function(space, name="T_inflow")
    # Boundary condition application equation
    InflowBCSolver(T_inflow_bc, T_inflow).solve()

    # Solution on the previous time level
    T_n = Function(space, name="T_n")
    # Solution on the next time level, subject to homogenized boundary
    # conditions
    T_np1_0 = Function(space, name="T_np1_0")

    T_np1 = T_inflow + trial
    T_nph = Constant(0.5, static=True) * (T_n + T_np1)

    def perp(v):
        return as_vector([-v[1], v[0]])

    # Timestep equation, subject to homogenized boundary conditions
    F = (inner(test, (T_np1 - T_n) / dt) * dx
         + inner(test, dot(perp(grad(psi)), grad(T_nph))) * dx
         + inner(grad(test), kappa * grad(T_nph)) * dx)
    timestep_eq = EquationSolver(
        lhs(F) == rhs(F),
        T_np1_0,
        HomogeneousDirichletBC(space, "fabs(x[0]) < DOLFIN_EPS"),
        solver_parameters={"linear_solver": "umfpack"})

    # Equation which constructs the complete solution on the next time level
    update_eq = AxpySolver(T_np1_0, 1.0, T_inflow, T_n)

    # Timestepping equations
    eqs = [timestep_eq, update_eq]

    if output_filename is not None:
        # Output the forward solution
        T_output = File(output_filename, "compressed")
        T_output << (T_n, 0.0)

    for n in range(N_t):
        # Timestep
        for eq in eqs:
            eq.solve()
        if output_filename is not None:
            # Output the forward solution
            T_output << (T_n, (n + 1) * float(dt))

    if T_N_ref is None:
        # Store the solution of the equation in a "reference" function
        T_N_ref = Function(space, name="T_N_ref")
        function_assign(T_N_ref, T_n)

    # First functional
    J = Functional(name="J")
    # Mis-match functional
    J.assign(inner(T_n - T_N_ref, T_n - T_N_ref)
             * ds(subdomain_data=boundary_markers)(1))
    # Regularization
    J.addto(alpha * inner(grad(T_inflow_bc), grad(T_inflow_bc)) * dx)

    # Second functional
    K = Functional(name="K")
    K.assign(inner(T_n, T_n) * dx)

    forward_calls[0] += 1
    info(f"Forward call {forward_calls[0]:d}, {time.time() - t0:.3f}s, J = {J.value():.16e}, K = {K.value():.16e}")  # noqa: E501
    return T_N_ref, J, K


# Generate a reference solution
# File("T_inflow.pvd", "compressed") << T_inflow
# T_N_ref, _, _ = forward(T_inflow, kappa, output_filename="forward.pvd")
T_N_ref, _, _ = forward(T_inflow, kappa)
# File("T_N_ref.pvd", "compressed") << T_N_ref

# Delete the original input
T_inflow = Function(inflow_space, name="T_inflow", static=True)

# Build the Hessian via brute-force
start_manager()
_, J, K = forward(T_inflow, kappa, T_N_ref=T_N_ref)
stop_manager()
ddJ = SingleBlockHessian(J)


def forward_T_inflow_ref_J(T_inflow):
    return forward(T_inflow, kappa, T_N_ref=T_N_ref)[1]


def forward_kappa_ref_J(kappa):
    return forward(T_inflow, kappa, T_N_ref=T_N_ref)[1]


def forward_T_inflow_ref_K(T_inflow):
    return forward(T_inflow, kappa, T_N_ref=T_N_ref)[2]


def forward_kappa_ref_K(kappa):
    return forward(T_inflow, kappa, T_N_ref=T_N_ref)[2]


H = np.full((function_local_size(T_inflow), function_local_size(T_inflow)),
            np.NAN, dtype=np.float64)
for i in range(H.shape[1]):
    info(f"Building Hessian column {i + 1:d} of {H.shape[1]:d}")
    dm = Function(inflow_space, static=True)
    dm.vector()[i] = 1.0
    H[:, i] = function_get_values(ddJ.action(T_inflow, dm)[2])
    clear_caches(dm)
    del dm
assert not np.isnan(H).any()
assert abs(H - H.T).max() < 1.0e-16

# Solve the optimization problem
_, dJ = ddJ.compute_gradient(T_inflow)
function_set_values(T_inflow, np.linalg.solve(H, -function_get_values(dJ)))
# File("T_inflow_inv.pvd", "compressed") << T_inflow
del ddJ

# Re-run the forward at the inverted state
reset_manager()
start_manager()
# _, J, K = forward(T_inflow, kappa, T_N_ref=T_N_ref,
#                   output_filename="inversion.pvd")
_, J, K = forward(T_inflow, kappa, T_N_ref=T_N_ref)
stop_manager()

# Forward model constrained derivatives
(dJ_dinflow, dJ_dkappa), (dK_dinflow, dK_dkappa) \
    = compute_gradient([J, K], [T_inflow, kappa])
if verify:
    # Verify the forward model constrained derivatives
    assert function_linf_norm(dJ_dinflow) < 1.0e-14
    min_order = taylor_test(forward_kappa_ref_J, kappa, J_val=J.value(),
                            dJ=dJ_dkappa, seed=1.0e-6)
    assert min_order > 1.99
    min_order = taylor_test(forward_T_inflow_ref_K, T_inflow, J_val=K.value(),
                            dJ=dK_dinflow, seed=1.0e-4)
    assert min_order > 1.99
    min_order = taylor_test(forward_kappa_ref_K, kappa, J_val=K.value(),
                            dJ=dK_dkappa, seed=1.0e-4)
    assert min_order > 1.99

    min_order = taylor_test_tlm(forward_kappa_ref_J, kappa, tlm_order=1,
                                seed=1.0e-6)
    assert min_order > 1.99
    min_order = taylor_test_tlm(forward_T_inflow_ref_K, T_inflow, tlm_order=1,
                                seed=1.0e-4)
    assert min_order > 1.99
    min_order = taylor_test_tlm(forward_kappa_ref_K, kappa, tlm_order=1,
                                seed=1.0e-6)
    assert min_order > 1.99


def project(b, space, name):
    x = Function(space, name=name)
    test, trial = TestFunction(space), TrialFunction(space)
    M = assemble(inner(test, trial) * dx)
    LUSolver(M, "umfpack").solve(x.vector(), b.vector().copy())
    return x


# File("dJ_dinflow.pvd", "compressed") << project(dJ_dinflow, inflow_space,
#                                                 name="dJ_dinflow")
# File("dJ_dkappa.pvd", "compressed") << project(dJ_dkappa, kappa_space,
#                                                name="dJ_dkappa")
# File("dK_dinflow.pvd", "compressed") << project(dK_dinflow, inflow_space,
#                                                 name="dK_dinflow")
# File("dK_dkappa.pvd", "compressed") << project(dK_dkappa, kappa_space,
#                                                name="dK_dkappa")

# Optimality constrained derivative
dJs1 = Function(inflow_space, static=True)
function_set_values(dJs1, np.linalg.solve(H, function_get_values(dK_dinflow)))
reset_manager()
add_tlm(T_inflow, dJs1)
start_manager()
_, J, K = forward(T_inflow, kappa, T_N_ref=T_N_ref)
dJ = J.tlm(T_inflow, dJs1)
stop_manager()
dJs2 = compute_gradient(dJ, kappa)
function_axpy(dK_dkappa, -1.0, dJs2)

# File("dK_dkappa_2.pvd", "compressed") << project(dK_dkappa, kappa_space,
#                                                  name="dK_dkappa_2")

if verify:
    # Verify the optimality constrained derivative

    def inversion(kappa, T_N_ref):
        T_inflow = Function(inflow_space, name="T_inflow", static=True)

        reset_manager()
        start_manager()
        _, J, K = forward(T_inflow, kappa, T_N_ref=T_N_ref)
        stop_manager()

        ddJ = SingleBlockHessian(J)
        H = np.full((function_local_size(T_inflow),
                     function_local_size(T_inflow)),
                    np.NAN, dtype=np.float64)
        for i in range(H.shape[1]):
            info(f"Building Hessian column {i + 1:d} of {H.shape[1]:d}")
            dm = Function(inflow_space, static=True)
            dm.vector()[i] = 1.0
            H[:, i] = function_get_values(ddJ.action(T_inflow, dm)[2])
            clear_caches(dm)
            del dm
        del ddJ
        assert not np.isnan(H).any()
        assert abs(H - H.T).max() < 1.0e-16

        dJ = compute_gradient(J, T_inflow)
        function_set_values(T_inflow,
                            -np.linalg.solve(H, function_get_values(dJ)))

        return T_inflow

    perturb = function_new(kappa, name="perturb")
    function_set_values(perturb,
                        2.0 * np.random.random(function_local_size(perturb))
                        - 1.0)
    # File("taylor_perturb.pvd", "compressed") << perturb

    K_val = K.value()
    K_vals = []
    error_norms_0 = []
    error_norms_1 = []
    eps_values = np.array([1.0e-4 * (2 ** -p) for p in range(6)],
                          dtype=np.float64)
    for eps in eps_values:
        kappa_perturb = function_copy(kappa, name="kappa_perturb", static=True)
        function_axpy(kappa_perturb, eps, perturb)
        T_inflow_perturb = inversion(kappa_perturb, T_N_ref)
        _, J_perturb, K_perturb = forward(T_inflow_perturb, kappa_perturb,
                                          T_N_ref=T_N_ref)
        K_vals.append(K_perturb.value())
        error_norms_0.append(abs(K_vals[-1] - K_val))
        error_norms_1.append(abs(K_vals[-1] - K_val
                                 - eps * function_inner(dK_dkappa, perturb)))
    K_vals = np.array(K_vals, dtype=np.float64)
    info(f"Functional values: {K_vals}")
    error_norms_0 = np.array(error_norms_0, dtype=np.float64)
    orders_0 = np.log(error_norms_0[1:] / error_norms_0[:-1]) / np.log(0.5)
    info(f"Error norms 0: {error_norms_0}")
    info(f"Orders 0     : {orders_0}")
    error_norms_1 = np.array(error_norms_1, dtype=np.float64)
    orders_1 = np.log(error_norms_1[1:] / error_norms_1[:-1]) / np.log(0.5)
    info(f"Error norms 1: {error_norms_1}")
    info(f"Orders 1     : {orders_1}")

    # h = h5py.File("taylor.hdf5", "w")
    # h.create_dataset("eps_values", data=eps_values, compression=True,
    #                  fletcher32=True, shuffle=True)
    # h.create_dataset("K_vals", data=K_vals, compression=True,
    #                  fletcher32=True, shuffle=True)
    # h.create_dataset("error_norms_0", data=error_norms_0, compression=True,
    #                  fletcher32=True, shuffle=True)
    # h.create_dataset("error_norms_1", data=error_norms_1, compression=True,
    #                  fletcher32=True, shuffle=True)
    # h.close()

    assert orders_1.min() > 1.99
