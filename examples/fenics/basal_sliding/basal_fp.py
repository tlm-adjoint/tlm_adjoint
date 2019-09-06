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

from fenics import *
from tlm_adjoint_fenics import *
from tlm_adjoint_fenics.hessian_optimization import *

# import h5py
import numpy as np
# import petsc4py.PETSc as PETSc
import slepc4py.SLEPc as SLEPc
import ufl

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -march=native"
parameters["form_compiler"]["optimize"] = True
stop_manager()
np.random.seed(12143432)
# PETSc.Options().setValue("citations", "petsc.bib")

# References:
# GHS09 D. Golgberg, D. M. Holland, and C. Schoof, "Grounding line movement and
#       ice shelf buttressing in marine ice sheets", Journal of Geophysical
#       Research: Earth Surface, 114(F4), F04026, 2009
# G11   D. N. Goldberg, "A variationally derived, depth-integrated
#       approximation to a higher-order glaciological flow model", Journal of
#       Glaciology, 57(201), pp. 157--170, 2011
# GH13  D. N. Golberg and P. Heimbach, "Parameter and state estimation with a
#       time-dependent adjoint marine ice sheet model", The Cyrosphere, 7(6),
#       pp. 1659--1678, 2013

L_x, L_y = 40.0e3, 40.0e3  # GH13 experiment 3
H_0 = Constant(1.0e3, static=True)  # GH13 experiment 3
theta = -0.5 * 2.0 * np.pi / 360.0  # GH13 experiment 3
# Here b is the topography height
grad_b_x = Constant(np.tan(theta), static=True)
# N_x, N_y = 40, 40  # GH13 experiment 3
# N_x, N_y = 20, 20
# timesteps = 120  # Close to (identical to?) GH13 experiment 3
# debug = False
# Debug configuration
N_x, N_y, timesteps, debug = 3, 3, 10, True
assert(timesteps % 10 == 0)

# Observe the flow and surface elevation at the end of these timesteps
# Similar to GH13 experiment 3
timestep_obs = [(timesteps // 2 - 1) + i * (timesteps // 10) for i in range(6)]

T = 10.0  # GH13 experiment 3
dt = Constant(T / timesteps, static=True)
B = Constant(2.1544e5, static=True)  # G11 table 1
g = Constant(9.81, static=True)  # G11 table 1
rho = Constant(910.0, static=True)  # G11 table 1
n = Constant(3.0, static=True)  # G11 table 1
sigma_u = Constant(1.0, static=True)  # GH13 experiment 3
sigma_h = Constant(0.02, static=True)  # GH13 experiment 3

mesh = RectangleMesh(Point(0.0, 0.0), Point(L_x, L_y), N_x, N_y, "crossed")


class PeriodicBoundaryCondition(SubDomain):
    # Following FEniCS 2019.1.0 API
    def inside(self, x, on_boundary):
        return ((abs(x[0]) < 1.0e-8 or abs(x[1]) < 1.0e-8)
                and abs(x[0] - L_x) > 1.0e-8 and abs(x[1] - L_y) > 1.0e-8)

    def map(self, x, y):
        if abs(x[0] - L_x) < 1.0e-8:
            if abs(x[1] - L_y) < 1.0e-8:
                y[0] = 0.0
                y[1] = 0.0
            else:
                y[0] = 0.0
                y[1] = x[1]
        elif abs(x[1] - L_y) < 1.0e-8:
            y[0] = x[0]
            y[1] = 0.0
        else:
            y[:] = -1.0e10


periodic_bc = PeriodicBoundaryCondition()

space = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=periodic_bc)
test, trial = TestFunction(space), TrialFunction(space)
# Velocity space
space_u_f = FunctionSpace(mesh, "Lagrange", 2)
space_U = FunctionSpace(mesh,
                        space_u_f.ufl_element() * space_u_f.ufl_element(),
                        constrained_domain=periodic_bc)
# Thickness perturbation space
space_h = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=periodic_bc)
test_h, trial_h = TestFunction(space_h), TrialFunction(space_h)

space_S = FunctionSpace(mesh, "Discontinuous Lagrange", 2,
                        constrained_domain=periodic_bc)

beta_sq_ref = Function(space, name="beta_sq_ref", static=True)
# GH13 eqn (16)
beta_sq_ref.interpolate(Expression(
    "1000.0 - 750.0 * exp(-(pow(x[0] - (L_x / 2.0), 2.0) + pow(x[1] - (L_y / 2.0), 2.0)) / pow(5.0e3, 2.0))",  # noqa: E501
    L_x=L_x, L_y=L_y,
    element=space.ufl_element()))
# File("beta_sq_ref.pvd", "compressed") << beta_sq_ref

forward_calls = [0]


def forward(beta_sq, ref=None, h_filename=None, speed_filename=None):
    forward_calls[0] += 1
    clear_caches()

    class VectorNormSolver(Equation):
        def __init__(self, U, U_norm):
            # Assumes compatible spaces
            Equation.__init__(self, U_norm, [U_norm, U], nl_deps=[U],
                              ic_deps=[])

        def forward_solve(self, x, deps=None):
            _, U = self.dependencies() if deps is None else deps
            u_split = U.split(deepcopy=True)
            U_norm_arr = np.zeros(function_local_size(x), dtype=np.float64)
            for u in u_split:
                assert(function_local_size(u) == function_local_size(x))
                U_norm_arr[:] += function_get_values(u) ** 2
            function_set_values(x, np.sqrt(U_norm_arr))

    h = [Function(space_h, name="h_n"),
         Function(space_h, name="h_np1")]

    F_h = [Function(space_h, name="F_h_nm2"),
           Function(space_h, name="F_h_nm1"),
           Function(space_h, name="F_h_n")]

    U = [Function(space_U, name="U_n"),
         Function(space_U, name="U_np1")]

    S = Function(space_S, name="S")
    nu = Function(space_S, name="nu")

    def momentum(U, h, initial_guess=None):
        h = h + H_0
        spaces = U.function_space()
        tests, trials = TestFunction(spaces), TrialFunction(spaces)
        test_u, test_v = split(tests)
        test_u_x, test_u_y = test_u.dx(0), test_u.dx(1)
        test_v_x, test_v_y = test_v.dx(0), test_v.dx(1)
        u, v = split(U)
        u_x, u_y = u.dx(0), u.dx(1)
        v_x, v_y = v.dx(0), v.dx(1)

        # G11 eqn (3)
        eps = Constant(5.0e-9, static=True)
        S_eq = LocalProjectionSolver((u_x ** 2) + (v_y ** 2) + u_x * v_y
                                     + 0.25 * ((u_y + v_x) ** 2) + eps, S)
        nu_eq = ExprEvaluationSolver(0.5 * B * (S ** ((1.0 - n) / (2.0 * n))),
                                     nu)

        # GHS09 eqns (1)--(2)
        F = (- inner(tests, -beta_sq * U) * dx
             + inner(test_u_x, nu * h * (4.0 * u_x + 2.0 * v_y)) * dx
             + inner(test_u_y, nu * h * (u_y + v_x)) * dx
             + inner(test_v_y, nu * h * (4.0 * v_y + 2.0 * u_x)) * dx
             + inner(test_v_x, nu * h * (u_y + v_x)) * dx
             + inner(test_u, rho * g * h * grad_b_x) * dx
             + inner(tests, rho * g * h * grad(h)) * dx)
        F = ufl.replace(F, {U: trials})
        U_eq = EquationSolver(
            lhs(F) == rhs(F), U,
            solver_parameters={"linear_solver": "cg",
                               "preconditioner": "hypre_amg",
                                "krylov_solver": {"relative_tolerance": 1.0e-12,  # noqa: E501
                                                  "absolute_tolerance": 1.0e-16}},  # noqa: E501
            adjoint_solver_parameters={"linear_solver": "umfpack"},
            tlm_solver_parameters={"linear_solver": "umfpack"},
            cache_adjoint_jacobian=True,
            cache_tlm_jacobian=True)

        return FixedPointSolver(
            [S_eq, nu_eq, U_eq],
            solver_parameters={"absolute_tolerance": 1.0e-16,
                               "relative_tolerance": 1.0e-11},
            initial_guess=initial_guess)

    def solve_momentum(U, h, initial_guess=None):
        momentum(U, h, initial_guess=initial_guess).solve()

    def elevation_rhs(U, h, F_h):
        # GHS09 eqn (11) right-hand-side (times timestep size)
        return EquationSolver(
            inner(test_h, trial_h) * dx ==
            - dt * inner(test_h, div(U * (h + H_0))) * dx,
            F_h, solver_parameters={"linear_solver": "cg",
                                    "preconditioner": "sor",
                                    "krylov_solver": {"relative_tolerance": 1.0e-12,  # noqa: E501
                                                      "absolute_tolerance": 1.0e-16}})  # noqa: E501

    def solve_elevation_rhs(U, h, F_h):
        elevation_rhs(U, h, F_h).solve()

    def axpy(x, *args):
        return LinearCombinationSolver(x, *args)

    def solve_axpy(x, *args):
        axpy(x, *args).solve()

    def cycle(x_np1, x_n):
        return AssignmentSolver(x_np1, x_n)

    def solve_cycle(x_np1, x_n):
        cycle(x_np1, x_n).solve()

    if h_filename is not None:
        h_file = File(h_filename, "compressed")
    if speed_filename is not None:
        speed_file = File(speed_filename, "compressed")
        speed_n = Function(space_u_f, name="speed_n")
        speed_eq = VectorNormSolver(U[0], speed_n)

    def output(t):
        if h_filename is not None:
            h_file << (h[0], t)
        if speed_filename is not None:
            speed_eq.solve(annotate=False, tlm=False)
            speed_file << (speed_n, t)

    # Initialization
    solve_momentum(U[0], h[0])

    output(t=0.0)

    # RK2
    # Stage 1
    solve_elevation_rhs(U[0], h[0], F_h[2])
    solve_axpy(h[1], (1.0, h[0]), (0.5, F_h[2]))
    solve_momentum(U[1], h[1], initial_guess=U[0])
    solve_cycle(F_h[2], F_h[1])
    # Stage 2
    solve_elevation_rhs(U[1], h[1], F_h[2])
    solve_axpy(h[1], (1.0, h[0]), (1.0, F_h[2]))
    solve_momentum(U[1], h[1], initial_guess=U[0])

    solve_cycle(h[1], h[0])
    solve_cycle(U[1], U[0])
    output(t=float(dt))

    # AB2
    solve_elevation_rhs(U[0], h[0], F_h[2])
    solve_axpy(h[1], (1.0, h[0]), (3.0 / 2.0, F_h[2]), (-1.0 / 2.0, F_h[1]))
    solve_momentum(U[1], h[1], initial_guess=U[0])
    solve_cycle(F_h[1], F_h[0])
    solve_cycle(F_h[2], F_h[1])
    solve_cycle(h[1], h[0])
    solve_cycle(U[1], U[0])
    output(t=2 * float(dt))

    # AB3
    eqs = [elevation_rhs(U[0], h[0], F_h[2]),
           axpy(h[1],
                (1.0, h[0]),
                (23.0 / 12.0, F_h[2]),
                (-4.0 / 3.0, F_h[1]),
                (5.0 / 12.0, F_h[0])),
           momentum(U[1], h[1], initial_guess=U[0]),
           cycle(F_h[1], F_h[0]),
           cycle(F_h[2], F_h[1]),
           cycle(h[1], h[0]),
           cycle(U[1], U[0])]

    gather_ref = ref is None
    if gather_ref:
        ref = {}
    J = Functional(name="J")

    for timestep in range(2, timesteps):
        for eq in eqs:
            eq.solve()
        if timestep in timestep_obs:
            if gather_ref:
                ref[timestep] = \
                    (function_copy(U[0], name=f"U_ref_{timestep + 1:d}"),
                     function_copy(h[0], name=f"h_ref_{timestep + 1:d}"))
            # Similar to GH13 equation (17)
            J.addto((1.0 / (sigma_u ** 2)) * inner(U[0] - ref[timestep][0],
                                                   U[0] - ref[timestep][0]) * dx  # noqa: E501
                    + (1.0 / (sigma_h ** 2)) * inner(h[0] - ref[timestep][1],
                                                     h[0] - ref[timestep][1]) * dx)  # noqa: E501
        else:
            J.addto()
        output(t=(timestep + 1) * float(dt))

    info(f"forward call {forward_calls[0]:d}, J = {J.value():.16e}")
    return ref, J


start_manager()
# ref, J = forward(beta_sq_ref, h_filename="h.pvd", speed_filename="speed.pvd")
ref, J = forward(beta_sq_ref)
stop_manager()

ddJ = SingleBlockHessian(J)
M_solver = KrylovSolver("cg", "sor")
M_solver.parameters.update({"relative_tolerance": 1.0e-12,
                            "absolute_tolerance": 1.0e-16})
M_solver.set_operator(assemble(inner(test, trial) * dx))
A_action_calls = [0]


def A_action(x):
    A_action_calls[0] += 1
    info(f"A_action call {A_action_calls[0]:d}")
    _, _, H_action = ddJ.action(beta_sq_ref, x)
    A_action = Function(space)
    M_solver.solve(A_action.vector(), H_action.vector())
    return function_get_values(A_action)


def eigendecompose_configure(esolver):
    esolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    esolver.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    esolver.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
    esolver.setDimensions(nev=function_global_size(beta_sq_ref),
                          ncv=SLEPc.DECIDE, mpd=SLEPc.DECIDE)
    esolver.setConvergenceTest(SLEPc.EPS.Conv.REL)
    esolver.setTolerances(tol=1.0e-8, max_it=SLEPc.DECIDE)


lam, (V, V_i) = eigendecompose(space, A_action,
                               configure=eigendecompose_configure)
assert(abs(lam.imag).max() == 0.0)
for v_i in V_i:
    assert(function_linf_norm(v_i) == 0.0)
del(v_i, V_i)

lam = lam.real
pack = sorted(zip(lam, V), key=lambda p: p[0], reverse=True)
lam = [p[0] for p in pack]
V = [p[1] for p in pack]

# h = h5py.File("eigenvalues.hdf5", "w")
# h.create_dataset("lam", data=lam, compression=True, fletcher32=True,
#                  shuffle=True)
# h.close()

for i, lam_val in enumerate(lam):
    info(f"Eigenvalue {i + 1:d} = {lam_val:.16e}")

# v_file = File("eigenvectors.pvd", "compressed")
# for i, v in enumerate(V):
#     function_set_values(v,
#                         function_get_values(v)
#                         / np.sqrt(assemble(inner(v, v) * dx)))
#     v.rename("eigenvector", "a Function")
#     v_file << (v, float(i + 1))

if debug:
    del(beta_sq_ref, A_action)
    beta_sq = Function(space, name="beta_sq", static=True)
    function_assign(beta_sq, 400.0)  # As in GH13 experiment 3

    reset_manager()
    start_manager()
    _, J = forward(beta_sq, ref=ref)
    stop_manager()

    dJ = compute_gradient(J, beta_sq)
    ddJ = SingleBlockHessian(J)

    def forward_ref_J(beta_sq):
        return forward(beta_sq, ref=ref)[1]

    min_order = taylor_test(forward_ref_J, beta_sq, J_val=J.value(), dJ=dJ,
                            seed=1.0e-4)
    assert(min_order > 2.00)

    min_order = taylor_test(forward_ref_J, beta_sq, J_val=J.value(), ddJ=ddJ,
                            seed=1.0e-3)
    assert(min_order > 2.99)

    min_order = taylor_test_tlm(forward_ref_J, beta_sq,
                                tlm_order=1, seed=1.0e-3)
    assert(min_order > 2.00)

    min_order = taylor_test_tlm_adjoint(forward_ref_J, beta_sq,
                                        adjoint_order=1, seed=1.0e-3)
    assert(min_order > 1.99)

    min_order = taylor_test_tlm_adjoint(forward_ref_J, beta_sq,
                                        adjoint_order=2, seed=1.0e-3)
    assert(min_order > 1.99)
