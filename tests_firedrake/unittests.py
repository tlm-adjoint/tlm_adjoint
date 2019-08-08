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
from tlm_adjoint_firedrake import *
from tlm_adjoint_firedrake import manager as _manager
from tlm_adjoint_firedrake.backend import backend_Function

import gc
import numpy as np
import os
import unittest
import weakref

Function_ids = {}
_orig_Function_init = backend_Function.__init__


def _Function__init__(self, *args, **kwargs):
    _orig_Function_init(self, *args, **kwargs)
    Function_ids[self.id()] = weakref.ref(self)


backend_Function.__init__ = _Function__init__

ls_parameters_cg = {"ksp_type": "cg",
                    "pc_type": "sor",
                    "ksp_rtol": 1.0e-14,
                    "ksp_atol": 1.0e-16}
ns_parameters_newton_cg = {"snes_type": "newtonls",
                           "ksp_type": "cg",
                           "pc_type": "sor",
                           "ksp_rtol": 1.0e-14,
                           "ksp_atol": 1.0e-16,
                           "snes_rtol": 1.0e-13,
                           "snes_atol": 1.0e-15}
ns_parameters_newton_gmres = {"snes_type": "newtonls",
                              "ksp_type": "gmres",
                              "pc_type": "sor",
                              "ksp_rtol": 1.0e-14,
                              "ksp_atol": 1.0e-16,
                              "snes_rtol": 1.0e-13,
                              "snes_atol": 1.0e-15}


def interpolate_expression(F, ex):
    F.interpolate(ex)


def leak_check(test):
    def wrapped_test(self, *args, **kwargs):
        Function_ids.clear()

        test(self, *args, **kwargs)

        # Clear some internal storage that is allowed to keep references
        clear_caches()
        manager = _manager()
        manager._cp.clear(clear_refs=True)
        tlm_values = manager._tlm.values()  # noqa: F841
        manager._tlm.clear()
        tlm_eqs_values = manager._tlm_eqs.values()  # noqa: F841
        manager._tlm_eqs.clear()

        gc.collect()

        refs = 0
        for F in Function_ids.values():
            F = F()
            if F is not None and function_name(F) != "Coordinates":
                info(f"{function_name(F):s} referenced")
                refs += 1
        if refs == 0:
            info("No references")

        Function_ids.clear()
        self.assertEqual(refs, 0)
    return wrapped_test


class tests(unittest.TestCase):
    @leak_check
    def test_Nullspace(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        def forward(F):
            psi = Function(space, name="psi")

            solve(inner(grad(test), grad(trial)) * dx
                  == -inner(test, F * F) * dx, psi,
                  solver_parameters=ls_parameters_cg,
                  nullspace=VectorSpaceBasis(constant=True),
                  transpose_nullspace=VectorSpaceBasis(constant=True))

            J = Functional(name="J")
            J.assign(inner(grad(psi), grad(psi)) * dx)

            return psi, J

        F = Function(space, name="F", static=True)
        interpolate_expression(F, sqrt(sin(pi * X[1])))

        start_manager()
        psi, J = forward(F)
        stop_manager()

        self.assertLess(abs(function_sum(psi)), 1.0e-15)

        dJ = compute_gradient(J, F)
        min_order = taylor_test(lambda F: forward(F)[1], F, J_val=J.value(),
                                dJ=dJ)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(lambda F: forward(F)[1])
        min_order = taylor_test(lambda F: forward(F)[1], F, J_val=J.value(),
                                ddJ=ddJ)
        self.assertGreater(min_order, 3.00)

    @leak_check
    def test_Storage(self):
        # Ensure creation of checkpoints~ directory
        reset_manager("periodic_disk")
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        space = FunctionSpace(mesh, "Lagrange", 1)

        def forward(x, d=None, h=None):
            y = Function(space, name="y")
            x_s = Function(space, name="x_s")
            y_s = Function(space, name="y_s")

            if d is None:
                function_assign(x_s, x)
                d = {}
            MemoryStorage(x_s, d, function_name(x_s)).solve()

            ExprEvaluationSolver(x * x * x * x_s, y).solve()

            if h is None:
                function_assign(y_s, y)
                comm = manager().comm()
                import h5py
                if comm.size > 1:
                    h = h5py.File(os.path.join("checkpoints~",
                                               "storage.hdf5"),
                                  "w", driver="mpio", comm=comm)
                else:
                    h = h5py.File(os.path.join("checkpoints~",
                                               "storage.hdf5"),
                                  "w")
            HDF5Storage(y_s, h, function_name(y_s)).solve()

            J = Functional(name="J")
            InnerProductSolver(y, y_s, J.fn()).solve()

            return d, h, J

        x = Function(space, name="x", static=True)
        function_set_values(x, np.random.random(function_local_size(x)))

        start_manager()
        d, h, J = forward(x)
        stop_manager()

        self.assertEqual(len(manager()._cp._refs), 1)
        self.assertEqual(tuple(manager()._cp._refs.keys()), (x.id(),))
        self.assertEqual(len(manager()._cp._cp), 0)

        dJ = compute_gradient(J, x)
        min_order = taylor_test(lambda x: forward(x, d=d, h=h)[2], x,
                                J_val=J.value(), dJ=dJ)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(lambda x: forward(x, d=d, h=h)[2])
        min_order = taylor_test(lambda x: forward(x, d=d, h=h)[2], x,
                                J_val=J.value(), ddJ=ddJ)
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_AssembleSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test = TestFunction(space)

        def forward(F):
            x = Function(space, name="x")
            y = Function(RealFunctionSpace(), name="y")

            AssembleSolver(inner(test, F * F) * dx
                           + inner(test, F) * dx, x).solve()
            AssembleSolver(inner(F, x) * dx, y).solve()

            J = Functional(name="J")
            J.assign(y)

            return J

        F = Function(space, name="F", static=True)
        interpolate_expression(F, X[0] * sin(pi * X[1]))

        start_manager()
        J = forward(F)
        stop_manager()

        dJ = compute_gradient(J, F)
        min_order = taylor_test(forward, F, J_val=J.value(), dJ=dJ)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(forward)
        min_order = taylor_test(forward, F, J_val=J.value(), dJ=dJ, ddJ=ddJ)
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_clear_caches(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitIntervalMesh(20)
        space = FunctionSpace(mesh, "Lagrange", 1)
        F = Function(space, name="F", cache=True)

        def cache_item(F):
            form = inner(TestFunction(F.function_space()), F) * dx
            cached_form, _ = assembly_cache().assemble(form)
            return cached_form

        def test_not_cleared(F, cached_form):
            self.assertEqual(len(assembly_cache()), 1)
            self.assertIsNotNone(cached_form())
            self.assertEqual(len(function_caches(F)), 1)

        def test_cleared(F, cached_form):
            self.assertEqual(len(assembly_cache()), 0)
            self.assertIsNone(cached_form())
            self.assertEqual(len(function_caches(F)), 0)

        self.assertEqual(len(assembly_cache()), 0)

        # Clear default
        cached_form = cache_item(F)
        test_not_cleared(F, cached_form)
        clear_caches()
        test_cleared(F, cached_form)

        # Clear Function caches
        cached_form = cache_item(F)
        test_not_cleared(F, cached_form)
        clear_caches(F)
        test_cleared(F, cached_form)

        # Clear on cache update
        cached_form = cache_item(F)
        test_not_cleared(F, cached_form)
        function_update_state(F)
        test_not_cleared(F, cached_form)
        update_caches([F])
        test_cleared(F, cached_form)

        # Clear on cache update, new Function
        cached_form = cache_item(F)
        test_not_cleared(F, cached_form)
        update_caches([F], [Function(space)])
        test_cleared(F, cached_form)

        # Clear on cache update, ReplacementFunction
        cached_form = cache_item(F)
        test_not_cleared(F, cached_form)
        update_caches([replaced_function(F)])
        test_cleared(F, cached_form)

        # Clear on cache update, ReplacementFunction with new Function
        cached_form = cache_item(F)
        test_not_cleared(F, cached_form)
        update_caches([replaced_function(F)], [Function(space)])
        test_cleared(F, cached_form)

    @leak_check
    def test_LocalProjectionSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(10, 10)
        X = SpatialCoordinate(mesh)
        space_1 = FunctionSpace(mesh, "Discontinuous Lagrange", 1)
        space_2 = FunctionSpace(mesh, "Lagrange", 2)
        test_1, trial_1 = TestFunction(space_1), TrialFunction(space_1)

        def forward(G):
            F = Function(space_1, name="F")
            LocalProjectionSolver(G, F).solve()
            J = Functional(name="J")
            J.assign((F ** 2 + F ** 3) * dx)
            return F, J

        G = Function(space_2, name="G", static=True)
        interpolate_expression(G, sin(pi * X[0]) * sin(2.0 * pi * X[1]))

        start_manager()
        F, J = forward(G)
        stop_manager()

        F_ref = Function(space_1, name="F_ref")
        solve(inner(test_1, trial_1) * dx == inner(test_1, G) * dx, F_ref,
              solver_parameters=ls_parameters_cg)
        F_error = Function(space_1, name="F_error")
        function_assign(F_error, F_ref)
        function_axpy(F_error, -1.0, F)

        F_error_norm = function_linf_norm(F_error)
        info(f"Error norm = {F_error_norm:.16e}")
        self.assertLess(F_error_norm, 1.0e-14)

        dJ = compute_gradient(J, G)
        min_order = taylor_test(lambda G: forward(G)[1], G, J_val=J.value(),
                                dJ=dJ)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(lambda G: forward(G)[1])
        min_order = taylor_test(lambda G: forward(G)[1], G, J_val=J.value(),
                                dJ=dJ, ddJ=ddJ)
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_LongRange(self):
        n_steps = 200
        reset_manager("multistage",
                      {"blocks": n_steps,
                       "snaps_on_disk": 0,
                       "snaps_in_ram": 2,
                       "verbose": True})
        clear_caches()
        stop_manager()

        mesh = UnitIntervalMesh(20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)

        def forward(F, x_ref=None):
            G = Function(space, name="G")
            AssignmentSolver(F, G).solve()

            x_old = Function(space, name="x_old")
            x = Function(space, name="x")
            AssignmentSolver(G, x_old).solve()
            J = Functional(name="J")
            gather_ref = x_ref is None
            if gather_ref:
                x_ref = {}
            for n in range(n_steps):
                terms = [(1.0, x_old)]
                if n % 11 == 0:
                    terms.append((1.0, G))
                LinearCombinationSolver(x, *terms).solve()
                if n % 17 == 0:
                    if gather_ref:
                        x_ref[n] = function_copy(x, name="x_ref_%i" % n)
                    J.addto(inner(x * x * x, x_ref[n]) * dx)
                AssignmentSolver(x, x_old).solve()
                if n < n_steps - 1:
                    new_block()

            return x_ref, J

        F = Function(space, name="F", static=True)
        interpolate_expression(F, sin(pi * X[0]))
        zeta = Function(space, name="zeta", static=True)
        interpolate_expression(zeta, exp(X[0]))
        add_tlm(F, zeta)
        start_manager()
        x_ref, J = forward(F)
        stop_manager()

        dJ = compute_gradient(J, F)
        min_order = taylor_test(lambda F: forward(F, x_ref=x_ref)[1], F,
                                J_val=J.value(), dJ=dJ)
        self.assertGreater(min_order, 2.00)

        dJ_tlm = J.tlm(F, zeta).value()
        dJ_adj = function_inner(dJ, zeta)
        error_norm = abs(dJ_tlm - dJ_adj)
        info("dJ/dF zeta, TLM     = %.16e" % dJ_tlm)
        info("dJ/dF zeta, adjoint = %.16e" % dJ_adj)
        info("Error norm          = %.16e" % error_norm)
        self.assertLess(error_norm, 1.0e-9)

        ddJ = Hessian(lambda F: forward(F, x_ref=x_ref)[1])
        min_order = taylor_test(lambda F: forward(F, x_ref=x_ref)[1], F,
                                J_val=J.value(), dJ=dJ, ddJ=ddJ)
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_ExprEvaluationSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitIntervalMesh(20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)

        def test_expression(y, y_int):
            return (y_int * y * (sin if is_function(y) else np.sin)(y)
                    + 2.0 + (y ** 2) + y / (1.0 + (y ** 2)))

        def forward(y):
            x = Function(space, name="x")
            y_int = Function(RealFunctionSpace(), name="y_int")
            AssembleSolver(y * dx, y_int).solve()
            ExprEvaluationSolver(test_expression(y, y_int), x).solve()

            J = Functional(name="J")
            J.assign(x * x * x * dx)
            return x, J

        y = Function(space, name="y", static=True)
        interpolate_expression(y, cos(3.0 * pi * X[0]))
        start_manager()
        x, J = forward(y)
        stop_manager()

        error_norm = abs(function_get_values(x)
                         - test_expression(function_get_values(y),
                                           assemble(y * dx))).max()
        info("Error norm = %.16e" % error_norm)
        self.assertEqual(error_norm, 0.0)

        dJ = compute_gradient(J, y)
        min_order = taylor_test(lambda y: forward(y)[1], y, J_val=J.value(),
                                dJ=dJ)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(lambda y: forward(y)[1])
        min_order = taylor_test(lambda y: forward(y)[1], y, J_val=J.value(),
                                dJ=dJ, ddJ=ddJ, seed=1.0e-3)
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_PointInterpolationSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitCubeMesh(5, 5, 5)
        X = SpatialCoordinate(mesh)
        y_space = FunctionSpace(mesh, "Lagrange", 3)
        space_0 = RealFunctionSpace()
        X_coords = np.array([[0.1, 0.1, 0.1],
                             [0.2, 0.3, 0.4],
                             [0.9, 0.8, 0.7],
                             [0.4, 0.2, 0.3]], dtype=np.float64)

        def forward(y):
            X_vals = [Function(space_0, name="x_%i" % i)
                      for i in range(X_coords.shape[0])]
            PointInterpolationSolver(y, X_vals, X_coords).solve()

            J = Functional(name="J")
            for x in X_vals:
                J.addto(x * x * x * dx)

            return X_vals, J

        y = Function(y_space, name="y", static=True)
        interpolate_expression(y, pow(X[0], 3) - 1.5 * X[0] * X[1] + 1.5)

        start_manager()
        X_vals, J = forward(y)
        stop_manager()

        def x_ref(x):
            return x[0] ** 3 - 1.5 * x[0] * x[1] + 1.5

        x_error_norm = 0.0
        for x, x_coord in zip(X_vals, X_coords):
            x_error_norm = max(x_error_norm, abs(function_max_value(x)
                                                 - x_ref(x_coord)))
        info("Error norm = %.16e" % x_error_norm)
        self.assertLess(x_error_norm, 1.0e-13)

        dJ = compute_gradient(J, y)
        min_order = taylor_test(lambda y: forward(y)[1], y, J_val=J.value(),
                                dJ=dJ, seed=1.0e-4)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(lambda y: forward(y)[1])
        min_order = taylor_test(lambda y: forward(y)[1], y, J_val=J.value(),
                                dJ=dJ, ddJ=ddJ)
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_FixedPointSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        space = RealFunctionSpace()

        x = Function(space, name="x")
        z = Function(space, name="z")

        a = Function(space, name="a", static=True)
        function_assign(a, 2.0)
        b = Function(space, name="b", static=True)
        function_assign(b, 3.0)

        def forward(a, b):
            eqs = [LinearCombinationSolver(z, (1.0, x), (1.0, b)),
                   AssembleSolver((a / sqrt(z)) * dx, x)]

            fp_parameters = {"absolute_tolerance": 0.0,
                             "relative_tolerance": 1.0e-14}
            FixedPointSolver(eqs, solver_parameters=fp_parameters).solve()

            J = Functional(name="J")
            J.assign(x)

            return J

        start_manager()
        J = forward(a, b)
        stop_manager()

        x_val = function_max_value(x)
        a_val = function_max_value(a)
        b_val = function_max_value(b)
        self.assertAlmostEqual(x_val * np.sqrt(x_val + b_val) - a_val, 0.0,
                               places=14)

        dJda, dJdb = compute_gradient(J, [a, b])
        dm = Function(space, name="dm", static=True)
        function_assign(dm, 1.0)
        min_order = taylor_test(lambda a: forward(a, b), a, J_val=J.value(),
                                dJ=dJda, dM=dm)
        self.assertGreater(min_order, 1.99)
        min_order = taylor_test(lambda b: forward(a, b), b, J_val=J.value(),
                                dJ=dJdb, dM=dm)
        self.assertGreater(min_order, 1.99)

        ddJ = Hessian(lambda a: forward(a, b))
        min_order = taylor_test(lambda a: forward(a, b), a, J_val=J.value(),
                                ddJ=ddJ, dM=dm)
        self.assertGreater(min_order, 2.99)

        ddJ = Hessian(lambda b: forward(a, b))
        min_order = taylor_test(lambda b: forward(a, b), b, J_val=J.value(),
                                ddJ=ddJ, dM=dm)
        self.assertGreater(min_order, 2.99)

        # Multi-control Taylor verification
        min_order = taylor_test(forward, [a, b], J_val=J.value(),
                                dJ=[dJda, dJdb], dM=[dm, dm])
        self.assertGreater(min_order, 1.99)

        # Multi-control Hessian action with Taylor verification
        ddJ = Hessian(forward)
        min_order = taylor_test(forward, [a, b], J_val=J.value(), ddJ=ddJ,
                                dM=[dm, dm])
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_higher_order_adjoint(self):
        n_steps = 20
        reset_manager("multistage",
                      {"blocks": n_steps,
                       "snaps_on_disk": 2,
                       "snaps_in_ram": 2,
                       "verbose": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        def forward(kappa):
            clear_caches()

            x_n = Function(space, name="x_n")
            interpolate_expression(x_n, sin(pi * X[0]) * sin(2.0 * pi * X[1]))
            x_np1 = Function(space, name="x_np1")
            dt = Constant(0.01, static=True)
            bc = DirichletBC(space, 0.0, "on_boundary",
                             static=True, homogeneous=True)

            eq = (inner(test, trial / dt) * dx
                  + inner(grad(test), kappa * grad(trial)) * dx
                  == inner(test, x_n / dt) * dx)
            eqs = [EquationSolver(eq, x_np1, bc,
                                  solver_parameters=ls_parameters_cg),
                   AssignmentSolver(x_np1, x_n)]

            for n in range(n_steps):
                for eq in eqs:
                    eq.solve()
                if n < n_steps - 1:
                    new_block()

            J = Functional(name="J")
            J.assign(inner(x_np1, x_np1) * dx)

            return J

        kappa = Function(space, name="kappa", static=True)
        function_assign(kappa, 1.0)

        for tlm_order in range(1, 4):
            min_order = taylor_test_tlm(forward, kappa, tlm_order=tlm_order,
                                        seed=1.0e-3)
            self.assertGreater(min_order, 1.99)

        for adjoint_order in range(1, 5):
            min_order = taylor_test_tlm_adjoint(forward, kappa,
                                                adjoint_order=adjoint_order,
                                                seed=1.0e-3)
            self.assertGreater(min_order, 1.99)

    @leak_check
    def test_minimize_scipy_multiple(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        def forward(alpha, beta, x_ref=None, y_ref=None):
            clear_caches()

            x = Function(space, name="x")
            solve(inner(test, trial) * dx == inner(test, alpha) * dx,
                  x, solver_parameters=ls_parameters_cg)

            y = Function(space, name="y")
            solve(inner(test, trial) * dx == inner(test, beta) * dx,
                  y, solver_parameters=ls_parameters_cg)

            if x_ref is None:
                x_ref = Function(space, name="x_ref", static=True)
                function_assign(x_ref, x)
            if y_ref is None:
                y_ref = Function(space, name="y_ref", static=True)
                function_assign(y_ref, y)

            J = Functional(name="J")
            J.assign(inner(x - x_ref, x - x_ref) * dx)
            J.addto(inner(y - y_ref, y - y_ref) * dx)
            return x_ref, y_ref, J

        alpha_ref = Function(space, name="alpha_ref", static=True)
        interpolate_expression(alpha_ref, exp(X[0] + X[1]))
        beta_ref = Function(space, name="beta_ref", static=True)
        interpolate_expression(beta_ref, sin(pi * X[0]) * sin(2.0 * pi * X[1]))
        x_ref, y_ref, _ = forward(alpha_ref, beta_ref)

        alpha0 = Function(space, name="alpha0", static=True)
        beta0 = Function(space, name="beta0", static=True)
        start_manager()
        _, _, J = forward(alpha0, beta0, x_ref=x_ref, y_ref=y_ref)
        stop_manager()

        def forward_J(alpha, beta):
            return forward(alpha, beta, x_ref=x_ref, y_ref=y_ref)[2]
        (alpha, beta), result = minimize_scipy(forward_J, (alpha0, beta0),
                                               J0=J,
                                               method="L-BFGS-B",
                                               options={"ftol": 0.0,
                                                        "gtol": 1.0e-11})
        self.assertTrue(result.success)

        error = Function(space, name="error")
        function_assign(error, alpha_ref)
        function_axpy(error, -1.0, alpha)
        self.assertLess(function_linf_norm(error), 1.0e-8)
        function_assign(error, beta_ref)
        function_axpy(error, -1.0, beta)
        self.assertLess(function_linf_norm(error), 1.0e-9)

    @leak_check
    def test_minimize_scipy(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        def forward(alpha, x_ref=None):
            clear_caches()

            x = Function(space, name="x")
            solve(inner(test, trial) * dx == inner(test, alpha) * dx,
                  x, solver_parameters=ls_parameters_cg)

            if x_ref is None:
                x_ref = Function(space, name="x_ref", static=True)
                function_assign(x_ref, x)

            J = Functional(name="J")
            J.assign(inner(x - x_ref, x - x_ref) * dx)
            return x_ref, J

        alpha_ref = Function(space, name="alpha_ref", static=True)
        interpolate_expression(alpha_ref, exp(X[0] + X[1]))
        x_ref, _ = forward(alpha_ref)

        alpha0 = Function(space, name="alpha0", static=True)
        start_manager()
        _, J = forward(alpha0, x_ref=x_ref)
        stop_manager()

        def forward_J(alpha):
            return forward(alpha, x_ref=x_ref)[1]
        alpha, result = minimize_scipy(forward_J, alpha0, J0=J,
                                       method="L-BFGS-B",
                                       options={"ftol": 0.0, "gtol": 1.0e-10})
        self.assertTrue(result.success)

        error = Function(space, name="error")
        function_assign(error, alpha_ref)
        function_axpy(error, -1.0, alpha)
        self.assertLess(function_linf_norm(error), 1.0e-7)

    @leak_check
    def test_overrides(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        F = Function(space, name="F", static=True)
        interpolate_expression(F, 1.0 + sin(pi * X[0]) * sin(3.0 * pi * X[1]))

        bc = DirichletBC(space, 1.0, "on_boundary",
                         static=True, homogeneous=False)

        def forward(F):
            G = [Function(space, name="G_%i" % i) for i in range(5)]

            G[0] = project(F, space)

            A = assemble(inner(test, trial) * dx, bcs=bc)
            b = assemble(inner(test, G[0]) * dx)
            solver = LinearSolver(A, solver_parameters=ls_parameters_cg)
            solver.solve(G[1].vector(), b)

            b = assemble(inner(test, G[1]) * dx)
            solve(A, G[2].vector(), b,
                  solver_parameters=ls_parameters_cg)

            eq = inner(test, trial) * dx == inner(test, G[2]) * dx
            problem = LinearVariationalProblem(eq.lhs, eq.rhs, G[3])
            solver = LinearVariationalSolver(problem)
            solver.parameters.update(solver_parameters=ls_parameters_cg)
            solver.solve()

            eq = inner(test, G[4]) * dx - inner(test, G[3]) * dx
            problem = NonlinearVariationalProblem(eq, G[4])
            solver = NonlinearVariationalSolver(problem)
            solver.parameters.update(solver_parameters=ns_parameters_newton_cg)
            solver.solve()

            J = Functional(name="J")
            J.assign(inner(G[-1], G[-1]) * dx)

            return G[-1], J

        start_manager()
        G, J = forward(F)
        stop_manager()

        self.assertAlmostEqual(assemble(inner(F - G, F - G) * dx), 0.0,
                               places=13)

        J_val = J.value()
        dJ = compute_gradient(J, F)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(lambda F: forward(F)[1], F, J_val=J_val, dJ=dJ)
        self.assertGreater(min_order, 1.98)

    @leak_check
    def test_bc(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(20, 20)
        X = SpatialCoordinate(mesh)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        F = Function(space, name="F", static=True)
        interpolate_expression(F, sin(pi * X[0]) * sin(3.0 * pi * X[1]))

        def forward(bc):
            x_0 = Function(space, name="x_0")
            x_1 = Function(space, name="x_1")
            x = Function(space, name="x")

            DirichletBCSolver(bc, x_1, "on_boundary").solve()

            solve(inner(grad(test), grad(trial)) * dx
                  == inner(test, F) * dx - inner(grad(test), grad(x_1)) * dx,
                  x_0, DirichletBC(space, 0.0, "on_boundary",
                                   static=True, homogeneous=True),
                  solver_parameters=ls_parameters_cg)

            AxpySolver(x_0, 1.0, x_1, x).solve()

            J = Functional(name="J")
            J.assign(inner(x, x) * dx)
            return x, J

        bc = Function(space, name="bc", static=True)
        function_assign(bc, 1.0)

        start_manager()
        x, J = forward(bc)
        stop_manager()

        x_ref = Function(space, name="x_ref")
        solve(inner(grad(test), grad(trial)) * dx == inner(test, F) * dx,
              x_ref,
              DirichletBC(space, 1.0, "on_boundary",
                          static=True, homogeneous=False),
              solver_parameters=ls_parameters_cg)
        error = Function(space, name="error")
        function_assign(error, x_ref)
        function_axpy(error, -1.0, x)
        self.assertEqual(function_linf_norm(error), 0.0)

        J_val = J.value()
        dJ = compute_gradient(J, bc)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(lambda bc: forward(bc)[1], bc, J_val=J_val,
                                dJ=dJ)
        self.assertGreater(min_order, 1.99)

    @leak_check
    def test_recursive_tlm(self):
        n_steps = 20
        reset_manager("multistage",
                      {"blocks": n_steps,
                       "snaps_on_disk": 4,
                       "snaps_in_ram": 2,
                       "verbose": True})
        clear_caches()
        stop_manager()

        # Use a mesh of non-unit size to test that volume factors are handled
        # correctly
        mesh = RectangleMesh(5, 5, 1.0, 2.0)
        r0 = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
        space = VectorFunctionSpace(mesh, r0)
        test = TestFunction(space)
        dt = Constant(0.01, static=True)
        control_space = FunctionSpace(mesh, r0)
        alpha = Function(control_space, name="alpha", static=True)
        function_assign(alpha, 1.0)
        inv_V = Constant(1.0 / assemble(Constant(1.0) * dx(mesh)), static=True)
        dalpha = Function(control_space, name="dalpha", static=True)
        function_assign(dalpha, 1.0)

        def forward(alpha, dalpha=None):
            clear_caches()

            T_n = Function(space, name="T_n")
            T_np1 = Function(space, name="T_np1")
            T_s = Constant(0.5, static=True) * (T_n + T_np1)

            # Forward model initialization and definition
            T_n.assign(Constant((1.0, 0.0)))
            eq = EquationSolver(inner(test[0], (T_np1[0] - T_n[0]) / dt) * dx
                                - inner(test[0], T_s[1]) * dx
                                + inner(test[1], (T_np1[1] - T_n[1]) / dt) * dx
                                + inner(test[1], sin(alpha * T_s[0])) * dx
                                == 0,
                                T_np1,
                                solver_parameters=ns_parameters_newton_gmres)
            cycle = AssignmentSolver(T_np1, T_n)
            J = Functional(name="J")

            for n in range(n_steps):
                eq.solve()
                cycle.solve()
                if n == n_steps - 1:
                    J.addto(inv_V * T_n[0] * dx)
                if n < n_steps - 1:
                    new_block()

            if dalpha is None:
                K = None
            else:
                K = J.tlm(alpha, dalpha)

            return J, K

        add_tlm(alpha, dalpha)
        start_manager()
        J, K = forward(alpha, dalpha=dalpha)
        stop_manager()

        J_val = J.value()
        info("J = %.16e" % J_val)
        self.assertAlmostEqual(J_val, 9.8320117858590805e-01, places=14)

        dJ = K.value()
        info("TLM sensitivity = %.16e" % dJ)

        # Run the adjoint of the forward+TLM system to compute the Hessian
        # action
        ddJ = compute_gradient(K, alpha)
        ddJ_val = function_max_value(ddJ) * function_global_size(ddJ)
        info("ddJ = %.16e" % ddJ_val)

        # Taylor verify the Hessian (and gradient)
        eps_vals = np.array([np.power(4.0, -p)
                             for p in range(1, 5)], dtype=np.float64)

        def control_value(value):
            alpha = Function(control_space, static=True)
            function_assign(alpha, value)
            return alpha
        J_vals = np.array([forward(control_value(1.0 + eps))[0].value()
                           for eps in eps_vals], dtype=np.float64)
        error_norms_0 = abs(J_vals - J_val)
        error_norms_1 = abs(J_vals - J_val - dJ * eps_vals)
        error_norms_2 = abs(J_vals - J_val - dJ * eps_vals
                            - 0.5 * ddJ_val * np.power(eps_vals, 2))
        orders_0 = (np.log(error_norms_0[1:] / error_norms_0[:-1])
                    / np.log(eps_vals[1:] / eps_vals[:-1]))
        orders_1 = (np.log(error_norms_1[1:] / error_norms_1[:-1])
                    / np.log(eps_vals[1:] / eps_vals[:-1]))
        orders_2 = (np.log(error_norms_2[1:] / error_norms_2[:-1])
                    / np.log(eps_vals[1:] / eps_vals[:-1]))
        info("dJ error norms, first order  = %s" % error_norms_0)
        info("dJ orders,      first order  = %s" % orders_0)
        info("dJ error norms, second order = %s" % error_norms_1)
        info("dJ orders,      second order = %s" % orders_1)
        info("dJ error norms, third order  = %s" % error_norms_2)
        info("dJ orders,      third order  = %s" % orders_2)
        self.assertGreater(orders_0[-1], 0.99)
        self.assertGreater(orders_1[-1], 2.00)
        self.assertGreater(orders_2[-1], 2.99)
        self.assertGreater(orders_0.min(), 0.87)
        self.assertGreater(orders_1.min(), 2.00)
        self.assertGreater(orders_2.min(), 2.94)

    @leak_check
    def test_timestepping(self):
        for n_steps in [1, 2, 5, 20]:
            reset_manager("multistage",
                          {"blocks": n_steps,
                           "snaps_on_disk": 4,
                           "snaps_in_ram": 2,
                           "verbose": True})
            clear_caches()
            stop_manager()

            mesh = UnitIntervalMesh(100)
            X = SpatialCoordinate(mesh)
            space = FunctionSpace(mesh, "Lagrange", 1)
            test, trial = TestFunction(space), TrialFunction(space)
            T_0 = Function(space, name="T_0", static=True)
            interpolate_expression(T_0, sin(pi * X[0]) + sin(10.0 * pi * X[0]))
            dt = Constant(0.01, static=True)
            space_r0 = FunctionSpace(mesh, "R", 0)
            kappa = Function(space_r0, name="kappa", static=True)
            function_assign(kappa, 1.0)

            def forward(T_0, kappa):
                from tlm_adjoint_firedrake.timestepping import N, \
                    TimeFunction, TimeLevels, TimeSystem, n

                levels = TimeLevels([n, n + 1], {n: n + 1})
                T = TimeFunction(levels, space, name="T")
                T[n].rename("T_n", "a Function")
                T[n + 1].rename("T_np1", "a Function")

                system = TimeSystem()

                system.add_solve(T_0, T[0])

                eq = (inner(test, trial) * dx
                      + dt * inner(grad(test), kappa * grad(trial)) * dx
                      == inner(test, T[n]) * dx)
                system.add_solve(eq, T[n + 1],
                                 DirichletBC(space, 1.0, "on_boundary",
                                             static=True, homogeneous=False),
                                 solver_parameters=ls_parameters_cg)

                for n_step in range(n_steps):
                    system.timestep()
                    if n_step < n_steps - 1:
                        new_block()
                system.finalize()

                J = Functional(name="J")
                J.assign(inner(T[N], T[N]) * dx)
                return J

            start_manager()
            J = forward(T_0, kappa)
            stop_manager()

            J_val = J.value()
            if n_steps == 20:
                self.assertAlmostEqual(J_val, 9.4790204396919131e-01,
                                       places=12)

            controls = [Control("T_0"), Control(kappa)]
            dJ = compute_gradient(J, controls)
            # Usage as in dolfin-adjoint tests
            min_order = taylor_test(lambda T_0: forward(T_0, kappa),
                                    controls[0], J_val=J_val, dJ=dJ[0])
            self.assertGreater(min_order, 1.99)
            dm = Function(space_r0, name="dm", static=True)
            function_assign(dm, 1.0)
            # Usage as in dolfin-adjoint tests
            min_order = taylor_test(lambda kappa: forward(T_0, kappa),
                                    controls[1], J_val=J_val, dJ=dJ[1], dM=dm)
            self.assertGreater(min_order, 1.99)

    @leak_check
    def test_second_order_adjoint(self):
        n_steps = 20
        reset_manager("multistage",
                      {"blocks": n_steps,
                       "snaps_on_disk": 4,
                       "snaps_in_ram": 2,
                       "verbose": True})
        clear_caches()
        stop_manager()

        mesh = UnitSquareMesh(5, 5)
        r0 = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), 0)
        space = VectorFunctionSpace(mesh, r0)
        test = TestFunction(space)
        T_0 = Function(space, name="T_0", static=True)
        T_0.assign(Constant((1.0, 0.0)))
        dt = Constant(0.01, static=True)

        def forward(T_0):
            T_n = Function(space, name="T_n")
            T_np1 = Function(space, name="T_np1")
            T_s = Constant(0.5, static=True) * (T_n + T_np1)

            AssignmentSolver(T_0, T_n).solve()

            eq = EquationSolver(inner(test[0], (T_np1[0] - T_n[0]) / dt) * dx
                                - inner(test[0], T_s[1]) * dx
                                + inner(test[1], (T_np1[1] - T_n[1]) / dt) * dx
                                + inner(test[1], sin(T_s[0])) * dx == 0,
                                T_np1,
                                solver_parameters=ns_parameters_newton_gmres)

            for n in range(n_steps):
                eq.solve()
                T_n.assign(T_np1)
                if n < n_steps - 1:
                    new_block()

            J = Functional(name="J")
            J.assign(T_n[0] * T_n[0] * dx)

            return J

        start_manager()
        J = forward(T_0)
        stop_manager()

        J_val = J.value()
        self.assertAlmostEqual(J_val, 9.8320117858590805e-01 ** 2, places=14)

        dJ = compute_gradient(J, T_0)
        dm = Function(space, name="dm", static=True)
        function_assign(dm, 1.0)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(forward, T_0, J_val=J_val, dJ=dJ, dM=dm)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(forward)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(forward, T_0, J_val=J_val, ddJ=ddJ, dM=dm)
        self.assertGreater(min_order, 2.99)

    @leak_check
    def test_AxpySolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        space = RealFunctionSpace()
        x = Function(space, name="x", static=True)
        function_assign(x, 1.0)

        def forward(x):
            y = [Function(space, name="y_%i" % i) for i in range(5)]
            z = [Function(space, name="z_%i" % i) for i in range(2)]
            function_assign(z[0], 7.0)

            AssignmentSolver(x, y[0]).solve()
            for i in range(len(y) - 1):
                AxpySolver(y[i], i + 1, z[0], y[i + 1]).solve()
            AssembleSolver(y[-1] * y[-1] * dx, z[1]).solve()

            J = Functional(name="J")
            J.assign(inner(z[1], z[1]) * dx)

            return J

        start_manager()
        J = forward(x)
        stop_manager()

        J_val = J.value()
        self.assertAlmostEqual(J_val, 25411681.0, places=8)

        dJ = compute_gradient(J, x)
        dm = Function(space, name="dm", static=True)
        function_assign(dm, 1.0)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(forward, x, J_val=J_val, dJ=dJ, dM=dm)
        self.assertGreater(min_order, 2.00)

    @leak_check
    def test_AssignmentSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        space = RealFunctionSpace()
        x = Function(space, name="x", static=True)
        function_assign(x, 16.0)

        def forward(x):
            y = [Function(space, name="y_%i" % i) for i in range(9)]
            z = Function(space, name="z")

            AssignmentSolver(x, y[0]).solve()
            for i in range(len(y) - 1):
                AssignmentSolver(y[i], y[i + 1]).solve()
            AssembleSolver(y[-1] * y[-1] * dx, z).solve()

            J = Functional(name="J")
            J.assign(inner(z, z) * dx)
            J.addto(2 * inner(x, x) * dx)

            K = Functional(name="K")
            K.assign(inner(z, z) * dx)

            return J, K

        start_manager()
        J, K = forward(x)
        stop_manager()

        J_val = J.value()
        K_val = K.value()
        self.assertEqual(J_val, 66048.0)
        self.assertEqual(K_val, 65536.0)

        dJs = compute_gradient([J, K], x)
        dm = Function(space, name="dm", static=True)
        function_assign(dm, 1.0)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(lambda x: forward(x)[0], x, J_val=J_val,
                                dJ=dJs[0], dM=dm)
        self.assertGreater(min_order, 2.00)
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(lambda x: forward(x)[1], x, J_val=K_val,
                                dJ=dJs[1], dM=dm)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(lambda m: forward(m)[0])
        # Usage as in dolfin-adjoint tests
        min_order = taylor_test(lambda x: forward(x)[0], x, J_val=J_val,
                                ddJ=ddJ, dM=dm)
        self.assertGreater(min_order, 3.00)

    @leak_check
    def test_HEP(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitIntervalMesh(20)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        M = assemble(inner(test, trial) * dx)
        M.force_evaluation()

        def M_action(F):
            G = function_new(F)
            with F.vector().dat.vec_ro as F_v, G.vector().dat.vec_wo as G_v:
                M.petscmat.mult(F_v, G_v)
            return function_get_values(G)

        import slepc4py.SLEPc as SLEPc
        lam, V_r = eigendecompose(space, M_action,
                                  problem_type=SLEPc.EPS.ProblemType.HEP)
        diff = Function(space)
        for lam_val, v_r in zip(lam, V_r):
            function_set_values(diff, M_action(v_r))
            function_axpy(diff, -lam_val, v_r)
            self.assertAlmostEqual(function_linf_norm(diff), 0.0, places=16)

    @leak_check
    def test_NHEP(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        mesh = UnitIntervalMesh(20)
        space = FunctionSpace(mesh, "Lagrange", 1)
        test, trial = TestFunction(space), TrialFunction(space)

        N = assemble(inner(test, trial.dx(0)) * dx)
        N.force_evaluation()

        def N_action(F):
            G = function_new(F)
            with F.vector().dat.vec_ro as F_v, G.vector().dat.vec_wo as G_v:
                N.petscmat.mult(F_v, G_v)
            return function_get_values(G)

        lam, (V_r, V_i) = eigendecompose(space, N_action)
        diff = Function(space)
        for lam_val, v_r, v_i in zip(lam, V_r, V_i):
            function_set_values(diff, N_action(v_r))
            function_axpy(diff, -float(lam_val.real), v_r)
            function_axpy(diff, +float(lam_val.imag), v_i)
            self.assertAlmostEqual(function_linf_norm(diff), 0.0, places=7)
            function_set_values(diff, N_action(v_i))
            function_axpy(diff, -float(lam_val.real), v_i)
            function_axpy(diff, -float(lam_val.imag), v_r)
            self.assertAlmostEqual(function_linf_norm(diff), 0.0, places=7)


if __name__ == "__main__":
    np.random.seed(1201)
    unittest.main()

#    tests().test_AssignmentSolver()
#    tests().test_AxpySolver()
#    tests().test_second_order_adjoint()
#    tests().test_timestepping()
#    tests().test_recursive_tlm()
#    tests().test_bc()
#    tests().test_overrides()
#    tests().test_minimize_scipy()
#    tests().test_minimize_scipy_multiple()
#    tests().test_higher_order_adjoint()
#    tests().test_FixedPointSolver()
#    tests().test_PointInterpolationSolver()
#    tests().test_ExprEvaluationSolver()
#    tests().test_LongRange()
#    tests().test_LocalProjectionSolver()
#    tests().test_clear_caches()
#    tests().test_AssembleSolver()
#    tests().test_Storage()
#    tests().test_Nullspace()

#    tests().test_HEP()
#    tests().test_NHEP()
