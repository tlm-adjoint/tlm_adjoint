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

from tlm_adjoint_numpy import *
from tlm_adjoint_numpy import manager as _manager

import gc
import numpy as np
import unittest
import weakref

Function_ids = {}
_orig_Function_init = Function.__init__


def _Function__init__(self, *args, **kwargs):
    _orig_Function_init(self, *args, **kwargs)
    Function_ids[self.id()] = weakref.ref(self)


Function.__init__ = _Function__init__


def leak_check(test):
    def wrapped_test(self, *args, **kwargs):
        Function_ids.clear()

        test(self, *args, **kwargs)

        # Clear some internal storage that is allowed to keep references
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
            if F is not None:
                info(f"{function_name(F):s} referenced")
                refs += 1
        if refs == 0:
            info("No references")

        Function_ids.clear()
        self.assertEqual(refs, 0)
    return wrapped_test


class tests(unittest.TestCase):
    @leak_check
    def test_ContractionSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        space_0 = FunctionSpace(1)
        space = FunctionSpace(3)
        A = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 6.0]],
                     dtype=np.float64)

        def forward(m):
            x = Function(space, name="x")
            ContractionSolver(A, (1,), (m,), x).solve()

            norm_sq = Function(space_0, name="norm_sq")
            NormSqSolver(x, norm_sq).solve()

            J = Functional(name="J")
            NormSqSolver(norm_sq, J.fn()).solve()

            return x, J

        m = Function(space, name="m", static=True)
        function_set_values(m, np.array([7.0, 8.0, 9.0], dtype=np.float64))

        start_manager()
        x, J = forward(m)
        stop_manager()

        self.assertEqual(abs(A.dot(m.vector()) - x.vector()).max(), 0.0)

        dJ = compute_gradient(J, m)
        min_order = taylor_test(lambda m: forward(m)[1], m, J_val=J.value(),
                                dJ=dJ)
        self.assertGreater(min_order, 2.00)

        ddJ = Hessian(lambda m: forward(m)[1])
        min_order = taylor_test(lambda m: forward(m)[1], m, J_val=J.value(),
                                dJ=dJ, ddJ=ddJ)
        self.assertGreater(min_order, 3.00)

    @leak_check
    def test_InnerProductSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        space = FunctionSpace(10)

        def forward(F):
            G = Function(space, name="G")
            AssignmentSolver(F, G).solve()

            J = Functional(name="J")
            InnerProductSolver(F, G, J.fn()).solve()

            return J

        F = Function(space, name="F", static=True)
        function_set_values(F, np.random.random(function_local_size(F)))

        start_manager()
        J = forward(F)
        stop_manager()

        dJ = compute_gradient(J, F)
        min_order = taylor_test(forward, F, J_val=J.value(), dJ=dJ)
        self.assertGreater(min_order, 1.99)

    @leak_check
    def test_SumSolver(self):
        reset_manager("memory", {"replace": True})
        clear_caches()
        stop_manager()

        space = FunctionSpace(10)

        def forward(F):
            G = Function(space, name="G")
            AssignmentSolver(F, G).solve()

            J = Functional(name="J")
            SumSolver(G, J.fn()).solve()

            return J

        F = Function(space, name="F", static=True)
        function_set_values(F, np.random.random(function_local_size(F)))

        start_manager()
        J = forward(F)
        stop_manager()

        self.assertEqual(J.value(), function_sum(F))

        dJ = compute_gradient(J, F)
        self.assertEqual(abs(function_get_values(dJ) - 1.0).max(), 0.0)


if __name__ == "__main__":
    np.random.seed(1201)
    unittest.main()

#    tests().test_ContractionSolver()
#    tests().test_InnerProductSolver()
#    tests().test_SumSolver()
