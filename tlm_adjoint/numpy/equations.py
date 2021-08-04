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

from ..interface import function_new

from ..equations import EquationException, LinearEquation, Matrix, RHS

import numpy as np
import warnings

__all__ = \
    [
        "ConstantMatrix",
        "ContractionRHS",
        "ContractionSolver"
    ]


class ConstantMatrix(Matrix):
    def __init__(self, A, A_T=None, ic=False, adj_ic=False):
        if A_T is not None:
            warnings.warn("A_T argument is deprecated and has no effect",
                          DeprecationWarning, stacklevel=2)

        super().__init__(nl_deps=[], ic=ic, adj_ic=adj_ic)
        self._A = A.copy()
        self._A_H = A.conjugate().T

    def A(self):
        A = self._A
        if isinstance(A, np.ndarray):
            A = A.view()
            A.setflags(write=False)
        return A

    def forward_action(self, nl_deps, x, b, method="assign"):
        sb = self._A.dot(x.vector())
        if method == "assign":
            b.vector()[:] = sb
        elif method == "add":
            b.vector()[:] += sb
        elif method == "sub":
            b.vector()[:] -= sb
        else:
            raise EquationException(f"Invalid method: '{method:s}'")

    def adjoint_action(self, nl_deps, adj_x, b, b_index=0, method="assign"):
        if b_index != 0:
            raise EquationException("Invalid index")
        sb = self._A_H.dot(adj_x.vector())
        if method == "assign":
            b.vector()[:] = sb
        elif method == "add":
            b.vector()[:] += sb
        elif method == "sub":
            b.vector()[:] -= sb
        else:
            raise EquationException(f"Invalid method: '{method:s}'")

    def forward_solve(self, x, nl_deps, b):
        x.vector()[:] = np.linalg.solve(self._A, b.vector())

    def adjoint_derivative_action(self, nl_deps, nl_dep_index, x, adj_x, b,
                                  method="assign"):
        raise EquationException("Unexpected call to adjoint_derivative_action")

    def adjoint_solve(self, adj_x, nl_deps, b):
        adj_x = function_new(b)
        adj_x.vector()[:] = np.linalg.solve(self._A_H, b.vector())
        return adj_x

    def tangent_linear_rhs(self, M, dM, tlm_map, x):
        return None


class ContractionArray:
    def __init__(self, A, I, A_T=None, alpha=1.0):  # noqa: E741
        if A_T is not None:
            warnings.warn("A_T argument is deprecated and has no effect",
                          DeprecationWarning, stacklevel=2)

        for i in range(len(I) - 1):
            if I[i + 1] <= I[i]:
                raise EquationException("Axes must be in ascending order")

        self._A = A.copy()
        self._A_conjugate = A.conjugate()
        self._I = tuple(I)
        self._alpha = A.dtype.type(alpha)

    def A(self):
        A = self._A
        if isinstance(A, np.ndarray):
            A = A.view()
            A.setflags(write=False)
        return A

    def A_conjugate(self):
        A_conjugate = self._A_conjugate
        if isinstance(A_conjugate, np.ndarray):
            A_conjugate = A_conjugate.view()
            A_conjugate.setflags(write=False)
        return A_conjugate

    def I(self):  # noqa: E741,E743
        return self._I

    def alpha(self):
        return self._alpha

    def value(self, X):
        if len(self._A.shape) == 2 and self._I == (0,):
            v = self._A.T.dot(X[0].vector())
        elif len(self._A.shape) == 2 and self._I == (1,):
            v = self._A.dot(X[0].vector())
        else:
            v = self._A
            if len(self._I) == 0:
                v = v.copy()
            else:
                assert len(self._I) == len(X)
                for i, (j, x) in enumerate(zip(self._I, X)):
                    v = np.tensordot(v, x.vector(), axes=((j - i,), (0,)))

        if self._alpha != 1.0:
            v *= self._alpha

        return v


class ContractionRHS(RHS):
    def __init__(self, A, I, X, A_T=None, alpha=1.0):  # noqa: E741
        if A_T is not None:
            warnings.warn("A_T argument is deprecated and has no effect",
                          DeprecationWarning, stacklevel=2)

        if len(X) != len(A.shape) - 1:
            raise EquationException("Contraction does not result in a vector")
        j = set(range(len(A.shape)))
        for i in I:
            # Raises an error if there is a duplicate element in I
            j.remove(i)
        assert len(j) == 1
        j = j.pop()

        super().__init__(X, nl_deps=[] if len(X) == 1 else X)
        self._c = ContractionArray(A, I, alpha=alpha)
        self._j = j

    def add_forward(self, b, deps):
        b.vector()[:] += self._c.value(deps)

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if dep_index < len(self._c.I()):
            A = self._c.A_conjugate()
            I = self._c.I()  # noqa: E741
            alpha = self._c.alpha().conjugate()
            X = [None for i in range(len(A.shape))]
            k = I[dep_index]
            if len(I) == 1:
                assert dep_index == 0
            else:
                assert len(I) == len(nl_deps)
                for j, (i, nl_dep) in enumerate(zip(I, nl_deps)):
                    if j != dep_index:
                        X[i] = nl_dep
            X[self._j] = adj_x

            A_c = ContractionArray(A,
                                   list(range(k))
                                   + list(range(k + 1, len(A.shape))),
                                   alpha=alpha)
            b.vector()[:] -= A_c.value(X[:k] + X[k + 1:])
        else:
            raise EquationException("dep_index out of bounds")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        X = list(self.dependencies())
        A, I, alpha = self._c.A(), self._c.I(), self._c.alpha()
        assert len(M) == len(dM)
        m_map = dict(zip(M, dM))

        J = []
        for j, x in enumerate(X):
            if x in m_map:
                X[j] = m_map[x]
            else:
                J.append(j)

        if len(J) == 0:
            return ContractionRHS(A, I, X, alpha=alpha)
        else:
            tlm_B = []
            for j in J:
                tau_x = tlm_map[X[j]]
                if tau_x is not None:
                    tlm_B.append(ContractionRHS(A, I,
                                                X[:j] + [tau_x] + X[j + 1:],
                                                alpha=alpha))
            return tlm_B


class ContractionSolver(LinearEquation):
    def __init__(self, A, I, Y, x, A_T=None, alpha=1.0):  # noqa: E741
        if A_T is not None:
            warnings.warn("A_T argument is deprecated and has no effect",
                          DeprecationWarning, stacklevel=2)

        super().__init__(ContractionRHS(A, I, Y, alpha=alpha), x)
