#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This module includes functionality for use with the tlm_adjoint NumPy
backend.
"""

from ..interface import function_get_values, function_new_conjugate, \
    function_set_values

from ..linear_equation import LinearEquation, Matrix, RHS

import numpy as np

__all__ = \
    [
        "ConstantMatrix",
        "ContractionRHS",
        "Contraction"
    ]


class ConstantMatrix(Matrix):
    r"""A matrix :math:`A` with no dependencies.

    :arg A: An ndim 2 :class:`numpy.ndarray` defining :math:`A`.
    :arg ic: Whether solution of a linear equation :math:`A x = b` for
        :math:`x` uses an initial guess.
    :arg adj_ic: Whether solution of an adjoint linear equation :math:`A^*
        \lambda = b` for :math:`\lambda` uses an initial guess.
    """

    def __init__(self, A, *,
                 ic=False, adj_ic=False):
        super().__init__(nl_deps=[], ic=ic, adj_ic=adj_ic)
        self._A = A.copy()
        self._A_H = A.conjugate().T

    def A(self):
        A = self._A
        if isinstance(A, np.ndarray):
            A = A.view()
            A.setflags(write=False)
        return A

    def forward_action(self, nl_deps, x, b, *, method="assign"):
        sb = self._A.dot(x.vector())
        if method == "assign":
            b.vector()[:] = sb
        elif method == "add":
            b.vector()[:] += sb
        elif method == "sub":
            b.vector()[:] -= sb
        else:
            raise ValueError(f"Invalid method: '{method:s}'")

    def adjoint_action(self, nl_deps, adj_x, b, b_index=0, *, method="assign"):
        if b_index != 0:
            raise IndexError("Invalid index")
        sb = self._A_H.dot(adj_x.vector())
        if method == "assign":
            b.vector()[:] = sb
        elif method == "add":
            b.vector()[:] += sb
        elif method == "sub":
            b.vector()[:] -= sb
        else:
            raise ValueError(f"Invalid method: '{method:s}'")

    def forward_solve(self, x, nl_deps, b):
        x.vector()[:] = np.linalg.solve(self._A, b.vector())

    def adjoint_derivative_action(self, nl_deps, nl_dep_index, x, adj_x, b, *,
                                  method="assign"):
        raise NotImplementedError("Unexpected call to "
                                  "adjoint_derivative_action")

    def adjoint_solve(self, adj_x, nl_deps, b):
        adj_x = self.new_adj_x()
        adj_x.vector()[:] = np.linalg.solve(self._A_H, b.vector())
        return adj_x

    def tangent_linear_rhs(self, M, dM, tlm_map, x):
        return None


class ContractionArray:
    def __init__(self, A, I, *,  # noqa: E741
                 alpha=1.0):
        for i in range(len(I) - 1):
            if I[i + 1] <= I[i]:
                raise ValueError("Axes must be in ascending order")

        self._A = A.copy()
        self._A_conjugate = A.conjugate()
        self._I = tuple(I)
        self._alpha = alpha

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
    r"""Represents a right-hand-side term corresponding to

    .. math::

        \sum_{i_0} \sum_{i_1} \ldots \sum_{i_{N - 1}}
            A_{i_0,i_1,\ldots,j,\ldots,i_{N - 1}}
            x_{i_0} x_{i_1} \ldots x_{i_{N - 1}},

    where :math:`A` has rank :math:`(N - 1)`.

    :arg A: An ndim :math:`(N - 1)` :class:`numpy.ndarray`.
    :arg I: A :class:`Sequence` of length :math:`(N - 1)` defining the
        :math:`i_0,\ldots,i_{N - 1}`.
    :arg X: A :class:`Sequence` of functions defining the
        :math:`x_0,\ldots,x_{N - 1}`.
    """

    def __init__(self, A, I, X, *,  # noqa: E741
                 alpha=1.0):
        if len(X) != len(A.shape) - 1:
            raise ValueError("Contraction does not result in a vector")
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
                        X[i] = function_new_conjugate(nl_dep)
                        function_set_values(
                            X[i], function_get_values(nl_dep).conjugate())
            X[self._j] = adj_x

            A_c = ContractionArray(A,
                                   list(range(k))
                                   + list(range(k + 1, len(A.shape))),
                                   alpha=alpha)
            b.vector()[:] -= A_c.value(X[:k] + X[k + 1:])
        else:
            raise IndexError("dep_index out of bounds")

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


class Contraction(LinearEquation):
    r"""
    Represents an assignment

    .. math::

        x = \alpha \sum_{i_0} \sum_{i_1} \ldots \sum_{i_{N - 1}}
            A_{i_0,i_1,\ldots,j,\ldots,i_{N - 1}}
            y_{i_0} y_{i_1} \ldots y_{i_{N - 1}},

    where :math:`A` has rank :math:`(N - 1)`.

    The forward residual is defined

    .. math::

        \mathcal{F} \left( x, y_0, \ldots, y_{N - 1} \right)
            = x - \alpha \sum_{i_0} \sum_{i_1} \ldots \sum_{i_{N - 1}}
            A_{i_0,i_1,\ldots,j,\ldots,i_{N - 1}}
            y_{i_0} y_{i_1} \ldots y_{i_{N - 1}}.

    :arg x: A function corresponding to `x`.
    :arg A: An ndim :math:`(N - 1)` :class:`numpy.ndarray`.
    :arg I: A :class:`Sequence` of length :math:`(N - 1)` defining the
        :math:`i_0,\ldots,i_{N - 1}`.
    :arg Y: A :class:`Sequence` of functions defining the
        :math:`y_0,\ldots,y_{N - 1}`.
    :arg alpha: A scalar defining :math:`\alpha`.
    """

    def __init__(self, x, A, I, Y, *,  # noqa: E741
                 alpha=1.0):
        super().__init__(x, ContractionRHS(A, I, Y, alpha=alpha))
