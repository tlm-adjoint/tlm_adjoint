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

from .interface import function_new, is_function

from .equation import Equation

__all__ = \
    [
        "ControlsMarker",
        "FunctionalMarker"
    ]


class ControlsMarker(Equation):
    r"""Represents

    .. math::

        m = m_\text{input},

    where :math:`m` is the control and :math:`m_\text{input}` the input value
    for the control. The forward residual is defined

    .. math::

        \mathcal{F} \left( m \right) = m - m_\text{input}.

    :arg M: A function or a :class:`Sequence` of functions defining the
        control :math:`m`. May be non-checkpointed.
    """

    def __init__(self, M):
        if is_function(M):
            M = (M,)

        super(Equation, self).__init__()
        self._X = tuple(M)
        self._deps = tuple(M)
        self._nl_deps = ()
        self._ic_deps = ()
        self._adj_ic_deps = ()
        self._adj_X_type = tuple("conjugate_dual" for m in M)

    def adjoint_jacobian_solve(self, adj_X, nl_deps, B):
        return B


class FunctionalMarker(Equation):
    r"""Represents

    .. math::

        J_\text{output} = J,

    where :math:`J` is the functional and :math:`J_\text{output}` is the output
    value for the functional. The forward residual is defined

    .. math::

        \mathcal{F} \left( J_\text{output}, J \right) = J_\text{output} - J.

    :arg J: A function or :class:`tlm_adjoint.functional.Functional` defining
        the functional :math:`J`.
    """

    def __init__(self, J):
        if not is_function(J):
            J = J.function()
        # Extra function allocation could be avoided
        J_ = function_new(J)
        super().__init__([J_], [J_, J], nl_deps=[], ic=False, adj_ic=False)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise IndexError("Unexpected dep_index")
        return (-1.0, adj_x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b
