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

from .interface import function_is_scalar, function_name, function_new, \
    function_scalar_value, function_space, functional_term_eq, is_function, \
    new_scalar_function, space_id, space_new

from .equations import AssignmentSolver, AxpySolver
from .manager import manager as _manager

import warnings

__all__ = \
    [
        "Functional",
        "FunctionalException"
    ]


class FunctionalException(Exception):
    pass


class Functional:
    _id_counter = [0]

    def __init__(self, fn=None, space=None, name=None, _fn=None):
        """
        A functional.

        Arguments:

        space  (Optional) The space for the functional.
        name   (Optional) The name of the functional.
        """

        if fn is None:
            fn = _fn
        else:
            warnings.warn("fn argument is deprecated ",
                          DeprecationWarning, stacklevel=2)
            if _fn is not None:
                raise FunctionalException("Cannot specify both fn and _fn "
                                          "arguments")
        del _fn

        if fn is None:
            if space is None:
                fn = new_scalar_function(name=name)
            else:
                fn = space_new(space, name=name)
        else:
            if space is not None \
                    and space_id(space) != space_id(function_space(fn)):
                raise FunctionalException("Invalid function space")
        if not function_is_scalar(fn):
            raise FunctionalException("fn must be a scalar")
        name = function_name(fn)

        self._name = name
        self._fn = fn
        self._id = self._id_counter[0]
        self._id_counter[0] += 1

    def id(self):
        return self._id

    def assign(self, term, manager=None, annotate=None, tlm=None):
        """
        Assign the functional.

        Arguments:

        term      A Form or function, to which the functional is assigned.
        manager   (Optional) The equation manager.
        annotate  (Optional) Whether the equation should be annotated.
        tlm       (Optional) Whether to derive (and solve) associated
                  tangent-linear equations.
        """

        if manager is None:
            manager = _manager()

        new_fn = function_new(self._fn, name=self._name)
        if is_function(term) and function_is_scalar(term):
            new_fn_eq = AssignmentSolver(term, new_fn)
        else:
            new_fn_eq = functional_term_eq(term, new_fn)
        new_fn_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
        self._fn = new_fn

    def addto(self, term=None, manager=None, annotate=None, tlm=None):
        """
        Add to the functional.

        Arguments:

        term      (Optional) A Form or function, which is added to the
                  functional. If not supplied then the functional is copied
                  into a new function (useful for avoiding long range
                  cross-block dependencies).
        manager   (Optional) The equation manager.
        annotate  (Optional) Whether the equations should be annotated.
        tlm       (Optional) Whether to derive (and solve) associated
                  tangent-linear equations.
        """

        if manager is None:
            manager = _manager()

        new_fn = function_new(self._fn, name=self._name)
        if term is None:
            new_fn_eq = AssignmentSolver(self._fn, new_fn)
            new_fn_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
        else:
            if is_function(term) and function_is_scalar(term):
                term_fn = term
            else:
                term_fn = function_new(self._fn, name=f"{self._name:s}_term")
                term_eq = functional_term_eq(term, term_fn)
                term_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
            new_fn_eq = AxpySolver(self._fn, 1.0, term_fn, new_fn)
            new_fn_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
        self._fn = new_fn

    def fn(self):
        """
        Return the function storing the functional value.
        """

        return self._fn

    def space(self):
        """
        Return the space for the functional.
        """

        return function_space(self._fn)

    def value(self):
        """
        Return the value of the functional.
        """

        return function_scalar_value(self._fn)

    def tlm(self, M, dM, max_depth=1, manager=None):
        """
        Return a Functional associated with evaluation of the tangent-linear of
        the functional.
        """

        if manager is None:
            manager = _manager()

        J_fn = manager.tlm(M, dM, self.fn(), max_depth=max_depth)

        return Functional(_fn=J_fn)
