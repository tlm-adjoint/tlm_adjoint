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

from .interface import check_space_type, function_is_scalar, function_name, \
    function_new, function_scalar_value, function_space, functional_term_eq, \
    is_function, space_id, space_new

from .equations import Assignment, Axpy
from .float import FloatSpace
from .manager import manager as _manager

import warnings

__all__ = \
    [
        "Functional"
    ]


class Functional:
    _id_counter = [0]

    def __init__(self, *, space=None, name=None, _fn=None):
        """
        A functional.

        Arguments:

        space  (Optional) The space for the functional.
        name   (Optional) The name of the functional.
        """

        if _fn is None:
            if space is None:
                space = FloatSpace()
            fn = space_new(space, name=name)
        else:
            fn = _fn
            del _fn
            if space is not None \
                    and space_id(space) != space_id(function_space(fn)):
                raise ValueError("Invalid function space")
        if not function_is_scalar(fn):
            raise ValueError("Functional must be a scalar")
        check_space_type(fn, "primal")

        name = function_name(fn)

        self._name = name
        self._fn = fn
        self._id = self._id_counter[0]
        self._id_counter[0] += 1

    def id(self):
        return self._id

    def assign(self, term, *, manager=None, annotate=None, tlm=None):
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
            new_fn_eq = Assignment(new_fn, term)
        else:
            new_fn_eq = functional_term_eq(new_fn, term)
        new_fn_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
        self._fn = new_fn

    def addto(self, term=None, *, manager=None, annotate=None, tlm=None):
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
            new_fn_eq = Assignment(new_fn, self._fn)
            new_fn_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
        else:
            if is_function(term) and function_is_scalar(term):
                term_fn = term
            else:
                term_fn = function_new(self._fn, name=f"{self._name:s}_term")
                term_eq = functional_term_eq(term_fn, term)
                term_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
            new_fn_eq = Axpy(new_fn, self._fn, 1.0, term_fn)
            new_fn_eq.solve(manager=manager, annotate=annotate, tlm=tlm)
        self._fn = new_fn

    def function(self):
        """
        Return the function storing the functional value.
        """

        return self._fn

    def fn(self):
        warnings.warn("Functional.fn method is deprecated -- "
                      "use Functional.function instead",
                      DeprecationWarning, stacklevel=2)

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

    def tlm_functional(self, *args, manager=None):
        """
        Return a Functional associated with evaluation of a tangent-linear
        associated with the functional.
        """

        if manager is None:
            manager = _manager()

        return Functional(_fn=manager.function_tlm(self.function(), *args))

    def tlm(self, M, dM, *, max_depth=1, manager=None):
        warnings.warn("Functional.tlm method is deprecated -- "
                      "use Functional.tlm_functional instead",
                      DeprecationWarning, stacklevel=2)

        if manager is None:
            manager = _manager()

        J_fn = manager.function_tlm(
            self.function(), *[(M, dM) for depth in range(max_depth)])

        return Functional(_fn=J_fn)
