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
from .manager import manager as _manager
from .overloaded_float import Float, FloatSpace

import warnings

__all__ = \
    [
        "Functional"
    ]


class Functional:
    """A convenience class for defining functionals.

    This allocates and stores an internal function, but note that this function
    can change e.g. after adding terms.

    :arg space: The space for the :class:`Functional`. Internal functions are
        in this space. Defaults to `FloatSpace(Float)`.
    :arg name: A :class:`str` name for the functional.
    """

    _id_counter = [0]

    def __init__(self, *, space=None, name=None, _fn=None):
        if _fn is None:
            if space is None:
                space = FloatSpace(Float)
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

    def __float__(self):
        return float(self.value())

    def __complex__(self):
        return complex(self.value())

    def id(self):
        """Return the unique :class:`int` ID associated with this
        :class:`Functional`.

        :returns: The unique :class:`int` ID.
        """

        return self._id

    def assign(self, term, *, manager=None, annotate=None, tlm=None):
        r"""Assign to the functional,

        .. math::

            \mathcal{J} = b.

        Note that this method allocates a new internal function.

        :arg term: The value. Defines the value of :math:`b`. Valid types
            depend upon the backend. :math:`b` may be a function, and with the
            FEniCS or Firedrake backends may be a rank zero UFL :class:`Form`.
        :arg manager: The :class:`tlm_adjoint.tlm_adjoint.EquationManager`.
            Defaults to `manager()`.
        :arg annotate: Whether the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
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
        r"""Add to the functional. Performs two assignments,

        .. math::

            \mathcal{J}_\text{term} = b,

        .. math::

            \mathcal{J}_{new} = \mathcal{J}_\text{old} + J_\text{term},

        where :math:`\mathcal{J}_\text{old}` and :math:`\mathcal{J}_\text{new}`
        are, respectively, the old and new values for the functional, and
        :math:`b` is the term to add.

        Note that this method allocates a new internal function.

        :arg term: The value. Defines the value of :math:`b`. Valid types
            depend upon the backend. :math:`b` may be a function, and with
            the FEniCS or Firedrake backends may be a rank zero UFL
            :class:`Form`.
        :arg manager: The :class:`tlm_adjoint.tlm_adjoint.EquationManager`.
            Defaults to `manager()`.
        :arg annotate: Whether the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
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
        """Return the internal function currently storing the value.

        :returns: The internal function.
        """

        return self._fn

    def fn(self):
        warnings.warn("Functional.fn method is deprecated -- "
                      "use Functional.function instead",
                      DeprecationWarning, stacklevel=2)

        return self._fn

    def space(self):
        """Return the space for the functional.

        :returns: The space
        """

        return function_space(self._fn)

    def value(self):
        """Return the current value of the functional.

        The value may also be accessed by casting using :func:`float` or
        :func:`complex`.

        :returns: The scalar value.
        """

        return function_scalar_value(self._fn)

    def tlm_functional(self, *args, manager=None):
        """Return a :class:`Functional` associated with a tangent-linear
        variable associated with the functional.

        :arg args: A :class:`Sequence` of `(M, dM)` pairs. Here `M` and `dM`
            are each a function or a sequence of functions defining a
            derivative and derivative direction. The tangent-linear variable is
            the derivative of the functional with respect to each `M` and with
            direction defined by each `dM`. Supplying a single pair requests a
            :class:`Functional` associated with a first order tangent-linear
            variable. Supplying multiple pairs leads to a :class:`Functional`
            associated with higher order tangent-linear variables. The relevant
            tangent-linear models must have been configured for the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` `manager`.
        :arg manager: The :class:`tlm_adjoint.tlm_adjoint.EquationManager`.
            Defaults to `manager()`.
        :returns: A :class:`Functional` associated with the tangent-linear
            variable.
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
