#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import (
    check_space_type, functional_term_eq, is_var, space_id, space_new,
    var_is_scalar, var_name, var_new, var_scalar_value, var_space)

from .equations import Assignment, Axpy
from .manager import var_tlm
from .overloaded_float import Float, FloatSpace

import warnings

__all__ = \
    [
        "Functional"
    ]


class Functional:
    """A convenience class for defining functionals.

    This allocates and stores an internal variable, but note that this variable
    can change e.g. after adding terms.

    :arg space: The space for the :class:`Functional`. Internal variables are
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
                    and space_id(space) != space_id(var_space(fn)):
                raise ValueError("Invalid space")
        if not var_is_scalar(fn):
            raise ValueError("Functional must be a scalar")
        check_space_type(fn, "primal")

        name = var_name(fn)

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

    def assign(self, term, *, annotate=None, tlm=None):
        r"""Assign to the functional,

        .. math::

            \mathcal{J} = b.

        Note that this method allocates a new internal variable.

        :arg term: The value. Defines the value of :math:`b`. Valid types
            depend upon the backend. :math:`b` may be a variable, and with the
            FEniCS or Firedrake backends may be an arity zero
            :class:`ufl.Form`.
        :arg annotate: Whether the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        """

        new_fn = var_new(self._fn, name=self._name)
        if is_var(term) and var_is_scalar(term):
            new_fn_eq = Assignment(new_fn, term)
        else:
            new_fn_eq = functional_term_eq(new_fn, term)
        new_fn_eq.solve(annotate=annotate, tlm=tlm)
        self._fn = new_fn

    def addto(self, term=None, *, annotate=None, tlm=None):
        r"""Add to the functional. Performs two assignments,

        .. math::

            \mathcal{J}_\text{term} = b,

        .. math::

            \mathcal{J}_{new} = \mathcal{J}_\text{old}
                + \mathcal{J}_\text{term},

        where :math:`\mathcal{J}_\text{old}` and :math:`\mathcal{J}_\text{new}`
        are, respectively, the old and new values for the functional, and
        :math:`b` is the term to add.

        Note that this method allocates a new internal variable.

        :arg term: The value. Defines the value of :math:`b`. Valid types
            depend upon the backend. :math:`b` may be a variable, and with the
            FEniCS or Firedrake backends may be an arity zero
            :class:`ufl.Form`.
        :arg annotate: Whether the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        """

        new_fn = var_new(self._fn, name=self._name)
        if term is None:
            new_fn_eq = Assignment(new_fn, self._fn)
            new_fn_eq.solve(annotate=annotate, tlm=tlm)
        else:
            if is_var(term) and var_is_scalar(term):
                term_fn = term
            else:
                term_fn = var_new(self._fn, name=f"{self._name:s}_term")
                term_eq = functional_term_eq(term_fn, term)
                term_eq.solve(annotate=annotate, tlm=tlm)
            new_fn_eq = Axpy(new_fn, self._fn, 1.0, term_fn)
            new_fn_eq.solve(annotate=annotate, tlm=tlm)
        self._fn = new_fn

    def function(self):
        """
        """

        # warnings.warn("function method is deprecated -- "
        #               "use var instead",
        #               DeprecationWarning, stacklevel=2)

        return self.var()

    def var(self):
        """Return the internal variable currently storing the value.

        :returns: The internal variable.
        """

        return self._fn

    def space(self):
        """Return the space for the functional.

        :returns: The space
        """

        return var_space(self._fn)

    def value(self):
        """Return the current value of the functional.

        The value may also be accessed by casting using :class:`float` or
        :class:`complex`.

        :returns: The scalar value.
        """

        return var_scalar_value(self._fn)

    def tlm_functional(self, *args):
        """Return a :class:`Functional` associated with a tangent-linear
        variable associated with the functional.

        :arg args: A :class:`Sequence` of `(M, dM)` pairs. Here `M` and `dM`
            are each a variable or a sequence of variables defining a
            derivative and derivative direction. The tangent-linear variable is
            the derivative of the functional with respect to each `M` and with
            direction defined by each `dM`. Supplying a single pair requests a
            :class:`Functional` associated with a first order tangent-linear
            variable. Supplying multiple pairs leads to a :class:`Functional`
            associated with higher order tangent-linear variables. The relevant
            tangent-linear models must have been configured for the
            :class:`tlm_adjoint.tlm_adjoint.EquationManager` `manager`.
        :returns: A :class:`Functional` associated with the tangent-linear
            variable.
        """

        return Functional(_fn=var_tlm(self.var(), *args))

    def tlm(self, M, dM, *, max_depth=1):
        warnings.warn("Functional.tlm method is deprecated -- "
                      "use Functional.tlm_functional instead",
                      DeprecationWarning, stacklevel=2)

        J_fn = var_tlm(
            self.var(), *((M, dM) for _ in range(max_depth)))

        return Functional(_fn=J_fn)
