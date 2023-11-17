#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .interface import functional_term_eq, is_var, var_is_scalar

from .equations import Assignment, Axpy
from .manager import var_tlm
from .overloaded_float import Float

import numbers
import sympy as sp
import warnings

__all__ = \
    [
        "Functional"
    ]


class Functional(Float):
    """A convenience class for defining functionals.

    Arguments are as for the :class:`.Float` class.
    """

    def __init__(self, *args, space=None, **kwargs):
        if space is not None:
            warnings.warn("space argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def assign(self, y, *, annotate=None, tlm=None):
        """Assign to the :class:`.Functional`.

        :arg y: The value.
        :arg annotate: Whether the :class:`.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        :returns: The :class:`.Functional`.
        """

        if is_var(y) and var_is_scalar(y):
            Assignment(self, y).solve(annotate=annotate, tlm=tlm)
            return self
        elif isinstance(y, (numbers.Complex, sp.Expr)):
            return super().assign(y, annotate=annotate, tlm=tlm)
        else:
            functional_term_eq(self, y).solve(annotate=annotate, tlm=tlm)
            return self

    def addto(self, y, *, annotate=None, tlm=None):
        """Add to the :class:`.Functional`.

        :arg y: The value to add.
        :arg annotate: Whether the :class:`.EquationManager` should record the
            solution of equations.
        :arg tlm: Whether tangent-linear equations should be solved.
        """

        if is_var(y) and var_is_scalar(y):
            J_old = self.new(self, annotate=annotate, tlm=tlm)
            Axpy(self, J_old, 1.0, y).solve(annotate=annotate, tlm=tlm)
        elif isinstance(y, (numbers.Complex, sp.Expr)):
            super().addto(y, annotate=annotate, tlm=tlm)
        else:
            J_old = self.new(self, annotate=annotate, tlm=tlm)
            b = self.new()
            functional_term_eq(b, y).solve(annotate=annotate, tlm=tlm)
            Axpy(self, J_old, 1.0, b).solve(annotate=annotate, tlm=tlm)

    def function(self):
        ""

        warnings.warn("Functional.function method is deprecated")
        return self

    def tlm(self, M, dM, *, max_depth=1):
        warnings.warn("Functional.tlm method is deprecated",
                      DeprecationWarning, stacklevel=2)

        return var_tlm(self, *((M, dM) for _ in range(max_depth)))
