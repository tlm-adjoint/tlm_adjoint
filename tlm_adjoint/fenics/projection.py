"""Projection operations with FEniCS.
"""

from .backend import TestFunction, TrialFunction

from .solve import EquationSolver, LocalEquationSolver

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

__all__ = \
    [
        "LocalProjection",
        "Projection"
    ]


class Projection(EquationSolver):
    """Represents the solution of a finite element variational problem
    performing a projection onto the space for `x`.

    :arg x: A DOLFIN `Function` defining the forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to project
        onto the space for `x`. Should not depend on `x`.

    Remaining arguments are passed to the
    :class:`tlm_adjoint.fenics.solve.EquationSolver` constructor.
    """

    def __init__(self, x, rhs, *args, **kwargs):
        space = x.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        lhs = ufl.inner(trial, test) * ufl.dx
        rhs = ufl.inner(rhs, test) * ufl.dx

        super().__init__(lhs == rhs, x, *args, **kwargs)


class LocalProjection(LocalEquationSolver):
    """Represents the solution of a finite element variational problem
    performing a projection onto the space for `x`, for the case where the mass
    matrix is element-wise local block diagonal.

    :arg x: A DOLFIN `Function` defining the forward solution.
    :arg rhs: A :class:`ufl.core.expr.Expr` defining the expression to project
        onto the space for `x`. Should not depend on `x`.

    Remaining arguments are passed to the
    :class:`tlm_adjoint.fenics.solve.LocalEquationSolver` constructor.
    """

    def __init__(self, x, rhs, *args, **kwargs):
        space = x.function_space()
        test, trial = TestFunction(space), TrialFunction(space)
        lhs = ufl.inner(trial, test) * ufl.dx
        rhs = ufl.inner(rhs, test) * ufl.dx

        super().__init__(lhs == rhs, x, *args, **kwargs)
