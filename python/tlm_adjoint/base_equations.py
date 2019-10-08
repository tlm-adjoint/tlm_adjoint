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

from .backend_interface import *

from .manager import manager as _manager

import numpy as np

__all__ = \
    [
        "EquationException",

        "AdjointBlockRHS",
        "AdjointEquationRHS",
        "AdjointModelRHS",
        "AdjointRHS",

        "Equation",
        "EquationAlias",

        "ControlsMarker",
        "FunctionalMarker",

        "get_tangent_linear",

        "AssignmentSolver",
        "AxpySolver",
        "FixedPointSolver",
        "LinearCombinationSolver",
        "NullSolver",
        "ScaleSolver",

        "LinearEquation",
        "Matrix",
        "RHS",

        "InnerProductRHS",
        "InnerProductSolver",
        "MatrixActionRHS",
        "MatrixActionSolver",
        "NormSqRHS",
        "NormSqSolver",
        "SumRHS",
        "SumSolver",

        "Storage",

        "HDF5Storage",
        "MemoryStorage"
    ]


class EquationException(Exception):
    pass


class AdjointRHS:
    def __init__(self, space):
        self._space = space
        self._b = None

    def b(self):
        self.finalize()
        return self._b

    def initialize(self):
        if self._b is None:
            self._b = space_new(self._space)

    def finalize(self):
        self.initialize()
        finalize_adjoint_derivative_action(self._b)

    def sub(self, b):
        if b is not None:
            self.initialize()
            subtract_adjoint_derivative_action(self._b, b)

    def is_empty(self):
        return self._b is None


class AdjointEquationRHS:
    def __init__(self, eq):
        self._B = tuple(AdjointRHS(function_space(x)) for x in eq.X())

    def __getitem__(self, key):
        return self._B[key]

    def b(self):
        if len(self._B) != 1:
            raise EquationException("Right-hand-side does not consist of exactly one function")  # noqa: E501
        return self._B[0].b()

    def B(self):
        return tuple(B.b() for B in self._B)

    def finalize(self):
        for b in self._B:
            b.finalize()

    def is_empty(self):
        for b in self._B:
            if not b.is_empty():
                return False
        return True


class AdjointBlockRHS:
    def __init__(self, block):
        self._B = [AdjointEquationRHS(eq) for eq in block]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._B[key]
        else:
            k, m = key
            return self._B[k][m]

    def pop(self):
        return self._B.pop()

    def finalize(self):
        for B in self._B:
            B.finalize()

    def is_empty(self):
        return len(self._B) == 0


class AdjointModelRHS:
    def __init__(self, blocks):
        self._B = [AdjointBlockRHS(block) for block in blocks]
        self._pop_empty()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._B[key]
        elif len(key) == 2:
            p, k = key
            return self._B[p][k]
        else:
            p, k, m = key
            return self._B[p][k][m]

    def pop(self):
        B = self._B[-1].pop()
        self._pop_empty()
        return B

    def _pop_empty(self):
        while len(self._B) > 0 and self._B[-1].is_empty():
            self._B.pop()

    def is_empty(self):
        return len(self._B) == 0


class Equation:
    _id_counter = [0]

    def __init__(self, X, deps, nl_deps=None, ic_deps=None):
        """
        An equation. The equation is expressed in the form:
            F ( X, y_0, y_1, ... ) = 0,
        where X is the equation solution and F is a residual function.
        Information regarding F is provided by methods which should be
        overridden as required by derived classes.

        Arguments:

        X        A function, or a list or tuple of functions. The solution to
                 the equation.
        deps     A list or tuple of dependencies, which must include the
                 solution itself.
        nl_deps  (Optional) A list or tuple of non-linear dependencies. Must be
                 a subset of deps. Defaults to deps.
        ic_deps  (Optional) A list or tuple of dependencies whose initial value
                 should be available prior to solving the forward equation.
                 Must be a subset of deps. Defaults to the elements of X which
                 are in nl_deps.
        """

        if is_function(X):
            X = (X,)
        for x in X:
            if not is_function(x):
                raise EquationException("Solution must be a function")
            if not function_is_checkpointed(x):
                raise EquationException("Solution must be checkpointed")
            if x not in deps:
                raise EquationException("Solution must be a dependency")
        dep_ids = {function_id(dep): i for i, dep in enumerate(deps)}
        if len(dep_ids) != len(deps):
            raise EquationException("Duplicate dependency")
        if nl_deps is None:
            nl_deps_map = tuple(range(len(deps)))
        else:
            if len(set(function_id(dep) for dep in nl_deps)) != len(nl_deps):
                raise EquationException("Duplicate non-linear dependency")
            for dep in nl_deps:
                if function_id(dep) not in dep_ids:
                    raise EquationException("Non-linear dependency is not a dependency")  # noqa: E501
            nl_deps_map = tuple(dep_ids[function_id(dep)] for dep in nl_deps)
        if ic_deps is None:
            if nl_deps is None:
                ic_deps = list(X)
            else:
                ic_deps = []
                for x in X:
                    if x in nl_deps:
                        ic_deps.append(x)
        else:
            if len(set(function_id(dep) for dep in ic_deps)) != len(ic_deps):
                raise EquationException("Duplicate initial condition dependency")  # noqa: E501
            for dep in ic_deps:
                if function_id(dep) not in dep_ids:
                    raise EquationException("Initial condition dependency is not a dependency")  # noqa: E501

        self._X = tuple(X)
        self._deps = tuple(deps)
        self._nl_deps = None if nl_deps is None else tuple(nl_deps)
        self._nl_deps_map = nl_deps_map
        self._ic_deps = tuple(ic_deps)
        self._id = self._id_counter[0]
        self._id_counter[0] += 1

    def id(self):
        return self._id

    def replace(self, replace_map):
        """
        Replace all internal functions using the supplied replace map. Must
        call the base class replace method.
        """

        self._X = tuple(replace_map.get(x, x) for x in self._X)
        self._deps = tuple(replace_map.get(dep, dep) for dep in self._deps)
        if self._nl_deps is not None:
            self._nl_deps = tuple(replace_map.get(dep, dep)
                                  for dep in self._nl_deps)
        if self._ic_deps is not None:
            self._ic_deps = tuple(replace_map.get(dep, dep)
                                  for dep in self._ic_deps)

    def x(self):
        """
        If the equation solves for exactly one function, return it. Otherwise
        raise an error.
        """

        if len(self._X) != 1:
            raise EquationException("Equation does not solve for exactly one "
                                    "function")
        return self._X[0]

    def X(self):
        """
        A tuple of functions. The solution to the equation.
        """

        return self._X

    def dependencies(self):
        return self._deps

    def nonlinear_dependencies(self):
        if self._nl_deps is None:
            return self.dependencies()
        else:
            return self._nl_deps

    def nonlinear_dependencies_map(self):
        return self._nl_deps_map

    def initial_condition_dependencies(self):
        return self._ic_deps

    def _pre_process(self, manager=None, annotate=None):
        if manager is None:
            manager = _manager()
        for dep in self.initial_condition_dependencies():
            manager.add_initial_condition(dep, annotate=annotate)

    def _post_process(self, manager=None, annotate=None, tlm=None,
                      tlm_skip=None):
        if manager is None:
            manager = _manager()
        manager.add_equation(self, annotate=annotate, tlm=tlm,
                             tlm_skip=tlm_skip)

    def solve(self, manager=None, annotate=None, tlm=None, _tlm_skip=None):
        """
        Solve the equation.

        Arguments:

        manager   (Optional) The equation manager.
        annotate  (Optional) Whether the equation should be annotated.
        tlm       (Optional) Whether to derive (and solve) associated
                  tangent-linear equations.
        """

        if manager is None:
            manager = _manager()

        self._pre_process(manager=manager, annotate=annotate)

        annotation_enabled, tlm_enabled = manager.stop()
        self.forward(self.X())
        manager.start(annotation=annotation_enabled, tlm=tlm_enabled)

        self._post_process(manager=manager, annotate=annotate, tlm=tlm,
                           tlm_skip=_tlm_skip)

    def forward(self, X, deps=None):
        """
        Solve the equation. The manager is stopped when this method is called.
        Lower-level version than forward_solve, and need not generally be
        overridden by custom Equation classes.

        Arguments:

        X     A list or tuple of functions. The solution, which should be set
              by this method.
        deps  (Optional) A list or tuple of functions defining the values of
              dependencies.
        """

        self.forward_solve(X[0] if len(X) == 1 else X, deps=deps)
        function_update_state(*X)

    def forward_solve(self, X, deps=None):
        """
        Solve the equation. The manager is stopped when this method is called.

        The form:
            forward_solve(self, x, deps=None)
        should be used for equations which solve for a single function.

        Arguments:

        x/X   The solution, which should be set by this method.
        deps  (Optional) A list or tuple of functions defining the values of
              dependencies. self.dependencies() should be used if this is not
              supplied.
        """

        raise EquationException("Method not overridden")

    def reset_adjoint(self):
        """
        Can be used to clear adjoint model caches. Called at the start of an
        adjoint calculation.
        """

        pass

    def adjoint(self, J, nl_deps, B, B_indices, Bs):
        """
        Solve the adjoint equation with the given right-hand-side, and subtract
        corresponding adjoint terms from other adjoint equations.

        Arguments:

        J          Adjoint model functional.
        nl_deps    A list or tuple of functions defining the values of
                   non-linear dependencies.
        B          A list or tuple of functions defining the right-hand-side.
                   May be modified by this method. May not have previously have
                   had boundary conditions applied.
        b_indices  A dictionary of j:(p, k, m) pairs. Bs[p][k][m] has an
                   adjoint term arising from a derivative action,
                   differentiating with respect to the dependency for this
                   equation with index j.
        Bs         An AdjointModelRHS, storing adjoint RHS data.

        Returns the solution of the adjoint equation as a tuple of functions.
        The result must have relevant boundary conditions applied, and should
        never be modified by calling code.
        """

        self.initialize_adjoint(J, nl_deps)

        adj_X = self.adjoint_jacobian_solve(nl_deps,
                                            B[0] if len(B) == 1 else B)
        if adj_X is not None:
            for j, (p, k, m) in B_indices.items():
                Bs[p][k][m].sub(self.adjoint_derivative_action(nl_deps, j,
                                                               adj_X))
            if is_function(adj_X):
                adj_X = (adj_X,)

        self.finalize_adjoint(J)

        return adj_X

    def initialize_adjoint(self, J, nl_deps):
        """
        Adjoint initialization. Called prior to calling adjoint_jacobian_solve
        or adjoint_derivative_action methods.

        Arguments:

        J        Adjoint model functional.
        nl_deps  A list or tuple of functions defining the values of non-linear
                 dependencies.
        """

        pass

    def finalize_adjoint(self, J):
        """
        Adjoint finalization. Called after calling adjoint_jacobian_solve or
        adjoint_derivative_action methods.

        Arguments:

        J          Adjoint model functional.
        """

        pass

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        """
        Return the action of the adjoint of a derivative of the RHS.

        Boundary conditions need not be applied in the returned result. The
        return value should never be modified by calling code.

        The form:
            adjoint_derivative_action(self, nl_deps, dep_index, adj_x)
        should be used for equations which solve for a single function.

        Arguments:

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        dep_index    The index of the dependency in self.dependencies() with
                     respect to which a derivative should be taken.
        adj_x/adj_X  The direction of the adjoint derivative action.
        """

        raise EquationException("Method not overridden")

    def adjoint_jacobian_solve(self, nl_deps, B):
        """
        Solve an adjoint equation, returning the result. The result must have
        relevant boundary conditions applied, and should never be modified by
        calling code.

        The form:
            adjoint_jacobian_solve(self, nl_deps, b)
        should be used for equations which solve for a single function.

        Arguments:

        nl_deps    A list or tuple of functions defining the values of
                   non-linear dependencies.
        b/B        The right-hand-side. May be modified by this method. May not
                   have previously have had boundary conditions applied.
        """

        raise EquationException("Method not overridden")

    def tangent_linear(self, M, dM, tlm_map):
        """
        Return an Equation corresponding to a tangent linear equation,
        computing derivatives with respect to the control M in the direction
        dM.

        Arguments:

        M        A list or tuple of functions defining the control.
        dM       A list or tuple of functions defining the direction.
        tlm_map  The TangentLinearMap.
        """

        raise EquationException("Method not overridden")


class EquationAlias(Equation):
    def __init__(self, eq):
        Equation.__setattr__(
            self, "_tlm_adjoint__alias__dict__",
            eq.__dict__)
        Equation.__setattr__(
            self, "_tlm_adjoint__alias__str__",
            f"{type(eq).__name__:s} (aliased)")

    def __new__(cls, obj):
        class EquationAlias(cls, type(obj)):
            pass
        return object.__new__(EquationAlias)

    def __str__(self):
        return self._tlm_adjoint__alias__str__

    def __getattr__(self, key):
        if key not in self._tlm_adjoint__alias__dict__:
            raise AttributeError(f"No attribute '{key:s}'")
        return self._tlm_adjoint__alias__dict__[key]

    def __setattr__(self, key, value):
        self._tlm_adjoint__alias__dict__[key] = value
        return value

    def __delattr__(self, key):
        del(self._tlm_adjoint__alias__dict__[key])

    def __dir__(self):
        return list(self._tlm_adjoint__alias__dict__.keys())


class ControlsMarker(Equation):
    def __init__(self, M):
        """
        Represents the equation "controls = inputs".

        Arguments:

        M  A function, or a list or tuple of functions. May be static.
        """

        if is_function(M):
            M = (M,)

        self._X = tuple(M)
        self._deps = tuple(M)
        self._nl_deps = ()
        self._nl_deps_map = ()
        self._ic_deps = ()
        self._id = self._id_counter[0]
        self._id_counter[0] += 1

    def adjoint_jacobian_solve(self, nl_deps, B):
        return B


class FunctionalMarker(Equation):
    def __init__(self, J):
        """
        Represents the equation "output = functional".

        Arguments:

        J  A function. The functional.
        """

        # Any function in the correct space suffices here
        J_alias = function_alias(J)
        Equation.__init__(self, J_alias, [J_alias, J], nl_deps=[], ic_deps=[])

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index != 1:
            raise EquationException("Unexpected dep_index")
        return (-1.0, adj_x)

    def adjoint_jacobian_solve(self, nl_deps, b):
        return b


def get_tangent_linear(x, M, dM, tlm_map):
    try:
        return dM[M.index(x)]
    except ValueError:
        return tlm_map[x]


class NullSolver(Equation):
    def __init__(self, X):
        if is_function(X):
            X = (X,)
        Equation.__init__(self, X, X, nl_deps=[], ic_deps=[])

    def forward_solve(self, X, deps=None):
        if is_function(X):
            X = (X,)
        for x in X:
            function_zero(x)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_function(adj_X):
            adj_X = (adj_X,)
        if dep_index < len(adj_X):
            return adj_X[dep_index]
        else:
            return None

    def adjoint_jacobian_solve(self, nl_deps, B):
        return B

    def tangent_linear(self, M, dM, tlm_map):
        return NullSolver([tlm_map[x] for x in self.X()])


class AssignmentSolver(Equation):
    def __init__(self, y, x):
        if function_local_size(x) != function_local_size(y):
            raise EquationException("Invalid function space")
        Equation.__init__(self, x, [x, y], nl_deps=[], ic_deps=[])

    def forward_solve(self, x, deps=None):
        _, y = self.dependencies() if deps is None else deps
        function_assign(x, y)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        elif dep_index == 1:
            return (-1.0, adj_x)
        else:
            return None

    def adjoint_jacobian_solve(self, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        x, y = self.dependencies()
        tau_y = get_tangent_linear(y, M, dM, tlm_map)
        if tau_y is None:
            return NullSolver(tlm_map[x])
        else:
            return AssignmentSolver(tau_y, tlm_map[x])


class LinearCombinationSolver(Equation):
    def __init__(self, x, *args):
        alpha = tuple(float(arg[0]) for arg in args)
        Y = [arg[1] for arg in args]
        for y in Y:
            if function_local_size(x) != function_local_size(y):
                raise EquationException("Invalid function space")

        Equation.__init__(self, x, [x] + Y, nl_deps=[], ic_deps=[])
        self._alpha = alpha

    def forward_solve(self, x, deps=None):
        deps = self.dependencies() if deps is None else tuple(deps)
        function_zero(x)
        for alpha, y in zip(self._alpha, deps[1:]):
            function_axpy(x, alpha, y)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        elif dep_index <= len(self._alpha):
            return (-self._alpha[dep_index - 1], adj_x)
        else:
            return None

    def adjoint_jacobian_solve(self, nl_deps, b):
        return b

    def tangent_linear(self, M, dM, tlm_map):
        deps = self.dependencies()
        x, ys = deps[0], deps[1:]
        args = []
        for alpha, y in zip(self._alpha, ys):
            tau_y = get_tangent_linear(y, M, dM, tlm_map)
            if tau_y is not None:
                args.append((alpha, tau_y))
        return LinearCombinationSolver(tlm_map[x], *args)


class ScaleSolver(LinearCombinationSolver):
    def __init__(self, alpha, y, x):
        LinearCombinationSolver.__init__(self, x, (alpha, y))


class AxpySolver(LinearCombinationSolver):
    def __init__(self, x_old, alpha, y, x_new):
        LinearCombinationSolver.__init__(self, x_new, (1.0, x_old),
                                         (alpha, y))


class FixedPointSolver(Equation):
    # Derives tangent-linear and adjoint information using the approach
    # described in
    #   J. G. Gilbert, "Automatic differentiation and iterative processes",
    #     Optimization Methods and Software, 1(1), pp. 13--21, 1992
    #   B. Christianson, "Reverse accumulation and attractive fixed points",
    #     Optimization Methods and Software, 3(4), pp. 311--326, 1994
    def __init__(self, eqs, solver_parameters):
        """
        A fixed point solver.

        Arguments:

        eqs
            A list or tuple of Equation objects. The last equation defines the
            solution of the fixed point iteration. A single function cannot
            appear as the solution to two or more equations.
        solver_parameters
            Solver parameters dictionary. Parameters (based on KrylovSolver
            parameters in FEniCS 2017.2.0):
                absolute_tolerance
                    Absolute tolerance for the solution change 2-norm. Float,
                    required.
                relative_tolerance
                    Relative tolerance for the solution change 2-norm. Float,
                    required.
                maximum_iterations
                    Maximum permitted iterations. Positive integer, optional,
                    default 1000.
                nonzero_initial_guess
                    Whether to use a non-zero initial guess for the forward
                    solve (for the final equation in eqs). Logical, optional,
                    default True.
                nonzero_adjoint_initial_guess
                    Whether to use a non-zero initial guess for the adjoint
                    solve. If True, the solution on the previous
                    adjoint_jacobian_solve call is retained and used as an
                    initial guess for a later call. If False, or on the first
                    call, a zero initial guess is used. Logical, optional,
                    default False.
                report
                    Whether to display output. Optional, default False.
        """

        X_ids = set()
        for eq in eqs:
            for x in eq.X():
                x_id = function_id(x)
                if x_id in X_ids:
                    raise EquationException("Duplicate solve")
                X_ids.add(x_id)

        solver_parameters = copy_parameters_dict(solver_parameters)
        # Based on KrylovSolver parameters in FEniCS 2017.2.0
        for key, default_value in [("maximum_iterations", 1000),
                                   ("nonzero_initial_guess", True),
                                   ("nonzero_adjoint_initial_guess", False),
                                   ("report", False)]:
            if key not in solver_parameters:
                solver_parameters[key] = default_value

        X = []
        deps = []
        dep_ids = {}
        nl_deps = []
        nl_dep_ids = {}
        ic_deps = {}

        eq_X_indices = tuple([] for eq in eqs)
        eq_dep_indices = tuple([] for eq in eqs)
        eq_nl_dep_indices = tuple([] for eq in eqs)

        previous_x_ids = set()
        remaining_x_ids = X_ids
        del(X_ids)

        for i, eq in enumerate(eqs):
            for x in eq.X():
                X.append(x)
                eq_X_indices[i].append(len(X) - 1)

            for dep in eq.dependencies():
                dep_id = function_id(dep)
                if dep_id not in dep_ids:
                    deps.append(dep)
                    dep_ids[dep_id] = len(deps) - 1
                eq_dep_indices[i].append(dep_ids[dep_id])
                if dep_id in remaining_x_ids and dep_id not in ic_deps:
                    ic_deps[dep_id] = dep

            for dep in eq.nonlinear_dependencies():
                dep_id = function_id(dep)
                if dep_id not in nl_dep_ids:
                    nl_deps.append(dep)
                    nl_dep_ids[dep_id] = len(nl_deps) - 1
                eq_nl_dep_indices[i].append(nl_dep_ids[dep_id])

            for dep in eq.initial_condition_dependencies():
                dep_id = function_id(dep)
                # Could exclude eqs[-1].X() here if nonzero_initial_guess is
                # False
                if dep_id not in previous_x_ids and dep_id not in ic_deps:
                    ic_deps[dep_id] = dep

            for x in eq.X():
                x_id = function_id(x)
                if x_id in remaining_x_ids:
                    remaining_x_ids.remove(x_id)
                previous_x_ids.add(x_id)

        del(previous_x_ids, remaining_x_ids, dep_ids, nl_dep_ids)
        ic_deps = tuple(ic_deps.values())

        eq_dep_ids = tuple({function_id(eq_dep): i
                            for i, eq_dep in enumerate(eq.dependencies())}
                           for eq in eqs)

        dep_map = {}
        for i, eq in enumerate(eqs):
            for m, x in enumerate(eq.X()):
                dep_map[function_id(x)] = (i, m)
        tdeps = tuple([] for eq in eqs)
        for k, eq in enumerate(eqs):
            X_ids = set(function_id(x) for x in eq.X())
            for j, dep in enumerate(eq.dependencies()):
                dep_id = function_id(dep)
                if dep_id not in X_ids and dep_id in dep_map:
                    i, m = dep_map[dep_id]
                    tdeps[i].append((j, k, m))
        del(dep_map)

        Equation.__init__(self, X, deps, nl_deps=nl_deps, ic_deps=ic_deps)
        self._eqs = tuple(eqs)
        self._eq_X_indices = eq_X_indices
        self._eq_dep_indices = eq_dep_indices
        self._eq_nl_dep_indices = eq_nl_dep_indices
        self._eq_dep_ids = eq_dep_ids
        self._solver_parameters = solver_parameters

        self._tdeps = tdeps

        self._adj_X_cache = {}

    def replace(self, replace_map):
        Equation.replace(self, replace_map)
        for eq in self._eqs:
            eq.replace(replace_map)

    def forward_solve(self, X, deps=None):
        if is_function(X):
            X = (X,)

        # Based on KrylovSolver parameters in FEniCS 2017.2.0
        absolute_tolerance = self._solver_parameters["absolute_tolerance"]
        relative_tolerance = self._solver_parameters["relative_tolerance"]
        maximum_iterations = self._solver_parameters["maximum_iterations"]
        nonzero_initial_guess = \
            self._solver_parameters["nonzero_initial_guess"]
        report = self._solver_parameters["report"]

        eq_X = tuple(tuple(X[j] for j in self._eq_X_indices[i])
                     for i in range(len(self._eqs)))
        if deps is None:
            eq_deps = tuple(None for i in range(len(self._eqs)))
        else:
            eq_deps = tuple(tuple(deps[j] for j in self._eq_dep_indices[i])
                            for i in range(len(self._eqs)))

        if not nonzero_initial_guess:
            for x in eq_X[-1]:
                function_zero(x)

        it = 0
        X_0 = tuple(function_copy(x) for x in eq_X[-1])
        while True:
            it += 1

            for i, eq in enumerate(self._eqs):
                eq.forward(eq_X[i], deps=eq_deps[i])

            R = X_0
            del(X_0)
            R_norm_sq = 0.0
            for r, x in zip(R, eq_X[-1]):
                function_axpy(r, -1.0, x)
                R_norm_sq += function_inner(r, r)
            if relative_tolerance == 0.0:
                tolerance_sq = absolute_tolerance ** 2
            else:
                X_norm_sq = 0.0
                for x in eq_X[-1]:
                    X_norm_sq += function_inner(x, x)
                tolerance_sq = max(absolute_tolerance ** 2,
                                   X_norm_sq * (relative_tolerance ** 2))
            if report:
                info(f"Fixed point iteration, forward iteration {it:d}, "
                     f"change norm {np.sqrt(R_norm_sq):.16e} "
                     f"(tolerance {np.sqrt(tolerance_sq):.16e})")
            if np.isnan(R_norm_sq):
                raise EquationException(
                    f"Fixed point iteration, forward iteration {it:d}, "
                    f"NaN encountered")
            if R_norm_sq < tolerance_sq or R_norm_sq == 0.0:
                break
            if it >= maximum_iterations:
                raise EquationException(
                    f"Fixed point iteration, forward iteration {it:d}, "
                    f"failed to converge")

            X_0 = R
            del(R)
            for x_0, x in zip(X_0, eq_X[-1]):
                function_assign(x_0, x)

    def reset_adjoint(self):
        for eq in self._eqs:
            eq.reset_adjoint()

        self._adj_X_cache.clear()

    def initialize_adjoint(self, J, nl_deps):
        self._eq_nl_deps = tuple(tuple(nl_deps[j]
                                       for j in self._eq_nl_dep_indices[i])
                                 for i in range(len(self._eqs)))

        for eq, eq_nl_deps in zip(self._eqs, self._eq_nl_deps):
            eq.initialize_adjoint(J, eq_nl_deps)

        if self._solver_parameters["nonzero_adjoint_initial_guess"]:
            J_id = J.id()
            if J_id not in self._adj_X_cache:
                self._adj_X_cache[J_id] = [function_new(x) for x in self.X()]
            self._adj_X = self._adj_X_cache[J_id]
        else:
            self._adj_X = [function_new(x) for x in self.X()]
        self._eq_adj_X = [tuple(self._adj_X[j] for j in self._eq_X_indices[i])
                          for i in range(len(self._eqs))]

    def finalize_adjoint(self, J):
        del(self._eq_nl_deps)
        del(self._adj_X)
        del(self._eq_adj_X)

    def adjoint_jacobian_solve(self, nl_deps, B):
        if is_function(B):
            B = (B,)

        # Based on KrylovSolver parameters in FEniCS 2017.2.0
        absolute_tolerance = self._solver_parameters["absolute_tolerance"]
        relative_tolerance = self._solver_parameters["relative_tolerance"]
        maximum_iterations = self._solver_parameters["maximum_iterations"]
        report = self._solver_parameters["report"]

        adj_X = self._adj_X
        eq_adj_X = self._eq_adj_X

        it = 0
        X_0 = tuple(function_copy(x) for x in eq_adj_X[-1])
        while True:
            it += 1

            for i in range(len(self._eqs) - 1, - 1, -1):
                i = (i - 1) % len(self._eqs)
                eq_B = tuple(function_copy(B[j])
                             for j in self._eq_X_indices[i])

                for j, k, m in self._tdeps[i]:
                    if len(eq_adj_X[k]) == 1:
                        sb = self._eqs[k].adjoint_derivative_action(
                            self._eq_nl_deps[k], j, eq_adj_X[k][0])
                    else:
                        sb = self._eqs[k].adjoint_derivative_action(
                            self._eq_nl_deps[k], j, eq_adj_X[k])
                    subtract_adjoint_derivative_action(eq_B[m], sb)
                    del(sb)
                for b in eq_B:
                    finalize_adjoint_derivative_action(b)

                eq_adj_X[i] = self._eqs[i].adjoint_jacobian_solve(
                    self._eq_nl_deps[i], eq_B[0] if len(eq_B) == 1 else eq_B)
                if eq_adj_X[i] is None:
                    eq_adj_X[i] = tuple(function_new(b) for b in eq_B)
                elif is_function(eq_adj_X[i]):
                    eq_adj_X[i] = (eq_adj_X[i],)
                for j, x in zip(self._eq_X_indices[i], eq_adj_X[i]):
                    adj_X[j] = x

            R = X_0
            del(X_0)
            R_norm_sq = 0.0
            for r, x in zip(R, eq_adj_X[-1]):
                function_axpy(r, -1.0, x)
                R_norm_sq += function_inner(r, r)
            if relative_tolerance == 0.0:
                tolerance_sq = absolute_tolerance ** 2
            else:
                X_norm_sq = 0.0
                for x in eq_adj_X[-1]:
                    X_norm_sq += function_inner(x, x)
                tolerance_sq = max(absolute_tolerance ** 2,
                                   X_norm_sq * (relative_tolerance ** 2))
            if report:
                info(f"Fixed point iteration, adjoint iteration {it:d}, "
                     f"change norm {np.sqrt(R_norm_sq):.16e} "
                     f"(tolerance {np.sqrt(tolerance_sq):.16e})")
            if np.isnan(R_norm_sq):
                raise EquationException(
                    f"Fixed point iteration, adjoint iteration {it:d}, "
                    f"NaN encountered")
            if R_norm_sq < tolerance_sq or R_norm_sq == 0.0:
                break
            if it >= maximum_iterations:
                raise EquationException(
                    f"Fixed point iteration, adjoint iteration {it:d}, "
                    f"failed to converge")

            X_0 = R
            del(R)
            for x_0, x in zip(X_0, eq_adj_X[-1]):
                function_assign(x_0, x)

        return adj_X

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_function(adj_X):
            adj_X = (adj_X,)

        dep = self.dependencies()[dep_index]
        dep_id = function_id(dep)
        F = function_new(dep)
        for eq, eq_nl_deps, eq_dep_ids, eq_adj_X in zip(self._eqs,
                                                        self._eq_nl_deps,
                                                        self._eq_dep_ids,
                                                        self._eq_adj_X):
            if dep_id in eq_dep_ids:
                sb = eq.adjoint_derivative_action(
                    eq_nl_deps, eq_dep_ids[dep_id],
                    eq_adj_X[0] if len(eq_adj_X) == 1 else eq_adj_X)
                subtract_adjoint_derivative_action(F, sb)
                del(sb)
        finalize_adjoint_derivative_action(F)

        return (-1.0, F)

    def tangent_linear(self, M, dM, tlm_map):
        tlm_eqs = []
        for eq in self._eqs:
            tlm_eq = eq.tangent_linear(M, dM, tlm_map)
            if tlm_eq is None:
                tlm_eq = NullSolver([tlm_map[x] for x in eq.X()])
            tlm_eqs.append(tlm_eq)

        return FixedPointSolver(tlm_eqs,
                                solver_parameters=self._solver_parameters)


class LinearEquation(Equation):
    def __init__(self, B, X, A=None):
        if isinstance(B, RHS):
            B = (B,)
        if is_function(X):
            X = (X,)

        deps = []
        dep_ids = {}
        nl_deps = []
        nl_dep_ids = {}

        x_ids = set()
        for x in X:
            x_id = function_id(x)
            if x_id in x_ids:
                raise EquationException("Duplicate solve")
            x_ids.add(x_id)
            deps.append(x)
            dep_ids[x_id] = len(deps) - 1

        b_dep_indices = tuple([] for b in B)
        b_nl_dep_indices = tuple([] for b in B)

        for i, b in enumerate(B):
            for dep in b.dependencies():
                dep_id = function_id(dep)
                if dep_id in x_ids:
                    raise EquationException("Invalid dependency in linear Equation")  # noqa: E501
                if dep_id not in dep_ids:
                    deps.append(dep)
                    dep_ids[dep_id] = len(deps) - 1
                b_dep_indices[i].append(dep_ids[dep_id])
            for dep in b.nonlinear_dependencies():
                dep_id = function_id(dep)
                if dep_id not in nl_dep_ids:
                    nl_deps.append(dep)
                    nl_dep_ids[dep_id] = len(nl_deps) - 1
                b_nl_dep_indices[i].append(nl_dep_ids[dep_id])

        b_dep_ids = tuple({function_id(b_dep): i
                           for i, b_dep in enumerate(b.dependencies())}
                          for b in B)

        if A is None:
            ic_deps = []
        else:
            A_dep_indices = []
            A_nl_dep_indices = []
            for dep in A.nonlinear_dependencies():
                dep_id = function_id(dep)
                if dep_id not in dep_ids:
                    deps.append(dep)
                    dep_ids[dep_id] = len(deps) - 1
                A_dep_indices.append(dep_ids[dep_id])
                if dep_id not in nl_dep_ids:
                    nl_deps.append(dep)
                    nl_dep_ids[dep_id] = len(nl_deps) - 1
                A_nl_dep_indices.append(nl_dep_ids[dep_id])

            A_nl_dep_ids = {function_id(A_nl_dep): i
                            for i, A_nl_dep
                            in enumerate(A.nonlinear_dependencies())}

            if len(A.nonlinear_dependencies()) > 0:
                A_x_indices = []
                for x in X:
                    x_id = function_id(x)
                    if x_id not in nl_dep_ids:
                        nl_deps.append(x)
                        nl_dep_ids[x_id] = len(nl_deps) - 1
                    A_x_indices.append(nl_dep_ids[x_id])

            ic_deps = X if A.has_initial_condition_dependency() else []

        del(x_ids, dep_ids, nl_dep_ids)

        Equation.__init__(self, X, deps, nl_deps=nl_deps, ic_deps=ic_deps)
        self._B = tuple(B)
        self._b_dep_indices = b_dep_indices
        self._b_nl_dep_indices = b_nl_dep_indices
        self._b_dep_ids = b_dep_ids
        self._A = A
        if A is not None:
            self._A_dep_indices = A_dep_indices
            self._A_nl_dep_indices = A_nl_dep_indices
            self._A_nl_dep_ids = A_nl_dep_ids
            if len(A.nonlinear_dependencies()) > 0:
                self._A_x_indices = A_x_indices

    def replace(self, replace_map):
        Equation.replace(self, replace_map)
        for b in self._B:
            b.replace(replace_map)
        if self._A is not None:
            self._A.replace(replace_map)

    def forward_solve(self, X, deps=None):
        if is_function(X):
            X = (X,)
        if deps is None:
            deps = self.dependencies()

        if self._A is None:
            B = X
        else:
            B = tuple(function_new(x) for x in X)

        for i, b in enumerate(self._B):
            b.add_forward(B[0] if len(B) == 1 else B,
                          [deps[j] for j in self._b_dep_indices[i]])

        if self._A is not None:
            self._A.forward_solve(X[0] if len(X) == 1 else X,
                                  [deps[j] for j in self._A_dep_indices],
                                  B[0] if len(B) == 1 else B)

    def reset_adjoint(self):
        if self._A is not None:
            self._A.reset_adjoint()

    def initialize_adjoint(self, J, nl_deps):
        if self._A is not None:
            self._A.initialize_adjoint(J, nl_deps)

    def finalize_adjoint(self, J):
        if self._A is not None:
            self._A.finalize_adjoint(J)

    def adjoint_jacobian_solve(self, nl_deps, B):
        if self._A is None:
            return B
        else:
            return self._A.adjoint_solve([nl_deps[j]
                                          for j in self._A_nl_dep_indices],
                                         B)

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_X):
        if is_function(adj_X):
            adj_X = (adj_X,)
        if dep_index < len(self.X()):
            if self._A is None:
                return adj_X[dep_index]
            else:
                dep = self.dependencies()[dep_index]
                F = function_new(dep)
                self._A.adjoint_action([nl_deps[j]
                                        for j in self._A_nl_dep_indices],
                                       adj_X[0] if len(adj_X) == 1 else adj_X,
                                       F, b_index=dep_index, method="assign")
                return F
        else:
            dep = self.dependencies()[dep_index]
            dep_id = function_id(dep)
            F = function_new(dep)
            for i, (b, b_dep_ids) in enumerate(zip(self._B, self._b_dep_ids)):
                try:
                    b_dep_index = b_dep_ids[dep_id]
                except KeyError:
                    continue
                b_nl_deps = [nl_deps[j] for j in self._b_nl_dep_indices[i]]
                b.subtract_adjoint_derivative_action(
                    b_nl_deps, b_dep_index,
                    adj_X[0] if len(adj_X) == 1 else adj_X,
                    F)
            if self._A is not None and dep_id in self._A_nl_dep_ids:
                A_nl_dep_index = self._A_nl_dep_ids[dep_id]
                A_nl_deps = [nl_deps[j] for j in self._A_nl_dep_indices]
                X = [nl_deps[j] for j in self._A_x_indices]
                self._A.adjoint_derivative_action(
                    A_nl_deps, A_nl_dep_index,
                    X[0] if len(X) == 1 else X,
                    adj_X[0] if len(adj_X) == 1 else adj_X,
                    F, method="add")
            return F

    def tangent_linear(self, M, dM, tlm_map):
        X = self.X()

        if self._A is None:
            tlm_B = []
        else:
            tlm_B = self._A.tangent_linear_rhs(M, dM, tlm_map,
                                               X[0] if len(X) == 1 else X)
            if tlm_B is None:
                tlm_B = []
            elif isinstance(tlm_B, RHS):
                tlm_B = [tlm_B]
        for b in self._B:
            tlm_b = b.tangent_linear_rhs(M, dM, tlm_map)
            if tlm_b is None:
                pass
            elif isinstance(tlm_b, RHS):
                tlm_B.append(tlm_b)
            else:
                tlm_B.extend(tlm_b)

        if len(tlm_B) == 0:
            return NullSolver([tlm_map[x] for x in self.X()])
        else:
            return LinearEquation(tlm_B, [tlm_map[x] for x in self.X()],
                                  A=self._A)


class Matrix:
    def __init__(self, nl_deps=None, has_ic_dep=False):
        if nl_deps is not None:
            if len(set(function_id(dep) for dep in nl_deps)) != len(nl_deps):
                raise EquationException("Duplicate non-linear dependency")

        self._nl_deps = () if nl_deps is None else tuple(nl_deps)
        self._has_ic_dep = has_ic_dep

    def replace(self, replace_map):
        self._nl_deps = tuple(replace_map.get(dep, dep)
                              for dep in self._nl_deps)

    def nonlinear_dependencies(self):
        return self._nl_deps

    def has_initial_condition_dependency(self):
        return self._has_ic_dep

    def forward_action(self, nl_deps, X, B, method="assign"):
        """
        Evaluate the (forward) action of the matrix.

        The form:
            forward_action(self, nl_deps, x, b, method="assign")
        should be used for matrices which act on a single function.

        Arguments:

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        x/X          The argument of the matrix action.
        b/B          The result of the matrix action.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise EquationException("Method not overridden")

    def adjoint_action(self, nl_deps, adj_X, b, b_index=0, method="assign"):
        """
        Evaluate the adjoint action of the matrix.

        The form:
            adjoint_action(self, nl_deps, adj_x, b, b_index=0, method="assign")
        should be used for matrices which act on a single function.

        Arguments:

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        adj_x/adj_X  The argument of the matrix action.
        b            The result of the matrix action.
        b_index      (Optional) The element of the matrix action B to compute.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise EquationException("Method not overridden")

    def forward_solve(self, X, nl_deps, B):
        raise EquationException("Method not overridden")

    def reset_adjoint(self):
        pass

    def initialize_adjoint(self, J, nl_deps):
        pass

    def finalize_adjoint(self, J):
        pass

    def adjoint_derivative_action(self, nl_deps, nl_dep_index, X, adj_X, b,
                                  method="assign"):
        """
        Evaluate the action of the adjoint of a derivative of the matrix
        action.

        The form:
            adjoint_derivative_action(self, nl_deps, nl_dep_index, x, adj_x, b,
                                      method="assign")
        should be used for matrices which act on a single function.

        Arguments:

        nl_deps      A list or tuple of functions defining the values of
                     non-linear dependencies.
        nl_dep_index The index of the dependency in
                     self.nonlinear_dependencies() with respect to which a
                     derivative should be taken.
        x/X          The argument of the forward matrix action.
        adj_x/adj_X  The direction of the adjoint derivative action.
        b            The result.
        method       (Optional) One of {"assign", "add", "sub"}.
        """

        raise EquationException("Method not overridden")

    def adjoint_solve(self, nl_deps, B):
        raise EquationException("Method not overridden")

    def tangent_linear_rhs(self, M, dM, tlm_map, X):
        raise EquationException("Method not overridden")


class RHS:
    def __init__(self, deps, nl_deps=None):
        dep_ids = set(function_id(dep) for dep in deps)
        if len(dep_ids) != len(deps):
            raise EquationException("Duplicate dependency")
        if nl_deps is not None:
            nl_dep_ids = set(function_id(dep) for dep in nl_deps)
            if len(nl_dep_ids) != len(nl_deps):
                raise EquationException("Duplicate non-linear dependency")
            if len(dep_ids.intersection(nl_dep_ids)) != len(nl_deps):
                raise EquationException("Non-linear dependency is not a dependency")  # noqa: E501

        self._deps = tuple(deps)
        self._nl_deps = None if nl_deps is None else tuple(nl_deps)

    def replace(self, replace_map):
        self._deps = tuple(replace_map.get(dep, dep) for dep in self._deps)
        if self._nl_deps is not None:
            self._nl_deps = tuple(replace_map.get(dep, dep)
                                  for dep in self._nl_deps)

    def dependencies(self):
        return self._deps

    def nonlinear_dependencies(self):
        if self._nl_deps is None:
            return self.dependencies()
        else:
            return self._nl_deps

    def add_forward(self, B, deps):
        raise EquationException("Method not overridden")

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_X, b):
        raise EquationException("Method not overridden")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        raise EquationException("Method not overridden")


class MatrixActionSolver(LinearEquation):
    def __init__(self, Y, A, X):
        LinearEquation.__init__(self, MatrixActionRHS(A, Y), X)


class InnerProductSolver(LinearEquation):
    def __init__(self, y, z, x, alpha=1.0, M=None):
        LinearEquation.__init__(self, InnerProductRHS(y, z, alpha=alpha, M=M),
                                x)


class NormSqSolver(InnerProductSolver):
    def __init__(self, y, x, alpha=1.0, M=None):
        InnerProductSolver.__init__(self, y, y, x, alpha=alpha, M=M)


class SumSolver(LinearEquation):
    def __init__(self, y, x):
        LinearEquation.__init__(self, SumRHS(y), x)


class MatrixActionRHS(RHS):
    def __init__(self, A, X):
        if is_function(X):
            X = (X,)
        if len(set(function_id(x) for x in X)) != len(X):
            raise EquationException("Invalid dependency")

        A_nl_deps = A.nonlinear_dependencies()
        if len(A_nl_deps) == 0:
            x_indices = {i: i for i in range(len(X))}
            RHS.__init__(self, X, nl_deps=[])
        else:
            nl_deps = list(A_nl_deps)
            nl_dep_ids = {function_id(dep): i for i, dep in enumerate(nl_deps)}
            x_indices = {}
            for i, x in enumerate(X):
                x_id = function_id(x)
                if x_id not in nl_dep_ids:
                    nl_deps.append(x)
                    nl_dep_ids[x_id] = len(nl_deps) - 1
                x_indices[nl_dep_ids[x_id]] = i
            RHS.__init__(self, nl_deps, nl_deps=nl_deps)

        self._A = A
        self._x_indices = x_indices

    def replace(self, replace_map):
        RHS.replace(self, replace_map)
        self._A.replace(replace_map)

    def add_forward(self, B, deps):
        if is_function(B):
            B = (B,)
        X = [deps[j] for j in self._x_indices]
        self._A.forward_action(deps[:len(self._A.nonlinear_dependencies())],
                               X[0] if len(X) == 1 else X,
                               B[0] if len(B) == 1 else B,
                               method="add")

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_X, b):
        if is_function(adj_X):
            adj_X = (adj_X,)
        N_A_nl_deps = len(self._A.nonlinear_dependencies())
        if dep_index < N_A_nl_deps:
            X = [nl_deps[j] for j in self._x_indices]
            self._A.adjoint_derivative_action(
                nl_deps[:N_A_nl_deps], dep_index,
                X[0] if len(X) == 1 else X,
                adj_X[0] if len(adj_X) == 1 else adj_X,
                b, method="sub")
        if dep_index in self._x_indices:
            self._A.adjoint_action(nl_deps[:N_A_nl_deps],
                                   adj_X[0] if len(adj_X) == 1 else adj_X,
                                   b, b_index=self._x_indices[dep_index],
                                   method="sub")

    def tangent_linear_rhs(self, M, dM, tlm_map):
        deps = self.dependencies()
        N_A_nl_deps = len(self._A.nonlinear_dependencies())

        X = [deps[j] for j in self._x_indices]
        tlm_X = tuple(get_tangent_linear(x, M, dM, tlm_map) for x in X)
        tlm_B = [MatrixActionRHS(self._A, tlm_X)]

        if N_A_nl_deps > 0:
            tlm_b = self._A.tangent_linear_rhs(M, dM, tlm_map, X)
            if tlm_b is None:
                pass
            elif isinstance(tlm_b, RHS):
                tlm_B.append(tlm_b)
            else:
                tlm_B.extend(tlm_b)

        return tlm_B


class InnerProductRHS(RHS):
    def __init__(self, x, y, alpha=1.0, M=None):
        """
        An equation representing an inner product.

        Arguments:

        x, y   Inner product arguments. May be the same function.
        alpha  (Optional) Scale the result of the inner product by alpha.
        M      (Optional) Matrix defining the inner product. Assumed symmetric,
               and must have no non-linear dependencies. Defaults to an
               identity matrix.
        """

        if M is not None and len(M.nonlinear_dependencies()) > 0:
            raise EquationException("Non-linear matrix dependencies not supported")  # noqa: E501

        norm_sq = x == y
        if norm_sq:
            deps = [x]
        else:
            deps = [x, y]

        RHS.__init__(self, deps, nl_deps=deps)
        self._x = x
        self._y = y
        self._norm_sq = norm_sq
        self._alpha = alpha
        self._M = M

    def replace(self, replace_map):
        RHS.replace(self, replace_map)
        self._x = replace_map.get(self._x, self._x)
        self._y = replace_map.get(self._y, self._y)
        if self._M is not None:
            self._M.replace(replace_map)

    def add_forward(self, b, deps):
        if self._norm_sq:
            x, y = deps[0], deps[0]
            M_deps = deps[1:]
        else:
            x, y = deps[:2]
            M_deps = deps[2:]

        if self._M is None:
            Y = y
        else:
            Y = function_new(x)
            self._M.forward_action(M_deps, y, Y, method="assign")

        function_set_values(b,
                            function_get_values(b) + self._alpha
                            * function_inner(x, Y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if self._norm_sq:
            if dep_index == 0:
                x = nl_deps[0]
                M_deps = nl_deps[1:]

                if self._M is None:
                    X = x
                else:
                    X = function_new(x)
                    self._M.forward_action(M_deps, x, X, method="assign")

                function_axpy(b, -2.0 * self._alpha * function_sum(adj_x), X)
        elif dep_index == 0:
            x, y = nl_deps[:2]
            M_deps = nl_deps[2:]

            if self._M is None:
                Y = y
            else:
                Y = function_new(x)
                self._M.forward_action(M_deps, y, Y, method="assign")

            function_axpy(b, -self._alpha * function_sum(adj_x), Y)
        elif dep_index == 1:
            x, y = nl_deps[:2]
            M_deps = nl_deps[2:]

            if self._M is None:
                X = x
            else:
                X = function_new(y)
                self._M.forward_action(M_deps, x, X, method="assign")

            function_axpy(b, -self._alpha * function_sum(adj_x), X)

    def tangent_linear_rhs(self, M, dM, tlm_map):
        tlm_B = []

        if self._norm_sq:
            x = self.dependencies()[0]
            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                tlm_B.append(InnerProductRHS(x, tlm_x, alpha=2.0 * self._alpha,
                                             M=self._M))
        else:
            x, y = self.dependencies()[:2]

            tlm_x = get_tangent_linear(x, M, dM, tlm_map)
            if tlm_x is not None:
                tlm_B.append(InnerProductRHS(tlm_x, y, alpha=self._alpha,
                                             M=self._M))

            tlm_y = get_tangent_linear(y, M, dM, tlm_map)
            if tlm_y is not None:
                tlm_B.append(InnerProductRHS(x, tlm_y, alpha=self._alpha,
                                             M=self._M))

        return tlm_B


class NormSqRHS(InnerProductRHS):
    def __init__(self, x, alpha=1.0, M=None):
        InnerProductRHS.__init__(self, x, x, alpha=alpha, M=M)


class SumRHS(RHS):
    def __init__(self, x):
        RHS.__init__(self, [x], nl_deps=[])

    def add_forward(self, b, deps):
        y, = deps
        function_set_values(b, function_get_values(b) + function_sum(y))

    def subtract_adjoint_derivative_action(self, nl_deps, dep_index, adj_x, b):
        if dep_index == 0:
            function_set_values(b,
                                function_get_values(b) - function_sum(adj_x))

    def tangent_linear_rhs(self, M, dM, tlm_map):
        y, = self.dependencies()
        tau_y = get_tangent_linear(y, M, dM, tlm_map)
        if tau_y is None:
            return None
        else:
            return SumRHS(tau_y)


class Storage(Equation):
    def __init__(self, x, key):
        Equation.__init__(self, x, [x], nl_deps=[], ic_deps=[])
        self._key = key

    def key(self):
        return self._key

    def is_saved(self):
        raise EquationException("Method not overridden")

    def load(self, x):
        raise EquationException("Method not overridden")

    def save(self, x):
        raise EquationException("Method not overridden")

    def forward_solve(self, x, deps=None):
        if self.is_saved():
            self.load(x)
        else:
            self.save(x)

    def adjoint_jacobian_solve(self, nl_deps, b):
        return b

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        else:
            return None

    def tangent_linear(self, M, dM, tlm_map):
        return NullSolver(tlm_map[self.x()])


class MemoryStorage(Storage):
    def __init__(self, x, d, key):
        Storage.__init__(self, x, key)
        self._d = d

    def is_saved(self):
        return self.key() in self._d

    def load(self, x):
        function_set_values(x, self._d[self.key()])

    def save(self, x):
        self._d[self.key()] = function_get_values(x)


class HDF5Storage(Storage):
    def __init__(self, x, h, key):
        Storage.__init__(self, x, key)
        self._h = h

    def is_saved(self):
        return self.key() in self._h

    def load(self, x):
        d = self._h[self.key()]["value"]
        function_set_values(x, d[function_local_indices(x)])

    def save(self, x):
        key = self.key()
        self._h.create_group(key)
        values = function_get_values(x)
        d = self._h[key].create_dataset("value",
                                        shape=(function_global_size(x),),
                                        dtype=values.dtype)
        d[function_local_indices(x)] = values
