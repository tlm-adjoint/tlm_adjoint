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

from .base_equations import AssignmentSolver, Equation
try:
    from .equations import EquationSolver
except ImportError:
    pass

from collections import OrderedDict

__all__ = \
    [
        "TimesteppingException",

        "BaseTimeLevel",
        "InitialTimeLevel",
        "TimeLevel",
        "FinalTimeLevel",

        "n",
        "N",

        "TimeLevels",
        "TimeFunction",
        "TimeSystem"
    ]

# Aim for a degree of consistency with the API of the 'timestepping' library,
# as in dolfin-adjoint timestepping_2017.1.0 branch, and as described in
#   J. R. Maddison and P. E. Farrell, "Rapid Development and adjoining of
#   transient finite element models", Computer Methods in Applied Mechanics and
#   Engineering, 276(1), pp. 95--121, 2014


class TimesteppingException(Exception):
    pass


class BaseTimeLevel:
    def __init__(self, order, i):
        self._order = order
        self._i = i

    def __hash__(self):
        return hash((self._order, self._i))

    def __eq__(self, other):
        if isinstance(other, BaseTimeLevel):
            return self._order == other._order and self._i == other._i
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, BaseTimeLevel):
            if self._order < other._order:
                return True
            elif self._order > other._order:
                return False
            else:
                return self._i < other._i
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, BaseTimeLevel):
            if self._order > other._order:
                return True
            elif self._order < other._order:
                return False
            else:
                return self._i > other._i
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        return not self > other

    def __ge__(self, other):
        return not self < other

    def i(self):
        return self._i


class InitialTimeLevel(BaseTimeLevel):
    def __init__(self, i=0):
        super().__init__(order=-1, i=i)

    def __add__(self, other):
        return InitialTimeLevel(self._i + other)

    def __sub__(self, other):
        return InitialTimeLevel(self._i - other)


class TimeLevel(BaseTimeLevel):
    def __init__(self, i=0):
        super().__init__(order=0, i=i)

    def __add__(self, other):
        return TimeLevel(self._i + other)

    def __sub__(self, other):
        return TimeLevel(self._i - other)


class FinalTimeLevel(BaseTimeLevel):
    def __init__(self, i=0):
        super().__init__(order=1, i=i)

    def __add__(self, other):
        return FinalTimeLevel(self._i + other)

    def __sub__(self, other):
        return FinalTimeLevel(self._i - other)


n = TimeLevel()
N = FinalTimeLevel()


class TimeLevels:
    def __init__(self, levels, cycle_map):
        levels = tuple(sorted(set(levels)))
        # Always assign to earlier time levels first in the cycle
        cycle_map = OrderedDict(sorted(cycle_map.items(), key=lambda i: i[0]))

        self._levels = levels
        self._cycle_map = cycle_map

    def __iter__(self):
        for level in self._levels:
            yield level

    def __len__(self):
        return len(self._levels)


class Alias:
    """
    Alias of obj, holding a reference to obj.
    """

    def __init__(self, obj):
        if isinstance(obj, Alias):
            raise TimesteppingException("Cannot alias Alias")
        super().__setattr__("_tlm_adjoint__alias", obj)

    def __new__(cls, obj):
        obj_cls = type(obj)

        class Alias(cls, obj_cls):
            def __new__(cls, *args, **kwargs):
                return obj_cls(*args, **kwargs)

        Alias.__name__ = f"{obj_cls.__name__:s}Alias"
        return super().__new__(Alias)

    def __getattr__(self, key):
        return getattr(self._tlm_adjoint__alias, key)

    def __setattr__(self, key, value):
        return setattr(self._tlm_adjoint__alias, key, value)

    def __delattr__(self, key):
        delattr(self._tlm_adjoint__alias, key)

    def __dir__(self):
        return dir(self._tlm_adjoint__alias)


class TimeFunction:
    def __init__(self, levels, *args, cls=Function, **kwargs):
        # Note that this keeps references to the functions on each time level
        self._fns = {}
        for level in levels:
            fn = cls(*args, **kwargs)
            fn._tlm_adjoint__tfn = self
            fn._tlm_adjoint__level = level
            self._fns[level] = fn

            initial_level = InitialTimeLevel(level.i())
            initial_fn = self._fns[initial_level] = Alias(fn)
            super(Alias, initial_fn).__setattr__("_tlm_adjoint__tfn", self)
            super(Alias, initial_fn).__setattr__("_tlm_adjoint__level",
                                                 initial_level)

            final_level = FinalTimeLevel(level.i())
            final_fn = self._fns[final_level] = Alias(fn)
            super(Alias, final_fn).__setattr__("_tlm_adjoint__tfn", self)
            super(Alias, final_fn).__setattr__("_tlm_adjoint__level",
                                               final_level)

        self._levels = levels
        self._cycle_eqs = None

    def __getitem__(self, level):
        if isinstance(level, BaseTimeLevel):
            return self._fns[level]
        else:
            return self._fns[InitialTimeLevel(level)]

    def __len__(self):
        return len(self._fns)

    def levels(self):
        return self._levels

    def cycle(self, manager=None):
        if self._cycle_eqs is None:
            self._cycle_eqs = tuple(AssignmentSolver(self[source_level],
                                                     self[target_level])
                                    for target_level, source_level
                                    in self._levels._cycle_map.items())
        for eq in self._cycle_eqs:
            eq.solve(manager=manager)


class TimeSystem:
    def __init__(self):
        self._state = "initial"
        self._initial_eqs = []
        self._timestep_eqs = []
        self._final_eqs = []
        self._sorted_eqs = None
        self._tfns = None

    def add_assignment(self, y, x):
        self.add_solve(y, x)

    def add_solve(self, *args, **kwargs):
        if self._state != "initial":
            raise TimesteppingException("Invalid state")

        if len(args) == 1 and isinstance(args[0], Equation):
            eq = args[0]
        elif (len(args) == 2
              and is_function(args[0])
              and is_function(args[1])
              and hasattr(args[1], "_tlm_adjoint__tfn")):
            eq = AssignmentSolver(args[0], args[1])
        else:
            eq = EquationSolver(*args, **kwargs)

        X = eq.X()
        level = X[0]._tlm_adjoint__level
        for x in X[1:]:
            if x._tlm_adjoint__level != level:
                raise TimesteppingException("Inconsistent time levels")
        if isinstance(level, InitialTimeLevel):
            self._initial_eqs.append(eq)
        elif isinstance(level, TimeLevel):
            self._timestep_eqs.append(eq)
        elif isinstance(level, FinalTimeLevel):
            self._final_eqs.append(eq)
        else:
            raise TimesteppingException(f"Invalid time level: {level}")

    def assemble(self, initialize=True, manager=None):
        if self._state != "initial":
            raise TimesteppingException("Invalid state")
        self._state = "assembled"
        assert self._sorted_eqs is None

        for eqs in [self._initial_eqs, self._timestep_eqs, self._final_eqs]:
            x_ids = set()
            for eq in eqs:
                for x in eq.X():
                    x_id = function_id(x)
                    if x_id in x_ids:
                        raise TimesteppingException("Duplicate solve")
                    x_ids.add(x_id)
            del x_ids

        # Dependency resolution
        def add_eq_deps(eq, eq_xs, eqs, parent_ids=None):
            X = eq.X()
            process = X[0] in eq_xs
            for x in X[1:]:
                assert (x in eq_xs) == process
            if not process:
                return
            if parent_ids is None:
                parent_ids = set()
            for x in X:
                parent_ids.add(function_id(x))
            for dep in eq.dependencies():
                if dep not in X and hasattr(dep, "_tlm_adjoint__tfn"):
                    if function_id(dep) in parent_ids:
                        raise TimesteppingException("Circular dependency")
                    elif dep in eq_xs:
                        add_eq_deps(eq_xs[dep], eq_xs, eqs, parent_ids)
            eqs.append(eq)
            for x in X:
                del eq_xs[x]
                parent_ids.remove(function_id(x))

        self._sorted_eqs = [[], [], []]
        for i, eqs in enumerate([self._initial_eqs,
                                 self._timestep_eqs,
                                 self._final_eqs]):
            eq_xs = {}
            for eq in eqs:
                for x in eq.X():
                    eq_xs[x] = eq
            for eq in eqs:
                add_eq_deps(eq, eq_xs, self._sorted_eqs[i])

        # Work around lack of OrderedSet class here
        tfns = {}
        for eq in (self._sorted_eqs[0]
                   + self._sorted_eqs[1]
                   + self._sorted_eqs[2]):
            for x in eq.X():
                x_tfn = x._tlm_adjoint__tfn
                if x_tfn not in tfns:
                    tfns[x_tfn] = x_tfn
        self._tfns = tuple(tfns.keys())

        if initialize:
            self.initialize(manager=manager)

        return self

    def initialize(self, manager=None):
        if self._state != "assembled":
            raise TimesteppingException("Invalid state")
        self._state = "initialized"

        for eq in self._sorted_eqs[0]:
            eq.solve(manager=manager)

        self._initial_eqs = []
        self._sorted_eqs[0] = []

    def timestep(self, s=1, manager=None):
        if self._state == "initial":
            self.assemble(initialize=True, manager=manager)
        if self._state not in ["initialized", "timestepping"]:
            raise TimesteppingException("Invalid state")
        self._state = "timestepping"

        for n in range(s):
            # Timestep solve
            for eq in self._sorted_eqs[1]:
                eq.solve(manager=manager)
            # Timestep cycle
            for tfn in self._tfns:
                tfn.cycle(manager=manager)

    def finalize(self, manager=None):
        if self._state == "initial":
            self.assemble(initialize=True, manager=manager)
        if self._state not in ["initialized", "timestepping"]:
            raise TimesteppingException("Invalid state")
        self._state = "final"

        self._timestep_eqs = []
        self._sorted_eqs[1] = []
        for tfn in self._tfns:
            if tfn._cycle_eqs is not None:
                tfn._cycle_eqs = None
        self._tfns = None

        for eq in self._sorted_eqs[2]:
            eq.solve(manager=manager)

        self._final_eqs = []
        self._sorted_eqs = None
