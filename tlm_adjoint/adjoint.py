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

from .interface import finalize_adjoint_derivative_action, function_copy, \
    function_space, function_space_type, space_new, \
    subtract_adjoint_derivative_action

from collections.abc import Sequence

__all__ = \
    [
        "AdjointRHS",
        "AdjointEquationRHS",
        "AdjointBlockRHS",
        "AdjointModelRHS"
    ]


class AdjointRHS:
    """The right-hand-side of an adjoint equation, for an adjoint variable
    associated with an equation solving for a forward variable `x`.

    :arg x: A function defining the forward variable.
    """

    def __init__(self, x):
        self._space = function_space(x)
        self._space_type = function_space_type(x, rel_space_type="conjugate_dual")  # noqa: E501
        self._b = None

    def b(self, *, copy=False):
        """Return the right-hand-side, as a function. Note that any deferred
        contributions *are* added to the function before it is returned -- see
        :meth:`finalize`.

        :arg copy: If `True` then a copy of the internal function storing the
            right-hand-side value is returned. If `False` the internal function
            itself is returned.
        :returns: A function storing the right-hand-side value.
        """

        self.finalize()
        if copy:
            return function_copy(self._b)
        else:
            return self._b

    def initialize(self):
        """Allocate an internal function to store the right-hand-side. Called
        by the :meth:`finalize` and :meth:`sub` methods, and typically need not
        be called manually.
        """

        if self._b is None:
            self._b = space_new(self._space, space_type=self._space_type)

    def finalize(self):
        """Subtracting of terms from the internal function storing the
        right-hand-side may be deferred. In particular finite element assembly
        may be deferred until a more complete expression, consisting of
        multiple terms, has been constructed. This method updates the internal
        function so that all deferred contributions are subtracted.
        """

        self.initialize()
        finalize_adjoint_derivative_action(self._b)

    def sub(self, b):
        """Subtract a term from the right-hand-side.

        :arg b: The term to subtract.
            :func:`subtract_adjoint_derivative_action` is used to subtract the
            term.
        """

        if b is not None:
            self.initialize()
            subtract_adjoint_derivative_action(self._b, b)

    def is_empty(self):
        """Return whether the right-hand-side is 'empty', meaning that the
        :meth:`initialize` method has not been called.

        :returns: `True` if the :meth:`initialize` method has not been called,
            and `False` otherwise.
        """

        return self._b is None


class AdjointEquationRHS:
    """The right-hand-side of an adjoint equation, for adjoint variables
    associated with an equation solving for multiple forward variables `X`.

    Multiple :class:`AdjointRHS` objects. The :class:`AdjointRHS` objects may
    be accessed by index, e.g.

    .. code-block:: python

        adj_eq_rhs = AdjointEquationRHS(eq)
        adj_rhs = adj_eq_rhs[m]

    :arg eq: An :class:`Equation`. `eq.X()` defines the forward variables.
    """

    def __init__(self, eq):
        self._B = tuple(AdjointRHS(x) for x in eq.X())

    def __getitem__(self, key):
        return self._B[key]

    def b(self, *, copy=False):
        """For the case where there is a single forward variable, return a
        function associated with the right-hand-side.

        :arg copy: If `True` then a copy of the internal function storing the
            right-hand-side value is returned. If `False` the internal function
            itself is returned.
        :returns: A function storing the right-hand-side value.
        """

        b, = self._B
        return b.b(copy=copy)

    def B(self, *, copy=False):
        """Return functions associated with the right-hand-sides.

        :arg copy: If `True` then copies of the internal functions storing the
            right-hand-side values are returned. If `False` the internal
            functions themselves are returned.
        :returns: A :class:`tuple` of functions storing the right-hand-side
            values.
        """

        return tuple(B.b(copy=copy) for B in self._B)

    def finalize(self):
        """Call the :meth:`finalize` methods of all :class:`AdjointRHS`
        objects.
        """

        for b in self._B:
            b.finalize()

    def is_empty(self):
        """Return whether all of the :class:`AdjointRHS` objects are 'empty',
        meaning that the :meth:`initialize` method has not been called for any
        :class:`AdjointRHS`.

        :returns: `True` if the :meth:`initialize` method has not been called
            for any :class:`AdjointRHS`, and `False` otherwise.
        """

        for b in self._B:
            if not b.is_empty():
                return False
        return True


class AdjointBlockRHS:
    """The right-hand-side of multiple adjoint equations.

    Multiple :class:`AdjointEquationRHS` objects. The
    :class:`AdjointEquationRHS` objects may be accessed by index, e.g.

    .. code-block:: python

        adj_block_rhs = AdjointBlockRHS(block)
        adj_eq_rhs = adj_block_rhs[k]

    :class:`AdjointRHS` objects may be accessed e.g.

    .. code-block:: python

        adj_rhs = adj_block_rhs[(k, m)]

    :arg block: A :class:`Sequence` of :class:`Equation` objects.
    """

    def __init__(self, block):
        self._B = [AdjointEquationRHS(eq) for eq in block]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._B[key]
        else:
            k, m = key
            return self._B[k][m]

    def pop(self):
        """Remove and return the last :class:`AdjointEquationRHS` in the
        :class:`AdjointBlockRHS`.

        :returns: The last :class:`AdjointEquationRHS` in the
            :class:`AdjointBlockRHS`.
        """

        return len(self._B) - 1, self._B.pop()

    def finalize(self):
        """Call the :meth:`finalize` methods of all :class:`AdjointEquationRHS`
        objects.
        """

        for B in self._B:
            B.finalize()

    def is_empty(self):
        """Return whether there are no :class:`AdjointEquationRHS` objects in
        the :class:`AdjointBlockRHS`.

        :returns: `True` if there are no :class:`AdjointEquationRHS` objects in
            the :class:`AdjointBlockRHS`, and `False` otherwise.
        """

        return len(self._B) == 0


class AdjointModelRHS:
    """The right-hand-side of multiple blocks of adjoint equations.

    Multiple :class:`AdjointBlockRHS` objects. The :class:`AdjointBlockRHS`
    objects may be accessed by index, e.g.

    .. code-block:: python

        adj_model_rhs = AdjointModelRHS(block)
        adj_block_rhs = adj_block_rhs[p]

    :class:`AdjointEquationRHS` objects may be accessed e.g.

    .. code-block:: python

        adj_eq_rhs = adj_block_rhs[(p, k)]

    :class:`AdjointRHS` objects may be accessed e.g.

    .. code-block:: python

        adj_rhs = adj_block_rhs[(p, k, m)]

    If the last block of adjoint equations contains no equations then it is
    automatically removed from the :class:`AdjointModelRHS`.

    :arg blocks: A :class:`Sequence` of :class:`Sequence` objects each
        containing :class:`Equation` objects, or a :class:`Mapping` with items
        `(index, block)` where `index` is an :class:`int` and `block` a
        :class:`Sequence` of :class:`Equation` objects. In the latter case
        blocks are ordered by `index`.
    """

    def __init__(self, blocks):
        if isinstance(blocks, Sequence):
            # Sequence
            self._blocks_n = list(range(len(blocks)))
        else:
            # Mapping
            self._blocks_n = sorted(blocks.keys())
        self._B = {n: AdjointBlockRHS(blocks[n]) for n in self._blocks_n}
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
        """Remove and return the last :class:`AdjointEquationRHS` in the last
        :class:`AdjointBlockRHS` in the :class:`AdjointModelRHS`.

        :returns: The last :class:`AdjointEquationRHS` in the last
            :class:`AdjointBlockRHS` in the :class:`AdjointModelRHS`.
        """

        n = self._blocks_n[-1]
        i, B = self._B[n].pop()
        self._pop_empty()
        return (n, i), B

    def _pop_empty(self):
        while len(self._B) > 0 and self._B[self._blocks_n[-1]].is_empty():
            del self._B[self._blocks_n.pop()]

    def is_empty(self):
        """Return whether there are no :class:`AdjointBlockRHS` objects in the
        :class:`AdjointModelRHS`.

        :returns: `True` if there are no :class:`AdjointBlockRHS` objects in
            the :class:`AdjointModelRHS`, and `False` otherwise.
        """

        return len(self._B) == 0
