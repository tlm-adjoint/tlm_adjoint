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

from .interface import function_get_values, function_global_size, \
    function_local_indices, function_set_values

from .equation import Equation
from .equations import ZeroAssignment

__all__ = \
    [
        "Storage",

        "HDF5Storage",
        "MemoryStorage"
    ]


class Storage(Equation):
    r"""Used to save and load a forward solution value.

    With `save=True` the first forward solve saves the value of `x`. With
    `save=False`, or on any subsequent forward solve, the value of the forward
    solution is loaded into `x`.

    When processed by the :class:`EquationManager` this is equivalent to an
    assignment

    .. math::

        x = x_\text{value},

    where :math:`x_\text{value}` is the value which is saved or loaded. The
    forward residual is defined

    .. math::

        \mathcal{F} \left( x \right) = x - x_\text{value}.

    This is an abstract base class. Information required to save and load data
    is provided by overloading abstract methods. This class does *not* inherit
    from :class:`abc.ABC`, so that methods may be implemented as needed.

    :arg x: A function defining the forward solution, whose value is saved or
        loaded.
    :arg key: A :class:`str` key for the saved or loaded data.
    :arg save: If `True` then the first forward solve saves the value of `x`.
        If `False` then the first forward solve loads the value of `x`.
    """

    def __init__(self, x, key, *, save=False):
        super().__init__(x, [x], nl_deps=[], ic=False, adj_ic=False)
        self._key = key
        self._save = save

    def key(self):
        """Return the key associated with saved or loaded data.

        :returns: The :class:`str` key.
        """

        return self._key

    def is_saved(self):
        """Return whether a value can be loaded.

        :returns: `True` if a value can be loaded, and `False` otherwise.
        """

        raise NotImplementedError("Method not overridden")

    def load(self, x):
        """Load data, storing the result in `x`.

        :arg x: A function in which the loaded data is stored.
        """

        raise NotImplementedError("Method not overridden")

    def save(self, x):
        """Save the value of `x`.

        :arg x: A function whose value should be saved.
        """

        raise NotImplementedError("Method not overridden")

    def forward_solve(self, x, deps=None):
        if not self._save or self.is_saved():
            self.load(x)
        else:
            self.save(x)

    def adjoint_jacobian_solve(self, adj_x, nl_deps, b):
        return b

    def adjoint_derivative_action(self, nl_deps, dep_index, adj_x):
        if dep_index == 0:
            return adj_x
        else:
            raise IndexError("dep_index out of bounds")

    def tangent_linear(self, M, dM, tlm_map):
        return ZeroAssignment(tlm_map[self.x()])


class MemoryStorage(Storage):
    """A :class:`Storage` which stores the value in memory.

    :arg x: A function defining the forward solution, whose value is saved or
        loaded.
    :arg d: A :class:`dict` in which data is stored with key `key`.
    :arg key: A :class:`str` key for the saved or loaded data.
    :arg save: If `True` then the first forward solve saves the value of `x`.
        If `False` then the first forward solve loads the value of `x`
    """

    def __init__(self, x, d, key, *, save=False):
        super().__init__(x, key, save=save)
        self._d = d

    def is_saved(self):
        return self.key() in self._d

    def load(self, x):
        function_set_values(x, self._d[self.key()])

    def save(self, x):
        self._d[self.key()] = function_get_values(x)


class HDF5Storage(Storage):
    """A :class:`Storage` which stores the value on disk using the h5py
    library.

    :arg x: A function defining the forward solution, whose value is saved or
        loaded.
    :arg h: An h5py :class:`File`.
    :arg key: A :class:`str` key for the saved or loaded data.
    :arg save: If `True` then the first forward solve saves the value of `x`.
        If `False` then the first forward solve loads the value of `x`
    """

    def __init__(self, x, h, key, *, save=False):
        super().__init__(x, key, save=save)
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
