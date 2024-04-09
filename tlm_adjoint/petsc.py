from .interface import (
    var_dtype, var_get_values, var_global_size, var_local_size, var_set_values)

import numpy as np
try:
    import petsc4py.PETSc as PETSc
except ImportError:
    PETSc = None
import weakref

__all__ = \
    [
    ]


class PETScOptions:
    def __init__(self, options_prefix):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        self._options_prefix = options_prefix
        self._options = PETSc.Options()
        self._keys = {}

        def finalize_callback(options_prefix, options, keys):
            for key in keys:
                key = f"{options_prefix:s}{key:s}"
                if key in options:
                    del options[key]

        finalize = weakref.finalize(
            self, finalize_callback,
            self._options_prefix, self._options, self._keys)
        finalize.atexit = False

    @property
    def options_prefix(self):
        return self._options_prefix

    def __getitem__(self, key):
        if key not in self._keys:
            raise KeyError(key)
        return self._options[f"{self.options_prefix:s}{key:s}"]

    def __setitem__(self, key, value):
        self._keys[key] = None
        self._options[f"{self.options_prefix:s}{key:s}"] = value

    def __delitem__(self, key):
        del self._keys[key]
        del self._options[f"{self.options_prefix:s}{key:s}"]

    def clear(self):
        for key in tuple(self._keys):
            del self[key]


class PETScVecInterface:
    def __init__(self, X, *, dtype=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        if dtype is None:
            dtype = PETSc.ScalarType
        dtype = np.dtype(dtype).type

        indices = []
        n = 0
        N = 0
        for x in X:
            indices.append((n, n + var_local_size(x)))
            n += var_local_size(x)
            N += var_global_size(x)

        self._dtype = dtype
        self._indices = tuple(indices)
        self._n = n
        self._N = N

    @property
    def dtype(self):
        return self._dtype

    @property
    def indices(self):
        return self._indices

    @property
    def n(self):
        return self._n

    @property
    def N(self):
        return self._N

    def from_petsc(self, y, X):
        y_a = y.getArray(True)

        if y_a.shape != (self.n,):
            raise ValueError("Invalid shape")
        if len(X) != len(self.indices):
            raise ValueError("Invalid length")
        for (i0, i1), x in zip(self.indices, X):
            if not np.can_cast(y_a.dtype, var_dtype(x)):
                raise ValueError("Invalid dtype")
            if var_local_size(x) != i1 - i0:
                raise ValueError("Invalid length")

        for (i0, i1), x in zip(self.indices, X):
            var_set_values(x, y_a[i0:i1])

    def to_petsc(self, x, Y):
        if len(Y) != len(self.indices):
            raise ValueError("Invalid length")
        for (i0, i1), y in zip(self.indices, Y):
            if not np.can_cast(var_dtype(y), self.dtype):
                raise ValueError("Invalid dtype")
            if var_local_size(y) != i1 - i0:
                raise ValueError("Invalid length")

        x_a = np.zeros(self.n, dtype=self.dtype)
        for (i0, i1), y in zip(self.indices, Y):
            x_a[i0:i1] = var_get_values(y)
        x.setArray(x_a)
