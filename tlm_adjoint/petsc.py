from .interface import (
    space_comm, space_global_size, space_local_size, var_from_petsc,
    var_to_petsc)

from contextlib import contextmanager
try:
    import mpi4py.MPI as MPI
except ModuleNotFoundError:
    MPI = None
import numpy as np
try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
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

    def update(self, other):
        for key, value in other.items():
            self[key] = value


@contextmanager
def petsc_option_setdefault(key, value):
    if PETSc is None:
        raise RuntimeError("PETSc not available")

    options = PETSc.Options()
    set_option = key not in options
    if set_option:
        options[key] = value
    try:
        yield
    finally:
        if set_option:
            del options[key]


class PETScVecInterface:
    def __init__(self, spaces, *, dtype=None, comm=None):
        if PETSc is None:
            raise RuntimeError("PETSc not available")

        if comm is None:
            comm = space_comm(spaces[0])
        if dtype is None:
            dtype = PETSc.ScalarType
        dtype = np.dtype(dtype).type

        n = sum(map(space_local_size, spaces))
        N = sum(map(space_global_size, spaces))

        isets = []
        self._i0 = i0 = comm.scan(n, op=MPI.SUM) - n
        for space in spaces:
            i1 = i0 + space_local_size(space)
            iset = PETSc.IS().createStride(
                size=i1 - i0, first=i0, step=1, comm=comm)
            isets.append(iset)
            i0 = i1

        self._comm = comm
        self._dtype = dtype
        self._isets = tuple(isets)
        self._n = n
        self._N = N

        attach_destroy_finalizer(self, *self._isets)

    @property
    def comm(self):
        return self._comm

    @property
    def dtype(self):
        return self._dtype

    @property
    def local_size(self):
        return self._n

    @property
    def global_size(self):
        return self._N

    def from_petsc(self, y, X):
        if len(X) != len(self._isets):
            raise ValueError("Invalid length")
        if y.getLocalSize() != self.local_size:
            raise ValueError("Invalid size")
        if y.getSize() != self.global_size:
            raise ValueError("Invalid size")
        if y.getOwnershipRange()[0] != self._i0:
            raise ValueError("Invalid decomposition")
        for x, iset in zip(X, self._isets):
            y_sub = y.getSubVector(iset)
            var_from_petsc(x, y_sub)
            y.restoreSubVector(iset, y_sub)

    def to_petsc(self, x, Y):
        if len(Y) != len(self._isets):
            raise ValueError("Invalid length")
        if x.getLocalSize() != self.local_size:
            raise ValueError("Invalid size")
        if x.getSize() != self.global_size:
            raise ValueError("Invalid size")
        if x.getOwnershipRange()[0] != self._i0:
            raise ValueError("Invalid decomposition")
        for y, iset in zip(Y, self._isets):
            x_sub = x.getSubVector(iset)
            var_to_petsc(y, x_sub)
            x.restoreSubVector(iset, x_sub)

    def _new_petsc(self):
        vec = PETSc.Vec().create(comm=self.comm)
        vec.setSizes((self.local_size, self.global_size))
        vec.setUp()
        return vec

    def new_vec(self):
        return PETScVec(self)


class PETScVec:
    def __init__(self, vec_interface):
        self._vec_interface = vec_interface
        self._vec = vec_interface._new_petsc()

        attach_destroy_finalizer(self, self._vec)

    @property
    def vec(self):
        return self._vec

    def to_petsc(self, Y):
        self._vec_interface.to_petsc(self.vec, Y)

    def from_petsc(self, X):
        self._vec_interface.from_petsc(self.vec, X)


def attach_destroy_finalizer(obj, *args):
    def finalize_callback(*args):
        for arg in args:
            if arg is not None:
                arg.destroy()

    finalize = weakref.finalize(obj, finalize_callback,
                                *args)
    finalize.atexit = False
