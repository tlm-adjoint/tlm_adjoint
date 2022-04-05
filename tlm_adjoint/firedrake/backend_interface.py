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

from .backend import FunctionSpace, UnitIntervalMesh, backend, \
    backend_Constant, backend_Function, backend_FunctionSpace, \
    backend_ScalarType, info
from ..functional import Functional as _Functional
from ..hessian import GeneralGaussNewton as _GaussNewton
from ..hessian_optimization import CachedGaussNewton as _CachedGaussNewton
from ..interface import InterfaceException, SpaceInterface, \
    add_finalize_adjoint_derivative_action, add_functional_term_eq, \
    add_interface, add_subtract_adjoint_derivative_action, \
    add_time_system_eq, check_space_type, function_comm, function_dtype, \
    function_is_scalar, function_scalar_value, function_space, \
    new_function_id, new_space_id, space_id, space_new, \
    subtract_adjoint_derivative_action
from ..interface import FunctionInterface as _FunctionInterface
from .backend_code_generator_interface import assemble, is_valid_r0_space

from .caches import form_neg
from .equations import AssembleSolver, EquationSolver
from .functions import Caches, Constant, ConstantInterface, \
    ConstantSpaceInterface, Function, ReplacementFunction, Zero, \
    define_function_alias

import mpi4py.MPI as MPI
import numpy as np
import petsc4py.PETSc as PETSc
from pyadjoint.block_variable import BlockVariable
import ufl
import warnings

__all__ = \
    [
        "CachedGaussNewton",
        "Functional",
        "GaussNewton",
        "new_scalar_function",

        "RealFunctionSpace",
        "default_comm",
        "function_space_id",
        "function_space_new",
        "info",
        "warning"
    ]


def _BlockVariable__init__(self, output):
    # Prevent a circular reference. See Firedrake issue #1617.
    BlockVariable._tlm_adjoint__orig___init__(self, None)


assert not hasattr(BlockVariable, "_tlm_adjoint__orig___init__")
BlockVariable._tlm_adjoint__orig___init__ = BlockVariable.__init__
BlockVariable.__init__ = _BlockVariable__init__


# Aim for compatibility with Firedrake API, git master revision
# efb48f4f178ae4989c146640025641cf0cc00a0e, Apr 19 2021
def _Constant__init__(self, value, domain=None, *,
                      name=None, space=None, comm=MPI.COMM_WORLD,
                      **kwargs):
    backend_Constant._tlm_adjoint__orig___init__(self, value, domain=domain,
                                                 **kwargs)

    if name is None:
        # Following FEniCS 2019.1.0 behaviour
        name = f"f_{self.count():d}"

    if space is None:
        space = self.ufl_function_space()
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm, "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_function_id(), "name": name, "state": 0,
                   "space": space, "space_type": "primal",
                   "dtype": self.dat.dtype.type, "static": False,
                   "cache": False, "checkpoint": True})


assert not hasattr(backend_Constant, "_tlm_adjoint__orig___init__")
backend_Constant._tlm_adjoint__orig___init__ = backend_Constant.__init__
backend_Constant.__init__ = _Constant__init__


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self.comm

    def _dtype(self):
        return backend_ScalarType

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        return Function(self, name=name, space_type=space_type, static=static,
                        cache=cache, checkpoint=checkpoint)


def _FunctionSpace__init__(self, *args, **kwargs):
    backend_FunctionSpace._tlm_adjoint__orig___init__(self, *args, **kwargs)
    add_interface(self, FunctionSpaceInterface,
                  {"id": new_space_id()})


assert not hasattr(backend_FunctionSpace, "_tlm_adjoint__orig___init__")
backend_FunctionSpace._tlm_adjoint__orig___init__ = backend_FunctionSpace.__init__  # noqa: E501
backend_FunctionSpace.__init__ = _FunctionSpace__init__


class FunctionInterface(_FunctionInterface):
    def _comm(self):
        return self.comm

    def _space(self):
        return self.function_space()

    def _space_type(self):
        return self._tlm_adjoint__function_interface_attrs["space_type"]

    def _dtype(self):
        return self.dat.dtype.type

    def _id(self):
        return self._tlm_adjoint__function_interface_attrs["id"]

    def _name(self):
        return self.name()

    def _state(self):
        return self._tlm_adjoint__function_interface_attrs["state"]

    def _update_state(self):
        state = self._tlm_adjoint__function_interface_attrs["state"]
        self._tlm_adjoint__function_interface_attrs.d_setitem("state", state + 1)  # noqa: E501

    def _is_static(self):
        return self._tlm_adjoint__function_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__function_interface_attrs["cache"]

    def _is_checkpointed(self):
        return self._tlm_adjoint__function_interface_attrs["checkpoint"]

    def _caches(self):
        if "caches" not in self._tlm_adjoint__function_interface_attrs:
            self._tlm_adjoint__function_interface_attrs["caches"] \
                = Caches(self)
        return self._tlm_adjoint__function_interface_attrs["caches"]

    def _zero(self):
        with self.dat.vec_wo as x_v:
            x_v.zeroEntries()

    def _assign(self, y):
        if isinstance(y, backend_Function):
            with self.dat.vec as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise InterfaceException("Invalid function space")
                y_v.copy(result=x_v)
        elif isinstance(y, (int, np.integer,
                            float, np.floating,
                            complex, np.complexfloating)):
            dtype = function_dtype(self)
            if len(self.ufl_shape) == 0:
                self.assign(backend_Constant(dtype(y)),
                            annotate=False, tlm=False)
            else:
                y_arr = np.full(self.ufl_shape, dtype(y), dtype=dtype)
                self.assign(backend_Constant(y_arr),
                            annotate=False, tlm=False)
        elif isinstance(y, Zero):
            with self.dat.vec_wo as x_v:
                x_v.zeroEntries()
        else:
            assert isinstance(y, backend_Constant)
            self.assign(y, annotate=False, tlm=False)

        e = self.ufl_element()
        if e.family() == "Real" and e.degree() == 0:
            # Work around Firedrake issue #1459
            values = self.dat.data_ro.copy()
            values = function_comm(self).bcast(values, root=0)
            self.dat.data[:] = values

    def _axpy(self, *args):  # self, alpha, x
        alpha, x = args
        dtype = function_dtype(self)
        alpha = dtype(alpha)
        if isinstance(x, backend_Function):
            with self.dat.vec as y_v, x.dat.vec_ro as x_v:
                if y_v.getLocalSize() != x_v.getLocalSize():
                    raise InterfaceException("Invalid function space")
                y_v.axpy(alpha, x_v)
        elif isinstance(x, (int, np.integer,
                            float, np.floating,
                            complex, np.complexfloating)):
            self.assign(self + alpha * dtype(x),
                        annotate=False, tlm=False)
        elif isinstance(x, Zero):
            pass
        else:
            assert isinstance(x, backend_Constant)
            self.assign(self + alpha * x, annotate=False, tlm=False)

        e = self.ufl_element()
        if e.family() == "Real" and e.degree() == 0:
            # Work around Firedrake issue #1459
            values = self.dat.data_ro.copy()
            values = function_comm(self).bcast(values, root=0)
            self.dat.data[:] = values

    def _inner(self, y):
        if isinstance(y, backend_Function):
            with self.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise InterfaceException("Invalid function space")
                inner = x_v.dot(y_v)
        elif isinstance(y, Zero):
            inner = 0.0
        else:
            assert isinstance(y, backend_Constant)
            y_ = backend_Function(self.function_space())
            y_.assign(y, annotate=False, tlm=False)
            with self.dat.vec_ro as x_v, y_.dat.vec_ro as y_v:
                inner = x_v.dot(y_v)
        return inner

    def _max_value(self):
        with self.dat.vec_ro as x_v:
            max = x_v.max()[1]
        return max

    def _sum(self):
        with self.dat.vec_ro as x_v:
            sum = x_v.sum()
        return sum

    def _linf_norm(self):
        with self.dat.vec_ro as x_v:
            linf_norm = x_v.norm(norm_type=PETSc.NormType.NORM_INFINITY)
        return linf_norm

    def _local_size(self):
        with self.dat.vec_ro as x_v:
            local_size = x_v.getLocalSize()
        return local_size

    def _global_size(self):
        with self.dat.vec_ro as x_v:
            size = x_v.getSize()
        return size

    def _local_indices(self):
        with self.dat.vec_ro as x_v:
            local_range = x_v.getOwnershipRange()
        return slice(*local_range)

    def _get_values(self):
        with self.dat.vec_ro as x_v:
            with x_v as x_v_a:
                values = x_v_a.copy()
        return values

    def _set_values(self, values):
        if not np.can_cast(values, function_dtype(self)):
            raise InterfaceException("Invalid dtype")
        with self.dat.vec as x_v:
            if values.shape != (x_v.getLocalSize(),):
                raise InterfaceException("Invalid shape")
            x_v.setArray(values)

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = ReplacementFunction(self)
        return self._tlm_adjoint__replacement

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return (is_valid_r0_space(self.function_space())
                and len(self.ufl_shape) == 0)

    def _scalar_value(self):
        # assert function_is_scalar(self)
        with self.dat.vec_ro as x_v:
            value = x_v.sum() / x_v.getSize()
        return value

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__function_interface_attrs


def _Function__init__(self, *args, **kwargs):
    backend_Function._tlm_adjoint__orig___init__(self, *args, **kwargs)
    add_interface(self, FunctionInterface,
                  {"id": new_function_id(), "state": 0, "space_type": "primal",
                   "static": False, "cache": False, "checkpoint": True})


assert not hasattr(backend_Function, "_tlm_adjoint__orig___init__")
backend_Function._tlm_adjoint__orig___init__ = backend_Function.__init__
backend_Function.__init__ = _Function__init__


def _Function__getattr__(self, key):
    if "_data" not in self.__dict__:
        raise AttributeError(f"No attribute '{key:s}'")
    return backend_Function._tlm_adjoint__orig__getattr__(self, key)


assert not hasattr(backend_Function, "_tlm_adjoint__orig__getattr__")
backend_Function._tlm_adjoint__orig__getattr__ = backend_Function.__getattr__
backend_Function.__getattr__ = _Function__getattr__


# Aim for compatibility with Firedrake API, git master revision
# ac22e4c55d6fad32ddc9e936cd3674fb8a75f1da, Mar 16 2022
def _Function_split(self):
    Y = backend_Function._tlm_adjoint__orig_split(self)
    for i, y in enumerate(Y):
        define_function_alias(y, self, key=("split", i))
    return Y


assert not hasattr(backend_Function, "_tlm_adjoint__orig_split")
backend_Function._tlm_adjoint__orig_split = backend_Function.split
backend_Function.split = _Function_split


# Aim for compatibility with Firedrake API, git master revision
# ac22e4c55d6fad32ddc9e936cd3674fb8a75f1da, Mar 16 2022
def _Function_sub(self, i):
    y = backend_Function._tlm_adjoint__orig_sub(self, i)
    define_function_alias(y, self, key=("sub", i))
    return y


assert not hasattr(backend_Function, "_tlm_adjoint__orig_sub")
backend_Function._tlm_adjoint__orig_sub = backend_Function.sub
backend_Function.sub = _Function_sub


def new_scalar_function(*, name=None, comm=None, static=False, cache=None,
                        checkpoint=None):
    return Constant(0.0, name=name, comm=comm, static=static, cache=cache,
                    checkpoint=checkpoint)


class Functional(_Functional):
    def __init__(self, *, space=None, name=None, _fn=None):
        if space is None and _fn is None:
            space = function_space(new_scalar_function())

        super().__init__(space=space, name=name, _fn=_fn)


class GaussNewton(_GaussNewton):
    def __init__(self, forward, R_inv_action, B_inv_action=None,
                 *, J_space=None, manager=None):
        if J_space is None:
            J_space = function_space(new_scalar_function())

        super().__init__(
            forward, J_space, R_inv_action, B_inv_action=B_inv_action,
            manager=manager)


class CachedGaussNewton(_CachedGaussNewton):
    def __init__(self, X, R_inv_action, B_inv_action=None,
                 *, J_space=None, manager=None):
        if J_space is None:
            J_space = function_space(new_scalar_function())

        super().__init__(
            X, J_space, R_inv_action, B_inv_action=B_inv_action,
            manager=manager)


def _subtract_adjoint_derivative_action(x, y):
    if isinstance(y, ufl.classes.Form) \
            and isinstance(x, (backend_Constant, backend_Function)):
        if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
            x._tlm_adjoint__firedrake_adj_b += form_neg(y)
        else:
            x._tlm_adjoint__firedrake_adj_b = form_neg(y)
    elif isinstance(x, backend_Constant):
        dtype = function_dtype(x)
        if isinstance(y, backend_Function) and function_is_scalar(y):
            alpha = 1.0
        elif isinstance(y, tuple) \
                and len(y) == 2 \
                and isinstance(y[0], (int, np.integer,
                                      float, np.floating,
                                      complex, np.complexfloating)) \
                and isinstance(y[1], backend_Function) \
                and function_is_scalar(y[1]):
            alpha, y = y
            alpha = dtype(alpha)
        else:
            return NotImplemented
        check_space_type(y, "conjugate_dual")
        y_value = function_scalar_value(y)
        x.assign(dtype(x) - alpha * y_value, annotate=False, tlm=False)
    else:
        return NotImplemented


add_subtract_adjoint_derivative_action(backend,
                                       _subtract_adjoint_derivative_action)


def _finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
        y = assemble(x._tlm_adjoint__firedrake_adj_b)
        subtract_adjoint_derivative_action(x, (-1.0, y))
        delattr(x, "_tlm_adjoint__firedrake_adj_b")


add_finalize_adjoint_derivative_action(backend,
                                       _finalize_adjoint_derivative_action)


def _functional_term_eq(term, x):
    if isinstance(term, ufl.classes.Form) \
            and len(term.arguments()) == 0 \
            and isinstance(x, (backend_Constant, backend_Function)):
        return AssembleSolver(term, x)
    else:
        return NotImplemented


add_functional_term_eq(backend, _functional_term_eq)


def _time_system_eq(*args, **kwargs):
    if len(args) >= 1:
        eq = args[0]
    elif "eq" in kwargs:
        eq = kwargs["eq"]
    else:
        return NotImplemented

    if len(args) >= 2:
        x = args[1]
    elif "x" in kwargs:
        x = kwargs["x"]
    else:
        return NotImplemented

    if isinstance(eq, ufl.classes.Equation) \
            and isinstance(x, backend_Function):
        return EquationSolver(*args, **kwargs)
    else:
        return NotImplemented


add_time_system_eq(backend, _time_system_eq)


def default_comm():
    warnings.warn("default_comm is deprecated -- "
                  "use mpi4py.MPI.COMM_WORLD instead",
                  DeprecationWarning, stacklevel=2)
    return MPI.COMM_WORLD


def RealFunctionSpace(comm=None):
    warnings.warn("RealFunctionSpace is deprecated -- "
                  "use new_scalar_function instead",
                  DeprecationWarning, stacklevel=2)
    if comm is None:
        comm = MPI.COMM_WORLD
    return FunctionSpace(UnitIntervalMesh(comm.size, comm=comm), "R", 0)


def function_space_id(*args, **kwargs):
    warnings.warn("function_space_id is deprecated -- use space_id instead",
                  DeprecationWarning, stacklevel=2)
    return space_id(*args, **kwargs)


def function_space_new(*args, **kwargs):
    warnings.warn("function_space_new is deprecated -- use space_new instead",
                  DeprecationWarning, stacklevel=2)
    return space_new(*args, **kwargs)


# def info(message):


def warning(message):
    warnings.warn("warning is deprecated -- use logging.warning instead",
                  DeprecationWarning, stacklevel=2)
    warnings.warn(message, RuntimeWarning)
