#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .backend import (
    backend_Cofunction, backend_CofunctionSpace, backend_Constant,
    backend_Function, backend_FunctionSpace, backend_ScalarType)
from ..interface import (
    DEFAULT_COMM, SpaceInterface, add_interface, check_space_type,
    comm_dup_cached, new_space_id, new_var_id, register_garbage_cleanup,
    register_finalize_adjoint_derivative_action, register_functional_term_eq,
    register_subtract_adjoint_derivative_action, relative_space_type,
    space_type_warning, subtract_adjoint_derivative_action,
    subtract_adjoint_derivative_action_base, var_caches, var_id, var_is_alias,
    var_is_cached, var_is_static, var_linf_norm, var_name, var_space,
    var_space_type)
from ..interface import VariableInterface as _VariableInterface
from .backend_code_generator_interface import assemble, r0_space

from ..equations import Conversion
from ..manager import manager_disabled
from ..override import override_method, override_property
from ..overloaded_float import SymbolicFloat

from .equations import Assembly
from .functions import (
    Caches, ConstantInterface, ConstantSpaceInterface, ReplacementFunction,
    ReplacementInterface, Zero, define_var_alias)

import mpi4py.MPI as MPI
import numpy as np
import petsc4py.PETSc as PETSc
import pyop2
import ufl

__all__ = \
    [
        "Cofunction",
        "Function",

        "ZeroFunction",

        "ReplacementCofunction",

        "to_firedrake"
    ]


# Aim for compatibility with Firedrake API


@override_method(backend_Constant, "__init__")
def Constant__init__(self, orig, orig_args, value, domain=None, *,
                     name=None, space=None, comm=None,
                     **kwargs):
    orig(self, value, domain=domain, name=name, **kwargs)

    if name is None:
        name = self.name
    if comm is None:
        comm = DEFAULT_COMM

    if space is None:
        if domain is None:
            cell = None
        else:
            cell = domain.ufl_cell()
        if len(self.ufl_shape) == 0:
            element = ufl.classes.FiniteElement("R", cell, 0)
        elif len(self.ufl_shape) == 1:
            element = ufl.classes.VectorElement("R", cell, 0,
                                                dim=self.ufl_shape[0])
        else:
            element = ufl.classes.TensorElement("R", cell, 0,
                                                shape=self.ufl_shape)
        space = ufl.classes.FunctionSpace(domain, element)
        add_interface(space, ConstantSpaceInterface,
                      {"comm": comm_dup_cached(comm), "domain": domain,
                       "dtype": backend_ScalarType, "id": new_space_id()})
    add_interface(self, ConstantInterface,
                  {"id": new_var_id(), "name": lambda x: name,
                   "state": [0], "space": space,
                   "derivative_space": lambda x: r0_space(x),
                   "space_type": "primal", "dtype": self.dat.dtype.type,
                   "static": False, "cache": False})


class FunctionSpaceInterface(SpaceInterface):
    def _comm(self):
        return self._tlm_adjoint__space_interface_attrs["comm"]

    def _dtype(self):
        return backend_ScalarType

    def _id(self):
        return self._tlm_adjoint__space_interface_attrs["id"]

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        space = self._tlm_adjoint__space_interface_attrs["space"]
        if space_type in {"primal", "conjugate"}:
            return Function(space, name=name, space_type=space_type,
                            static=static, cache=cache)
        elif space_type in {"dual", "conjugate_dual"}:
            return Cofunction(space.dual(), name=name, space_type=space_type,
                              static=static, cache=cache)
        else:
            raise ValueError("Invalid space type")


@override_method(backend_FunctionSpace, "__init__")
def FunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    add_interface(self, FunctionSpaceInterface,
                  {"space": self, "comm": comm_dup_cached(self.comm),
                   "id": new_space_id()})


@override_method(backend_CofunctionSpace, "__init__")
def CofunctionSpace__init__(self, orig, orig_args, *args, **kwargs):
    orig_args()
    add_interface(self, FunctionSpaceInterface,
                  {"space": self.dual(), "comm": comm_dup_cached(self.comm),
                   "id": new_space_id()})


class FunctionInterfaceBase(_VariableInterface):
    def _comm(self):
        return self._tlm_adjoint__var_interface_attrs["comm"]

    def _space(self):
        return self.function_space()

    def _space_type(self):
        return self._tlm_adjoint__var_interface_attrs["space_type"]

    def _dtype(self):
        return self.dat.dtype.type

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self.name()

    def _state(self):
        return self._tlm_adjoint__var_interface_attrs["state"][0]

    def _update_state(self):
        self._tlm_adjoint__var_interface_attrs["state"][0] += 1

    def _is_static(self):
        return self._tlm_adjoint__var_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__var_interface_attrs["cache"]

    def _caches(self):
        if "caches" not in self._tlm_adjoint__var_interface_attrs:
            self._tlm_adjoint__var_interface_attrs["caches"] \
                = Caches(self)
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _zero(self):
        with self.dat.vec_wo as x_v:
            x_v.zeroEntries()

    @manager_disabled()
    def _axpy(self, alpha, x, /):
        if isinstance(x, (backend_Cofunction, backend_Function)):
            with self.dat.vec as y_v, x.dat.vec_ro as x_v:
                if y_v.getLocalSize() != x_v.getLocalSize():
                    raise ValueError("Invalid function space")
                y_v.axpy(alpha, x_v)
        else:
            raise TypeError(f"Unexpected type: {type(x)}")

    @manager_disabled()
    def _inner(self, y):
        if isinstance(y, (backend_Cofunction, backend_Function)):
            with self.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise ValueError("Invalid function space")
                inner = x_v.dot(y_v)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")
        return inner

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
        with self.dat.vec_wo as x_v:
            x_v.setArray(values)

    def _is_replacement(self):
        return False

    def _is_scalar(self):
        return False

    def _is_alias(self):
        return "alias" in self._tlm_adjoint__var_interface_attrs


class FunctionInterface(FunctionInterfaceBase):
    def _derivative_space(self):
        return self.function_space()

    @manager_disabled()
    def _assign(self, y):
        if isinstance(y, SymbolicFloat):
            y = y.value
        if isinstance(y, (backend_Cofunction, backend_Function)):
            with self.dat.vec as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise ValueError("Invalid function space")
                y_v.copy(result=x_v)
        elif isinstance(y, (int, np.integer,
                            float, np.floating,
                            complex, np.complexfloating)):
            if len(self.ufl_shape) == 0:
                self.assign(backend_Constant(y))
            else:
                y_arr = np.full(self.ufl_shape, y, dtype=self.dat.dtype.type)
                self.assign(backend_Constant(y_arr))
        elif isinstance(y, backend_Constant):
            self.assign(y)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = ReplacementFunction(self)
        return self._tlm_adjoint__replacement


class Function(backend_Function):
    """Extends the backend `Function` class.

    :arg space_type: The space type for the :class:`Function`. `'primal'` or
        `'conjugate'`.
    :arg static: Defines whether the :class:`Function` is static, meaning that
        it is stored by reference in checkpointing/replay, and an associated
        tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`Function` may be
        cached. Default `static`.

    Remaining arguments are passed to the backend `Function` constructor.
    """

    def __init__(self, *args, space_type="primal", static=False, cache=None,
                 **kwargs):
        if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")
        if space_type not in {"primal", "conjugate"}:
            space_type_warning("Unexpected space type")
        if cache is None:
            cache = static

        super().__init__(*args, **kwargs)
        self._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        self._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        self._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)


class ZeroFunction(Function, Zero):
    """A :class:`Function` which is flagged as having a value of zero.

    Arguments are passed to the :class:`Function` constructor, together with
    `static=True` and `cache=True`.
    """

    def __init__(self, *args, **kwargs):
        Function.__init__(
            self, *args, **kwargs,
            static=True, cache=True)
        if var_linf_norm(self) != 0.0:
            raise RuntimeError("ZeroFunction is not zero-valued")

    def assign(self, *args, **kwargs):
        raise RuntimeError("Cannot call assign method of ZeroFunction")

    def interpolate(self, *args, **kwargs):
        raise RuntimeError("Cannot call interpolate method of ZeroFunction")

    def project(self, *args, **kwargs):
        raise RuntimeError("Cannot call project method of ZeroFunction")


@override_method(backend_Function, "__init__")
def Function__init__(self, orig, orig_args, function_space, val=None,
                     *args, **kwargs):
    orig_args()
    comm = self.comm
    if pyop2.mpi.is_pyop2_comm(comm):
        # Work around Firedrake issue #3043
        comm = self.function_space().comm
    add_interface(self, FunctionInterface,
                  {"comm": comm_dup_cached(comm), "id": new_var_id(),
                   "state": [0], "space_type": "primal", "static": False,
                   "cache": False})
    if isinstance(val, backend_Function):
        define_var_alias(self, val, key=("Function__init__",))


@override_method(backend_Function, "__getattr__")
def Function__getattr__(self, orig, orig_args, key):
    if "_data" not in self.__dict__:
        raise AttributeError(f"No attribute '{key:s}'")
    return orig_args()


@override_method(backend_Function, "riesz_representation")
def Function_riesz_representation(self, orig, orig_args,
                                  riesz_map="L2", *args, **kwargs):
    if riesz_map != "l2":
        check_space_type(self, "primal")
    return_value = orig_args()
    if riesz_map == "l2":
        define_var_alias(return_value, self,
                         key=("riesz_representation", "l2"))
    # define_var_alias sets the space_type, so this has to appear after
    return_value._tlm_adjoint__var_interface_attrs.d_setitem(
        "space_type",
        relative_space_type(self._tlm_adjoint__var_interface_attrs["space_type"], "conjugate_dual"))  # noqa: E501
    return return_value


@override_property(backend_Function, "subfunctions", cached=True)
def Function_subfunctions(self, orig):
    Y = orig()
    for i, y in enumerate(Y):
        define_var_alias(y, self, key=("subfunctions", i))
    return Y


@override_method(backend_Function, "sub")
def Function_sub(self, orig, orig_args, i):
    self.subfunctions
    y = orig_args()
    if not var_is_alias(y):
        define_var_alias(y, self, key=("sub", i))
    return y


class CofunctionInterface(FunctionInterfaceBase):
    @manager_disabled()
    def _assign(self, y):
        if isinstance(y, (backend_Cofunction, backend_Function)):
            with self.dat.vec as x_v, y.dat.vec_ro as y_v:
                if x_v.getLocalSize() != y_v.getLocalSize():
                    raise ValueError("Invalid function space")
                y_v.copy(result=x_v)
        else:
            raise TypeError(f"Unexpected type: {type(y)}")

    def _replacement(self):
        if not hasattr(self, "_tlm_adjoint__replacement"):
            self._tlm_adjoint__replacement = ReplacementCofunction(self)
        return self._tlm_adjoint__replacement


class Cofunction(backend_Cofunction):
    """Extends the backend `Cofunction` class.

    :arg space_type: The space type for the :class:`Cofunction`. `'conjugate'`
        or `'conjugate_dual'`.
    :arg static: Defines whether the :class:`Cofunction` is static, meaning
        that it is stored by reference in checkpointing/replay, and an
        associated tangent-linear variable is zero.
    :arg cache: Defines whether results involving the :class:`Cofunction` may
        be cached. Default `static`.

    Remaining arguments are passed to the backend `Cofunction` constructor.
    """

    def __init__(self, *args, space_type="conjugate_dual", static=False,
                 cache=None, **kwargs):
        if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
            raise ValueError("Invalid space type")
        if space_type not in {"dual", "conjugate_dual"}:
            space_type_warning("Unexpected space type")
        if cache is None:
            cache = static

        super().__init__(*args, **kwargs)
        self._tlm_adjoint__var_interface_attrs.d_setitem("space_type", space_type)  # noqa: E501
        self._tlm_adjoint__var_interface_attrs.d_setitem("static", static)
        self._tlm_adjoint__var_interface_attrs.d_setitem("cache", cache)


@override_method(backend_Cofunction, "__init__")
def Cofunction__init__(self, orig, orig_args, function_space, val=None,
                       *args, **kwargs):
    orig_args()
    add_interface(self, CofunctionInterface,
                  {"comm": comm_dup_cached(self.comm), "id": new_var_id(),
                   "state": [0], "space_type": "conjugate_dual",
                   "static": False, "cache": False})
    if isinstance(val, backend_Cofunction):
        define_var_alias(self, val, key=("Cofunction__init__",))


@override_method(backend_Cofunction, "riesz_representation")
def Cofunction_riesz_representation(self, orig, orig_args,
                                    riesz_map="L2", *args, **kwargs):
    if riesz_map != "l2":
        check_space_type(self, "conjugate_dual")
    return_value = orig_args()
    if riesz_map == "l2":
        define_var_alias(return_value, self,
                         key=("riesz_representation", "l2"))
    # define_var_alias sets the space_type, so this has to appear after
    return_value._tlm_adjoint__var_interface_attrs.d_setitem(
        "space_type",
        relative_space_type(self._tlm_adjoint__var_interface_attrs["space_type"], "conjugate_dual"))  # noqa: E501
    return return_value


class ReplacementCofunction(ufl.classes.Cofunction):
    """Represents a symbolic Firedrake `Cofunction`, but has no value.
    """

    def __init__(self, x):
        space = var_space(x)

        super().__init__(space, count=x.count())
        add_interface(self, ReplacementInterface,
                      {"id": var_id(x), "name": var_name(x),
                       "space": space,
                       "space_type": var_space_type(x),
                       "static": var_is_static(x),
                       "cache": var_is_cached(x),
                       "caches": var_caches(x)})

    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, var_space(x), *args, **kwargs)

    def function_space(self):
        return var_space(self)


def to_firedrake(y, space, *, name=None):
    """Convert a variable to a Firedrake `Function`.

    :arg y: A variable.
    :arg space: The space for the return value.
    :arg name: A :class:`str` name.
    :returns: The Firedrake `Function`.
    """

    space_type = var_space_type(y)
    if space_type in {"primal", "conjugate"}:
        if isinstance(space, backend_FunctionSpace):
            x = Function(space, space_type=space_type, name=name)
        else:
            x = Function(space.dual(), space_type=space_type, name=name)
    elif space_type in {"dual", "conjugate_dual"}:
        if isinstance(space, backend_FunctionSpace):
            x = Cofunction(space.dual(), space_type=space_type, name=name)
        else:
            x = Cofunction(space, space_type=space_type, name=name)
    else:
        raise ValueError("Invalid space type")
    Conversion(x, y).solve()
    return x


def garbage_cleanup_internal_comm(comm):
    if not MPI.Is_finalized() and not PETSc.Sys.isFinalized() \
            and not pyop2.mpi.PYOP2_FINALIZED \
            and comm.py2f() != MPI.COMM_NULL.py2f():
        if pyop2.mpi.is_pyop2_comm(comm):
            raise RuntimeError("Should not call garbage_cleanup directly on a "
                               "PyOP2 communicator")
        internal_comm = comm.Get_attr(pyop2.mpi.innercomm_keyval)
        if internal_comm is not None and internal_comm.py2f() != MPI.COMM_NULL.py2f():  # noqa: E501
            PETSc.garbage_cleanup(internal_comm)


register_garbage_cleanup(garbage_cleanup_internal_comm)


def subtract_adjoint_derivative_action_cofunction_form(x, alpha, y):
    check_space_type(x, "conjugate_dual")
    if alpha != 1.0:
        y = backend_Constant(alpha) * y
    if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
        x._tlm_adjoint__firedrake_adj_b = x._tlm_adjoint__firedrake_adj_b - y
    else:
        x._tlm_adjoint__firedrake_adj_b = -y


register_subtract_adjoint_derivative_action(
    (backend_Constant, backend_Cofunction, backend_Function), object,
    subtract_adjoint_derivative_action_base,
    replace=True)
register_subtract_adjoint_derivative_action(
    (backend_Constant, backend_Cofunction), ufl.classes.Form,
    subtract_adjoint_derivative_action_cofunction_form)


def finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__firedrake_adj_b"):
        y = assemble(x._tlm_adjoint__firedrake_adj_b)
        subtract_adjoint_derivative_action(x, (-1.0, y))
        delattr(x, "_tlm_adjoint__firedrake_adj_b")


register_finalize_adjoint_derivative_action(finalize_adjoint_derivative_action)


def functional_term_eq_form(x, term):
    if len(term.arguments()) > 0:
        raise ValueError("Invalid number of arguments")
    return Assembly(x, term)


register_functional_term_eq(
    ufl.classes.Form,
    functional_term_eq_form)
