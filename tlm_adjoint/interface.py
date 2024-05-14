r"""This module defines an interface for interaction with backend data types.
This is implemented via runtime binding of mixins. The
:class:`.VariableInterface` adds methods to 'variables' which can be used to
interact with backend variables. The :class:`.SpaceInterface` adds methods to
'spaces' which define the vector spaces in which those 'variables' are defined.

The extra methods are accessed using the callables defined in this module
(which also handle some extra details, e.g. related to cache invalidation and
space type checking). Typically these are prefixed with `space_` for spaces and
`var_` for variables.

The interface distinguishes between original backend 'variables', which both
define symbolic variables and store values, and replacement 'variables', which
define the same variables but which need not store values.

Variables have an associated 'space type', which indicates e.g. if the variable
is 'primal', meaning a member on an originating vector space, or 'conjugate
dual', meaning a member of the corresponding antidual space of antilinear
functionals from the originating vector space. Variables can also be 'dual',
meaning a member of the dual space of linear functionals, or 'conjugate',
meaning a member of a space defined by a conjugate operator from the primal
space. This conjugate operator is defined by complex conjugation of the vector
of degrees of freedom, and could e.g. correspond to complex conjugation of a
finite element discretized function.

The space type associated with a variable is defined relative to an originating
vector space (e.g. a finite element discrete function space). A 'relative space
type' is defined relative to one of the 'primal', 'conjugate', 'dual', or
'conjugate dual' spaces. For example the primal space associated with the dual
space is the dual space, and the dual space associated with the dual space is
the primal space.

This module defines a default communicator `DEFAULT_COMM`.
"""

from .manager import manager_disabled

from collections.abc import MutableMapping, Sequence
from collections import deque
import contextlib
import copy
import functools
import itertools
import logging
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None
import numbers
import numpy as np
try:
    from operator import call
except ImportError:
    # For Python < 3.11, following Python 3.11 API
    def call(obj, /, *args, **kwargs):
        return obj(*args, **kwargs)
try:
    import petsc4py.PETSc as PETSc
except ImportError:
    PETSc = None
try:
    import pyop2
except ImportError:
    pyop2 = None
import sys
import warnings
import weakref

__all__ = \
    [
        "DEFAULT_COMM",
        "comm_dup_cached",
        "comm_parent",
        "garbage_cleanup",

        "add_interface",

        "SpaceInterface",
        "is_space",
        "space_comm",
        "space_default_space_type",
        "space_dtype",
        "space_eq",
        "space_global_size",
        "space_id",
        "space_local_indices",
        "space_local_size",
        "space_new",

        "SpaceTypeError",
        "check_space_type",
        "check_space_types",
        "check_space_types_conjugate",
        "check_space_types_conjugate_dual",
        "check_space_types_dual",
        "conjugate_dual_space_type",
        "conjugate_space_type",
        "dual_space_type",
        "no_space_type_checking",
        "paused_space_type_checking",
        "relative_space_type",

        "VariableInterface",
        "is_var",
        "var_assign",
        "var_axpy",
        "var_caches",
        "var_comm",
        "var_copy",
        "var_dtype",
        "var_get_values",
        "var_global_size",
        "var_id",
        "var_inner",
        "var_is_cached",
        "var_is_replacement",
        "var_is_static",
        "var_linf_norm",
        "var_local_indices",
        "var_local_size",
        "var_name",
        "var_new",
        "var_new_conjugate",
        "var_new_conjugate_dual",
        "var_new_dual",
        "var_replacement",
        "var_set_values",
        "var_space",
        "var_space_type",
        "var_state",
        "var_update_caches",
        "var_update_state",
        "var_zero",

        "var_is_scalar",
        "var_scalar_value",

        "subtract_adjoint_derivative_action",

        "var_is_alias",

        "VariableStateLockDictionary",
        "var_lock_state",
        "var_locked",

        "is_function",
        "function_assign",
        "function_axpy",
        "function_caches",
        "function_comm",
        "function_copy",
        "function_dtype",
        "function_get_values",
        "function_global_size",
        "function_id",
        "function_inner",
        "function_is_cached",
        "function_is_checkpointed",
        "function_is_replacement",
        "function_is_static",
        "function_linf_norm",
        "function_local_indices",
        "function_local_size",
        "function_name",
        "function_new",
        "function_new_conjugate",
        "function_new_conjugate_dual",
        "function_new_dual",
        "function_replacement",
        "function_set_values",
        "function_space",
        "function_space_type",
        "function_state",
        "function_update_caches",
        "function_update_state",
        "function_zero",
        "function_is_scalar",
        "function_scalar_value",
        "function_is_alias",
        "var_is_checkpointed"
    ]


if pyop2 is not None:
    DEFAULT_COMM = pyop2.mpi.COMM_WORLD
elif MPI is not None:
    DEFAULT_COMM = MPI.COMM_WORLD
else:
    # As for mpi4py 3.1.4 API
    class SerialComm:
        _id_counter = itertools.count(start=-1, step=-1)

        def __init__(self, *, _id=None):
            self._id = _id
            if self._id is None:
                self._id = next(self._id_counter)

        @property
        def rank(self):
            return 0

        @property
        def size(self):
            return 1

        def Dup(self, info=None):
            return SerialComm()

        def Free(self):
            pass

        def allgather(self, sendobj):
            return [copy.deepcopy(sendobj)]

        def barrier(self):
            pass

        def bcast(self, obj, root=0):
            return copy.deepcopy(obj)

        def gather(self, sendobj, root=0):
            assert root == 0
            return [copy.deepcopy(sendobj)]

        def py2f(self):
            return self._id

        def f2py(self, arg):
            return SerialComm(_id=arg)

        def scatter(self, sendobj, root=0):
            assert root == 0
            sendobj, = sendobj
            return copy.deepcopy(sendobj)

    DEFAULT_COMM = SerialComm()

if MPI is None:
    def comm_finalize(comm, finalize_callback,
                      *args, **kwargs):
        weakref.finalize(comm, finalize_callback,
                         *args, **kwargs)
else:
    _comm_finalize_key = MPI.Comm.Create_keyval(
        delete_fn=lambda comm, key, finalizes:
        deque(map(call, finalizes), maxlen=0))

    # Similar to weakref.finalize behaviour with atexit=False
    def comm_finalize(comm, finalize_callback,
                      *args, **kwargs):
        finalizes = comm.Get_attr(_comm_finalize_key)
        if finalizes is None:
            finalizes = []
            comm.Set_attr(_comm_finalize_key, finalizes)
        finalizes.append(lambda: finalize_callback(*args, **kwargs))


_parent_comms = {}
_dup_comms = {}
_dupped_comms = {}


def comm_parent(dup_comm):
    while True:
        parent_comm = _parent_comms.get(dup_comm.py2f(), dup_comm)
        if MPI is None \
                or parent_comm.py2f() == MPI.COMM_NULL.py2f() \
                or parent_comm.py2f() == dup_comm.py2f():
            return parent_comm
        dup_comm = parent_comm


def comm_dup_cached(comm, *, key=None):
    """Return an internal duplicated communicator with key `key`.

    :arg comm: A communicator. Defines the base communicator.
    :arg key: The key.
    :returns: An internal duplicated communicator. May be `comm` itself. Freed
        when the original base communicator is freed.
    """

    if MPI is None:
        return comm

    comm = comm_parent(comm)
    if comm.py2f() == MPI.COMM_NULL.py2f():
        return comm

    key = (comm.py2f(), key)
    dup_comm = _dup_comms.get(key, None)

    if dup_comm is None:
        dup_comm = comm.Dup()
        _parent_comms[dup_comm.py2f()] = comm
        _dupped_comms.setdefault(comm.py2f(), {})[key] = dup_comm
        _dup_comms[key] = dup_comm

        def finalize_callback(comm_py2f, key, dup_comm):
            _parent_comms.pop(dup_comm.py2f(), None)
            _dupped_comms.pop(comm_py2f, None)
            _dup_comms.pop(key, None)
            garbage_cleanup(dup_comm)
            if MPI is not None and not MPI.Is_finalized():
                dup_comm.Free()

        comm_finalize(comm, finalize_callback,
                      comm.py2f(), key, dup_comm)

    return dup_comm


_garbage_cleanup = []


def register_garbage_cleanup(fn):
    _garbage_cleanup.append(fn)


if MPI is not None and PETSc is not None and hasattr(PETSc, "garbage_cleanup"):
    def garbage_cleanup_base(comm):
        if not MPI.Is_finalized() and not PETSc.Sys.isFinalized() \
                and comm.py2f() != MPI.COMM_NULL.py2f():
            PETSc.garbage_cleanup(comm)

    register_garbage_cleanup(garbage_cleanup_base)


def garbage_cleanup(comm=None):
    """Call `petsc4py.PETSc.garbage_cleanup(comm)` for a communicator, and any
    communicators duplicated from it using :func:`comm_dup_cached`.

    :arg comm: A communicator. Defaults to `DEFAULT_COMM`.
    """

    if MPI is None or MPI.Is_finalized():
        return

    if comm is None:
        comm = DEFAULT_COMM
    if comm.py2f() == MPI.COMM_NULL.py2f():
        return

    comm_stack = [comm]
    comms = {}
    while len(comm_stack) > 0:
        comm = comm_stack.pop()
        if MPI is not None \
                and comm.py2f() != MPI.COMM_NULL.py2f() \
                and comm.py2f() not in comms:
            comms[comm.py2f()] = comm
            comm_stack.extend(_dupped_comms.get(comm.py2f(), {}).values())

    for comm in comms.values():
        for fn in _garbage_cleanup:
            fn(comm)


def weakref_method(fn, obj):
    if not hasattr(obj, "_tlm_adjoint__weakref_method_self_ref"):
        obj._tlm_adjoint__weakref_method_self_ref = weakref.ref(obj)
    self_ref = obj._tlm_adjoint__weakref_method_self_ref

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        self = self_ref()
        if self is None:
            raise RuntimeError("Referent must be alive")
        return fn(self, *args, **kwargs)
    return wrapped_fn


class ProtectedDictionary(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        if key in self:
            raise RuntimeError(f"Key '{key}' already set")
        self._d[key] = value

    def __delitem__(self, key):
        if key in self:
            raise RuntimeError(f"Cannot delete key '{key}'")
        del self._d[key]  # Raises an Exception

    def __iter__(self):
        yield from self._d

    def __len__(self):
        return len(self._d)

    def d_delitem(self, key):
        del self._d[key]

    def d_setitem(self, key, value):
        self._d[key] = value


def add_interface(obj, interface_cls, attrs=None):
    """Attach a mixin `interface_cls`, defining an interface, to `obj`.

    :arg obj: An object to which the mixin should be attached.
    :arg interface_cls: A subclass of :class:`.SpaceInterface` or
        :class:`.VariableInterface` defining the interface.
    :arg attrs: A :class:`Mapping` defining any attributes. Used to set an
        attribute `_tlm_adjoint__space_interface_attrs` (for a
        :class:`.SpaceInterface`) or `_tlm_adjoint__var_interface_attrs`
        (for a :class:`.VariableInterface`).
    """

    if attrs is None:
        attrs = {}

    interface_name = f"{interface_cls.prefix:s}"
    assert not hasattr(obj, interface_name)
    setattr(obj, interface_name, interface_cls)

    for name in interface_cls.names:
        attr_name = f"{interface_cls.prefix:s}{name:s}"
        if not hasattr(obj, attr_name):
            setattr(obj, attr_name,
                    weakref_method(getattr(interface_cls, name), obj))

    attrs_name = f"{interface_cls.prefix:s}_attrs"
    assert not hasattr(obj, attrs_name)
    setattr(obj, attrs_name, ProtectedDictionary(attrs))


class SpaceInterface:
    """A mixin defining an interface for spaces. Space types do not inherit
    from this class -- instead an interface is defined by a
    :class:`.SpaceInterface` subclass, and methods are bound dynamically at
    runtime using :func:`.add_interface`.
    """

    prefix = "_tlm_adjoint__space_interface"
    names = ("_default_space_type", "_comm", "_dtype", "_id", "_eq",
             "_local_size", "_global_size", "_local_indices", "_new")

    def __init__(self):
        raise RuntimeError("Cannot instantiate SpaceInterface object")

    def _default_space_type(self):
        return "primal"

    def _comm(self):
        raise NotImplementedError("Method not overridden")

    def _dtype(self):
        raise NotImplementedError("Method not overridden")

    def _id(self):
        raise NotImplementedError("Method not overridden")

    def _eq(self, other):
        return space_id(self) == space_id(other)

    def _local_size(self):
        indices = space_local_indices(self)
        n0, n1 = indices.start, indices.stop
        return n1 - n0

    def _global_size(self):
        raise NotImplementedError("Method not overridden")

    def _local_indices(self):
        raise NotImplementedError("Method not overridden")

    def _new(self, *, name=None, space_type="primal", static=False,
             cache=None):
        raise NotImplementedError("Method not overridden")


def is_space(space):
    """Return whether `space` is a space -- i.e. has had a
    :class:`.SpaceInterface` attached.

    :arg space: An arbitrary :class:`object`.
    :returns: `True` if `space` is a space, and `False` otherwise.
    """

    return hasattr(space, "_tlm_adjoint__space_interface")


def space_comm(space):
    """
    :arg space: A space.
    :returns: The communicator associated with the space.
    """

    return space._tlm_adjoint__space_interface_comm()


def space_default_space_type(space):
    """
    :arg space: A space.
    :returns: The default space type associated with the space.
    """

    return space._tlm_adjoint__space_interface_default_space_type()


def space_dtype(space):
    """
    :arg space: A space.
    :returns: The data type associated with the space. Typically
        :class:`numpy.double` or :class:`numpy.cdouble`.
    """

    return space._tlm_adjoint__space_interface_dtype()


_space_id_counter = itertools.count()


def new_space_id():
    return next(_space_id_counter)


def space_id(space):
    """Return a unique :class:`int` ID associated with a space.

    :arg space: The space.
    :returns: The unique :class:`int` ID.
    """

    return space._tlm_adjoint__space_interface_id()


def space_eq(space, other):
    """
    :arg space: The space.
    :arg other: A second space, to compare to space.
    :returns: Whether the two spaces are equal.
    """

    return (is_space(other)
            and space._tlm_adjoint__space_interface_eq(other))


def space_local_size(space):
    """Return the process local number of degrees of freedom associated with
    a variable in a space. This is the number of 'owned' degrees of freedom.

    :arg x: The space.
    :returns: The process local number of degrees of freedom for a variable in
        the space.
    """

    return space._tlm_adjoint__space_interface_local_size()


def space_global_size(space):
    """Return the global number of degrees of freedom associated with a
    variable in a space. This is the total number of 'owned' degrees of
    freedom, summed across all processes.

    :arg x: The space.
    :returns: The global number of degrees of freedom for a variable in the
        space.
    """

    return space._tlm_adjoint__space_interface_global_size()


def space_local_indices(space):
    """Return the indices of process local degrees of freedom associated with
    a variable in a space.

    :arg x: The space.
    :returns: An :class:`slice`, yielding the indices of the process local
        elements.
    """

    return space._tlm_adjoint__space_interface_local_indices()


def space_new(space, *, name=None, space_type=None, static=False,
              cache=None):
    """Return a new variable.

    :arg space: The space.
    :arg name: A :class:`str` name for the variable.
    :arg space_type: The space type for the new variable. `'primal'`, `'dual'`,
        `'conjugate'`, or `'conjugate_dual'`. Defaults to
        `space_default_space_type(space)`.
    :arg static: Defines whether the new variable is static, meaning that it is
        stored by reference in checkpointing/replay, and an associated
        tangent-linear variable is zero.
    :arg cache: Defines whether results involving the new variable may be
        cached. Default `static`.
    :returns: The new variable.
    """

    if space_type is None:
        space_type = space_default_space_type(space)
    if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
        raise ValueError("Invalid space type")
    return space._tlm_adjoint__space_interface_new(
        name=name, space_type=space_type, static=static, cache=cache)


def relative_space_type(space_type, rel_space_type):
    """Return a relative space type. For example if `space_type` is `'dual'`
    and `rel_space_type` is `'conjugate_dual'`, this returns `'conjugate'`.

    :arg space_type: An input space type. One of `'primal'`, `'conjugate'`,
        `'dual'`, or `'conjugate_dual'`.
    :arg rel_space_type: The relative space type to return. One of `'primal'`,
        `'conjugate'`, `'dual'`, or `'conjugate_dual'`.
    :returns: A space type relative to `space_type`.
    """

    space_type_fn = {"primal": lambda space_type: space_type,
                     "conjugate": conjugate_space_type,
                     "dual": dual_space_type,
                     "conjugate_dual": conjugate_dual_space_type}[rel_space_type]  # noqa: E501
    return space_type_fn(space_type)


def conjugate_space_type(space_type):
    r"""Defines a map

        - `'primal'` :math:`\rightarrow` `'conjugate'`
        - `'conjugate'` :math:`\rightarrow` `'primal'`
        - `'dual'` :math:`\rightarrow` `'conjugate_dual'`
        - `'conjugate_dual'` :math:`\rightarrow` `'dual'`

    :returns: The space type conjugate to `space_type`.
    """

    return {"primal": "conjugate", "conjugate": "primal",
            "dual": "conjugate_dual", "conjugate_dual": "dual"}[space_type]


def dual_space_type(space_type):
    r"""Defines a map

        - `'primal'` :math:`\rightarrow` `'dual'`
        - `'conjugate'` :math:`\rightarrow` `'conjugate_dual'`
        - `'dual'` :math:`\rightarrow` `'primal'`
        - `'conjugate_dual'` :math:`\rightarrow` `'conjugate'`

    :returns: The space type dual to `space_type`.
    """

    return {"primal": "dual", "conjugate": "conjugate_dual",
            "dual": "primal", "conjugate_dual": "conjugate"}[space_type]


def conjugate_dual_space_type(space_type):
    r"""Defines a map

        - `'primal'` :math:`\rightarrow` `'conjugate_dual'`
        - `'conjugate'` :math:`\rightarrow` `'dual'`
        - `'dual'` :math:`\rightarrow` `'conjugate'`
        - `'conjugate_dual'` :math:`\rightarrow` `'primal'`

    :returns: The space type conjugate dual to `space_type`.
    """

    return {"primal": "conjugate_dual", "conjugate": "dual",
            "dual": "conjugate", "conjugate_dual": "primal"}[space_type]


_check_space_types = True


def no_space_type_checking(fn):
    """Decorator to disable space type checking.

    :arg fn: A callable for which space type checking should be disabled.
    :returns: A callable for which space type checking is disabled.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        with paused_space_type_checking():
            return fn(*args, **kwargs)
    return wrapped_fn


@contextlib.contextmanager
def paused_space_type_checking():
    """Construct a context manager which can be used to temporarily disable
    space type checking.

    :returns: A context manager which can be used to temporarily disable
        space type checking.
    """

    global _check_space_types
    check_space_types = _check_space_types
    _check_space_types = False
    try:
        yield
    finally:
        _check_space_types = check_space_types


class SpaceTypeError(RuntimeError):
    """Raised when an unexpected space type is encountered with space type
    checking enabled.
    """


def check_space_type(x, space_type):
    """Check that a variable has a given space type.

    Raises a :class:`.SpaceTypeError` if the check fails and space type
    checking is enabled.

    :arg x: A variable, whose space type should be checked.
    :arg space_type: The space type. One of `'primal'`, `'conjugate'`,
        `'dual'`, or `'conjugate_dual'`.
    """

    if space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
        raise ValueError("Invalid space type")
    if _check_space_types:
        x_space_type = var_space_type(x)
        if x_space_type != space_type:
            raise SpaceTypeError(f"Unexpected space type '{x_space_type}', "
                                 f"expected '{space_type}'")


def check_space_types(x, y, *, rel_space_type="primal"):
    """Check that `x` and `y` have compatible space types.

    Raises a :class:`.SpaceTypeError` if the check fails and space type
    checking is enabled.

    :arg x: A variable.
    :arg y: A variable.
    :arg rel_space_type: Check that the space type of `x` is `rel_space_type`
        relative to `y`. For example if `rel_space_type='dual'`, and the
        space type of `y` is `'conjugate_dual'`, checks that the space type of
        `x` is `'conjugate'`.
    """

    check_space_type(x, var_space_type(y, rel_space_type=rel_space_type))


def check_space_types_conjugate(x, y):
    """Check that `x` has space type conjugate to the space type for `y`.

    Raises a :class:`.SpaceTypeError` if the check fails and space type
    checking is enabled.

    :arg x: A variable.
    :arg y: A variable.
    """

    check_space_type(x, var_space_type(y, rel_space_type="conjugate"))


def check_space_types_dual(x, y):
    """Check that `x` has space type dual to the space type for `y`.

    Raises a :class:`.SpaceTypeError` if the check fails and space type
    checking is enabled.

    :arg x: A variable.
    :arg y: A variable.
    """

    check_space_type(x, var_space_type(y, rel_space_type="dual"))


def check_space_types_conjugate_dual(x, y):
    """Check that `x` has space type conjugate dual to the space type for `y`.

    Raises a :class:`.SpaceTypeError` if the check fails and space type
    checking is enabled.

    :arg x: A variable.
    :arg y: A variable.
    """

    check_space_type(x, var_space_type(y, rel_space_type="conjugate_dual"))


class VariableInterface:
    """A mixin defining an interface for variables. Variables types do not
    inherit from this class -- instead an interface is defined by a
    :class:`.VariableInterface` subclass, and methods are bound dynamically at
    runtime using :func:`.add_interface`.
    """

    prefix = "_tlm_adjoint__var_interface"
    names = ("_comm", "_space", "_space_type", "_dtype", "_id", "_name",
             "_state", "_update_state", "_is_static", "_is_cached", "_caches",
             "_zero", "_assign", "_axpy", "_inner", "_linf_norm",
             "_local_size", "_global_size", "_local_indices", "_get_values",
             "_set_values", "_new", "_copy", "_replacement", "_is_replacement",
             "_is_scalar", "_scalar_value", "_is_alias")

    def __init__(self):
        raise RuntimeError("Cannot instantiate VariableInterface object")

    def _comm(self):
        return space_comm(var_space(self))

    def _space(self):
        raise NotImplementedError("Method not overridden")

    def _space_type(self):
        raise NotImplementedError("Method not overridden")

    def _dtype(self):
        return space_dtype(var_space(self))

    def _id(self):
        raise NotImplementedError("Method not overridden")

    def _name(self):
        raise NotImplementedError("Method not overridden")

    def _state(self):
        raise NotImplementedError("Method not overridden")

    def _update_state(self):
        raise NotImplementedError("Method not overridden")

    def _is_static(self):
        raise NotImplementedError("Method not overridden")

    def _is_cached(self):
        raise NotImplementedError("Method not overridden")

    def _caches(self):
        raise NotImplementedError("Method not overridden")

    def _zero(self):
        raise NotImplementedError("Method not overridden")

    def _assign(self, y):
        raise NotImplementedError("Method not overridden")

    def _axpy(self, alpha, x, /):
        raise NotImplementedError("Method not overridden")

    def _inner(self, y):
        raise NotImplementedError("Method not overridden")

    def _linf_norm(self):
        raise NotImplementedError("Method not overridden")

    def _local_size(self):
        return space_local_size(var_space(self))

    def _global_size(self):
        return space_global_size(var_space(self))

    def _local_indices(self):
        return space_local_indices(var_space(self))

    def _get_values(self):
        raise NotImplementedError("Method not overridden")

    def _set_values(self, values):
        raise NotImplementedError("Method not overridden")

    def _new(self, *, name=None, static=False, cache=None,
             rel_space_type="primal"):
        space_type = var_space_type(self, rel_space_type=rel_space_type)
        return space_new(var_space(self), name=name,
                         space_type=space_type, static=static, cache=cache)

    def _copy(self, *, name=None, static=False, cache=None):
        y = var_new(self, name=name, static=static, cache=cache)
        var_assign(y, self)
        return y

    def _replacement(self):
        raise NotImplementedError("Method not overridden")

    def _is_replacement(self):
        raise NotImplementedError("Method not overridden")

    def _is_scalar(self):
        raise NotImplementedError("Method not overridden")

    def _scalar_value(self):
        raise NotImplementedError("Method not overridden")

    def _is_alias(self):
        return False


def is_var(x):
    """Return whether `x` is a variable -- i.e. has had a
    :class:`.VariableInterface` added.

    :arg x: An arbitrary :class:`object`.
    :returns: `True` if `x` is a variable, and `False` otherwise.
    """

    return hasattr(x, "_tlm_adjoint__var_interface")


def var_comm(x):
    """
    :arg x: A variable.
    :returns: The communicator associated with the variable.
    """

    return x._tlm_adjoint__var_interface_comm()


def var_space(x):
    """
    :arg x: A variable.
    :returns: The space associated with the variable.
    """

    return x._tlm_adjoint__var_interface_space()


def var_space_type(x, *, rel_space_type="primal"):
    """Return the space type of a variable.

    :arg x: The variable.
    :arg rel_space_type: If supplied then return a space type relative to the
        variable space type. One of `'primal'`, `'conjugate'`, `'dual'`, or
        `'conjugate_dual'`.
    :returns: The space type.
    """

    space_type = x._tlm_adjoint__var_interface_space_type()
    return relative_space_type(space_type, rel_space_type)


def var_dtype(x):
    """
    :arg x: A variable.
    :returns: The data type associated with the variable. Typically
        :class:`numpy.double` or :class:`numpy.cdouble`.
    """

    return x._tlm_adjoint__var_interface_dtype()


_var_id_counter = itertools.count()


def new_var_id():
    return next(_var_id_counter)


def var_id(x):
    """Return a unique :class:`int` ID associated with a variable.

    Note that two variables share the same ID if they represent the same
    symbolic variable -- for example if one variable represents both a variable
    and stores a value, and a second the same variable with no value (i.e. is a
    'replacement'), then the two variables share the same ID.

    :arg x: The variable.
    :returns: The :class:`int` ID.
    """

    return x._tlm_adjoint__var_interface_id()


def var_name(x):
    """
    :arg x: A variable.
    :returns: The :class:`str` name of the variable.
    """

    return x._tlm_adjoint__var_interface_name()


def var_state(x):
    """Return the value of the state counter for a variable. Updated when the
    value of the variable changes.

    :arg x: The variable.
    :returns: The :class:`int` state value.
    """

    return x._tlm_adjoint__var_interface_state()


def var_increment_state_lock(x, obj):
    if var_is_replacement(x):
        raise ValueError("x cannot be a replacement")
    var_check_state_lock(x)
    x_id = var_id(x)

    if not hasattr(x, "_tlm_adjoint__state_lock"):
        x._tlm_adjoint__state_lock = 0
    if x._tlm_adjoint__state_lock == 0:
        x._tlm_adjoint__state_lock_state = var_state(x)

    # Functionally similar to a weakref.WeakKeyDictionary, using the variable
    # ID as a key. This approach does not require obj to be hashable.
    if not hasattr(obj, "_tlm_adjoint__state_locks"):
        obj._tlm_adjoint__state_locks = {}

        def weakref_finalize(locks):
            for x_ref, count in locks.values():
                x = x_ref()
                if x is not None and hasattr(x, "_tlm_adjoint__state_lock"):
                    x._tlm_adjoint__state_lock -= count

        weakref.finalize(obj, weakref_finalize,
                         obj._tlm_adjoint__state_locks)
    if x_id not in obj._tlm_adjoint__state_locks:
        obj._tlm_adjoint__state_locks[x_id] = [weakref.ref(x), 0]

    x._tlm_adjoint__state_lock += 1
    obj._tlm_adjoint__state_locks[x_id][1] += 1


def var_decrement_state_lock(x, obj):
    if var_is_replacement(x):
        raise ValueError("x cannot be a replacement")
    var_check_state_lock(x)
    x_id = var_id(x)

    if x._tlm_adjoint__state_lock < obj._tlm_adjoint__state_locks[x_id][1]:
        raise RuntimeError("Invalid state lock")
    if obj._tlm_adjoint__state_locks[x_id][1] < 1:
        raise RuntimeError("Invalid state lock")

    x._tlm_adjoint__state_lock -= 1
    obj._tlm_adjoint__state_locks[x_id][1] -= 1
    if obj._tlm_adjoint__state_locks[x_id][1] == 0:
        del obj._tlm_adjoint__state_locks[x_id]


class VariableStateChangeError(RuntimeError):
    pass


def var_lock_state(x):
    """Lock the state of a variable.

    :arg x: The variable.
    """

    class Lock:
        pass

    lock = x._tlm_adjoint__state_lock_lock = Lock()
    var_increment_state_lock(x, lock)


def var_state_is_locked(x):
    count = getattr(x, "_tlm_adjoint__state_lock", 0)
    if count < 0:
        raise RuntimeError("Invalid state lock")
    return count > 0


def var_check_state_lock(x):
    if var_state_is_locked(x) \
            and x._tlm_adjoint__state_lock_state != var_state(x):
        raise VariableStateChangeError("State change while locked")


class VariableStateLockDictionary(MutableMapping):
    """A dictionary-like class. If a value is a variable and not a replacement
    then the variable state is 'locked' so that a state update, with the lock
    active, will raise an exception.

    State locks are automatically released when the
    :class:`.VariableStateLockDictionary` is destroyed. Consequently objects of
    this type should be used with caution. In particular object destruction via
    the garbage collector may lead to non-deterministic release of the state
    lock.
    """

    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        for value in self._d.values():
            if is_var(value) and not var_is_replacement(value):
                var_increment_state_lock(value, self)

    def __getitem__(self, key):
        value = self._d[key]
        if is_var(value) and not var_is_replacement(value):
            var_check_state_lock(value)
        return value

    def __setitem__(self, key, value):
        oldvalue = self._d.get(key, None)
        self._d[key] = value
        if is_var(value) and not var_is_replacement(value):
            var_increment_state_lock(value, self)
        if is_var(oldvalue) and not var_is_replacement(oldvalue):
            var_decrement_state_lock(oldvalue, self)

    def __delitem__(self, key):
        oldvalue = self._d.get(key, None)
        del self._d[key]
        if is_var(oldvalue) and not var_is_replacement(oldvalue):
            var_decrement_state_lock(oldvalue, self)

    def __iter__(self):
        yield from self._d

    def __len__(self):
        return len(self._d)


@contextlib.contextmanager
def var_locked(*X):
    """Construct a context manager which can be used to temporarily lock the
    state of one or more variables.

    :arg X: A :class:`tuple` of variables.
    :returns: A context manager which can be used to temporarily lock the state
        of the variables in `X`.
    """

    class Lock:
        pass

    lock = Lock()
    for x in X:
        var_increment_state_lock(x, lock)

    try:
        yield
    finally:
        for x in X:
            var_decrement_state_lock(x, lock)


def var_update_state(*X):
    """Ensure that variable state is updated, and check for cache invalidation.
    May delegate updating of the state to a backend library.

    :arg X: A :class:`tuple` of variables.
    """

    for x in X:
        if var_is_replacement(x):
            raise ValueError("x cannot be a replacement")
        var_check_state_lock(x)
        if var_state_is_locked(x):
            raise VariableStateChangeError("Cannot update state for locked "
                                           "variable")
        x._tlm_adjoint__var_interface_update_state()
    var_update_caches(*X)


def var_is_static(x):
    """Return whether a variable is flagged as 'static'. A static variable
    is stored by reference in checkpointing/replay, and the associated
    tangent-linear variable is zero.

    :arg x: The variable.
    :returns: Whether the variable is flagged as static.
    """

    return x._tlm_adjoint__var_interface_is_static()


def var_is_cached(x):
    """Return whether results involving this variable may be cached.

    :arg x: The variable.
    :returns: Whether results involving the variable may be cached.
    """

    return x._tlm_adjoint__var_interface_is_cached()


def var_is_checkpointed(x):
    ""

    warnings.warn("var_is_checkpointed is deprecated -- "
                  "use `not var_is_static(x)` instead",
                  DeprecationWarning, stacklevel=2)
    return var_is_static(x)


def var_caches(x):
    """Return the :class:`.Caches` associated with a variable.

    :arg x: The variable.
    :returns: The :class:`.Caches` associated with the variable.
    """

    return x._tlm_adjoint__var_interface_caches()


def var_update_caches(*X, value=None):
    """Check for cache invalidation associated with a possible change in value.

    :arg X: A :class:`tuple` of variables whose value may have changed.
    :arg value: A variable or a :class:`Sequence` of variables defining the
        possible new values. `X` is used if not supplied.
    """

    if value is None:
        for x in X:
            if var_is_replacement(x):
                raise TypeError("value required")
            var_check_state_lock(x)
            var_caches(x).update(x)
    else:
        if is_var(value):
            value = (value,)
        var_update_caches(*value)
        assert len(X) == len(value)
        for x, x_value in zip(X, value):
            var_check_state_lock(x_value)
            var_caches(x).update(x_value)


@manager_disabled()
def var_zero(x):
    """Zero a variable.

    :arg x: The variable.
    """

    x._tlm_adjoint__var_interface_zero()
    var_update_state(x)


@manager_disabled()
def var_assign(x, y):
    """Perform an assignment `x = y`.

    :arg x: A variable.
    :arg y: A variable.
    """

    if is_var(y):
        check_space_types(x, y)
    x._tlm_adjoint__var_interface_assign(y)
    var_update_state(x)


@manager_disabled()
def var_axpy(y, alpha, x, /):
    """Perform an in-place addition `y += alpha * x`.

    :arg y: A variable.
    :arg alpha: A scalar.
    :arg x: A variable.
    """

    if is_var(x):
        check_space_types(y, x)
    y._tlm_adjoint__var_interface_axpy(alpha, x)
    var_update_state(y)


@manager_disabled()
def var_inner(x, y):
    """Compute the :math:`l_2` inner product of the degrees of freedom vectors
    associated with `x` and `y`. By convention if `y` is in the conjugate dual
    space associated with `x`, this returns the complex conjugate of the
    functional associated with `y` evaluated at `x`.

    :arg x: A variable.
    :arg y: A variable.
    :returns: The result of the inner product.
    """

    if is_var(y):
        check_space_types_conjugate_dual(x, y)
    return x._tlm_adjoint__var_interface_inner(y)


@manager_disabled()
def var_linf_norm(x):
    r"""Compute the :math:`l_\infty` norm of the degrees of freedom vector
    associated with a variable.

    :arg x: The variable.
    :returns: The :math:`l_\infty` norm of the degrees of freedom vector.
    """

    return x._tlm_adjoint__var_interface_linf_norm()


def var_local_size(x):
    """Return the process local number of degrees of freedom associated with
    a variable. This is the number of 'owned' degrees of freedom.

    :arg x: The variable.
    :returns: The process local number of degrees of freedom for the variable.
    """

    return x._tlm_adjoint__var_interface_local_size()


def var_global_size(x):
    """Return the global number of degrees of freedom associated with a
    variable. This is the total number of 'owned' degrees of freedom, summed
    across all processes.

    :arg x: The variable.
    :returns: The global number of degrees of freedom for the variable.
    """

    return x._tlm_adjoint__var_interface_global_size()


def var_local_indices(x):
    """Return the indices of process local degrees of freedom associated with
    a variable.

    :arg x: The variable.
    :returns: An :class:`slice`, yielding the indices of the process local
        elements.
    """

    return x._tlm_adjoint__var_interface_local_indices()


@manager_disabled()
def var_get_values(x):
    """Return a copy of the process local degrees of freedom vector associated
    with a variable.

    :arg x: The variable.
    :returns: A :class:`numpy.ndarray` containing a copy of the degrees of
        freedom.
    """

    values = x._tlm_adjoint__var_interface_get_values()
    if not np.can_cast(values, var_dtype(x)):
        raise ValueError("Invalid dtype")
    if values.shape != (var_local_size(x),):
        raise ValueError("Invalid shape")
    return values


@manager_disabled()
def var_set_values(x, values):
    """Set the process local degrees of freedom vector associated with a
    variable.

    :arg x: The variable.
    :arg values: A :class:`numpy.ndarray` containing the degrees of freedom
        values.
    """

    if not np.can_cast(values, var_dtype(x)):
        raise ValueError("Invalid dtype")
    if values.shape != (var_local_size(x),):
        raise ValueError("Invalid shape")
    x._tlm_adjoint__var_interface_set_values(values)
    var_update_state(x)


@manager_disabled()
def var_new(x, *, name=None, static=False, cache=None, checkpoint=None,
            rel_space_type="primal"):
    """Return a new variable defined using the same space as `x`.

    :arg x: A variable.
    :arg name: A :class:`str` name for the new variable.
    :arg static: Defines whether the new variable is static, meaning that it is
        stored by reference in checkpointing/replay, and an associated
        tangent-linear variable is zero.
    :arg cache: Defines whether results involving the new variable may be
        cached. Default `static`.
    :arg checkpoint: Deprecated.
    :arg rel_space_type: Defines the space type of the new variable, relative
        to the space type of `x`.
    :returns: The new variable.
    """

    if checkpoint is not None:
        if checkpoint == static:
            warnings.warn("checkpoint argument is deprecated",
                          DeprecationWarning, stacklevel=2)
        else:
            raise ValueError("checkpoint argument is deprecated")

    if rel_space_type not in {"primal", "conjugate", "dual", "conjugate_dual"}:
        raise ValueError("Invalid relative space type")
    return x._tlm_adjoint__var_interface_new(
        name=name, static=static, cache=cache, rel_space_type=rel_space_type)


def var_new_conjugate(x, *, name=None, static=False, cache=None):
    """Return a new conjugate variable. See :func:`.var_new`.

    :returns: A new variable defined using the same space as `x`, with space
        type conjugate to the space type for `x`.
    """

    return var_new(x, name=name, static=static, cache=cache,
                   rel_space_type="conjugate")


def var_new_dual(x, *, name=None, static=False, cache=None):
    """Return a new dual variable. See :func:`.var_new`.

    :returns: A new variable defined using the same space as `x`, with space
        type dual to the space type for `x`.
    """

    return var_new(x, name=name, static=static, cache=cache,
                   rel_space_type="dual")


def var_new_conjugate_dual(x, *, name=None, static=False, cache=None):
    """Return a new conjugate dual variable. See :func:`.var_new`.

    :returns: A new variable defined using the same space as `x`, with space
        type conjugate dual to the space type for `x`.
    """

    return var_new(x, name=name, static=static, cache=cache,
                   rel_space_type="conjugate_dual")


@manager_disabled()
def var_copy(x, *, name=None, static=False, cache=None):
    """Copy a variable. See :func:`.var_new`.

    :returns: The copied variable.
    """

    return x._tlm_adjoint__var_interface_copy(
        name=name, static=static, cache=cache)


def var_new_tangent_linear(x, *, name=None):
    if var_is_static(x):
        return None
    else:
        return var_new(x, name=name, static=False, cache=var_is_cached(x))


def var_replacement(x):
    """Return a variable, associated with the same variable as `x`, but
    possibly without a value.

    :arg x: The variable.
    :returns: A variable which symbolically represents the same variable as
        `x`, but which may not store a value. May return `x` itself.
    """

    if var_is_replacement(x):
        return x
    else:
        return x._tlm_adjoint__var_interface_replacement()


def var_is_replacement(x):
    """Return whether a variable is a 'replacement', meaning that it has no
    associated value.

    :arg x: The variable.
    :returns: Whether `x` is a replacement.
    """

    return x._tlm_adjoint__var_interface_is_replacement()


def var_is_scalar(x):
    """Return whether a variable defines a scalar variable.

    :arg x: The variable.
    :returns: Whether `x` defines a scalar variable.
    """

    return x._tlm_adjoint__var_interface_is_scalar()


@manager_disabled()
def var_scalar_value(x):
    """If `x` defines a scalar variable, returns its value.

    :arg x: The variable, defining a scalar variable.
    :returns: The scalar value.
    """

    if not var_is_scalar(x):
        raise ValueError("Invalid variable")
    return x._tlm_adjoint__var_interface_scalar_value()


def var_is_alias(x):
    """Return whether a variable is an 'alias', meaning part or all of the
    degree of freedom vector associated with the variable is shared with some
    different aliased variable. A variable may not appear as an equation
    dependency if it is an alias.

    :arg x: The variable.
    :returns: Whether the variable is an alias.
    """

    return x._tlm_adjoint__var_interface_is_alias()


def var_copy_conjugate(x):
    y = var_new_conjugate(x)
    var_set_values(y, var_get_values(x).conjugate())
    return y


def var_assign_conjugate(x, y):
    check_space_types_conjugate(x, y)
    var_assign(x, var_copy_conjugate(y))


def var_axpy_conjugate(y, alpha, x, /):
    check_space_types_conjugate(y, x)
    var_axpy(y, alpha, var_copy_conjugate(x))


def var_dot(x, y):
    return var_inner(x, var_copy_conjugate(y))


def vars_assign(X, Y):
    if len(X) != len(Y):
        raise ValueError("Incompatible lengths")
    for x, y in zip(X, Y):
        var_assign(x, y)


def vars_axpy(Y, alpha, X, /):
    if len(X) != len(Y):
        raise ValueError("Incompatible lengths")
    for y, x in zip(Y, X):
        var_axpy(y, alpha, x)


def vars_inner(X, Y):
    if len(X) != len(Y):
        raise ValueError("Incompatible lengths")
    return sum((var_inner(x, y) for x, y in zip(X, Y)), start=0.0)


def vars_linf_norm(X):
    return max(map(var_linf_norm, X), default=0.0)


def vars_new(X):
    return tuple(map(var_new, X))


def vars_new_conjugate_dual(X):
    return tuple(map(var_new_conjugate_dual, X))


def vars_copy(X):
    return tuple(map(var_copy, X))


class ReplacementInterface(VariableInterface):
    def _space(self):
        return self._tlm_adjoint__var_interface_attrs["space"]

    def _space_type(self):
        return self._tlm_adjoint__var_interface_attrs["space_type"]

    def _id(self):
        return self._tlm_adjoint__var_interface_attrs["id"]

    def _name(self):
        return self._tlm_adjoint__var_interface_attrs["name"]

    def _state(self):
        return -1

    def _is_static(self):
        return self._tlm_adjoint__var_interface_attrs["static"]

    def _is_cached(self):
        return self._tlm_adjoint__var_interface_attrs["cache"]

    def _caches(self):
        return self._tlm_adjoint__var_interface_attrs["caches"]

    def _replacement(self):
        return self

    def _is_replacement(self):
        return True


def add_replacement_interface(replacement, x):
    add_interface(replacement, ReplacementInterface,
                  {"id": var_id(x), "name": var_name(x),
                   "space": var_space(x),
                   "space_type": var_space_type(x),
                   "static": var_is_static(x),
                   "cache": var_is_cached(x),
                   "caches": var_caches(x)})


@functools.singledispatch
def subtract_adjoint_derivative_action(x, y):
    """Subtract an adjoint right-hand-side contribution defined by `y` from
    the right-hand-side defined by `x`.

    :arg x: A variable storing the adjoint right-hand-side.
    :arg y: A contribution to subtract from the adjoint right-hand-side. An
        :meth:`.Equation.adjoint_derivative_action` return value. Valid types
        depend upon the variable type. Typically this will be a variable, or a
        two element :class:`tuple` `(alpha, F)`, where `alpha` is a
        :class:`numbers.Complex` and `F` a variable, with the value to subtract
        defined by the product of `alpha` and `F`.
    """

    raise NotImplementedError("Unexpected case encountered")


def register_subtract_adjoint_derivative_action(x_cls, y_cls, fn, *,
                                                replace=False):
    if not isinstance(x_cls, Sequence):
        x_cls = (x_cls,)
    if not isinstance(y_cls, Sequence):
        y_cls = (y_cls,)
    for x_cls, y_cls in itertools.product(x_cls, y_cls):
        if x_cls not in subtract_adjoint_derivative_action.registry:
            @functools.singledispatch
            def _x_fn(y, alpha, x):
                raise NotImplementedError("Unexpected case encountered")

            @subtract_adjoint_derivative_action.register(x_cls)
            def x_fn(x, y):
                if isinstance(y, tuple) \
                        and len(y) == 2 \
                        and isinstance(y[0], numbers.Complex):
                    alpha, y = y
                else:
                    alpha = 1.0
                return subtract_adjoint_derivative_action.dispatch(type(x))._tlm_adjoint__x_fn(y, alpha, x)  # noqa: E501

            x_fn._tlm_adjoint__x_fn = _x_fn

        _x_fn = subtract_adjoint_derivative_action.registry[x_cls]._tlm_adjoint__x_fn  # noqa: E501
        if y_cls in _x_fn.registry and not replace:
            raise RuntimeError("Case already registered")

        @_x_fn.register(y_cls)
        def wrapped_fn(y, alpha, x):
            return_value = fn(x, alpha, y)
            var_update_state(x)
            return return_value


def subtract_adjoint_derivative_action_base(x, alpha, y):
    if y is None:
        pass
    elif is_var(y):
        check_space_types(x, y)
        var_axpy(x, -alpha, y)
    else:
        raise NotImplementedError("Unexpected case encountered")


@functools.singledispatch
def _functional_term_eq(term, x):
    raise NotImplementedError("Unexpected case encountered")


def functional_term_eq(x, term):
    return _functional_term_eq(term, x)


def register_functional_term_eq(term_cls, fn, *,
                                replace=False):
    if not isinstance(term_cls, Sequence):
        term_cls = (term_cls,)
    for term_cls in term_cls:
        if term_cls in _functional_term_eq.registry and not replace:
            raise RuntimeError("Case already registered")

        @_functional_term_eq.register(term_cls)
        def wrapped_fn(term, x):
            return fn(x, term)


_logger = logging.getLogger("tlm_adjoint")
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


def is_function(x):
    ""

    warnings.warn("is_function is deprecated -- "
                  "use is_var instead",
                  DeprecationWarning, stacklevel=2)
    return is_var(x)


def _function_warning(fn):
    def wrapped_fn(*args, **kwargs):
        ""

        warnings.warn("function_ prefixed functions are deprecated -- "
                      "use var_ prefixed functions instead",
                      DeprecationWarning, stacklevel=2)
        return fn(*args, **kwargs)

    return wrapped_fn


function_assign = _function_warning(var_assign)
function_axpy = _function_warning(var_axpy)
function_caches = _function_warning(var_caches)
function_comm = _function_warning(var_comm)
function_copy = _function_warning(var_copy)
function_dtype = _function_warning(var_dtype)
function_get_values = _function_warning(var_get_values)
function_global_size = _function_warning(var_global_size)
function_id = _function_warning(var_id)
function_inner = _function_warning(var_inner)
function_is_cached = _function_warning(var_is_cached)
function_is_checkpointed = _function_warning(var_is_checkpointed)
function_is_replacement = _function_warning(var_is_replacement)
function_is_static = _function_warning(var_is_static)
function_linf_norm = _function_warning(var_linf_norm)
function_local_indices = _function_warning(var_local_indices)
function_local_size = _function_warning(var_local_size)
function_name = _function_warning(var_name)
function_new = _function_warning(var_new)
function_new_conjugate = _function_warning(var_new_conjugate)
function_new_conjugate_dual = _function_warning(var_new_conjugate_dual)
function_new_dual = _function_warning(var_new_dual)
function_replacement = _function_warning(var_replacement)
function_set_values = _function_warning(var_set_values)
function_space = _function_warning(var_space)
function_space_type = _function_warning(var_space_type)
function_state = _function_warning(var_state)
function_update_caches = _function_warning(var_update_caches)
function_update_state = _function_warning(var_update_state)
function_zero = _function_warning(var_zero)
function_is_scalar = _function_warning(var_is_scalar)
function_scalar_value = _function_warning(var_scalar_value)
function_is_alias = _function_warning(var_is_alias)
