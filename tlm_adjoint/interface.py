#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""This module defines an interface for interaction with backend data types.
This is implemented via runtime binding of mixins. The
:class:`FunctionInterface` adds methods to 'functions' which can be used to
interact with backend variables. The :class:`SpaceInterface` adds methods to
'spaces' which define the vector spaces in which those 'functions' are defined.

The extra methods are accessed using the callables defined in this module
(which also handle some extra details, e.g. related to cache invalidation and
space type checking). Typically these are prefixed with `space_` for spaces and
`function_` for functions.

The term 'function' originates from finite element discrete functions, but
there is no assumption that these correspond to actual functions defined on any
particular computational domain. For example the :class:`SymbolicFloat` class
represents a scalar variable.

The interface distinguishes between original backend 'functions', which both
define symbolic variables and store values, and replacement 'functions', which
define the same variables but which need not store values.

Functions have an associated 'space type', which indicates e.g. if the variable
is 'primal', meaning a member on an originating vector space, or 'conjugate
dual', meaning a member of the corresponding antidual space of antilinear
functionals from the originating vector space. Functions can also be 'dual',
meaning a member of the dual space of linear functionals, or 'conjugate',
meaning a member of a space defined by a conjugate operator from the primal
space. This conjugate operator is defined by complex conjugation of the vector
of degrees of freedom, and could e.g. correspond to complex conjugation of a
finite element discretized function.

The space type associated with a function is defined relative to an originating
vector space (e.g. a finite element discrete function space). A 'relative space
type' is defined relative to one of the 'primal', 'conjugate', 'dual', or
'conjugate dual' spaces. For example the primal space associated with the dual
space is the dual space, and the dual space associated with the dual space is
the primal space.

This module defines a default communicator `DEFAULT_COMM`, which is
`mpi4py.MPI.COMM_WORLD` if mpi4py is available. If mpi4py is not available a
dummy 'serial' communicator is used, of type :class:`SerialComm`.
"""

from collections.abc import Mapping, Sequence
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
        "space_dtype",
        "space_id",
        "space_new",

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

        "FunctionInterface",
        "is_function",
        "function_assign",
        "function_axpy",
        "function_caches",
        "function_comm",
        "function_copy",
        "function_dtype",
        "function_form_derivative_space",
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
        "function_sum",
        "function_update_caches",
        "function_update_state",
        "function_zero",

        "function_is_scalar",
        "function_scalar_value",

        "subtract_adjoint_derivative_action",

        "function_is_alias"
    ]


DEFAULT_COMM = None
if MPI is None:
    # As for mpi4py 3.1.4 API
    class SerialComm:
        _id_counter = [-1]

        def __init__(self, *, _id=None):
            self._id = _id
            if self._id is None:
                self._id = self._id_counter[0]
                self._id_counter[0] -= 1

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

    f2py = DEFAULT_COMM.f2py

    def comm_finalize(comm, finalize_callback,
                      *args, **kwargs):
        weakref.finalize(comm, finalize_callback,
                         *args, **kwargs)
else:
    DEFAULT_COMM = MPI.COMM_WORLD

    f2py = MPI.Comm.f2py

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
    return _parent_comms.get(dup_comm.py2f(), dup_comm)


def comm_dup_cached(comm, *, key=None):
    """If the communicator `comm` with key `key` has previously been duplicated
    using :func:`comm_dup_cached`, then return the previous result. Otherwise
    duplicate the communicator and cache the result. The duplicated
    communicator is freed when the original base communicator is freed.

    :arg comm: A communicator, the base communicator to be duplicated.
    :arg key: The key.
    :returns: A communicator. A duplicated MPI communicator, or a previously
        cached duplicated MPI communicator, which is freed when the original
        base communicator is freed.
    """

    if MPI is not None and comm.py2f() == MPI.COMM_NULL.py2f():
        return comm

    if key is None:
        key = comm.py2f()
    else:
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
    """Call `petsc4py.PETSc.garbage_cleanup(comm)` for a communicator, any
    communicators duplicated from it, base communicators from which it was
    duplicated, and any communicators duplicated from those base communicators.

    :arg comm: A communicator. Defaults to `DEFAULT_COMM`.
    """

    if comm is None:
        comm = DEFAULT_COMM
    if MPI is None \
            or MPI.Is_finalized() \
            or comm.py2f() == MPI.COMM_NULL.py2f():
        return

    while True:
        parent_comm = comm_parent(comm)
        if MPI is None \
                or parent_comm.py2f() == MPI.COMM_NULL.py2f() \
                or parent_comm.py2f() == comm.py2f():
            break
        comm = parent_comm

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


class protecteddict(Mapping):  # noqa: N801
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Key '{key}' already set")
        self._d[key] = value

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
    :arg interface_cls: A subclass of :class:`SpaceInterface` or
        :class:`FunctionInterface` defining the interface.
    :arg attrs: A :class:`Mapping` defining any attributes. Used to set an
        attribute `_tlm_adjoint__space_interface_attrs` (for a
        :class:`SpaceInterface`) or `_tlm_adjoint__function_interface_attrs`
        (for a :class:`FunctionInterface`).
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
    setattr(obj, attrs_name, protecteddict(attrs))


class SpaceInterface:
    """A mixin defining an interface for spaces. Space types do not inherit
    from this class -- instead an interface is defined by a
    :class:`SpaceInterface` subclass, and methods are bound dynamically at
    runtime using :func:`add_interface`.
    """

    prefix = "_tlm_adjoint__space_interface"
    names = ("_comm", "_dtype", "_id", "_new")

    def __init__(self):
        raise RuntimeError("Cannot instantiate SpaceInterface object")

    def _comm(self):
        raise NotImplementedError("Method not overridden")

    def _dtype(self):
        raise NotImplementedError("Method not overridden")

    def _id(self):
        raise NotImplementedError("Method not overridden")

    def _new(self, *, name=None, space_type="primal", static=False, cache=None,
             checkpoint=None):
        raise NotImplementedError("Method not overridden")


def is_space(space):
    """Return whether `space` is a space -- i.e. has had a
    :class:`SpaceInterface` attached.

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


def space_dtype(space):
    """
    :arg space: A space.
    :returns: The data type associated with the space. Typically
        :class:`numpy.double` or :class:`numpy.cdouble`.
    """

    return space._tlm_adjoint__space_interface_dtype()


_space_id_counter = 0


def new_space_id():
    global _space_id_counter
    space_id = _space_id_counter
    _space_id_counter += 1
    return space_id


def space_id(space):
    """Return the unique :class:`int` ID associated with a space.

    :arg space: The space.
    :returns: The unique :class:`int` ID.
    """

    return space._tlm_adjoint__space_interface_id()


def space_new(space, *, name=None, space_type="primal", static=False,
              cache=None, checkpoint=None):
    """Return a new function.

    :arg space: The space.
    :arg name: A :class:`str` name for the function.
    :arg space_type: The space type for the new function. `'primal'`, `'dual'`,
        `'conjugate'`, or `'conjugate_dual'`.
    :arg static: Defines the default value for `cache` and `checkpoint`.
    :arg cache: Defines whether results involving this function may be cached.
        Default `static`.
    :arg checkpoint: Defines whether a
        :class:`tlm_adjoint.checkpointing.CheckpointStorage` should store this
        function by value (`checkpoint=True`) or reference
        (`checkpoint=False`). Default `not static`.
    :returns: The new function.
    """

    if space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
        raise ValueError("Invalid space type")
    return space._tlm_adjoint__space_interface_new(
        name=name, space_type=space_type, static=static, cache=cache,
        checkpoint=checkpoint)


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


_check_space_types = 0


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
    _check_space_types += 1
    try:
        yield
    finally:
        _check_space_types -= 1


def space_type_warning(msg, *, stacklevel=1):
    if _check_space_types == 0:
        warnings.warn(msg, stacklevel=stacklevel + 1)


def check_space_type(x, space_type):
    """Check that a function has a given space type.

    Emits a warning if the check fails and space type checking is enabled.

    :arg x: A function, whose space type should be checked.
    :arg space_type: The space type. One of `'primal'`, `'conjugate'`,
        `'dual'`, or `'conjugate_dual'`.
    """

    if space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
        raise ValueError("Invalid space type")
    if function_space_type(x) != space_type:
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types(x, y, *, rel_space_type="primal"):
    """Check that `x` and `y` have compatible space types.

    Emits a warning if the check fails and space type checking is enabled.

    :arg x: A function.
    :arg y: A function.
    :arg rel_space_type: Check that the space type of `x` is `rel_space_type`
        relative to `y`. For example if `rel_space_type='dual'`, and the
        space type of `y` is `'conjuguate_dual'`, checks that the space type of
        `x` is `'conjugate'`.
    """

    if function_space_type(x) != \
            function_space_type(y, rel_space_type=rel_space_type):
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types_conjugate(x, y):
    """Check that `x` has space type conjugate to the space type for `y`.

    Emits a warning if the check fails and space type checking is enabled.

    :arg x: A function.
    :arg y: A function.
    """

    if function_space_type(x) != \
            function_space_type(y, rel_space_type="conjugate"):
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types_dual(x, y):
    """Check that `x` has space type dual to the space type for `y`.

    Emits a warning if the check fails and space type checking is enabled.

    :arg x: A function.
    :arg y: A function.
    """

    if function_space_type(x) != \
            function_space_type(y, rel_space_type="dual"):
        space_type_warning("Unexpected space type", stacklevel=2)


def check_space_types_conjugate_dual(x, y):
    """Check that `x` has space type conjugate dual to the space type for `y`.

    Emits a warning if the check fails and space type checking is enabled.

    :arg x: A function.
    :arg y: A function.
    """

    if function_space_type(x) != \
            function_space_type(y, rel_space_type="conjugate_dual"):
        space_type_warning("Unexpected space type", stacklevel=2)


class FunctionInterface:
    """A mixin defining an interface for functions. Functions types do not
    inherit from this class -- instead an interface is defined by a
    :class:`FunctionInterface` subclass, and methods are bound dynamically at
    runtime using :func:`add_interface`.
    """

    prefix = "_tlm_adjoint__function_interface"
    names = ("_comm", "_space", "_form_derivative_space", "_space_type",
             "_dtype", "_id", "_name", "_state", "_update_state", "_is_static",
             "_is_cached", "_is_checkpointed", "_caches", "_zero", "_assign",
             "_axpy", "_inner", "_sum", "_linf_norm", "_local_size",
             "_global_size", "_local_indices", "_get_values", "_set_values",
             "_new", "_copy", "_replacement", "_is_replacement", "_is_scalar",
             "_scalar_value", "_is_alias")

    def __init__(self):
        raise RuntimeError("Cannot instantiate FunctionInterface object")

    def _comm(self):
        return space_comm(function_space(self))

    def _space(self):
        raise NotImplementedError("Method not overridden")

    def _form_derivative_space(self):
        raise NotImplementedError("Method not overridden")

    def _space_type(self):
        raise NotImplementedError("Method not overridden")

    def _dtype(self):
        return space_dtype(function_space(self))

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

    def _is_checkpointed(self):
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

    def _sum(self):
        raise NotImplementedError("Method not overridden")

    def _linf_norm(self):
        raise NotImplementedError("Method not overridden")

    def _local_size(self):
        raise NotImplementedError("Method not overridden")

    def _global_size(self):
        raise NotImplementedError("Method not overridden")

    def _local_indices(self):
        raise NotImplementedError("Method not overridden")

    def _get_values(self):
        raise NotImplementedError("Method not overridden")

    def _set_values(self, values):
        raise NotImplementedError("Method not overridden")

    def _new(self, *, name=None, static=False, cache=None, checkpoint=None,
             rel_space_type="primal"):
        space_type = function_space_type(self, rel_space_type=rel_space_type)
        return space_new(function_space(self), name=name,
                         space_type=space_type, static=static, cache=cache,
                         checkpoint=checkpoint)

    def _copy(self, *, name=None, static=False, cache=None, checkpoint=None):
        y = function_new(self, name=name, static=static, cache=cache,
                         checkpoint=checkpoint)
        function_assign(y, self)
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


def is_function(x):
    """Return whether `x` is a function -- i.e. has had a
    :class:`FunctionInterface` added.

    :arg x: An arbitrary :class:`object`.
    :returns: `True` if `x` is a function, and `False` otherwise.
    """

    return hasattr(x, "_tlm_adjoint__function_interface")


def function_comm(x):
    """
    :arg x: A function.
    :returns: The communicator associated with the function.
    """

    return x._tlm_adjoint__function_interface_comm()


def function_space(x):
    """
    :arg x: A function.
    :returns: The space associated with the function.
    """

    return x._tlm_adjoint__function_interface_space()


def function_form_derivative_space(x):
    """
    :returns: The space in which a derivative is defined when differentiating a
        UFL :class:`Form` with respect to the function.
    """

    return x._tlm_adjoint__function_interface_form_derivative_space()


def function_space_type(x, *, rel_space_type="primal"):
    """Return the space type of a function.

    :arg x: The function.
    :arg rel_space_type: If supplied then return a space type relative to the
        function space type. One of `'primal'`, `'conjugate'`, `'dual'`, or
        `'conjugate_dual'`.
    :returns: The space type.
    """

    space_type = x._tlm_adjoint__function_interface_space_type()
    return relative_space_type(space_type, rel_space_type)


def function_dtype(x):
    """
    :arg x: A function.
    :returns: The data type associated with the function. Typically
        :class:`numpy.double` or :class:`numpy.cdouble`.
    """

    return x._tlm_adjoint__function_interface_dtype()


_function_id_counter = 0


def new_function_id():
    global _function_id_counter
    function_id = _function_id_counter
    _function_id_counter += 1
    return function_id


def function_id(x):
    """Return the :class:`int` ID associated with a function.

    Note that two functions share the same ID if they represent the same
    variable -- for example if one function represents both a variable and
    stores a value, and a second the same variable with no value (i.e. is a
    'replacement'), then the two functions share the same ID.

    :arg x: The function.
    :returns: The :class:`int` ID.
    """

    return x._tlm_adjoint__function_interface_id()


def function_name(x):
    """
    :arg x: A function.
    :returns: The :class:`str` name of the function.
    """

    return x._tlm_adjoint__function_interface_name()


def function_state(x):
    """Return the value of the state counter for a function. Updated when the
    value of the function changes.

    :arg x: The function.
    :returns: The :class:`int` state value.
    """

    return x._tlm_adjoint__function_interface_state()


def function_update_state(*X):
    """Update the state counter for zero of more functions. Invalidates cache
    entries.

    :arg X: A :class:`tuple` of functions whose state value should be updated.
    """

    for x in X:
        x._tlm_adjoint__function_interface_update_state()
    function_update_caches(*X)


def function_is_static(x):
    """Return whether a function is flagged as 'static'.

    The 'static' flag is used when instantiating functions to set the default
    caching and checkpointing behaviour, but plays no other role.

    :arg x: The function.
    :returns: Whether the function is flagged as static.
    """

    return x._tlm_adjoint__function_interface_is_static()


def function_is_cached(x):
    """Return whether results involving this function may be cached.

    :arg x: The function.
    :returns: Whether results involving the function may be cached.
    """

    return x._tlm_adjoint__function_interface_is_cached()


def function_is_checkpointed(x):
    """Return whether the function is 'checkpointed', meaning that a
    :class:`tlm_adjoint.checkpointing.CheckpointStorage` stores this function
    by value. If not 'checkpointed' then a
    :class:`tlm_adjoint.checkpointing.CheckpointStorage` stores this function
    by reference.

    Only functions which are 'checkpointed' may appear as the solution of
    equations.

    :arg x: The function.
    :returns: Whether the function is 'checkpointed'.
    """

    return x._tlm_adjoint__function_interface_is_checkpointed()


def function_caches(x):
    """Return the :class:`tlm_adjoint.caches.Caches` associated with a
    function.

    :arg x: The function.
    :returns: The :class:`tlm_adjoint.caches.Caches` associated with the
        function.
    """

    return x._tlm_adjoint__function_interface_caches()


def function_update_caches(*X, value=None):
    """Check for cache invalidation associated with a possible change in value.

    :arg X: A :class:`tuple` of functions whose value may have changed.
    :arg value: A function or a :class:`Sequence` of functions defining the
        possible new values. `X` is used if not supplied.
    """

    if value is None:
        for x in X:
            if function_is_replacement(x):
                raise TypeError("value required")
            function_caches(x).update(x)
    else:
        if is_function(value):
            value = (value,)
        assert len(X) == len(value)
        for x, x_value in zip(X, value):
            function_caches(x).update(x_value)


def function_zero(x):
    """Zero a function.

    :arg x: The function.
    """

    x._tlm_adjoint__function_interface_zero()
    function_update_state(x)


def function_assign(x, y):
    """Perform an assignment `x = y`.

    :arg x: A function.
    :arg y: A function.
    """

    if is_function(y):
        check_space_types(x, y)
    x._tlm_adjoint__function_interface_assign(y)
    function_update_state(x)


def function_axpy(y, alpha, x, /):
    """Perform an in-place addition `y += alpha * x`.

    :arg y: A function.
    :arg alpha: A scalar.
    :arg x: A function.
    """

    if is_function(x):
        check_space_types(y, x)
    y._tlm_adjoint__function_interface_axpy(alpha, x)
    function_update_state(y)


def function_inner(x, y):
    """Compute the :math:`l_2` inner product of the degrees of freedom vectors
    associated with `x` and `y`. By convention if `y` is in the conjugate dual
    space associated with `x`, this returns the complex conjugate of the
    functional associated with `y` evaluated at `x`.

    :arg x: A function.
    :arg y: A function.
    :returns: The result of the inner product.
    """

    if is_function(y):
        check_space_types_conjugate_dual(x, y)
    return x._tlm_adjoint__function_interface_inner(y)


def function_sum(x):
    """Compute the sum of all degrees of freedom associated with a function.

    :arg x: The function.
    :returns: The sum of the degrees of freedom associated with `x`.
    """

    return x._tlm_adjoint__function_interface_sum()


def function_linf_norm(x):
    r"""Compute the :math:`l_\infty` norm of the degrees of freedom vector
    associated with a function.

    :arg x: The function.
    :returns: The :math:`l_\infty` norm of the degrees of freedom vector.
    """

    return x._tlm_adjoint__function_interface_linf_norm()


def function_local_size(x):
    """Return the process local number of degrees of freedom associated with
    a function. This is the number of 'owned' degrees of freedom.

    :arg x: The function.
    :returns: The process local number of degrees of freedom for the function.
    """

    return x._tlm_adjoint__function_interface_local_size()


def function_global_size(x):
    """Return the global number of degrees of freedom associated with a
    function. This is the total number of 'owned' degrees of freedom, summed
    across all processes.

    :arg x: The function.
    :returns: The global number of degrees of freedom for the function.
    """

    return x._tlm_adjoint__function_interface_global_size()


def function_local_indices(x):
    """Return the indices of process local degrees of freedom associated with
    a function.

    :arg x: The function.
    :returns: An :class:`Iterable`, yielding the indices of the process local
        elements.
    """

    return x._tlm_adjoint__function_interface_local_indices()


def function_get_values(x):
    """Return a copy of the process local degrees of freedom vector associated
    with a function.

    :arg x: The function.
    :returns: A :class:`numpy.ndarray` containing the degrees of freedom.
    """

    return x._tlm_adjoint__function_interface_get_values()


def function_set_values(x, values):
    """Set the process local degrees of freedom vector associated with a
    function.

    :arg x: The function.
    :arg values: A :class:`numpy.ndarray` containing the degrees of freedom
        values.
    """

    x._tlm_adjoint__function_interface_set_values(values)
    function_update_state(x)


def function_new(x, *, name=None, static=False, cache=None, checkpoint=None,
                 rel_space_type="primal"):
    """Return a new function defined using the same space as `x`.

    :arg x: A function.
    :arg name: A :class:`str` name for the new function.
    :arg static: Defines the default value for `cache` and `checkpoint`.
    :arg cache: Defines whether results involving the new function may be
        cached. Default `static`.
    :arg checkpoint: Defines whether a
        :class:`tlm_adjoint.checkpointing.CheckpointStorage` should store the
        new function by value (`checkpoint=True`) or reference
        (`checkpoint=False`). Default `not static`.
    :arg rel_space_type: Defines the space type of the new function, relative
        to the space type of `x`.
    :returns: The new function.
    """

    if rel_space_type not in ["primal", "conjugate", "dual", "conjugate_dual"]:
        raise ValueError("Invalid relative space type")
    return x._tlm_adjoint__function_interface_new(
        name=name, static=static, cache=cache, checkpoint=checkpoint,
        rel_space_type=rel_space_type)


def function_new_conjugate(x, *, name=None, static=False, cache=None,
                           checkpoint=None):
    """Return a new conjugate function. See :func:`function_new`.

    :returns: A new function defined using the same space as `x`, with space
        type conjugate to the space type for `x`.
    """

    return function_new(x, name=name, static=static, cache=cache,
                        checkpoint=checkpoint,
                        rel_space_type="conjugate")


def function_new_dual(x, *, name=None, static=False, cache=None,
                      checkpoint=None):
    """Return a new dual function. See :func:`function_new`.

    :returns: A new function defined using the same space as `x`, with space
        type dual to the space type for `x`.
    """

    return function_new(x, name=name, static=static, cache=cache,
                        checkpoint=checkpoint,
                        rel_space_type="dual")


def function_new_conjugate_dual(x, *, name=None, static=False, cache=None,
                                checkpoint=None):
    """Return a new conjugate dual function. See :func:`function_new`.

    :returns: A new function defined using the same space as `x`, with space
        type conjugate dual to the space type for `x`.
    """

    return function_new(x, name=name, static=static, cache=cache,
                        checkpoint=checkpoint,
                        rel_space_type="conjugate_dual")


def function_copy(x, *, name=None, static=False, cache=None, checkpoint=None):
    """Copy a function. See :func:`function_new`.

    :returns: The copied function.
    """

    return x._tlm_adjoint__function_interface_copy(
        name=name, static=static, cache=cache, checkpoint=checkpoint)


def function_new_tangent_linear(x, *, name=None):
    if function_is_checkpointed(x):
        return function_new(x, name=name, static=function_is_static(x),
                            cache=function_is_cached(x),
                            checkpoint=True)
    else:
        return None


def function_replacement(x):
    """Return a function, associated with the same variable as `x`, but
    possibly without a value.

    :arg x: The function.
    :returns: A function which symbolically represents the same variable as
        `x`, but which may not store a value. May return `x` itself.
    """

    return x._tlm_adjoint__function_interface_replacement()


def function_is_replacement(x):
    """Return whether a function is a 'replacement', meaning that it has no
    associated value.

    :arg x: The function.
    :returns: Whether `x` is a replacement.
    """

    return x._tlm_adjoint__function_interface_is_replacement()


def function_is_scalar(x):
    """Return whether a function defines a scalar variable.

    :arg x: The function.
    :returns: Whether `x` defines a scalar variable.
    """

    return x._tlm_adjoint__function_interface_is_scalar()


def function_scalar_value(x):
    """If `x` defines a scalar variable, returns its value.

    :arg x: The function, defining a scalar variable.
    :returns: The scalar value.
    """

    if not function_is_scalar(x):
        raise ValueError("Invalid function")
    return x._tlm_adjoint__function_interface_scalar_value()


def function_is_alias(x):
    """Return whether a function is an 'alias', meaning part or all of the
    degree of freedom vector associated with the function is shared with some
    different aliased function. A function may not appear as an equation
    dependency if it is an alias.

    :arg x: The function.
    :returns: Whether the function is an alias.
    """

    return x._tlm_adjoint__function_interface_is_alias()


@functools.singledispatch
def subtract_adjoint_derivative_action(x, y):
    """Subtract an adjoint right-hand-side contribution defined by `y` from
    the right-hand-side defined by `x`.

    :arg x: A function storing the adjoint right-hand-side.
    :arg y: A contribution to subtract from the adjoint right-hand-side. An
        :meth:`tlm_adjoint.equation.Equation.adjoint_derivative_action` return
        value. Valid types depend upon the backend used. Typically this will be
        a function, or a two element :class:`tuple` `(alpha, F)`, where `alpha`
        is a scalar and `F` a function, with the value to subtract defined by
        the product of `alpha` and `F`.
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
                        and isinstance(y[0], (int, np.integer,
                                              float, np.floating,
                                              complex, np.complexfloating)):
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
            return fn(x, alpha, y)


def subtract_adjoint_derivative_action_base(x, alpha, y):
    if y is None:
        pass
    elif is_function(y):
        check_space_types(x, y)
        function_axpy(x, -alpha, y)
    else:
        raise NotImplementedError("Unexpected case encountered")


_finalize_adjoint_derivative_action = []


def finalize_adjoint_derivative_action(x):
    for fn in _finalize_adjoint_derivative_action:
        fn(x)


def register_finalize_adjoint_derivative_action(fn):
    _finalize_adjoint_derivative_action.append(fn)


@functools.singledispatch
def functional_term_eq(x, term):
    raise NotImplementedError("Unexpected case encountered")


def register_functional_term_eq(x_cls, term_cls, fn, *,
                                replace=False):
    if not isinstance(x_cls, Sequence):
        x_cls = (x_cls,)
    if not isinstance(term_cls, Sequence):
        term_cls = (term_cls,)
    for x_cls, term_cls in itertools.product(x_cls, term_cls):
        if x_cls not in functional_term_eq.registry:
            @functools.singledispatch
            def _x_fn(term, x):
                raise NotImplementedError("Unexpected case encountered")

            @functional_term_eq.register(x_cls)
            def x_fn(x, term):
                return functional_term_eq.dispatch(type(x))._tlm_adjoint__x_fn(term, x)  # noqa: E501

            x_fn._tlm_adjoint__x_fn = _x_fn

        _x_fn = functional_term_eq.registry[x_cls]._tlm_adjoint__x_fn
        if term_cls in _x_fn.registry and not replace:
            raise RuntimeError("Case already registered")

        @_x_fn.register(term_cls)
        def wrapped_fn(term, x):
            return fn(x, term)


_logger = logging.getLogger("tlm_adjoint")
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
