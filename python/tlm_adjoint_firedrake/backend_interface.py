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

from .backend import *
from .backend_code_generator_interface import copy_parameters_dict, \
    is_real_function

from .caches import Function, ReplacementFunction, clear_caches, form_neg, \
    function_is_cached, function_is_checkpointed, function_is_static, \
    function_name, function_space_new, function_state, function_tlm_depth, \
    function_update_state, is_function, replaced_function

import ufl
import sys

__all__ = \
    [
        "Function",
        "FunctionSpace",
        "RealFunctionSpace",
        "ReplacementFunction",
        "clear_caches",
        "copy_parameters_dict",
        "default_comm",
        "function_alias",
        "function_assign",
        "function_axpy",
        "function_comm",
        "function_copy",
        "function_finalize_adjoint_derivative_action",
        "function_get_values",
        "function_global_size",
        "function_inner",
        "function_is_cached",
        "function_is_checkpointed",
        "function_is_static",
        "function_linf_norm",
        "function_local_indices",
        "function_local_size",
        "function_max_value",
        "function_name",
        "function_new",
        "function_set_values",
        "function_space_id",
        "function_space_new",
        "function_state",
        "function_subtract_adjoint_derivative_action",
        "function_sum",
        "function_tangent_linear",
        "function_tlm_depth",
        "function_update_state",
        "function_zero",
        "info",
        "is_function",
        "replaced_function",
        "warning"
    ]


# def clear_caches(*deps):


def info(message):
    sys.stdout.write(f"{message:s}\n")
    sys.stdout.flush()


def warning(message):
    sys.stderr.write(f"{message:s}\n")
    sys.stderr.flush()


# def copy_parameters_dict(parameters):


# class FunctionSpace:


def function_space_id(space):
    return id(space)


# def function_space_new(space, name=None, static=False, cache=None,
#                        checkpoint=None, tlm_depth=0):


def RealFunctionSpace(comm=None):
    if comm is None:
        comm = default_comm()
    return FunctionSpace(UnitIntervalMesh(comm.size, comm=comm), "R", 0)


# class Function:
#     def function_space(self):
#     def id(self):


backend_Function.id = lambda self: self.count()


# class ReplacementFunction:
#     def __init__(self, x):
#     def function_space(self):
#     def id(self):


# def replaced_function(x):


# def is_function(x):


# def function_name(x):


# def function_state(x):


# def function_update_state(*X):


# def function_is_static(x):


# def function_is_cached(x):


# def function_is_checkpointed(x):


# def function_tlm_depth(x):


def function_copy(x, name=None, static=False, cache=None, checkpoint=None,
                  tlm_depth=0):
    # This is much faster than x.copy(deepcopy=True)
    y = function_new(x, name=name, static=static, cache=cache,
                     checkpoint=checkpoint, tlm_depth=tlm_depth)
    function_assign(y, x)
    return y


def function_assign(x, y):
    if isinstance(y, (int, float)):
        if is_real_function(x):
            # Work around Firedrake issue #1459
            x.dat.data[:] = y
        else:
            with x.dat.vec_wo as x_v:
                x_v.set(float(y))
    else:
        if is_real_function(x):
            # Work around Firedrake bug (related to issue #1459?)
            x.dat.data[:] = y.dat.data
        else:
            with x.dat.vec_wo as x_v, y.dat.vec_ro as y_v:
                y_v.copy(result=x_v)


def function_axpy(x, alpha, y):
    if is_real_function(x):
        # Work around Firedrake bug (related to issue #1459?)
        x.dat.data[:] += alpha * y.dat.data
    else:
        with x.dat.vec as x_v, y.dat.vec_ro as y_v:
            x_v.axpy(alpha, y_v)


def default_comm():
    import mpi4py.MPI as MPI
    return MPI.COMM_WORLD


def function_comm(x):
    return x.comm


def function_inner(x, y):
    with x.dat.vec_ro as x_v, y.dat.vec_ro as y_v:
        inner = x_v.dot(y_v)
    return inner


def function_local_size(x):
    with x.dat.vec_ro as x_v:
        local_size = x_v.getLocalSize()
    return local_size


def function_get_values(x):
    with x.dat.vec_ro as x_v:
        values = x_v.getArray(readonly=True)
    return values


def function_set_values(x, values):
    with x.dat.vec_wo as x_v:
        x_v.setArray(values)


def function_max_value(x):
    with x.dat.vec_ro as x_v:
        max = x_v.max()[1]
    return max


def function_sum(x):
    with x.dat.vec_ro as x_v:
        sum = x_v.sum()
    return sum


def function_linf_norm(x):
    import petsc4py.PETSc as PETSc
    with x.dat.vec_ro as x_v:
        linf_norm = x_v.norm(norm_type=PETSc.NormType.NORM_INFINITY)
    return linf_norm


def function_new(x, name=None, static=False, cache=None, checkpoint=None,
                 tlm_depth=0):
    return Function(x.function_space(), name=name, static=static,
                    cache=cache, checkpoint=checkpoint, tlm_depth=tlm_depth)


def function_tangent_linear(x, name=None):
    if hasattr(x, "tangent_linear"):
        return x.tangent_linear(name=name)
    elif function_is_static(x):
        return None
    else:
        return function_new(x, name=name, static=False,
                            cache=function_is_cached(x),
                            checkpoint=function_is_checkpointed(x),
                            tlm_depth=function_tlm_depth(x) + 1)


def function_alias(x):
    return Function(x.function_space(), name=function_name(x),
                    static=function_is_static(x), cache=function_is_cached(x),
                    checkpoint=function_is_checkpointed(x),
                    tlm_depth=function_tlm_depth(x), val=x.dat)


def function_zero(x):
    with x.dat.vec_wo as x_v:
        x_v.zeroEntries()


def function_global_size(x):
    with x.dat.vec_ro as x_v:
        size = x_v.getSize()
    return size


def function_local_indices(x):
    with x.dat.vec_ro as x_v:
        local_range = x_v.getOwnershipRange()
    return slice(*local_range)


def function_subtract_adjoint_derivative_action(x, y):
    if y is None:
        pass
    elif isinstance(y, tuple):
        alpha, y = y
        function_axpy(x, -alpha, y)
    elif isinstance(y, ufl.classes.Form):
        if hasattr(x, "_tlm_adjoint__adj_b"):
            x._tlm_adjoint__adj_b += form_neg(y)
        else:
            x._tlm_adjoint__adj_b = form_neg(y)
    else:
        function_axpy(x, -1.0, y)


def function_finalize_adjoint_derivative_action(x):
    if hasattr(x, "_tlm_adjoint__adj_b"):
        function_axpy(x, 1.0, backend_assemble(x._tlm_adjoint__adj_b))
        delattr(x, "_tlm_adjoint__adj_b")
