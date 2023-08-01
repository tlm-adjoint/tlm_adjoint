#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tlm_adjoint.override import override_function
import tlm_adjoint.interface


__all__ = \
    [
    ]


def space_type_error(orig, orig_args, msg, *, stacklevel=1):
    if tlm_adjoint.interface._check_space_types == 0:
        raise RuntimeError(f"{msg}")


tlm_adjoint.interface.space_type_warning = override_function(
    tlm_adjoint.interface.space_type_warning)(space_type_error)
