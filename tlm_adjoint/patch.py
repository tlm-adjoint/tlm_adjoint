from .manager import (
    annotation_enabled, manager_disabled, paused_manager, tlm_enabled)

import functools

__all__ = []

_PATCH_PROPERTY_NAME_KEY = "_tlm_adjoint__patch_property_name_%i"
_PATCH_PROPERTY_COUNTER_KEY = "_tlm_adjoint__patch_property_counter"


def patch_method(cls, name):
    orig = getattr(cls, name)

    def wrapper(patch):
        @functools.wraps(orig)
        def wrapped_patch(self, *args, **kwargs):
            return patch(self, orig,
                         lambda: orig(self, *args, **kwargs),
                         *args, **kwargs)

        setattr(cls, name, wrapped_patch)
        return wrapped_patch

    return wrapper


def patch_property(cls, name, *,
                   fset=None, cached=False):
    orig = getattr(cls, name)

    def wrapper(patch):
        if fset is None:
            wrapped_fset = None
        else:
            @functools.wraps(fset)
            def wrapped_fset(self, *args, **kwargs):
                return fset(self, orig.fset,
                            lambda: orig.fset(self, *args, **kwargs),
                            *args, **kwargs)

        if cached:
            if fset is not None:
                raise TypeError("Cannot use fset with a cached_property")
            property_decorator = functools.cached_property
        else:
            def property_decorator(arg):
                return property(arg, fset=wrapped_fset)

        @property_decorator
        @functools.wraps(orig)
        def wrapped_patch(self):
            return patch(self, lambda: orig.__get__(self, type(self)))

        setattr(cls, name, wrapped_patch)
        if cached:
            patch_counter = getattr(cls, _PATCH_PROPERTY_COUNTER_KEY, -1) + 1
            setattr(cls, _PATCH_PROPERTY_COUNTER_KEY, patch_counter)
            wrapped_patch.__set_name__(
                wrapped_patch,
                _PATCH_PROPERTY_NAME_KEY % patch_counter)
        return wrapped_patch

    return wrapper


def patch_function(orig):
    def wrapper(patch):
        @functools.wraps(orig)
        def wrapped_patch(*args, **kwargs):
            return patch(orig,
                         lambda: orig(*args, **kwargs),
                         *args, **kwargs)

        return wrapped_patch

    return wrapper


def add_manager_controls(orig):
    def wrapped_orig(*args, annotate=None, tlm=None, **kwargs):
        if annotate is None or annotate:
            annotate = annotation_enabled()
        if tlm is None or tlm:
            tlm = tlm_enabled()
        with paused_manager(annotate=not annotate, tlm=not tlm):
            return orig(*args, **kwargs)

    return wrapped_orig


def manager_method(cls, name, *,
                   patch_without_manager=False,
                   pre_call=None, post_call=None):
    orig = getattr(cls, name)

    def wrapper(patch):
        @manager_disabled()
        @functools.wraps(orig)
        def wrapped_orig(self, *args, **kwargs):
            if pre_call is not None:
                args, kwargs = pre_call(self, *args, **kwargs)
            return_value = orig(self, *args, **kwargs)
            if post_call is not None:
                return_value = post_call(self, return_value, *args, **kwargs)
            return return_value

        def wrapped_patch(self, *args, annotate=None, tlm=None, **kwargs):
            if annotate is None or annotate:
                annotate = annotation_enabled()
            if tlm is None or tlm:
                tlm = tlm_enabled()
            if annotate or tlm or patch_without_manager:
                with paused_manager(annotate=not annotate, tlm=not tlm):
                    return patch(self, wrapped_orig,
                                 lambda: wrapped_orig(self, *args, **kwargs),
                                 *args, **kwargs)
            else:
                return wrapped_orig(self, *args, **kwargs)

        setattr(cls, name, wrapped_patch)
        return wrapped_patch

    return wrapper
