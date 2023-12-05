import functools
import gc

__all__ = \
    [
        "Alias",
        "WeakAlias",

        "gc_disabled"
    ]


def gc_disabled(fn):
    """Decorator to disable the Python garbage collector.

    :arg fn: A callable for which the Python garbage collector should be
        disabled.
    :returns: A callable for which the Python garbage collector is disabled.
    """

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            return fn(*args, **kwargs)
        finally:
            if gc_enabled:
                gc.enable()
    return wrapped_fn


class Alias:
    """An alias to an object. Holds a reference to the original object.

    :arg obj: Object to alias.
    """

    def __init__(self, obj):
        if isinstance(obj, Alias):
            raise TypeError("Cannot alias Alias")
        super().__setattr__("_tlm_adjoint__alias", obj)

    def __new__(cls, obj, *args, **kwargs):
        obj_cls = type(obj)

        class Alias(cls, obj_cls):
            def __new__(cls, *args, **kwargs):
                return obj_cls(*args, **kwargs)

        Alias.__name__ = f"{obj_cls.__name__:s}Alias"
        return super().__new__(Alias)

    def __getattr__(self, key):
        return getattr(self._tlm_adjoint__alias, key)

    def __setattr__(self, key, value):
        setattr(self._tlm_adjoint__alias, key, value)

    def __delattr__(self, key):
        delattr(self._tlm_adjoint__alias, key)

    def __dir__(self):
        return dir(self._tlm_adjoint__alias)


class WeakAlias:
    """An alias to an object. Does *not* hold a reference to the original
    object.

    Intended to be used in combination with `weakref.finalize`, so that object
    attributes may be updated when the original object is destroyed, but object
    methods may still be called after it is destroyed.

    :arg obj: Object to alias.
    """

    def __init__(self, obj):
        if hasattr(obj, "__slots__"):
            raise TypeError("Cannot alias object with __slots__ attribute")
        if isinstance(obj, WeakAlias):
            raise TypeError("Cannot alias WeakAlias")
        if len(self.__dict__) > 0:
            raise RuntimeError("Unexpected __dict__ entries")
        self.__dict__ = obj.__dict__

    def __new__(cls, obj, *args, **kwargs):
        obj_cls = type(obj)

        class WeakAlias(cls, obj_cls):
            pass

        WeakAlias.__name__ = f"{obj_cls.__name__:s}WeakAlias"
        return super().__new__(WeakAlias)
