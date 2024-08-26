from tlm_adjoint import DEFAULT_COMM, VectorEquation
from tlm_adjoint.patch import patch_function, patch_method

import functools
import hashlib
import inspect
try:
    import jax
except ModuleNotFoundError:
    jax = None
import json
import numpy as np
from operator import itemgetter
import os
import pytest
import runpy

__all__ = \
    [
        "chdir_tmp_path",
        "jax_tlm_config",
        "seed_test",
        "tmp_path"
    ]


if jax is not None:
    jax.config.update("jax_enable_x64", True)


def seed_test(fn):
    @patch_function(fn)
    def wrapped_fn(orig, orig_args, *args, **kwargs):
        args_ = list(args)
        for i, arg in enumerate(args_):
            if callable(arg):
                args_[i] = arg.__name__
        args = tuple(args_)
        if "tmp_path" in inspect.signature(fn).parameters:
            # Raises an error if tmp_path is a positional argument
            del kwargs["tmp_path"]
        for key, value in kwargs.items():
            if isinstance(value, functools.partial):
                kwargs[key] = ("partial", value.func.__name__)
            elif callable(value):
                kwargs[key] = value.__name__

        h = hashlib.sha256()
        h.update(fn.__name__.encode("utf-8"))
        h.update(str(args).encode("utf-8"))
        h.update(str(sorted(kwargs.items(), key=itemgetter(0))).encode("utf-8"))  # noqa: E501
        seed = int(h.hexdigest(), 16) + DEFAULT_COMM.rank
        seed %= 2 ** 32
        np.random.seed(seed)

        return orig_args()

    return wrapped_fn


@pytest.fixture
def tmp_path(tmp_path):
    if DEFAULT_COMM.rank != 0:
        tmp_path = None
    return DEFAULT_COMM.bcast(tmp_path, root=0)


@pytest.fixture
def chdir_tmp_path(tmp_path):
    cwd = os.getcwd()
    os.chdir(tmp_path)

    yield

    os.chdir(cwd)


_jax_with_tlm = True


@patch_method(VectorEquation, "__init__")
def VectorEquation__init__(self, orig, orig_args, *args, with_tlm=None,
                           **kwargs):
    if with_tlm is None:
        with_tlm = _jax_with_tlm
    orig(self, *args, with_tlm=with_tlm, **kwargs)


@pytest.fixture(params=[True, False])
def jax_tlm_config(request):
    global _jax_with_tlm
    _jax_with_tlm = request.param
    yield
    _jax_with_tlm = True


def run_example(filename, *,
                clear_forward_globals=True):
    gl = runpy.run_path(filename)

    if clear_forward_globals:
        # Clear objects created by the script. Requires the script to define a
        # 'forward' function.
        gl["forward"].__globals__.clear()


def run_example_notebook(filename, tmp_path):
    if DEFAULT_COMM.size > 1:
        raise RuntimeError("Serial only")

    tmp_filename = os.path.join(tmp_path, "tmp.py")

    with open(filename) as nb_h, open(tmp_filename, "w") as py_h:
        nb = json.load(nb_h)
        if nb["metadata"]["language_info"]["name"] != "python":
            raise RuntimeError("Expected a Python notebook")

        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                for line in cell["source"]:
                    if not line.startswith("%matplotlib "):
                        py_h.write(line)
                py_h.write("\n\n")

    run_example(tmp_filename, clear_forward_globals=False)
