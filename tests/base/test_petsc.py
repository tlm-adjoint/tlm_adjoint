from tlm_adjoint import DEFAULT_COMM
from tlm_adjoint.petsc import PETScOptions

import itertools
try:
    import petsc4py.PETSc as PETSc
except ModuleNotFoundError:
    PETSc = None
import pytest

from .test_base import seed_test, setup_test  # noqa: F401

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")
pytestmark = pytest.mark.skipif(
    PETSc is None,
    reason="PETSc not available")


_count = itertools.count()


def flatten(options):
    return dict(PETScOptions(
        f"_tlm_adjoint__test_petsc_{next(_count):d}",
        options))


@pytest.mark.base
@seed_test
def test_flatten_petsc_options(setup_test):  # noqa: F811
    assert flatten({"zero": 0,
                    "one": 1}) == {"zero": "0",
                                   "one": "1"}
    assert flatten({"zero": 0,
                    "one": {"two": 2,
                            "three": 3}}) == {"zero": "0",
                                              "one_two": "2",
                                              "one_three": "3"}
    assert flatten({"zero": 0,
                    "one": {"two": {"three": 3, "four": 4},
                            "five": 5}}) == {"zero": "0",
                                             "one_two_three": "3",
                                             "one_two_four": "4",
                                             "one_five": "5"}


@pytest.mark.base
@seed_test
def test_flatten_petsc_options_duplicate_key(setup_test):  # noqa: F811
    with pytest.raises(ValueError):
        flatten({"zero_one": 1, "zero": {"one": 1}})
    with pytest.raises(ValueError):
        flatten({"zero": {"one": 1}, "zero_one": 1})
