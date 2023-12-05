from tlm_adjoint.fenics import *

from .test_base import *

import os
import pytest

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.fenics
@pytest.mark.example
@seed_test
def test_diffusion(setup_test, test_leaks):
    run_example(os.path.join("diffusion", "diffusion.py"))


@pytest.mark.fenics
@pytest.mark.example
@seed_test
def test_poisson(setup_test, test_leaks):
    run_example(os.path.join("poisson", "poisson.py"))
