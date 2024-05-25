# flake8: noqa

def _init():
    try:
        import petsc4py
    except ModuleNotFoundError:
        petsc4py = None
    try:
        import slepc4py
    except ModuleNotFoundError:
        slepc4py = None
    import sys

    if slepc4py is not None:
        slepc4py.init(sys.argv)
    if petsc4py is not None:
        petsc4py.init(sys.argv)


_init()
del _init

from .cached_hessian import *
from .caches import *
from .eigendecomposition import *
from .equation import *
from .equations import *
from .fixed_point import *
from .functional import *
from .hessian import *
from .hessian_system import *
from .instructions import *
from .interface import *
from .jax import *
from .linear_equation import *
from .manager import *
from .optimization import *
from .overloaded_float import *
from .patch import *
from .storage import *
from .tangent_linear import *
from .tlm_adjoint import *
from .verification import *

from .hessian import GeneralGaussNewton as GaussNewton
from .hessian import GeneralHessian as Hessian
