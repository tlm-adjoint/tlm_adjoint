from firedrake import *
from firedrake.assemble import \
    BaseFormAssembler, ExplicitMatrixAssembler as TwoFormAssembler, \
    OneFormAssembler  # noqa: F401
from firedrake.projection import ProjectorBase  # noqa: F401
from firedrake.utils import complex_mode  # noqa: F401
import firedrake
from firedrake.petsc import \
    garbage_cleanup as backend_garbage_cleanup  # noqa: F401

backend = "Firedrake"

backend_RealType = firedrake.utils.RealType.type
backend_ScalarType = firedrake.utils.ScalarType.type
if not issubclass(backend_RealType, np.floating):
    raise ImportError(f"Invalid backend real type: {backend_RealType}")
if complex_mode:
    if not issubclass(backend_ScalarType, np.complexfloating):
        raise ImportError(f"Invalid backend scalar type: {backend_ScalarType}")
else:
    if not issubclass(backend_ScalarType, np.floating):
        raise ImportError(f"Invalid backend scalar type: {backend_ScalarType}")

extract_args = firedrake.solving._extract_args

backend_Cofunction = Cofunction
backend_CofunctionSpace = firedrake.functionspaceimpl.FiredrakeDualSpace
backend_Constant = Constant
backend_DirichletBC = DirichletBC
backend_Function = Function
backend_FunctionSpace = firedrake.functionspaceimpl.WithGeometry
backend_Matrix = firedrake.matrix.Matrix
backend_Vector = Vector
backend_action = action
backend_assemble = assemble
backend_interpolate = interpolate
backend_project = project
backend_solve = solve
