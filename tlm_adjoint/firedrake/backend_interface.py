from ..interface import register_garbage_cleanup

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc
import pyop2

__all__ = []


def garbage_cleanup_internal_comm(comm):
    if not MPI.Is_finalized() and not PETSc.Sys.isFinalized() \
            and not pyop2.mpi.PYOP2_FINALIZED \
            and comm.py2f() != MPI.COMM_NULL.py2f():
        if pyop2.mpi.is_pyop2_comm(comm):
            raise RuntimeError("Should not call garbage_cleanup directly on a "
                               "PyOP2 communicator")
        internal_comm = comm.Get_attr(pyop2.mpi.innercomm_keyval)
        if internal_comm is not None and internal_comm.py2f() != MPI.COMM_NULL.py2f():  # noqa: E501
            PETSc.garbage_cleanup(internal_comm)


register_garbage_cleanup(garbage_cleanup_internal_comm)
