from checkpoint_schedules import StorageType
from checkpoint_schedules import (
    MultistageCheckpointSchedule as _MultistageCheckpointSchedule,
    TwoLevelCheckpointSchedule as _TwoLevelCheckpointSchedule)

from .translation import translation

import functools

__all__ = \
    [
        "MultistageCheckpointSchedule",
        "TwoLevelCheckpointSchedule"
    ]


def cache_step(fn):
    _cache = {}

    @functools.wraps(fn)
    def wrapped_fn(n, s):
        # Avoid some cache misses
        s = min(s, n - 1)
        if (n, s) not in _cache:
            _cache[(n, s)] = fn(n, s)
        return _cache[(n, s)]

    return wrapped_fn


@cache_step
def optimal_extra_steps(n, s):
    if n <= 0:
        raise ValueError("Invalid number of steps")
    if s < min(1, n - 1) or s > n - 1:
        raise ValueError("Invalid number of snapshots")

    if n == 1:
        return 0
    # Equation (2) of
    #   A. Griewank and A. Walther, "Algorithm 799: Revolve: An implementation
    #   of checkpointing for the reverse or adjoint mode of computational
    #   differentiation", ACM Transactions on Mathematical Software, 26(1), pp.
    #   19--45, 2000
    elif s == 1:
        return n * (n - 1) // 2
    else:
        m = None
        for i in range(1, n):
            m1 = (i
                  + optimal_extra_steps(i, s)
                  + optimal_extra_steps(n - i, s - 1))
            if m is None or m1 < m:
                m = m1
        if m is None:
            raise RuntimeError("Failed to determine number of extra steps")
        return m


def optimal_steps(n, s):
    return n + optimal_extra_steps(n, s)


class MultistageCheckpointSchedule(translation(_MultistageCheckpointSchedule)):
    """A binomial checkpointing schedule using the approach described in

        - Andreas Griewank and Andrea Walther, 'Algorithm 799: revolve: an
          implementation of checkpointing for the reverse or adjoint mode of
          computational differentiation', ACM Transactions on Mathematical
          Software, 26(1), pp. 19--45, 2000, doi: 10.1145/347837.347846

    hereafter referred to as GW2000.

    Uses a 'MultiStage' distribution of checkpoints between RAM and disk
    equivalent to that described in

        - Philipp Stumm and Andrea Walther, 'MultiStage approaches for optimal
          offline checkpointing', SIAM Journal on Scientific Computing, 31(3),
          pp. 1946--1967, 2009, doi: 10.1137/080718036

    The distribution between RAM and disk is determined using an initial run of
    the schedule.

    Offline, one adjoint calculation permitted.

    :arg max_n: The number of forward steps in the initial forward calculation.
    :arg snapshots_in_ram: The maximum number of forward restart checkpoints
        to store in memory.
    :arg snapshots_on_disk: The maximum number of forward restart checkpoints
        to store on disk.
    :arg trajectory: When advancing `n` forward steps with `s` checkpointing
        units available there are in general multiple solutions to the problem
        of determining the number of forward steps to advance before storing
        a new forward restart checkpoint -- see Fig. 4 of GW2000. This argument
        selects a solution:

            - `'revolve'`: The standard revolve solution, as specified in the
              equation at the bottom of p. 34 of GW2000.
            - `'maximum'`: The maximum possible number of steps, corresponding
              to the maximum step size compatible with the optimal region in
              Fig. 4 of GW2000.

    The argument names `snaps_in_ram` and `snaps_on_disk` originate from the
    corresponding arguments for the `dolfin_adjoint.solving.adj_checkpointing`
    function in dolfin-adjoint (see e.g. version 2017.1.0).
    """


class TwoLevelCheckpointSchedule(translation(_TwoLevelCheckpointSchedule)):
    """A two-level mixed periodic/binomial checkpointing schedule using the
    approach described in

        - Gavin J. Pringle, Daniel C. Jones, Sudipta Goswami, Sri Hari Krishna
          Narayanan, and Daniel Goldberg, 'Providing the ARCHER community with
          adjoint modelling tools for high-performance oceanographic and
          cryospheric computation', version 1.1, EPCC, 2016

    and in the supporting information for

        - D. N. Goldberg, T. A. Smith, S. H. K. Narayanan, P. Heimbach, and M.
          Morlighem, 'Bathymetric influences on Antarctic ice-shelf melt
          rates', Journal of Geophysical Research: Oceans, 125(11),
          e2020JC016370, 2020, doi: 10.1029/2020JC016370

    Online, unlimited adjoint calculations permitted.

    :arg period: Forward restart checkpoints are stored to disk every `period`
        forward steps in the initial forward calculation.
    :arg binomial_snapshots: The maximum number of additional forward restart
        checkpointing units to use when advancing the adjoint between periodic
        disk checkpoints.
    :arg binomial_storage: The storage to use for the additional forward
        restart checkpoints generated when advancing the adjoint between
        periodic disk checkpoints. Either `'RAM'` or `'disk'`.
    :arg binomial_trajectory: See the `trajectory` constructor argument for
        :class:`.MultistageCheckpointSchedule`.
    """

    def __init__(self, period, binomial_snapshots, *,
                 binomial_storage="disk",
                 binomial_trajectory="maximum"):
        super().__init__(
            period, binomial_snapshots,
            binomial_storage={"RAM": StorageType.RAM,
                              "disk": StorageType.DISK}[binomial_storage],
            binomial_trajectory=binomial_trajectory)
