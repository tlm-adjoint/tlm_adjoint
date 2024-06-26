from checkpoint_schedules import StorageType
from checkpoint_schedules import (
    MixedCheckpointSchedule as _MixedCheckpointSchedule)
from checkpoint_schedules.mixed import (  # noqa: F401
    mixed_step_memoization, mixed_steps_tabulation,
    optimal_steps_mixed as optimal_steps)

from .translation import translation

__all__ = \
    [
        "MixedCheckpointSchedule"
    ]


class MixedCheckpointSchedule(translation(_MixedCheckpointSchedule)):
    """A checkpointing schedule which mixes storage of forward restart data and
    non-linear dependency data in checkpointing units. Assumes that the data
    required to restart the forward has the same size as the data required to
    advance the adjoint over a step.

    An updated version of the algorithm described in

        - James R. Maddison, 'On the implementation of checkpointing with
          high-level algorithmic differentiation',
          https://arxiv.org/abs/2305.09568v1, 2023

    Offline, one adjoint calculation permitted.

    :arg max_n: The number of forward steps in the initial forward calculation.
    :arg snapshots: The number of available checkpointing units.
    :arg storage: Checkpointing unit storage location. Either `'RAM'` or
        `'disk'`.
    """

    def __init__(self, max_n, snapshots, *, storage="disk"):
        super().__init__(
            max_n, snapshots,
            storage={"RAM": StorageType.RAM,
                     "disk": StorageType.DISK}[storage])
