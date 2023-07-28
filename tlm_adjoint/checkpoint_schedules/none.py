from checkpoint_schedules import (
    NoneCheckpointSchedule as _NoneCheckpointSchedule)

from .translation import translation

__all__ = \
    [
        "NoneCheckpointSchedule"
    ]


class NoneCheckpointSchedule(translation(_NoneCheckpointSchedule)):
    """A checkpointing schedule for the case where no adjoint calculation is
    performed.

    Online, zero adjoint calculations permitted.
    """
