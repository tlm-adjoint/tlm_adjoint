from checkpoint_schedules import (
    SingleMemoryStorageSchedule as _SingleMemoryStorageSchedule)

from .translation import translation

__all__ = \
    [
        "MemoryCheckpointSchedule"
    ]


class MemoryCheckpointSchedule(translation(_SingleMemoryStorageSchedule)):
    """A checkpointing schedule where all non-linear dependency data are stored
    in memory.

    Online, unlimited adjoint calculations permitted.
    """
