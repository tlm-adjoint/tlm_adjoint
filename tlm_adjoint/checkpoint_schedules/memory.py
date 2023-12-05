from .schedule import (
    CheckpointSchedule, Configure, Forward, Reverse, EndForward, EndReverse)

import sys

__all__ = \
    [
        "MemoryCheckpointSchedule"
    ]


class MemoryCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule where all forward restart and non-linear
    dependency data are stored in memory.

    Online, unlimited adjoint calculations permitted.
    """

    def iter(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")
        yield Configure(True, True)

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1)

        yield EndForward()

        while True:
            if self._r == 0:
                # Reverse

                self._r = self._max_n
                yield Reverse(self._max_n, 0)
            elif self._r == self._max_n:
                # Reset for new reverse

                self._r = 0
                yield EndReverse(False)
            else:
                raise RuntimeError("Invalid checkpointing state")

    @property
    def is_exhausted(self):
        return False

    @property
    def uses_disk_storage(self):
        return False
