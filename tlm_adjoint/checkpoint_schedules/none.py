#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .schedule import CheckpointSchedule, Configure, Forward, EndForward

import sys

__all__ = \
    [
        "NoneCheckpointSchedule"
    ]


class NoneCheckpointSchedule(CheckpointSchedule):
    """A checkpointing schedule for the case where no adjoint calculation is
    performed.

    Online, zero adjoint calculations permitted.
    """

    def __init__(self):
        super().__init__()
        self._exhausted = False

    def iter(self):
        # Forward

        if self._max_n is not None:
            # Unexpected finalize
            raise RuntimeError("Invalid checkpointing state")
        yield Configure(False, False)

        while self._max_n is None:
            n0 = self._n
            n1 = n0 + sys.maxsize
            self._n = n1
            yield Forward(n0, n1)

        self._exhausted = True
        yield EndForward()

    @property
    def is_exhausted(self):
        return self._exhausted

    @property
    def uses_disk_storage(self):
        return False
