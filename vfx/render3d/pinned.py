"""Pinned host staging buffers for small per-draw uploads (camera basis etc.) that are safe however
many draws are queued without a host sync.

A pinned buffer handed to `dev.copy_(buf, non_blocking=True)` must not be rewritten before that copy
has executed on the GPU. The old fixed ring of 4 buffers was overwritten when more than 4 draws were
queued ahead of the GPU, so some frames silently rendered another draw's camera (reported by the data
engineer). Here every slot remembers a CUDA event recorded right after its copy was enqueued; a slot is
reused only when its event has completed (`query()`, no sync). If all slots are still in flight, a new
slot is allocated (no sync) up to `max_slots`; beyond that the oldest slot's event is waited on (a host
sync — never a wrong frame).
"""
from __future__ import annotations

import torch


class PinnedRing:
    def __init__(self, shape, dtype=torch.float32, slots: int = 4, max_slots: int = 256):
        self.shape, self.dtype, self.max_slots = tuple(shape) if not isinstance(shape, int) else (shape,), dtype, max_slots
        self.bufs = [torch.zeros(self.shape, dtype=dtype).pin_memory() for _ in range(slots)]
        self.events: list = [None] * slots
        self.i = 0
        self.waits = 0                    # host syncs forced by a full ring (diagnostics)
        self._cur = None

    def acquire(self) -> torch.Tensor:
        n = len(self.bufs)
        for k in range(n):
            j = (self.i + k) % n
            e = self.events[j]
            if e is None or e.query():
                self.i, self._cur = (j + 1) % n, j
                return self.bufs[j]
        if n < self.max_slots:
            self.bufs.append(torch.zeros(self.shape, dtype=self.dtype).pin_memory())
            self.events.append(None)
            self._cur = n
            return self.bufs[n]
        j = self.i                        # every slot in flight and the ring is at its cap: wait
        self.events[j].synchronize()
        self.waits += 1
        self.i, self._cur = (j + 1) % n, j
        return self.bufs[j]

    def record(self, stream=None) -> None:
        """Call right after the non_blocking copy from the acquired buffer was enqueued."""
        e = torch.cuda.Event()
        e.record(stream)
        self.events[self._cur] = e
