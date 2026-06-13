from __future__ import annotations

from collections import deque

from microserve.sequence import Sequence, Status


class Scheduler:
    """First-come-first-served continuous-batching scheduler.

    Owns:
      - `waiting`: FIFO queue of sequences awaiting admission
      - `running`: sequences currently in flight
      - `free_slots`: pool of unused KV-cache slot indices

    Does NOT run the model. Each step the Engine asks the scheduler what to
    do; the scheduler decides who to admit/retire and the Engine runs the
    forward pass and updates Sequence state.

    Policy: prefer prefill over decode whenever capacity allows. Matches the
    "prefill-priority" scheduling used by early vLLM versions — simple, gets
    new requests into the running set as fast as memory allows.
    """

    def __init__(self, max_batch_size: int):
        self.max_batch_size = max_batch_size
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        # Lower indices are handed out first; retired slots come back in
        # ascending order so the running set stays compact at low indices.
        self.free_slots: list[int] = list(range(max_batch_size))

    def add(self, seq: Sequence) -> None:
        seq.status = Status.WAITING
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def admit_one(self) -> Sequence | None:
        """Move one waiting sequence into running. Returns it, or None if
        either the waiting queue is empty or the running set is at capacity.
        """
        if not self.waiting or len(self.running) >= self.max_batch_size:
            return None
        seq = self.waiting.popleft()
        seq.status = Status.RUNNING
        seq.slot_idx = self.free_slots.pop(0)
        self.running.append(seq)
        return seq

    def retire(self, seq: Sequence) -> None:
        """Move a finished sequence out of running, returning its slot to the pool."""
        seq.status = Status.FINISHED
        if seq in self.running:
            self.running.remove(seq)
        if seq.slot_idx >= 0:
            self.free_slots.append(seq.slot_idx)
            self.free_slots.sort()
            seq.slot_idx = -1
