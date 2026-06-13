"""Pure unit tests for the continuous-batching scheduler.

No model, no CUDA required. Tests the state machine of admit/retire/queueing
in isolation from the forward pass.
"""

from microserve.scheduler import Scheduler
from microserve.sequence import Sequence, Status


def _seq(prompt_len: int = 5) -> Sequence:
    return Sequence(prompt_token_ids=list(range(prompt_len)))


def test_scheduler_starts_empty():
    sched = Scheduler(max_batch_size=4)
    assert not sched.has_work()
    assert sched.admit_one() is None
    assert len(sched.waiting) == 0
    assert len(sched.running) == 0
    assert sched.free_slots == [0, 1, 2, 3]


def test_scheduler_admit_up_to_capacity():
    sched = Scheduler(max_batch_size=4)
    seqs = [_seq() for _ in range(6)]
    for s in seqs:
        sched.add(s)
    assert len(sched.waiting) == 6
    assert all(s.status is Status.WAITING for s in seqs)

    # Admit 4 (up to capacity)
    for _ in range(4):
        s = sched.admit_one()
        assert s is not None
        assert s.status is Status.RUNNING
        assert s.slot_idx >= 0

    assert len(sched.running) == 4
    assert len(sched.waiting) == 2
    assert sched.free_slots == []

    # No more capacity — 5th admit returns None even with waiting requests
    assert sched.admit_one() is None


def test_scheduler_slots_are_unique():
    sched = Scheduler(max_batch_size=4)
    for _ in range(4):
        sched.add(_seq())
    admitted = [sched.admit_one() for _ in range(4)]
    slots = [s.slot_idx for s in admitted if s is not None]
    assert len(set(slots)) == 4  # all distinct
    assert set(slots) == {0, 1, 2, 3}


def test_scheduler_slot_reuse_after_retire():
    sched = Scheduler(max_batch_size=2)
    a, b, c = _seq(), _seq(), _seq()
    sched.add(a)
    sched.add(b)
    sched.add(c)

    sched.admit_one()  # a
    sched.admit_one()  # b
    assert sched.admit_one() is None  # full

    freed_slot = a.slot_idx
    sched.retire(a)
    assert a.status is Status.FINISHED
    assert a.slot_idx == -1  # cleared
    assert freed_slot in sched.free_slots

    # Now c can be admitted into the freed slot
    admitted = sched.admit_one()
    assert admitted is c
    assert c.slot_idx == freed_slot


def test_scheduler_fifo_admission_order():
    sched = Scheduler(max_batch_size=10)
    seqs = [_seq() for _ in range(5)]
    for s in seqs:
        sched.add(s)
    admitted_order = []
    while True:
        s = sched.admit_one()
        if s is None:
            break
        admitted_order.append(s)
    assert admitted_order == seqs


def test_scheduler_has_work_lifecycle():
    sched = Scheduler(max_batch_size=2)
    assert not sched.has_work()
    s = _seq()
    sched.add(s)
    assert sched.has_work()  # waiting
    sched.admit_one()
    assert sched.has_work()  # running
    sched.retire(s)
    assert not sched.has_work()  # all gone


def test_scheduler_lowest_slot_first():
    """The scheduler always hands out the lowest free slot so the running set
    stays compact at small indices. Makes future batched indexing cleaner.
    """
    sched = Scheduler(max_batch_size=4)
    a, b, c = _seq(), _seq(), _seq()
    sched.add(a)
    sched.add(b)
    sched.add(c)
    sched.admit_one()
    sched.admit_one()
    sched.admit_one()
    assert a.slot_idx == 0
    assert b.slot_idx == 1
    assert c.slot_idx == 2

    sched.retire(b)  # frees slot 1

    d = _seq()
    sched.add(d)
    sched.admit_one()
    assert d.slot_idx == 1  # lowest free
