"""Temporary planning must account for actual zero-offset allocations."""

from ops.compiler.memory import _assign_slots, _Interval
from ops.tensor.types import DType, TensorSpec


def interval(identity, start, end, size):
    return _Interval(("value", identity, 0), start, end, size, 16, TensorSpec((size,), DType.U8))


def test_smaller_lifetime_does_not_split_a_reusable_native_allocation():
    slots, size = _assign_slots(
        (
            interval(0, 0, 0, 1024),
            interval(1, 1, 1, 512),
            interval(2, 2, 2, 1024),
        ),
        16,
    )
    assert size == 1024
    assert set(slots.values()) == {0}


def test_simultaneous_lifetimes_keep_separate_physical_capacity():
    slots, size = _assign_slots(
        (
            interval(0, 0, 0, 1024),
            interval(1, 1, 1, 512),
            interval(2, 1, 1, 512),
            interval(3, 2, 2, 1024),
        ),
        16,
    )
    assert size == 1536
    assert slots[("value", 1, 0)] != slots[("value", 2, 0)]
    assert slots[("value", 3, 0)] == slots[("value", 0, 0)]
