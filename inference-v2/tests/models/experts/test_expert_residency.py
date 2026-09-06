import pytest

from magnitude_engine.models.experts.residency import Residency


def load(directory, expert):
    transfer = directory.reserve(expert)
    assert transfer is not None
    assert directory.resolve(expert) is None
    directory.finish(transfer, publish=True)


def test_leased_slots_cannot_be_overwritten_and_failed_load_does_not_restore_old_bytes():
    directory = Residency(4, 2)
    load(directory, 0)
    load(directory, 1)
    lease = directory.pin((0, 0))
    transfer = directory.reserve(2)
    assert transfer is not None
    assert directory.resolve(1) is None
    assert directory.reserve(2) == transfer
    with pytest.raises(MemoryError):
        directory.reserve(3)
    directory.finish(transfer, publish=False)
    assert directory.resolve(1) is None and directory.resolve(2) is None
    newer = directory.reserve(3)
    assert newer is not None
    with pytest.raises(ValueError, match="stale"):
        directory.finish(transfer, publish=True)
    directory.finish(newer, publish=True)
    lease.close()
    lease.close()
    directory.validate()


def test_failed_pin_is_atomic_and_lru_prefers_empty_then_oldest_slots():
    directory = Residency(6, 3)
    load(directory, 0)
    with pytest.raises(ValueError):
        directory.pin((0, 1))
    load(directory, 1)
    load(directory, 2)
    directory.touch((0,))
    load(directory, 3)
    assert directory.resolve(1) is None
    assert directory.resolve(0) is not None
    directory.validate()
