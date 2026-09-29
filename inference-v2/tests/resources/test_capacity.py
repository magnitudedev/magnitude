import pytest

from magnitude_engine.resources.capacity import (
    CapacityObservation,
    StartupHeadroom,
    startup_allowance,
)


def test_startup_allowance_uses_the_tighter_resource_and_does_not_subtract_own_gpu_allocations():
    observation = CapacityObservation(10, 0.1, 8000, 3000, True)
    policy = StartupHeadroom(fraction=0.1, floor_bytes=100, freshness_seconds=5)
    allowance = startup_allowance(10000, 10000, observation, policy, now=10.2)
    assert allowance.available_bytes == 7000
    assert allowance.reserved_headroom == 700
    assert allowance.allocation_bytes == 6300
    assert allowance.limiting_resources == ("gpu_available",)
    capped = startup_allowance(3000, 10000, observation, policy, now=10.2)
    assert capped.allocation_bytes == 3000
    assert capped.limiting_resources == ("requested_weight_budget",)


@pytest.mark.parametrize(
    "observation,now",
    [
        (CapacityObservation(10, 0.1, 8000, None, False), 10.2),
        (CapacityObservation(10, 0.1, 8000, 0, True), 16),
        (CapacityObservation(10, 0.1, 8000, 0, True), 9),
        (CapacityObservation(10, 0.5, 8000, 0, True), 10.2),
    ],
)
def test_ambiguous_stale_or_inconsistent_observations_cannot_authorize_allocation(observation, now):
    with pytest.raises(MemoryError):
        startup_allowance(1000, 10000, observation, StartupHeadroom(floor_bytes=100), now=now)
