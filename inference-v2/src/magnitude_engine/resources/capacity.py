"""Translate observed machine capacity into an optional startup allocation allowance."""

from dataclasses import dataclass
from math import ceil, isfinite


@dataclass(frozen=True)
class CapacityObservation:
    """Global occupancy includes this worker's existing allocations."""

    started_at: float
    sampling_seconds: float
    cpu_free_inactive: int
    gpu_occupied: int | None
    valid: bool


@dataclass(frozen=True)
class StartupHeadroom:
    """Explicit opt-in policy, independent of model format and cache residency strategy."""

    fraction: float = 0.1
    floor_bytes: int = 256 << 20
    freshness_seconds: float = 5.0

    def __post_init__(self) -> None:
        if (
            not isfinite(self.fraction)
            or not 0 < self.fraction < 1
            or type(self.floor_bytes) is not int
            or self.floor_bytes < 0
            or not isfinite(self.freshness_seconds)
            or self.freshness_seconds <= 0
        ):
            raise ValueError("invalid startup headroom policy")


@dataclass(frozen=True)
class StartupAllowance:
    requested_bytes: int
    available_bytes: int
    reserved_headroom: int
    allocation_bytes: int
    limiting_resources: tuple[str, ...]


def startup_allowance(
    requested_bytes: int,
    device_limit: int,
    observation: CapacityObservation,
    policy: StartupHeadroom,
    *,
    now: float,
) -> StartupAllowance:
    """Preserve separate CPU and accelerator constraints; neither is a performance estimate."""
    if any(type(value) is not int or value <= 0 for value in (requested_bytes, device_limit)):
        raise ValueError("hard allocation and device ceilings must be positive integers")
    age = now - observation.started_at
    if (
        not observation.valid
        or not isfinite(age)
        or not 0 <= age <= policy.freshness_seconds
        or not isfinite(observation.sampling_seconds)
        or not 0 <= observation.sampling_seconds <= age
        or type(observation.gpu_occupied) is not int
        or observation.gpu_occupied < 0
        or type(observation.cpu_free_inactive) is not int
        or observation.cpu_free_inactive < 0
    ):
        raise MemoryError("startup allowance requires fresh, unambiguous capacity observations")
    capacities = {
        "cpu_free_inactive": observation.cpu_free_inactive,
        "gpu_available": max(0, device_limit - observation.gpu_occupied),
    }
    available = min(capacities.values())
    reserve = max(policy.floor_bytes, ceil(available * policy.fraction))
    allocation = min(requested_bytes, available - reserve)
    if allocation <= 0:
        raise MemoryError("startup capacity does not cover the required headroom")
    limiting = (
        ("requested_weight_budget",)
        if allocation == requested_bytes
        else tuple(sorted(name for name, value in capacities.items() if value == available))
    )
    return StartupAllowance(requested_bytes, available, reserve, allocation, limiting)
