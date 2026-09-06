from dataclasses import field

from magnitude_engine.composition import Blueprint, component

from .contracts import MemoryPolicy as Budget
from .contracts import PressurePolicy as Pressure


@component
class EvictPrefixesBeforeRejecting(Blueprint[Pressure]):
    @staticmethod
    def implementation() -> type[Pressure]:
        from .policy import EvictPrefixesBeforeRejecting

        return EvictPrefixesBeforeRejecting


@component
class Budgeted(Blueprint[Budget]):
    limit_bytes: int = 28 << 30
    pressure: Blueprint[Pressure] = field(default_factory=EvictPrefixesBeforeRejecting)

    def __post_init__(self) -> None:
        if self.limit_bytes < 1:
            raise ValueError("engine memory limit must be positive")

    @staticmethod
    def implementation() -> type[Budget]:
        from .policy import Budgeted

        return Budgeted
