from dataclasses import field

from magnitude_engine.composition import Blueprint, blueprint

from .contracts import PrefixIndex as Prefixes
from .contracts import RetentionPolicy as Retention


@blueprint
class LeastRecentlyUsed(Blueprint[Retention]):
    max_entries: int = 32
    max_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_entries < 0:
            raise ValueError("retention capacity cannot be negative")
        if self.max_bytes is not None and self.max_bytes < 0:
            raise ValueError("retention byte capacity cannot be negative")

    @staticmethod
    def implementation() -> type[Retention]:
        from .retention import LeastRecentlyUsed

        return LeastRecentlyUsed


@blueprint
class Radix(Blueprint[Prefixes]):
    retention: Blueprint[Retention] = field(default_factory=LeastRecentlyUsed)

    @staticmethod
    def implementation() -> type[Prefixes]:
        from .radix import Radix

        return Radix
