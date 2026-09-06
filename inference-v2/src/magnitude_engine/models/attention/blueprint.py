from dataclasses import field

from magnitude_engine.composition import Blueprint, component

from .contracts import PagedAttention


@component
class Paged(Blueprint[PagedAttention]):
    prefill: Blueprint[PagedAttention] = field(default_factory=lambda: Gathered())
    heads_per_group: int = 1

    def __post_init__(self) -> None:
        if self.heads_per_group not in (1, 2, 4):
            raise ValueError("unsupported attention head grouping")

    @staticmethod
    def implementation() -> type[PagedAttention]:
        from .metal import MetalPagedAttention

        return MetalPagedAttention


@component
class Gathered(Blueprint[PagedAttention]):
    @staticmethod
    def implementation() -> type[PagedAttention]:
        from .gathered import GatheredAttention

        return GatheredAttention
