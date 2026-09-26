from dataclasses import field

from magnitude_engine.composition import Blueprint, blueprint

from .contracts import PagedAttention


@blueprint
class Paged(Blueprint[PagedAttention]):
    prefill: Blueprint[PagedAttention] = field(default_factory=lambda: Gathered())
    heads_per_group: int = 2

    def __post_init__(self) -> None:
        if self.heads_per_group not in (1, 2, 4):
            raise ValueError("unsupported attention head grouping")

    @staticmethod
    def implementation() -> type[PagedAttention]:
        from .metal import MetalPagedAttention

        return MetalPagedAttention


@blueprint
class Gathered(Blueprint[PagedAttention]):
    @staticmethod
    def implementation() -> type[PagedAttention]:
        from .gathered import GatheredAttention

        return GatheredAttention
