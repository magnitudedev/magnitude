from magnitude_engine.composition import Blueprint, blueprint

from ..contracts import ProgramSource
from .contracts import StateFactory


@blueprint
class PagedHybrid(Blueprint[StateFactory]):
    page_size: int = 16
    slab_pages: int = 32

    def __post_init__(self) -> None:
        if self.page_size < 1 or self.slab_pages < 1:
            raise ValueError("page and slab dimensions must be positive")

    @staticmethod
    def implementation() -> type[StateFactory]:
        from .binding import PagedHybridFactory

        return PagedHybridFactory


@blueprint
class Native(Blueprint[StateFactory]):
    source: Blueprint[ProgramSource]

    @staticmethod
    def implementation() -> type[StateFactory]:
        from .binding import NativeStateFactory

        return NativeStateFactory
