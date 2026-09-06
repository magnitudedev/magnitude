from ..contracts import ProgramSource
from ..residency import (
    BoundProgram,
    HybridRequirements,
    ModelResources,
    NativeRequirements,
    PagedRequirements,
)
from .arena import KVArena
from .contracts import StateFactory
from .hybrid import HybridStateStore
from .native import LibraryStateStore
from .paged import PagedStateStore
from .pages import PageStore


class PagedHybridFactory(StateFactory):
    def __init__(self, *, page_size: int, slab_pages: int):
        self.page_size, self.slab_pages = page_size, slab_pages

    def validate(self, program: ProgramSource) -> None:
        if program.native_state_source() is not None:
            raise ValueError("native program cannot use paged state")

    def create(self, program: BoundProgram, resources: ModelResources):
        requirements = program.state
        if not isinstance(requirements, PagedRequirements):
            raise ValueError("program does not consume paged state")
        context = min(resources.context_tokens, program.descriptor.context_tokens)
        pages_per_sequence = (context + self.page_size - 1) // self.page_size
        required_pages = resources.max_active * pages_per_sequence
        # Capacity is advertised in tokens; physical growth is in complete slabs.
        # Rounding down would silently make the final context pages unreachable.
        slabs = (required_pages + self.slab_pages - 1) // self.slab_pages
        arena = resources.own(
            KVArena(
                requirements.attention,
                page_size=self.page_size,
                slab_pages=self.slab_pages,
                max_pages=max(1, slabs) * self.slab_pages,
                budget=resources.budget,
                dtype=requirements.dtype,
            )
        )
        pages = PageStore(arena)
        if isinstance(requirements, HybridRequirements):
            return HybridStateStore(pages, requirements.recurrence, resources.budget)
        return PagedStateStore(pages)


class NativeStateFactory(StateFactory):
    def __init__(self, *, source: ProgramSource):
        self.source = source

    def validate(self, program: ProgramSource) -> None:
        if program.native_state_source() is not self.source:
            raise ValueError("native program and state must share their source")

    def create(self, program: BoundProgram, resources: ModelResources):
        requirements = program.state
        if (
            not isinstance(requirements, NativeRequirements)
            or requirements.source is not self.source
        ):
            raise ValueError("native state and program must share the same upstream loader")
        return LibraryStateStore(requirements.make_cache, resources.budget, requirements.capacity)
