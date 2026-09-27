"""Schemas for the existing physical and logical state bindings."""

from magnitude_engine.models.state.hybrid import HybridStateStore
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.models.state.pages import PageStore, SequencePages
from magnitude_engine.models.state.recurrent import RecurrentImage, RecurrentLayout
from performance.bindings import Fields, Use, port, schema
from performance.facts import (
    Configuration,
    KVGeometry,
    KVStorage,
    NativeStorage,
    RecurrentStorage,
    TensorFacts,
)


def kv_geometry(a: PageStore) -> KVStorage:
    arena = a.arena
    return KVStorage(
        layers=tuple(
            KVGeometry(heads=g.heads, key_width=g.key_width, value_width=g.value_width)
            for g in arena.layers
        ),
        element_bytes=arena.dtype.size,
        page_size=arena.page_size,
        slab_pages=arena.allocator.slab_pages,
        max_pages=arena.allocator.max_pages,
    )


def recurrent_geometry(layouts: tuple[RecurrentLayout, ...]) -> RecurrentStorage:
    return RecurrentStorage(
        layouts=tuple(
            tuple(
                TensorFacts(
                    identity=f"recurrent.{i}.{j}", shape=t.shape, bytes=t.nbytes, dtype=str(t.dtype)
                )
                for j, t in enumerate(layout.tensors)
            )
            for i, layout in enumerate(layouts)
        )
    )


@schema(PageStore)
def pages(a: PageStore, _: None) -> Fields[KVStorage]:
    geometry = kv_geometry(a)
    return Fields(
        geometry,
        children={
            "append": port(a, SequencePages.write, geometry),
            "branch": port(a, PageStore.create),
        },
        sources=(a.arena,),
    )


@schema(HybridStateStore)
def hybrid(a: HybridStateStore, _: None) -> Fields[Configuration]:
    return Fields(
        Configuration(),
        children={
            "kv": Use(a.pages),
            "recurrent": port(a.layouts, RecurrentImage, recurrent_geometry(a.layouts)),
        },
    )


@schema(LibraryStateStore)
def library(a: LibraryStateStore, _: None) -> Fields[NativeStorage]:
    return Fields(
        NativeStorage(
            cache_types=tuple(
                type(cache).__module__ + "." + type(cache).__qualname__ for cache in a.make_cache()
            )
        )
    )


@schema(RecurrentImage)
def image(a: RecurrentImage, _: None) -> Fields[RecurrentStorage]:
    return Fields(recurrent_geometry(a.layouts))
