"""Schemas for the existing physical and logical state bindings."""

from magnitude_engine import components as c
from magnitude_engine.models.state.hybrid import HybridStateStore
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.models.state.pages import PageStore, SequencePages
from magnitude_engine.models.state.recurrent import RecurrentImage, RecurrentLayout
from performance.bindings import Fields, Use, port, schema


def kv_geometry(a: PageStore) -> c.KVStorage:
    arena = a.arena
    return c.KVStorage(
        layers=tuple(
            c.KVGeometry(heads=g.heads, key_width=g.key_width, value_width=g.value_width)
            for g in arena.layers
        ),
        element_bytes=arena.dtype.size,
        page_size=arena.page_size,
        slab_pages=arena.allocator.slab_pages,
        max_pages=arena.allocator.max_pages,
    )


def recurrent_geometry(layouts: tuple[RecurrentLayout, ...]) -> c.RecurrentStorage:
    return c.RecurrentStorage(
        layouts=tuple(
            tuple(
                c.TensorFacts(
                    identity=f"recurrent.{i}.{j}", shape=t.shape, bytes=t.nbytes, dtype=str(t.dtype)
                )
                for j, t in enumerate(layout.tensors)
            )
            for i, layout in enumerate(layouts)
        )
    )


@schema(PageStore)
def pages(a: PageStore, _: None) -> Fields[c.KVStorage]:
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
def hybrid(a: HybridStateStore, _: None) -> Fields[c.Configuration]:
    return Fields(
        c.Configuration(),
        children={
            "kv": Use(a.pages),
            "recurrent": port(a.layouts, RecurrentImage, recurrent_geometry(a.layouts)),
        },
    )


@schema(LibraryStateStore)
def library(a: LibraryStateStore, _: None) -> Fields[c.NativeStorage]:
    return Fields(
        c.NativeStorage(
            cache_types=tuple(
                type(cache).__module__ + "." + type(cache).__qualname__ for cache in a.make_cache()
            )
        )
    )


@schema(RecurrentImage)
def image(a: RecurrentImage, _: None) -> Fields[c.RecurrentStorage]:
    return Fields(recurrent_geometry(a.layouts))
