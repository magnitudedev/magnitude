"""Backend-free operation signatures; arrays and execution state are worker types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx

    from ..state.decode import DecodeKV
    from ..state.views import PagedKV


class PagedAttention(Protocol):
    def compute(
        self,
        queries: mx.array,
        kv: PagedKV,
        scale: float,
        *,
        window: int | None = None,
        key_ends: mx.array | None = None,
    ) -> mx.array: ...


@runtime_checkable
class DecodeAttention(PagedAttention, Protocol):
    """Attention that can bind a resident tensor step without host storage operations."""

    def decode(
        self,
        queries: mx.array,
        kv: DecodeKV,
        layer: int,
        scale: float,
        *,
        window: int | None = None,
    ) -> mx.array: ...
