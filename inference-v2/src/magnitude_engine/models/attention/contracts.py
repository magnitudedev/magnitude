"""Backend-free operation signatures; arrays and execution state are worker types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import mlx.core as mx

    from ..state.views import PagedKV


class PagedAttention(Protocol):
    def compute(
        self, queries: mx.array, kv: PagedKV, scale: float, *, window: int | None = None
    ) -> mx.array: ...
