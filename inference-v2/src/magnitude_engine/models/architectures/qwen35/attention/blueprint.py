from dataclasses import field

from magnitude_engine.composition import Blueprint, component
from magnitude_engine.models.attention.blueprint import Paged
from magnitude_engine.models.attention.contracts import PagedAttention

from ..contracts import (
    AttentionFactory,
)


@component
class Attention(Blueprint[AttentionFactory]):
    computation: Blueprint[PagedAttention] = field(default_factory=Paged)

    @staticmethod
    def implementation() -> type[AttentionFactory]:
        from .binding import Attention

        return Attention
