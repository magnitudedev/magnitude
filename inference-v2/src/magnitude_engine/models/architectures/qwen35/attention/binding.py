"""Qwen block construction. Tensor assignment and ordering stay architecture-owned."""

from dataclasses import dataclass

from magnitude_engine.models.attention.contracts import PagedAttention

from ..contracts import (
    AttentionFactory,
)
from .operation import GatedAttention


@dataclass(eq=False)
class Attention(AttentionFactory):
    computation: PagedAttention

    def bind(self, layer, slot: int) -> GatedAttention:
        return GatedAttention(
            slot,
            layer.q_proj,
            layer.k_proj,
            layer.v_proj,
            layer.o_proj,
            layer.q_norm,
            layer.k_norm,
            layer.rope,
            layer.num_attention_heads,
            layer.num_key_value_heads,
            layer.head_dim,
            self.computation,
        )
