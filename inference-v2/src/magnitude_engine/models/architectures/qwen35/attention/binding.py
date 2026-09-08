"""Qwen block construction. Tensor assignment and ordering stay architecture-owned."""

from dataclasses import dataclass

from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.projections import ParallelProjections, bind_linear

from ..contracts import (
    AttentionFactory,
)
from .operation import GatedAttention
from .rotary import QwenRotary


@dataclass(eq=False)
class Attention(AttentionFactory):
    computation: PagedAttention

    def bind(self, layer, slot: int, inputs: ParallelProjections) -> GatedAttention:
        return GatedAttention(
            slot,
            inputs,
            bind_linear(layer.o_proj),
            layer.q_norm,
            layer.k_norm,
            QwenRotary(layer.rope),
            layer.num_attention_heads,
            layer.num_key_value_heads,
            layer.head_dim,
            self.computation,
        )
