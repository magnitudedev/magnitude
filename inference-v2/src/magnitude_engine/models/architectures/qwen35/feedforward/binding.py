"""Qwen block construction. Tensor assignment and ordering stay architecture-owned."""

from dataclasses import dataclass

from magnitude_engine.models.experts.contracts import ExpertFactory

from ..contracts import (
    FeedForwardFactory,
)
from .operation import (
    DenseFeedForward,
    RoutedFeedForward,
)


@dataclass(eq=False)
class MoE(FeedForwardFactory):
    experts: ExpertFactory

    def bind(self, layer, expert):
        if expert is None:
            return DenseFeedForward(layer)
        return RoutedFeedForward(
            layer.gate,
            expert,
            layer.shared_expert,
            layer.shared_expert_gate,
            layer.top_k,
            layer.norm_topk_prob,
        )
