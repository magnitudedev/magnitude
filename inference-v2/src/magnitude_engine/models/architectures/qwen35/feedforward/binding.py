"""Qwen block construction. Tensor assignment and ordering stay architecture-owned."""

from dataclasses import dataclass

from magnitude_engine.models.experts.computation import affine_mlp
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

    def bind(self, layer, expert, routing):
        if expert is None:
            return DenseFeedForward(layer)
        assert routing is not None
        return RoutedFeedForward(
            routing,
            expert,
            layer.shared_expert,
            layer.top_k,
            layer.norm_topk_prob,
            affine_mlp(layer.shared_expert),
        )
