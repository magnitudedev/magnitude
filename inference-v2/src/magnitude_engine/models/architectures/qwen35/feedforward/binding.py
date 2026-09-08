"""Qwen block construction. Tensor assignment and ordering stay architecture-owned."""

from copy import copy
from dataclasses import dataclass

from magnitude_engine.models.experts.computation import affine_mlp
from magnitude_engine.models.experts.contracts import ExpertFactory
from magnitude_engine.models.projections import bind_linear

from ..contracts import (
    FeedForwardFactory,
)
from .operation import (
    DenseFeedForward,
    RoutedFeedForward,
)


def bind_dense(layer):
    # Keep the architecture's upstream MLP equation; borrow its tensors into
    # request-independent projection bindings without mutating the source module.
    bound = copy(layer)
    for name in ("gate_proj", "up_proj", "down_proj"):
        bound[name] = bind_linear(layer[name])
    return bound


@dataclass(eq=False)
class MoE(FeedForwardFactory):
    experts: ExpertFactory

    def bind(self, layer, expert, routing):
        if expert is None:
            return DenseFeedForward(bind_dense(layer))
        assert routing is not None
        return RoutedFeedForward(
            routing,
            expert,
            bind_dense(layer.shared_expert),
            layer.top_k,
            layer.norm_topk_prob,
            affine_mlp(layer.shared_expert),
        )
