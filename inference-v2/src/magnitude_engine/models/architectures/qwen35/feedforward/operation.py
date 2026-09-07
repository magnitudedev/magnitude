"""Qwen dense and routed feedforward execution, including shared-expert combination."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from magnitude_engine.models.activations import sigmoid_gate
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.experts.contracts import ExpertOperator

Transform = Callable[[mx.array], mx.array]


class FeedForward(Protocol):
    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array: ...


@dataclass(frozen=True)
class DenseFeedForward:
    call: Transform

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        return self.call(hidden)


@dataclass(frozen=True)
class RoutedFeedForward:
    router: Transform
    experts: ExpertOperator
    shared: Transform
    shared_gate: Transform
    top_k: int
    normalize: bool

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        return self.apply(hidden, lambda x, indices: self.experts.compute(x, indices, scope))

    def apply(
        self, hidden: mx.array, experts: Callable[[mx.array, mx.array], mx.array]
    ) -> mx.array:
        """Tensor routing and combination, independent of expert storage ownership."""
        probabilities = mx.softmax(self.router(hidden), axis=-1, precise=True)
        indices = mx.argpartition(probabilities, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        weights = mx.take_along_axis(probabilities, indices, axis=-1)
        if self.normalize:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        routed = experts(hidden, indices)
        combined = (routed * weights[..., None]).sum(axis=-2)
        return combined + sigmoid_gate(self.shared(hidden), self.shared_gate(hidden))
