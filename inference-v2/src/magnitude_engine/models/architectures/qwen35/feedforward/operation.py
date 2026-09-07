"""Qwen dense and routed feedforward execution, including shared-expert combination."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import mlx.core as mx
from mlx_lm.models.switch_layers import SwiGLU

from magnitude_engine.components import component
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.experts import metal
from magnitude_engine.models.experts.computation import (
    ExpertWeights,
    QuantizedProjection,
    ResidentExperts,
    affine_mlp,
)
from magnitude_engine.models.experts.contracts import ExpertOperator
from magnitude_engine.models.projections import ParallelProjections

from .routing import select

Transform = Callable[[mx.array], mx.array]


class FeedForward(Protocol):
    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array: ...


@dataclass(frozen=True)
@component("MODEL:QWEN35.FEEDFORWARD:MAG:DENSE")
class DenseFeedForward:
    call: Transform
    _weights: ExpertWeights | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        weights = affine_mlp(self.call)
        # Borrow the encoded projections once at binding, as the routed/shared
        # branches do. Execution must not rediscover parameters every token.
        expanded = (
            ExpertWeights(
                *(
                    QuantizedProjection(p.weight[None], p.scales[None], p.biases[None], p.encoding)
                    for p in (weights.up, weights.gate, weights.down)
                )
            )
            if weights is not None
            else None
        )
        object.__setattr__(self, "_weights", expanded)

    def __call__(self, hidden: mx.array) -> mx.array:
        if self._weights is not None and hidden.size // hidden.shape[-1] == 1:
            indices = mx.zeros(hidden.shape[:-1] + (1,), mx.int32)
            if metal.supported(self._weights, hidden, indices):
                activation = metal.activate(self._weights, hidden, indices).reshape(
                    *hidden.shape[:-1], self._weights.gate.weight.shape[1]
                )
                return self.call.down_proj(activation)
        return self.call(hidden)

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        return self(hidden)


@dataclass(frozen=True)
@component("MODEL:QWEN35.FEEDFORWARD:MAG:ROUTED")
class RoutedFeedForward:
    routing: ParallelProjections
    experts: ExpertOperator
    shared: Transform
    top_k: int
    normalize: bool
    shared_weights: ExpertWeights | None = None

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        return self.apply(
            hidden,
            self.experts
            if isinstance(self.experts, ResidentExperts)
            else lambda x, indices, scores: self.experts.compute(x, indices, scores, scope),
        )

    def route(self, hidden: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if (
            self.routing.packed
            and hidden.size // hidden.shape[-1] <= 8
            and self.routing.sizes[0] <= 1024
            and self.top_k <= 16
        ):
            return select(self.routing.operations[0](hidden), self.top_k, self.normalize)
        logits, shared = self.routing(hidden)
        probabilities = mx.softmax(logits, axis=-1, precise=True)
        indices = mx.argpartition(probabilities, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        weights = mx.take_along_axis(probabilities, indices, axis=-1)
        if self.normalize:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return indices, weights, mx.sigmoid(shared)

    def apply(
        self, hidden: mx.array, experts: Callable[[mx.array, mx.array, mx.array], mx.array]
    ) -> mx.array:
        """Tensor routing and combination, independent of expert storage ownership."""
        indices, weights, shared = self.route(hidden)
        if (
            isinstance(experts, ResidentExperts)
            and isinstance(experts.math.activation, SwiGLU)
            and self.shared_weights is not None
            and metal.shared_supported(experts.weights, self.shared_weights, hidden, indices)
        ):
            return metal.apply(
                experts.weights,
                hidden,
                indices,
                weights,
                shared=self.shared_weights,
                shared_score=shared,
            )
        combined = experts(hidden, indices, weights)
        return combined + self.shared(hidden) * shared
