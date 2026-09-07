"""Hybrid Qwen execution: model routing owns policy, operations own execution."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.hybrid import HybridState

from .decode import ResidentDecode
from .definition import DEFINITION
from .feedforward.operation import (
    FeedForward,
)

Transform = Callable[[mx.array], mx.array]


@component("MODEL:QWEN35.READOUT:MAG:STANDARD")
def readout(projection: Transform, hidden: mx.array) -> mx.array:
    return projection(hidden)


class Mixer(Protocol):
    def compute_batch(
        self, hidden: mx.array, states: tuple[HybridState, ...], scope: ExecutionScope
    ) -> mx.array: ...


@dataclass(frozen=True)
class HybridBlock:
    mixer_norm: Transform
    mixer: Mixer
    feedforward_norm: Transform
    feedforward: FeedForward


@component("MODEL:QWEN35:MAG:LAYERWISE", model=DEFINITION)
class Qwen35Program:
    conditioning: frozenset[str] = frozenset()

    def __init__(
        self,
        embedding: EmbeddingLookup,
        blocks: tuple[HybridBlock, ...],
        norm: Transform,
        output: Transform,
    ):
        self.embedding = embedding
        self.blocks = blocks
        self.norm = norm
        self.output = output
        self.features = frozenset(f"residual:{i}" for i in range(len(blocks) + 1))
        self.decode = ResidentDecode(self) if ResidentDecode.supports(self) else None

    def forward(
        self,
        inputs: ModelInputs,
        state: HybridState,
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        return self.forward_batch((inputs,), (state,), request, scope)

    def forward_batch(
        self,
        inputs: tuple[ModelInputs, ...],
        states: tuple[HybridState, ...],
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        arena = states[0].pages.store.arena
        if any(state.pages.store.arena is not arena for state in states):
            raise ValueError("Qwen batch states must share physical storage")
        scope.enter(arena.pin())
        tokens = (
            inputs[0].tokens if len(inputs) == 1 else mx.concatenate([row.tokens for row in inputs])
        )
        if (
            tokens.shape[1] == 1
            and self.decode is not None
            and self.blocks is self.decode.blocks
            and self.output is self.decode.output
        ):
            return self.decode.forward(tokens, states, request)
        hidden = self.embedding.lookup(tokens, scope)
        features = {}
        for index, block in enumerate(self.blocks):
            name = f"residual:{index}"
            if name in request.features:
                features[name] = hidden
            hidden = hidden + block.mixer.compute_batch(block.mixer_norm(hidden), states, scope)
            hidden = hidden + block.feedforward.compute(block.feedforward_norm(hidden), scope)
        name = f"residual:{len(self.blocks)}"
        if name in request.features:
            features[name] = hidden
        return ModelOutput(
            readout(self.output, self.norm(hidden)) if request.logits else None, features
        )
