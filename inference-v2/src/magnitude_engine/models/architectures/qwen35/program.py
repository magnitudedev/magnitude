"""Hybrid Qwen execution: model routing owns policy, operations own execution."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.embeddings.replacement import replace
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.normalization import residual_norm
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.hybrid import HybridState

from .attention.operation import GatedAttention
from .decode import ResidentDecode
from .definition import DEFINITION
from .feedforward.operation import (
    FeedForward,
)
from .inputs import QwenInputs, batch_positions

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
        tokens = (
            inputs[0].tokens if len(inputs) == 1 else mx.concatenate([row.tokens for row in inputs])
        )
        positions = batch_positions(inputs, tuple(state.position for state in states))
        replacements = tuple(
            row.data.embeddings if isinstance(row.data, QwenInputs) else () for row in inputs
        )
        compiled = tokens.shape[1] == 1 and self.decode is not None and self.decode.matches(self)
        compiled = compiled and not any(replacements)
        if not compiled:
            for state in states:
                state.pages.flush_tail()
        scope.enter(arena.pin())
        if compiled:
            assert self.decode is not None
            return self.decode.forward(tokens, states, request, scope, positions)
        hidden = self.embedding.lookup(tokens, scope)
        if any(replacements):
            hidden = replace(hidden, replacements)
        logits, features = evaluate(
            hidden,
            blocks=self.blocks,
            norm=self.norm,
            output=self.output,
            mix=lambda mixer, x: (
                mixer.compute_batch(x, states, scope, positions)
                if positions is not None and isinstance(mixer, GatedAttention)
                else mixer.compute_batch(x, states, scope)
            ),
            feed=lambda feedforward, x: feedforward.compute(x, scope),
            request=request,
        )
        return ModelOutput(logits[0] if logits else None, features)


def evaluate(
    hidden: mx.array,
    *,
    blocks: tuple[HybridBlock, ...],
    norm: Transform,
    output: Transform,
    mix: Callable[[Mixer, mx.array], mx.array],
    feed: Callable[[FeedForward, mx.array], mx.array],
    request: ForwardRequest,
) -> tuple[tuple[mx.array, ...], dict[str, mx.array]]:
    """The architecture's single residual stream, independent of state/residency binding."""
    features = {}
    final_required = request.logits or f"residual:{len(blocks)}" in request.features
    normalized = blocks[0].mixer_norm(hidden)
    for index, block in enumerate(blocks):
        name = f"residual:{index}"
        if name in request.features:
            features[name] = hidden
        mixed = mix(block.mixer, normalized)
        # The final mixer must advance state. Its stateless suffix is unnecessary
        # when neither logits nor the final residual were requested.
        if index + 1 == len(blocks) and not final_required:
            break
        hidden, normalized = residual_norm(hidden, mixed, block.feedforward_norm)
        value = feed(block.feedforward, normalized)
        if index + 1 == len(blocks) and not request.logits:
            hidden = hidden + value
        else:
            next_norm = blocks[index + 1].mixer_norm if index + 1 < len(blocks) else norm
            hidden, normalized = residual_norm(hidden, value, next_norm)
    name = f"residual:{len(blocks)}"
    if name in request.features:
        features[name] = hidden
    logits = (readout(output, normalized),) if request.logits else ()
    return logits, features
