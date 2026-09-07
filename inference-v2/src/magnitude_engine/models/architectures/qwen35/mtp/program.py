"""An attached MTP head is a conditioned model program with its own state."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.native import LibraryState

from ..definition import DEFINITION

Transform = Callable[[mx.array], mx.array]


@dataclass(frozen=True)
@component("MODEL:QWEN35.MTP:MAG:CONDITIONED", model=DEFINITION)
class MTPProgram:
    embedding: EmbeddingLookup
    normalize_embedding: Transform
    normalize_conditioning: Transform
    combine: Transform
    layers: tuple[Callable[[mx.array, Any], mx.array], ...]
    normalize_output: Transform
    project: Transform

    features = frozenset({"draft_hidden"})
    conditioning = frozenset({"previous_hidden"})

    def forward(
        self,
        inputs: ModelInputs,
        state: LibraryState,
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        caches = state.caches if state.batch is None else state.store.batch_caches((state,))
        output = self._forward(
            inputs.tokens, inputs.conditioning["previous_hidden"], caches, request, scope
        )
        state.store.publish_batch((state,))
        return output

    def forward_batch(self, inputs, states, request, scope) -> ModelOutput:
        output = self._forward(
            mx.concatenate([row.tokens for row in inputs]),
            mx.concatenate([row.conditioning["previous_hidden"] for row in inputs]),
            states[0].store.batch_caches(states),
            request,
            scope,
        )
        states[0].store.publish_batch(states)
        return output

    def _forward(self, tokens, previous, caches, request, scope) -> ModelOutput:
        embedded = self.embedding.lookup(tokens, scope)
        if embedded.shape != previous.shape or len(caches) != len(self.layers):
            raise ValueError("MTP conditioning or state geometry differs from the head")
        hidden = self.combine(
            mx.concatenate(
                [self.normalize_embedding(embedded), self.normalize_conditioning(previous)], axis=-1
            )
        )
        for layer, cache in zip(self.layers, caches, strict=True):
            hidden = layer(hidden, cache)
        hidden = self.normalize_output(hidden)
        return ModelOutput(
            self.project(hidden) if request.logits else None,
            {"draft_hidden": hidden} if "draft_hidden" in request.features else {},
        )
