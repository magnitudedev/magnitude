"""Gemma neural execution: KV producers, shared readers and explicit branch operations."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.experts.contracts import ExpertOperator
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.pages import SequencePages, append_layer
from magnitude_engine.models.state.views import read_layer
from magnitude_engine.models.transforms import PositionTransform, Transform


@dataclass(frozen=True)
class KVProducer:
    keys: Transform
    values: Transform | None
    key_norm: Transform
    value_norm: Transform
    positions: PositionTransform
    heads: int

    def project(self, hidden: mx.array, offsets: mx.array) -> tuple[mx.array, mx.array]:
        batch, count, _ = hidden.shape
        raw_keys = self.keys(hidden).reshape(batch, count, self.heads, -1)
        # K=V refers to the raw projection, before either branch's normalization.
        raw_values = (
            raw_keys
            if self.values is None
            else self.values(hidden).reshape(batch, count, self.heads, -1)
        )
        keys = self.positions(self.key_norm(raw_keys).transpose(0, 2, 1, 3), offset=offsets)
        return keys, self.value_norm(raw_values).transpose(0, 2, 1, 3)


@dataclass(frozen=True)
class GemmaAttention:
    source: int
    producer: KVProducer | None
    queries: Transform
    query_norm: Transform
    positions: PositionTransform
    output: Transform
    heads: int
    window: int | None
    operation: PagedAttention

    def compute(
        self, hidden: mx.array, states: tuple[SequencePages, ...], scope: ExecutionScope
    ) -> mx.array:
        batch, count, _ = hidden.shape
        offsets = mx.array([state.length for state in states], dtype=mx.int32)
        q = self.query_norm(self.queries(hidden).reshape(batch, count, self.heads, -1))
        q = self.positions(q.transpose(0, 2, 1, 3), offset=offsets)
        if self.producer is not None:
            keys, values = self.producer.project(hidden, offsets)
            append_layer(states, self.source, keys, values)
        kv = read_layer(states, self.source, pending_tokens=count)
        attended = self.operation.compute(q, kv, 1.0, window=self.window)
        if self.producer is not None and count > 1:
            scope.submit_state(kv.keys, kv.values)
        return self.output(attended.transpose(0, 2, 1, 3).reshape(batch, count, -1))


@dataclass(frozen=True)
class GeGLU:
    gate: Transform
    up: Transform
    down: Transform

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.down(nn.gelu_approx(self.gate(hidden)) * self.up(hidden))


@dataclass(frozen=True)
class GemmaRouter:
    projection: Transform
    scale: mx.array
    expert_scale: mx.array
    epsilon: float
    top_k: int

    def route(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        normalized = mx.fast.rms_norm(hidden, self.scale * hidden.shape[-1] ** -0.5, self.epsilon)
        scores = self.projection(normalized)
        ids = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        weights = mx.softmax(mx.take_along_axis(scores, ids, axis=-1), axis=-1)
        return ids, weights * self.expert_scale[ids]


@dataclass(frozen=True)
class ExpertBranch:
    router: GemmaRouter
    operation: ExpertOperator
    input_norm: Transform
    output_norm: Transform

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        ids, weights = self.router.route(hidden)
        selected = self.operation.compute(self.input_norm(hidden), ids, scope)
        return self.output_norm((selected * weights[..., None]).sum(axis=-2))


@dataclass(frozen=True)
class GemmaFeedForward:
    input_norm: Transform
    dense: GeGLU
    output_norm: Transform
    dense_norm: Transform | None = None
    experts: ExpertBranch | None = None

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        dense = self.dense(self.input_norm(hidden))
        if self.dense_norm is not None:
            dense = self.dense_norm(dense)
        if self.experts is not None:
            dense = dense + self.experts.compute(hidden, scope)
        return self.output_norm(dense)


@dataclass(frozen=True)
class PerLayerInputs:
    embedding: EmbeddingLookup
    projection: Transform
    norm: Transform
    layers: int
    width: int
    embedding_scale: float
    projection_scale: float
    combination_scale: float

    def prepare(self, tokens: mx.array, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        shape = (*tokens.shape, self.layers, self.width)
        lookup = (self.embedding.lookup(tokens, scope) * self.embedding_scale).reshape(shape)
        projected = (self.projection(hidden) * self.projection_scale).reshape(shape)
        return (self.norm(projected) + lookup) * self.combination_scale


@dataclass(frozen=True)
class LayerInput:
    gate: Transform
    projection: Transform
    norm: Transform

    def apply(self, hidden: mx.array, inputs: mx.array) -> mx.array:
        return self.norm(self.projection(nn.gelu_approx(self.gate(hidden)) * inputs))


@dataclass(frozen=True)
class GemmaBlock:
    input_norm: Transform
    attention: GemmaAttention
    attention_norm: Transform
    feedforward: GemmaFeedForward
    layer_input: LayerInput | None
    scalar: mx.array


class Gemma4Program:
    conditioning: frozenset[str] = frozenset()

    def __init__(
        self,
        embedding: EmbeddingLookup,
        embedding_scale: float,
        blocks: tuple[GemmaBlock, ...],
        norm: Transform,
        output: Transform,
        per_layer: PerLayerInputs | None,
        softcap: float | None,
    ):
        self.embedding, self.embedding_scale = embedding, embedding_scale
        self.blocks, self.norm, self.output = blocks, norm, output
        self.per_layer, self.softcap = per_layer, softcap
        self.features = frozenset(f"residual:{i}" for i in range(len(blocks) + 1))
        self.kv_layers = sum(block.attention.producer is not None for block in blocks)
        if softcap is not None and softcap <= 0:
            raise ValueError("Gemma logit softcap must be positive")
        if any((block.layer_input is not None) != (per_layer is not None) for block in blocks):
            raise ValueError("Gemma per-layer input producers and consumers must agree")

    def forward(
        self,
        inputs: ModelInputs,
        state: SequencePages,
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        return self.forward_batch((inputs,), (state,), request, scope)

    def forward_batch(
        self,
        inputs: tuple[ModelInputs, ...],
        states: tuple[SequencePages, ...],
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        arena = states[0].store.arena
        if len(arena.layers) != self.kv_layers or any(s.store.arena is not arena for s in states):
            raise ValueError("Gemma batch requires the bound physical KV layout")
        scope.enter(arena.pin())
        tokens = (
            inputs[0].tokens if len(inputs) == 1 else mx.concatenate([i.tokens for i in inputs])
        )
        hidden = self.embedding.lookup(tokens, scope) * self.embedding_scale
        per_layer = self.per_layer.prepare(tokens, hidden, scope) if self.per_layer else None
        features = {}
        for index, block in enumerate(self.blocks):
            name = f"residual:{index}"
            if name in request.features:
                features[name] = hidden
            mixed = block.attention.compute(block.input_norm(hidden), states, scope)
            hidden = hidden + block.attention_norm(mixed)
            hidden = hidden + block.feedforward.compute(hidden, scope)
            if block.layer_input is not None and per_layer is not None:
                hidden = hidden + block.layer_input.apply(hidden, per_layer[:, :, index, :])
            hidden = hidden * block.scalar
        name = f"residual:{len(self.blocks)}"
        if name in request.features:
            features[name] = hidden
        logits = self.output(self.norm(hidden)) if request.logits else None
        if logits is not None and self.softcap is not None:
            logits = mx.tanh(logits / self.softcap) * self.softcap
        return ModelOutput(logits, features)
