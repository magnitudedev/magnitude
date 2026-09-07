"""Gemma neural execution: KV producers, shared readers and explicit branch operations."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.components import component
from magnitude_engine.models.attention.contracts import DecodeAttention, PagedAttention
from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.experts.contracts import ExpertOperator
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.decode import DecodeKV
from magnitude_engine.models.state.pages import SequencePages, append_layer
from magnitude_engine.models.state.views import read_layer
from magnitude_engine.models.transforms import PositionTransform, Transform

from .decode import ResidentDecode
from .definition import DEFINITION

ExpertCall = Callable[[mx.array, mx.array, mx.array], mx.array]


@partial(mx.compile, shapeless=True)
def _combine_inputs(lookup, projected, embedding_scale, combination_scale):
    return (projected + lookup * embedding_scale) * combination_scale


@component("MODEL:GEMMA4.READOUT:MAG:SOFTCAPPED")
def readout(projection: Transform, hidden: mx.array, softcap: float | None) -> mx.array:
    logits = projection(hidden)
    return logits if softcap is None else mx.tanh(logits / softcap) * softcap


@dataclass(frozen=True)
@component("MODEL:GEMMA4.KV:MAG:PRODUCER")
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
@component("MODEL:GEMMA4.ATTENTION:MAG:SHARED_KV")
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

    def project_queries(self, hidden: mx.array, offsets: mx.array) -> mx.array:
        batch, count, _ = hidden.shape
        q = self.query_norm(self.queries(hidden).reshape(batch, count, self.heads, -1))
        return self.positions(q.transpose(0, 2, 1, 3), offset=offsets)

    def finish(self, attended: mx.array) -> mx.array:
        batch, _, count, _ = attended.shape
        return self.output(attended.transpose(0, 2, 1, 3).reshape(batch, count, -1))

    def compute(
        self, hidden: mx.array, states: tuple[SequencePages, ...], scope: ExecutionScope
    ) -> mx.array:
        count = hidden.shape[1]
        offsets = mx.array([state.length for state in states], dtype=mx.int32)
        q = self.project_queries(hidden, offsets)
        if self.producer is not None:
            keys, values = self.producer.project(hidden, offsets)
            append_layer(states, self.source, keys, values)
        kv = read_layer(states, self.source, pending_tokens=count)
        attended = self.operation.compute(q, kv, 1.0, window=self.window)
        if self.producer is not None and count > 1:
            scope.submit_state(kv.keys, kv.values)
        return self.finish(attended)

    def decode(self, hidden: mx.array, kv: DecodeKV) -> mx.array:
        assert isinstance(self.operation, DecodeAttention)
        queries = self.project_queries(hidden, kv.positions)
        if self.producer is not None:
            keys, values = self.producer.project(hidden, kv.positions)
            kv.append(self.source, keys, values)
        return self.finish(self.operation.decode(queries, kv, self.source, 1.0, window=self.window))


@dataclass(frozen=True)
@component("MODEL:GEMMA4.MLP:MAG:GEGLU")
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
@component("MODEL:GEMMA4.EXPERT_BRANCH:MAG:ROUTED")
class ExpertBranch:
    router: GemmaRouter
    operation: ExpertOperator
    input_norm: Transform
    output_norm: Transform

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        return self.apply(hidden, partial(self.operation.compute, scope=scope))

    def apply(self, hidden: mx.array, experts: ExpertCall) -> mx.array:
        ids, weights = self.router.route(hidden)
        combined = experts(self.input_norm(hidden), ids, weights)
        return self.output_norm(combined)


@dataclass(frozen=True)
@component("MODEL:GEMMA4.FEEDFORWARD:MAG:BRANCHED")
class GemmaFeedForward:
    input_norm: Transform
    dense: GeGLU
    output_norm: Transform
    dense_norm: Transform | None = None
    experts: ExpertBranch | None = None

    def compute(self, hidden: mx.array, scope: ExecutionScope) -> mx.array:
        experts = partial(self.experts.operation.compute, scope=scope) if self.experts else None
        return self.apply(hidden, experts)

    def apply(self, hidden: mx.array, experts: ExpertCall | None) -> mx.array:
        dense = self.dense(self.input_norm(hidden))
        if self.dense_norm is not None:
            dense = self.dense_norm(dense)
        if self.experts is not None:
            assert experts is not None
            dense = dense + self.experts.apply(hidden, experts)
        return self.output_norm(dense)


@dataclass(frozen=True)
@component("MODEL:GEMMA4.INPUTS:MAG:PER_LAYER")
class PerLayerInputs:
    embedding: EmbeddingLookup
    projection: Transform
    norm: Transform
    layers: int
    width: int
    embedding_scale: float
    projection_scale: float
    combination_scale: float

    def combine(self, lookup: mx.array, hidden: mx.array) -> mx.array:
        shape = (*hidden.shape[:-1], self.layers, self.width)
        projected = (self.projection(hidden) * self.projection_scale).reshape(shape)
        return _combine_inputs(
            lookup.reshape(shape),
            self.norm(projected),
            self.embedding_scale,
            self.combination_scale,
        )


@dataclass(frozen=True)
@component(PerLayerInputs)
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


@component("MODEL:GEMMA4:MAG:LAYERWISE", model=DEFINITION)
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
        self.decode = ResidentDecode(self) if ResidentDecode.supports(self) else None

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
        tokens = (
            inputs[0].tokens if len(inputs) == 1 else mx.concatenate([i.tokens for i in inputs])
        )
        compiled = (
            tokens.shape[1] == 1
            and self.decode is not None
            and self.decode.matches(self)
            and all(g.key_width in (32, 64, 128, 256, 512) for g in arena.layers)
        )
        if not compiled:
            for state in states:
                state.flush_tail()
        scope.enter(arena.pin())
        if compiled:
            assert self.decode is not None
            return self.decode.forward(tokens, states, request, scope)
        hidden = self.embedding.lookup(tokens, scope)
        lookup = self.per_layer.embedding.lookup(tokens, scope) if self.per_layer else None
        logits, features = evaluate(
            hidden,
            lookup,
            blocks=self.blocks,
            embedding_scale=self.embedding_scale,
            per_layer=self.per_layer,
            norm=self.norm,
            output=self.output,
            softcap=self.softcap,
            attend=lambda attention, x: attention.compute(x, states, scope),
            feed=lambda feedforward, x: feedforward.compute(x, scope),
            request=request,
        )
        return ModelOutput(logits[0] if logits else None, features)


def evaluate(
    hidden: mx.array,
    layer_lookup: mx.array | None,
    *,
    blocks: tuple[GemmaBlock, ...],
    embedding_scale: float,
    per_layer: PerLayerInputs | None,
    norm: Transform,
    output: Transform,
    softcap: float | None,
    attend: Callable[[GemmaAttention, mx.array], mx.array],
    feed: Callable[[GemmaFeedForward, mx.array], mx.array],
    request: ForwardRequest,
) -> tuple[tuple[mx.array, ...], dict[str, mx.array]]:
    """One Gemma composition for compiled and resource-scoped execution."""
    hidden = hidden * embedding_scale
    layer_inputs = None
    if per_layer is not None:
        assert layer_lookup is not None
        layer_inputs = per_layer.combine(layer_lookup, hidden)
    features = {}
    final_required = request.logits or f"residual:{len(blocks)}" in request.features
    for index, block in enumerate(blocks):
        name = f"residual:{index}"
        if name in request.features:
            features[name] = hidden
        mixed = attend(block.attention, block.input_norm(hidden))
        if index + 1 == len(blocks) and not final_required:
            break
        hidden = hidden + block.attention_norm(mixed)
        hidden = hidden + feed(block.feedforward, hidden)
        if block.layer_input is not None and layer_inputs is not None:
            hidden = hidden + block.layer_input.apply(hidden, layer_inputs[:, :, index, :])
        hidden = hidden * block.scalar
    name = f"residual:{len(blocks)}"
    if name in request.features:
        features[name] = hidden
    logits = (readout(output, norm(hidden), softcap),) if request.logits else ()
    return logits, features
