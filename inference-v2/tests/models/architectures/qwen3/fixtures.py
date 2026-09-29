"""Small Qwen3 reference program for page-state integration tests."""

from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.embeddings.contracts import EmbeddingLookup
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput
from magnitude_engine.models.state.pages import SequencePages, append_layer
from magnitude_engine.models.state.views import read_layer
from magnitude_engine.models.transforms import PositionTransform, Transform


@dataclass(frozen=True)
class QwenAttention:
    queries: Transform
    keys: Transform
    values: Transform
    output: Transform
    query_norm: Transform
    key_norm: Transform
    positions: PositionTransform
    query_heads: int
    kv_heads: int
    head_width: int

    def project(self, hidden: mx.array, position: int) -> tuple[mx.array, mx.array, mx.array]:
        batch, length, _ = hidden.shape
        q = self.query_norm(self.queries(hidden).reshape(batch, length, self.query_heads, -1))
        k = self.key_norm(self.keys(hidden).reshape(batch, length, self.kv_heads, -1))
        v = self.values(hidden).reshape(batch, length, self.kv_heads, -1).transpose(0, 2, 1, 3)
        return (
            self.positions(q.transpose(0, 2, 1, 3), offset=position),
            self.positions(k.transpose(0, 2, 1, 3), offset=position),
            v,
        )


@dataclass(frozen=True)
class QwenBlock:
    attention_norm: Transform
    attention: QwenAttention
    feedforward_norm: Transform
    feedforward: Transform


class Qwen3Program:
    forward_batch = None

    conditioning: frozenset[str] = frozenset()

    def __init__(
        self,
        embedding: EmbeddingLookup,
        blocks: tuple[QwenBlock, ...],
        norm: Transform,
        output: Transform,
        attention: PagedAttention,
    ):
        self.embedding = embedding
        self.blocks = blocks
        self.norm = norm
        self.output = output
        self.attention = attention
        # Features name pre-block residual streams, so their position semantics
        # are independent of the library's optional capture argument conventions.
        self.features = frozenset(f"residual:{i}" for i in range(len(blocks) + 1))

    def forward(
        self,
        inputs: ModelInputs,
        state: SequencePages,
        request: ForwardRequest,
        scope: ExecutionScope,
    ) -> ModelOutput:
        if len(self.blocks) != len(state.store.arena.layers):
            raise ValueError("Qwen program and physical state layer layouts differ")
        scope.enter(state.store.arena.pin())
        hidden = self.embedding.lookup(inputs.tokens, scope)
        captured = {}
        for index, block in enumerate(self.blocks):
            name = f"residual:{index}"
            if name in request.features:
                captured[name] = hidden
            q, k, v = block.attention.project(block.attention_norm(hidden), state.length)
            append_layer((state,), index, k, v)
            attended = self.attention.compute(
                q,
                read_layer((state,), index, pending_tokens=inputs.count),
                block.attention.head_width**-0.5,
            )
            hidden = hidden + block.attention.output(
                attended.transpose(0, 2, 1, 3).reshape(hidden.shape[0], hidden.shape[1], -1)
            )
            hidden = hidden + block.feedforward(block.feedforward_norm(hidden))
            if inputs.count > 1:
                scope.submit_state(state.store.arena.keys[index], state.store.arena.values[index])
        name = f"residual:{len(self.blocks)}"
        if name in request.features:
            captured[name] = hidden
        logits = self.output(self.norm(hidden)) if request.logits else None
        return ModelOutput(logits, captured)
