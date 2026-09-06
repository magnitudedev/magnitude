"""Qwen gated attention: projections, rotary positions, KV append and injected attention."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.state.hybrid import HybridState
from magnitude_engine.models.state.pages import append_layer
from magnitude_engine.models.state.views import read_layer

from .rotary import QwenRotary

Transform = Callable[[mx.array], mx.array]


@dataclass(frozen=True)
class GatedAttention:
    index: int
    queries_and_gate: Transform
    keys: Transform
    values: Transform
    output: Transform
    query_norm: Transform
    key_norm: Transform
    positions: QwenRotary
    query_heads: int
    kv_heads: int
    head_width: int
    attention: PagedAttention

    def compute_batch(
        self, hidden: mx.array, states: tuple[HybridState, ...], scope: ExecutionScope
    ) -> mx.array:
        batch, count, _ = hidden.shape
        positions = (
            states[0].position if batch == 1 else mx.array([s.position for s in states], mx.int32)
        )
        q, gate = mx.split(
            self.queries_and_gate(hidden).reshape(batch, count, self.query_heads, -1), 2, axis=-1
        )
        q = self.query_norm(q).transpose(0, 2, 1, 3)
        k = self.keys(hidden).reshape(batch, count, self.kv_heads, -1)
        k = self.key_norm(k).transpose(0, 2, 1, 3)
        q, k = self.positions(q, k, offset=positions)
        v = self.values(hidden).reshape(batch, count, self.kv_heads, -1).transpose(0, 2, 1, 3)
        pages = tuple(s.pages for s in states)
        append_layer(pages, self.index, k, v)
        attended = self.attention.compute(
            q, read_layer(pages, self.index, pending_tokens=count), self.head_width**-0.5
        )
        if count > 1:
            arena = states[0].pages.store.arena
            scope.submit_state(arena.keys[self.index], arena.values[self.index])
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, count, -1)
        return self.output(attended * mx.sigmoid(gate.reshape(batch, count, -1)))
