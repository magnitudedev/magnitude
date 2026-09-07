"""Qwen gated attention: projections, rotary positions, KV append and injected attention."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.components import component
from magnitude_engine.models.activations import sigmoid_gate
from magnitude_engine.models.attention.contracts import DecodeAttention, PagedAttention
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.projections import ParallelProjections
from magnitude_engine.models.state.decode import DecodeKV
from magnitude_engine.models.state.hybrid import HybridState
from magnitude_engine.models.state.pages import append_layer
from magnitude_engine.models.state.views import read_layer

from .preparation import prepare
from .rotary import QwenRotary

Transform = Callable[[mx.array], mx.array]


@dataclass(frozen=True)
@component("MODEL:QWEN35.ATTENTION:MAG:GROUPED_PROJECTIONS")
class GatedAttention:
    index: int
    inputs: ParallelProjections
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
        q, k, v, gate = self.project(hidden, positions)
        pages = tuple(s.pages for s in states)
        append_layer(pages, self.index, k, v)
        attended = self.attention.compute(
            q, read_layer(pages, self.index, pending_tokens=count), self.head_width**-0.5
        )
        if count > 1:
            arena = states[0].pages.store.arena
            scope.submit_state(arena.keys[self.index], arena.values[self.index])
        return self.finish(attended, gate)

    def project(
        self, hidden: mx.array, positions: int | mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        batch, count, _ = hidden.shape
        rotation = self.positions.rotation
        if (
            self.inputs.packed
            and batch * count <= 8
            and 32 <= self.head_width <= 1024
            and rotation is not None
            and rotation.dim <= self.head_width
            and isinstance(self.query_norm, nn.RMSNorm)
            and isinstance(self.key_norm, nn.RMSNorm)
            and self.query_norm.weight.dtype == self.key_norm.weight.dtype == hidden.dtype
            and hidden.dtype in (mx.float32, mx.float16, mx.bfloat16)
        ):
            return cast(
                tuple[mx.array, mx.array, mx.array, mx.array],
                prepare(
                    self.inputs.operations[0](hidden),
                    self.query_norm.weight,
                    self.key_norm.weight,
                    positions,
                    rotation.inv_freq,
                    query_heads=self.query_heads,
                    kv_heads=self.kv_heads,
                    width=self.head_width,
                    query_eps=self.query_norm.eps,
                    key_eps=self.key_norm.eps,
                ),
            )
        queries_and_gate, keys, values = self.inputs(hidden)
        q, gate = mx.split(queries_and_gate.reshape(batch, count, self.query_heads, -1), 2, axis=-1)
        q = self.query_norm(q).transpose(0, 2, 1, 3)
        k = keys.reshape(batch, count, self.kv_heads, -1)
        k = self.key_norm(k).transpose(0, 2, 1, 3)
        q, k = self.positions(q, k, offset=positions)
        v = values.reshape(batch, count, self.kv_heads, -1).transpose(0, 2, 1, 3)
        return q, k, v, gate.reshape(batch, count, -1)

    def finish(self, attended: mx.array, gate: mx.array) -> mx.array:
        batch, count = gate.shape[:2]
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, count, -1)
        return self.output(sigmoid_gate(attended, gate))

    def decode(self, hidden: mx.array, kv: DecodeKV) -> mx.array:
        assert isinstance(self.attention, DecodeAttention)
        queries, keys, values, gate = self.project(hidden, kv.positions)
        kv.append(self.index, keys, values)
        attended = self.attention.decode(queries, kv, self.index, self.head_width**-0.5)
        return self.finish(attended, gate)
