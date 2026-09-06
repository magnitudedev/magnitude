"""Matched physical KV append and attention, including completion and layout cost."""

import hashlib
from itertools import pairwise
from typing import cast

import mlx.core as mx
import numpy as np

from benchmarks.contracts import Observation
from magnitude_engine.models.attention.contracts import PagedAttention
from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.state.arena import KVArena, LayerGeometry
from magnitude_engine.models.state.pages import PageStore, append_layer
from magnitude_engine.models.state.views import read_layer
from magnitude_engine.resources.budget import MemoryBudget


class AttentionTrace:
    def __init__(
        self,
        *,
        computation: PagedAttention,
        prefix_tokens: int,
        query_tokens: int,
        query_heads: int,
        kv_heads: int,
        width: int,
        fragmented: bool,
    ):
        if prefix_tokens < 0 or query_tokens < 1:
            raise ValueError("invalid attention operating point")
        self.operator = computation
        self.prefix, self.count = prefix_tokens, query_tokens
        self.scale = width**-0.5
        total = prefix_tokens + query_tokens
        page_count = (total + 15) // 16
        self.budget = MemoryBudget(2 << 30)
        self.arena = KVArena(
            (LayerGeometry(kv_heads, width, width),),
            page_size=16,
            slab_pages=32,
            max_pages=max(32, 2 * page_count + 32),
            budget=self.budget,
        )
        self.guards = ()
        if fragmented:
            allocated = self.arena.allocate(2 * page_count)
            self.guards = allocated[::2]
            self.arena.release(allocated[1::2])
        self.state = PageStore(self.arena).create()
        self.state.reserve(total)
        mx.random.seed(131)
        self.keys = mx.random.normal((1, kv_heads, total, width)).astype(mx.bfloat16)
        self.values = mx.random.normal(self.keys.shape).astype(mx.bfloat16)
        self.queries = mx.random.normal((1, query_heads, query_tokens, width)).astype(mx.bfloat16)
        mask = mx.arange(total)[None, :] <= (prefix_tokens + mx.arange(query_tokens))[:, None]
        self.oracle = mx.fast.scaled_dot_product_attention(
            self.queries.astype(mx.float32),
            self.keys.astype(mx.float32),
            self.values.astype(mx.float32),
            scale=self.scale,
            mask=mask,
        ).astype(mx.bfloat16)
        if prefix_tokens:
            self.state.write(
                0, 0, self.keys[0, :, :prefix_tokens], self.values[0, :, :prefix_tokens]
            )
            self.state.commit(prefix_tokens)
        mx.eval(self.queries, self.keys, self.values, self.oracle)
        self.arena.complete()
        self.output = None

    def reset(self) -> None:
        self.arena.complete()
        self.output = None
        self.state.trim(self.prefix)
        self.state.reserve(self.prefix + self.count)

    def invoke(self) -> None:
        append_layer(
            (self.state,), 0, self.keys[:, :, self.prefix :], self.values[:, :, self.prefix :]
        )
        kv = read_layer((self.state,), 0, pending_tokens=self.count)
        if isinstance(self.operator, MetalPagedAttention) and not self.operator.supports(
            self.queries, kv
        ):
            raise ValueError("case must exercise native attention, not prefill fallback")
        self.output = self.operator.compute(self.queries, kv, self.scale)
        self.state.commit(self.prefix + self.count)

    def complete(self) -> None:
        assert self.output is not None
        mx.eval(self.output)
        self.arena.complete()

    def observe(self) -> Observation:
        assert self.output is not None
        error = cast(float, mx.max(mx.abs(self.output.astype(mx.float32) - self.oracle)).item())
        if not mx.allclose(self.output, self.oracle, atol=2e-3, rtol=2e-3).item():
            raise ValueError(f"attention disagrees with rounded FP32 equation: {error}")
        return Observation(
            hashlib.sha256(np.asarray(self.output.astype(mx.float32)).tobytes()).hexdigest(),
            {
                "prefix_tokens": self.prefix,
                "query_tokens": self.count,
                "logical_kv_bytes": self.keys.nbytes + self.values.nbytes,
                "physical_kv_bytes": self.budget.snapshot().reserved,
                "physical_runs": 1 + sum(b != a + 1 for a, b in pairwise(self.state.addresses)),
                "max_output_error": error,
            },
            {
                "implementation": type(self.operator).__qualname__,
                "oracle": "FP32 attention rounded to BF16",
            },
        )

    def close(self) -> None:
        self.arena.complete()
        self.output = None
        self.state.close()
        self.arena.release(self.guards)
        self.arena.close()
