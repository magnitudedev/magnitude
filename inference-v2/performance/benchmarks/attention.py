"""Attention through completed output, with the actual selected KV implementation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from magnitude_engine.components import component
from performance.assembly import Binding, inspect_component
from performance.benchmarks.numerics import compare
from performance.facts import AttentionGeometry
from performance.records import digest
from performance.runner import recording

if TYPE_CHECKING:
    import mlx.core as mx

    from magnitude_engine.models.state.views import PagedKV


@dataclass
class Inputs:
    queries: mx.array
    kv: PagedKV
    scale: float
    reference: mx.array
    _identity: str | None = field(default=None, init=False, repr=False)

    def identity(self):
        if self._identity is None:
            import hashlib

            import mlx.core as mx
            import numpy as np

            h = hashlib.sha256()
            for value in (
                self.queries,
                *(a for row in range(len(self.kv.lengths)) for a in self.kv.gather(row)),
            ):
                h.update(str((value.shape, value.dtype)).encode())
                h.update(np.asarray(value.astype(mx.float32)).tobytes())
            self._identity = h.hexdigest()
        return self._identity


@contextmanager
def prepare(
    *,
    context_tokens,
    query_tokens,
    geometry: AttentionGeometry,
    dtype="bfloat16",
    fragmented=False,
    seed=131,
):
    import mlx.core as mx

    from magnitude_engine.models.state.arena import KVArena, LayerGeometry
    from magnitude_engine.models.state.pages import PageStore
    from magnitude_engine.models.state.views import read_layer
    from magnitude_engine.resources.budget import MemoryBudget

    if context_tokens < 0 or query_tokens < 1:
        raise ValueError("invalid attention lengths")
    hq, hk, dk, dv = (
        geometry.query_heads,
        geometry.kv_heads,
        geometry.key_width,
        geometry.value_width,
    )
    dtype = getattr(mx, dtype)
    total = context_tokens + query_tokens
    pages = (total + 15) // 16
    arena = KVArena(
        (LayerGeometry(hk, dk, dv),),
        page_size=16,
        slab_pages=32,
        max_pages=max(32, 2 * pages + 32),
        budget=MemoryBudget(4 << 30),
        dtype=dtype,
    )
    state, guards = None, ()
    try:
        if fragmented:
            allocated = arena.allocate(2 * pages)
            guards = allocated[::2]
            arena.release(allocated[1::2])
        state = PageStore(arena).create()
        state.reserve(total)
        mx.random.seed(seed)
        keys = mx.random.normal((1, hk, total, dk)).astype(dtype)
        values = mx.random.normal((1, hk, total, dv)).astype(dtype)
        queries = mx.random.normal((1, hq, query_tokens, dk)).astype(dtype)
        state.write(0, 0, keys[0], values[0])
        state.commit(total)
        kv = read_layer((state,), 0)
        scale = dk**-0.5
        mask = mx.arange(total)[None, :] <= (context_tokens + mx.arange(query_tokens))[:, None]
        if geometry.window is not None:
            first = context_tokens + mx.arange(query_tokens) - geometry.window + 1
            mask = mask & (mx.arange(total)[None, :] >= first[:, None])
        reference = mx.fast.scaled_dot_product_attention(
            queries.astype(mx.float32),
            keys.astype(mx.float32),
            values.astype(mx.float32),
            scale=scale,
            mask=mask,
        ).astype(dtype)
        mx.eval(queries, reference)
        arena.complete()
        with arena.pin():
            prepared = Inputs(queries, kv, scale, reference)
            prepared._identity = digest(
                {
                    "recipe": "normal-qkv-v1",
                    "seed": seed,
                    "geometry": geometry.model_dump(mode="json"),
                    "dtype": str(dtype),
                    "context_tokens": context_tokens,
                    "query_tokens": query_tokens,
                }
            )
            yield prepared
            mx.synchronize()
    finally:
        arena.complete()
        if state is not None:
            state.close()
        arena.release(guards)
        arena.close()


def benchmark(
    component,
    *,
    context_tokens,
    query_tokens=1,
    geometry: AttentionGeometry | None = None,
    dtype="bfloat16",
    fragmented=False,
    inputs=None,
    profile=None,
    output=None,
    warmup=2,
    repetitions=7,
):
    import mlx.core as mx

    if isinstance(component, Binding):
        binding = component
        if geometry is None:
            parameters = binding.node.parameters
            if not isinstance(parameters, AttentionGeometry):
                raise TypeError("bound attention is missing its geometry")
            geometry = parameters
    else:
        if geometry is None:
            raise ValueError("standalone attention requires typed geometry")
        binding = inspect_component(component, context=geometry).at("component")
    if not isinstance(geometry, AttentionGeometry):
        raise TypeError("attention requires AttentionGeometry")
    workload = {
        "histories": [context_tokens],
        "batch_size": 1,
        "query_tokens": query_tokens,
        "geometry": geometry.model_dump(mode="json"),
        "dtype": str(dtype),
        "fragmented": fragmented,
        "fixture": "synthetic.normal",
        "seed": 131,
        "numerical_contract": "fp32-attention-rounded-atol2e-3-rtol2e-3",
    }
    with recording(
        binding,
        benchmark="attention.execute",
        workload=workload,
        profile=profile,
        output=output,
        warmup=warmup,
        repetitions=repetitions,
    ) as run:

        def execute(prepared):
            shape = (1, geometry.query_heads, query_tokens, geometry.key_width)
            if (
                prepared.queries.shape != shape
                or prepared.queries.dtype.size != geometry.element_bytes
            ):
                raise ValueError("prepared attention inputs differ from declared geometry")
            if prepared.kv.lengths != (context_tokens + query_tokens,) or (
                prepared.kv.keys.shape[0] != geometry.kv_heads
                or prepared.kv.keys.shape[-1] != geometry.key_width
                or prepared.kv.values.shape[-1] != geometry.value_width
            ):
                raise ValueError("prepared KV differs from declared geometry")
            workload["input_digest"] = prepared.identity()
            run.measure(
                lambda: binding.instance.compute(
                    prepared.queries, prepared.kv, prepared.scale, window=geometry.window
                ),
                complete=mx.eval,
                validate=lambda value: compare(value, prepared.reference, atol=2e-3, rtol=2e-3),
                deterministic=True,
            )

        if inputs is not None:
            execute(inputs)
        else:
            with prepare(
                context_tokens=context_tokens,
                query_tokens=query_tokens,
                geometry=geometry,
                dtype=dtype,
                fragmented=fragmented,
            ) as prepared:
                execute(prepared)
    return run


def decode(component, *, geometry=None, contexts=(4096, 16384, 65536), **options):
    return [
        benchmark(component, geometry=geometry, context_tokens=context, **options)
        for context in contexts
    ]


def verification(component, *, geometry=None, contexts=(4096, 65536), queries=(2, 4, 8), **options):
    return [
        benchmark(component, geometry=geometry, context_tokens=context, query_tokens=q, **options)
        for context in contexts
        for q in queries
    ]


@component("MODEL:ATTENTION:MLX:DENSE")
def dense_attention(queries, keys, values, *, scale, mask):
    import mlx.core as mx

    return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)


def dense(
    *,
    context_tokens,
    query_tokens=1,
    geometry: AttentionGeometry,
    dtype="bfloat16",
    component=None,
    **record,
):
    import mlx.core as mx

    from performance.assembly import bind_operation
    from performance.benchmarks.references import attention_core_equation

    bound = bind_operation(
        component or dense_attention,
        dense_attention,
        parameters=geometry,
    )
    workload = {
        "histories": [context_tokens],
        "query_tokens": query_tokens,
        "batch_size": 1,
        "geometry": geometry.model_dump(mode="json"),
        "dtype": str(dtype),
        "fixture": "synthetic.normal",
        "seed": 131,
        "numerical_contract": "fp32-equation-atol2e-3-rtol2e-3",
    }
    with recording(
        bound,
        benchmark="attention.dense",
        workload=workload,
        boundary="dense-qkv-through-output-ready",
        **record,
    ) as run:
        dtype = getattr(mx, dtype)
        if dtype.size != geometry.element_bytes:
            raise ValueError("attention dtype differs from declared representation")
        mx.random.seed(131)
        q = mx.random.normal((1, geometry.query_heads, query_tokens, geometry.key_width)).astype(
            dtype
        )
        k = mx.random.normal(
            (1, geometry.kv_heads, context_tokens + query_tokens, geometry.key_width)
        ).astype(dtype)
        v = mx.random.normal(
            (1, geometry.kv_heads, context_tokens + query_tokens, geometry.value_width)
        ).astype(dtype)
        expected = attention_core_equation(q, k, v)
        mx.eval(q, k, v, expected)
        run.measure(
            lambda: bound.instance(
                q,
                k,
                v,
                scale=geometry.key_width**-0.5,
                mask="causal" if query_tokens > 1 else None,
            ),
            complete=mx.eval,
            validate=lambda result: compare(result, expected, atol=0.002, rtol=0.002),
            deterministic=True,
        )
    return run
