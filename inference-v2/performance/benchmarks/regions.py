"""Capture architecture inputs once; measure each real component independently."""

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import Any, cast

from magnitude_engine.components import component_id
from magnitude_engine.models.architectures.qwen35.attention.operation import GatedAttention
from magnitude_engine.models.architectures.qwen35.feedforward.operation import RoutedFeedForward
from magnitude_engine.models.architectures.qwen35.program import readout as qwen_readout
from magnitude_engine.models.architectures.qwen35.recurrence.operation import RecurrentMixer
from magnitude_engine.models.embeddings.resident import ResidentEmbedding
from magnitude_engine.models.experts.computation import ResidentExperts
from performance.assembly import Binding, BoundAssembly, inspect_engine, source_files
from performance.benchmarks import references
from performance.benchmarks.fixtures import record_inputs, tokens
from performance.benchmarks.model import prefix
from performance.benchmarks.numerics import compare
from performance.records import Assembly, Observation, digest
from performance.runner import recording


@dataclass
class CaptureMixer:
    inner: Any
    index: int
    captured: dict

    def compute_batch(self, hidden, states, scope):
        if hasattr(self.inner, "operation") and hasattr(self.inner.operation, "graph"):
            slot = states[0].slots[self.inner.index]
            self.captured[self.index] = (hidden, *slot.values)
        else:
            self.captured[self.index] = (hidden,)
        return self.inner.compute_batch(hidden, states, scope)


@dataclass
class CaptureFeedForward:
    inner: Any
    index: int
    captured: dict

    def compute(self, hidden, scope):
        self.captured[self.index] = hidden
        return self.inner.compute(hidden, scope)


@contextmanager
def capture(
    engine,
    *,
    context_tokens,
    query_tokens,
    fixture="prose.moby-dick",
    prompt=None,
    continuation=None,
):
    import mlx.core as mx

    from magnitude_engine.models.architectures.qwen35.program import Qwen35Program
    from magnitude_engine.models.runtime import ForwardRequest

    model = engine.engine.generation.model
    assembly = inspect_engine(engine)
    program = assembly.at("target").instance
    if not isinstance(program, Qwen35Program):
        raise TypeError("this input capture binds the Qwen layer contract")
    provenance = {}
    if prompt is None:
        prepared = tokens(
            engine.properties["target_path"],
            fixture=fixture,
            context_tokens=context_tokens,
            continuation_tokens=query_tokens,
        )
        prompt, continuation, provenance = (
            prepared.prompt,
            prepared.continuation,
            prepared.provenance,
        )
    if continuation is None or len(continuation) < query_tokens:
        raise ValueError("insufficient captured continuation")
    values = {
        "tokens": mx.array([continuation[:query_tokens]], mx.int32),
        "prompt": tuple(prompt),
        "producer": assembly.graph.component_keys()["target"],
        "ff": {},
        "mixer": {},
        "kv": [],
        "provenance": provenance,
        "context_tokens": len(prompt),
        "query_tokens": query_tokens,
    }
    with prefix(model, prompt) as checkpoint:
        values["checkpoint"] = checkpoint
        row = model.create(checkpoint)
        blocks = program.blocks
        try:
            values["kv"] = [
                tuple(a[None] for a in row.state.pages.read(i))
                for i in range(len(row.state.pages.store.arena.layers))
            ]
            mx.eval(values["kv"])
            program.blocks = tuple(
                replace(
                    block,
                    mixer=CaptureMixer(block.mixer, i, values["mixer"]),
                    feedforward=CaptureFeedForward(block.feedforward, i, values["ff"]),
                )
                for i, block in enumerate(blocks)
            )
            last = f"residual:{len(blocks)}"
            step = model.forward(
                row,
                continuation[:query_tokens],
                ForwardRequest(
                    features=frozenset(("residual:0", last)), committed_inputs=query_tokens
                ),
            )
            step.accept(query_tokens)
            step.complete()
            values["last_hidden"] = step.output.features[last]
            values["readout"] = step.output.logits
            values["embedded"] = step.output.features["residual:0"]
            mx.eval(values["ff"], values["mixer"], values["last_hidden"], values["embedded"])
        finally:
            program.blocks = blocks
            row.close()
        yield values


def benchmark(
    engine,
    path,
    *,
    context_tokens,
    query_tokens=1,
    rows=1,
    fixture="prose.moby-dick",
    prepared=None,
    reference=False,
    attention_oracle="fp32-equation",
    **record,
):
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, KVCache

    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.state.views import read_layer

    assembly = inspect_engine(engine)
    bound = assembly.at(path)
    model = engine.engine.generation.model
    program = assembly.at("target").instance
    op = bound.instance
    component = bound.node.component
    if rows < 1 or (
        rows > 1
        and component in (component_id(GatedAttention).kind, component_id(RecurrentMixer).kind)
    ):
        raise ValueError("multiple captured rows require a stateless region")
    index = int(path.split(".layers.")[1].split(".")[0]) if ".layers." in path else None
    # Reference adapters borrow the exact bound weights. They are isolated controls;
    # their measurements do not pretend the production parent invoked the adapter.
    if reference:
        sources = source_files(references.attention)
        node = replace(
            bound.node,
            binding=component_id(
                {
                    component_id(GatedAttention).kind: references.attention_equation
                    if attention_oracle == "fp32-equation"
                    else references.attention,
                    component_id(RecurrentMixer).kind: references.recurrence,
                    component_id(RoutedFeedForward).kind: references.feedforward,
                    component_id(ResidentExperts).kind: references.experts,
                    component_id(ResidentEmbedding).kind: references.embedding,
                    component_id(qwen_readout).kind: references.readout,
                }[component]
            ),
            source=digest(sources),
            children={},
            dependencies={},
        )
        bound = Binding(
            BoundAssembly(
                Assembly(path, {path: node}, node.implementation, assembly.graph.artifacts),
                {path: op},
                sources,
            ),
            path,
        )
    workload = {
        "histories": [context_tokens] * rows,
        "context_tokens": context_tokens,
        "query_tokens": query_tokens,
        "batch_size": rows,
        "input_layout": "captured-row" if rows == 1 else "replicated-captured-row",
        "fixture": fixture,
        "attention_oracle": attention_oracle,
        "numerical_contract": {"atol": 0.002, "rtol": 0.002},
    }
    with (
        recording(bound, benchmark="neural.region", workload=workload, **record) as run,
        ExitStack() as life,
    ):
        p = prepared or life.enter_context(
            capture(
                engine, context_tokens=context_tokens, query_tokens=query_tokens, fixture=fixture
            )
        )
        if p["context_tokens"] != context_tokens or p["query_tokens"] != query_tokens:
            raise ValueError("prepared input operating point differs")
        record_inputs(
            run,
            provenance=p["provenance"],
            tokens=p["tokens"].tolist(),
            producer=p["producer"],
            prompt=p["prompt"],
        )
        control: Callable[..., Any] | None = None
        hidden = indices = scores = None
        if component == component_id(GatedAttention).kind:
            hidden = p["mixer"][index][0]
            control = (
                references.attention_equation(op)
                if attention_oracle == "fp32-equation"
                else references.attention(op)
            )
        elif component == component_id(RecurrentMixer).kind:
            hidden = p["mixer"][index][0]
            control = references.recurrence(op.operation)
        elif component == component_id(RoutedFeedForward).kind:
            hidden = p["ff"][index]
            control = (
                references.feedforward(op) if hasattr(op, "experts") else references.dense(op.call)
            )
        elif component == component_id(ResidentExperts).kind:
            hidden = p["ff"][index]
            parent = program.blocks[index].feedforward
            indices, scores, _ = parent.route(hidden)
            mx.eval(indices, scores)
            workload["distinct_experts"] = len(set(cast(list[int], indices.reshape(-1).tolist())))
            control = references.experts(op)
        elif component == component_id(ResidentEmbedding).kind:
            control = references.embedding(op)
            workload["distinct_input_tokens"] = len(set(p["tokens"].reshape(-1).tolist()))
        elif component == component_id(qwen_readout).kind:
            hidden = program.norm(p["last_hidden"])
            control = references.readout(op)
        else:
            raise TypeError(f"no invocation contract for {component}")
        if hidden is not None:
            hidden = mx.contiguous(mx.repeat(hidden, rows, axis=0)) if rows > 1 else hidden
            mx.eval(hidden)
        if indices is not None and rows > 1:
            assert scores is not None
            indices, scores = (mx.contiguous(mx.repeat(a, rows, axis=0)) for a in (indices, scores))
            mx.eval(indices, scores)
        input_tokens = mx.repeat(p["tokens"], rows, axis=0) if rows > 1 else p["tokens"]
        mx.eval(input_tokens)
        row: Any = None
        transaction = None
        caches = []

        def reset():
            nonlocal row, transaction, caches
            if transaction is not None:
                transaction.close()
                transaction = None
            if row is not None:
                row.close()
                row = None
            if component in (component_id(GatedAttention).kind, component_id(RecurrentMixer).kind):
                row = model.create(p["checkpoint"])
                model.reserve(row, query_tokens)
                if component == component_id(RecurrentMixer).kind:
                    transaction = model.states.begin(
                        row.state, ModelInputs(p["tokens"]), committed_inputs=query_tokens
                    )
            caches = []
            for keys, values in p["kv"]:
                cache = KVCache()
                cache.keys, cache.values, cache.offset = keys, values, keys.shape[2]
                caches.append(cache)

        def execute(use_reference):
            call = cast(Callable[..., Any], control)
            with model.owner.scope() as scope:
                if component == component_id(ResidentEmbedding).kind:
                    outputs = [
                        call(input_tokens) if use_reference else op.lookup(input_tokens, scope)
                    ]
                elif component == component_id(qwen_readout).kind:
                    outputs = [call(hidden) if use_reference else op(hidden)]
                elif component == component_id(GatedAttention).kind:
                    scope.enter(row.state.pages.store.arena.pin())
                    outputs = [
                        call(
                            hidden,
                            cache=caches[op.index],
                            mask="causal" if query_tokens > 1 else None,
                        )
                        if use_reference
                        else op.compute_batch(hidden, (row.state,), scope)
                    ]
                elif component == component_id(RecurrentMixer).kind:
                    if use_reference:
                        cache = ArraysCache(2)
                        cache[0], cache[1] = p["mixer"][index][1:]
                        value = call(hidden, cache=cache)
                        outputs = [value, cache[0], cache[1]]
                    else:
                        value = op.compute_batch(hidden, (row.state,), scope)
                        outputs = [value, *row.state.slots[op.index].pending.values]
                elif component == component_id(ResidentExperts).kind:
                    outputs = [
                        call(hidden, indices, scores)
                        if use_reference
                        else op.compute(hidden, indices, scores, scope)
                    ]
                else:
                    outputs = [call(hidden) if use_reference else op.compute(hidden, scope)]
                pending = scope.seal(*outputs)
            return outputs, pending

        def complete(result):
            result[1].complete()

        try:
            reset()
            wanted = execute(True)
            complete(wanted)
            expected = wanted[0]
            expected_kv = None
            if component == component_id(GatedAttention).kind:
                cache = caches[op.index]
                end = context_tokens + query_tokens
                expected_kv = (cache.keys[:, :, :end], cache.values[:, :, :end])
                mx.eval(expected_kv)

            def validate(result):
                observed = compare(result[0], expected, atol=0.002, rtol=0.002)
                if component == component_id(RecurrentMixer).kind:
                    compare(result[0][-1], expected[-1], atol=1e-5, rtol=1e-5)
                if expected_kv is not None:
                    if reference:
                        cache = caches[op.index]
                        current = cache.keys[:, :, :end], cache.values[:, :, :end]
                    else:
                        view = read_layer((row.state.pages,), op.index, pending_tokens=query_tokens)
                        current = tuple(a[None] for a in view.gather(0))
                    compare(current, expected_kv)
                return Observation(
                    observed.output_digest,
                    observed.counters,
                    {"fixture": p["provenance"], "prepared_inputs": True},
                )

            run.measure(
                lambda: execute(reference),
                prepare=reset,
                complete=complete,
                validate=validate,
                deterministic=True,
            )
        finally:
            if transaction is not None:
                transaction.close()
            if row is not None:
                row.close()
    return run


def qwen_layers(
    engine, *, context_tokens=4096, query_tokens=1, fixture="prose.moby-dick", **record
):
    graph = inspect_engine(engine).graph
    results = []
    # Shared capture is deliberately outside all parent measurements.
    with capture(
        engine, context_tokens=context_tokens, query_tokens=query_tokens, fixture=fixture
    ) as inputs:
        for path, node in graph.nodes.items():
            if node.component in (
                "MODEL:EMBEDDING",
                "MODEL:QWEN35.ATTENTION",
                "MODEL:QWEN35.RECURRENCE",
                "MODEL:QWEN35.FEEDFORWARD",
                "MODEL:EXPERTS",
                "MODEL:QWEN35.READOUT",
            ):
                results.append(
                    benchmark(
                        engine,
                        path,
                        context_tokens=context_tokens,
                        query_tokens=query_tokens,
                        fixture=fixture,
                        prepared=inputs,
                        **record,
                    )
                )
    return results
