"""Attached-head execution and restoration with captured target conditioning."""

from contextlib import ExitStack
from dataclasses import replace
from typing import Any

from magnitude_engine import components as c
from performance.assembly import Binding, BoundAssembly, inspect_engine, source_files
from performance.benchmarks.fixtures import record_inputs, tokens
from performance.benchmarks.numerics import compare
from performance.benchmarks.references import embedding
from performance.records import Assembly, Observation, digest
from performance.runner import recording


def benchmark(
    engine,
    *,
    context_tokens,
    history_tokens=0,
    query_tokens=1,
    mode="execute",
    reference=False,
    fixture="prose.moby-dick",
    prompt=None,
    continuation=None,
    **record,
):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.runtime import ForwardRequest

    assembly = inspect_engine(engine)
    if "draft" not in assembly.graph.nodes:
        raise ValueError("engine has no attached MTP head")
    generation = engine.engine.generation
    target, head = generation.model, generation.method.head
    program = assembly.at("draft").instance
    bound = assembly.at("draft" if mode == "execute" else "draft.state")
    if reference:
        sources = source_files(benchmark)
        node = replace(
            bound.node,
            binding=c.Implementation(c.QWEN_MTP, c.Source.MAG, "UPSTREAM_ADAPTER"),
            source=digest(sources),
            children={},
            dependencies={},
        )
        bound = Binding(
            BoundAssembly(
                Assembly(
                    "draft", {"draft": node}, "MTP upstream control", assembly.graph.artifacts
                ),
                {"draft": program},
                sources,
            ),
            "draft",
        )
    workload = {
        "context_tokens": context_tokens,
        "head_history_tokens": history_tokens,
        "histories": [history_tokens],
        "query_tokens": query_tokens,
        "batch_size": 1,
        "mode": mode,
        "fixture": fixture,
        "restore_mode": "saved_boundary",
        "numerical_contract": {"outputs": 0.002, "state": 0.00001},
    }
    with (
        recording(bound, benchmark="mtp." + mode, workload=workload, **record) as run,
        ExitStack() as life,
    ):
        if (
            not 0 <= history_tokens < context_tokens
            or query_tokens < 1
            or mode not in ("execute", "restore")
        ):
            raise ValueError("invalid MTP operating point")
        if mode == "restore" and reference:
            raise ValueError("restoration uses the checkpoint oracle")
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
        if (
            len(prompt) != context_tokens
            or continuation is None
            or len(continuation) < query_tokens
        ):
            raise ValueError("MTP conditioning inputs differ from operating point")
        record_inputs(
            run,
            prompt=tuple(prompt),
            continuation=tuple(continuation[:query_tokens]),
            producer=assembly.graph.component_keys()["target"],
        )
        all_tokens = (*prompt, *continuation[:query_tokens])
        start = context_tokens - history_tokens - 1
        target_row = target.create()
        life.callback(target_row.close)
        conditioning = []
        feature = generation.method.target_feature
        for offset in range(0, start, 512):
            target.prefill(target_row, all_tokens[offset : min(start, offset + 512)])
        for offset in range(start, context_tokens + query_tokens - 1, 512):
            values = all_tokens[offset : min(context_tokens + query_tokens - 1, offset + 512)]
            step = target.forward(
                target_row,
                values,
                ForwardRequest(
                    logits=False, features=frozenset((feature,)), committed_inputs=len(values)
                ),
            )
            step.accept(len(values))
            step.complete()
            conditioning.append(step.output.features[feature])
        previous = mx.concatenate(conditioning, axis=1)
        mx.eval(previous)
        target_row.close()
        initial = head.create()
        life.callback(initial.close)
        for offset in range(0, history_tokens, 512):
            end = min(history_tokens, offset + 512)
            inputs = ModelInputs(
                mx.array(
                    [
                        all_tokens[
                            context_tokens - history_tokens + offset : context_tokens
                            - history_tokens
                            + end
                        ]
                    ],
                    mx.int32,
                ),
                {"previous_hidden": previous[:, offset:end]},
            )
            step = head.forward(
                initial, inputs, ForwardRequest(logits=False, committed_inputs=end - offset)
            )
            step.accept(end - offset)
            step.complete()
        checkpoint = initial.checkpoint()
        life.callback(checkpoint.close)
        initial.close()
        inputs = ModelInputs(
            mx.array([continuation[:query_tokens]], mx.int32),
            {"previous_hidden": previous[:, history_tokens : history_tokens + query_tokens]},
        )
        embedding_path = assembly.graph.nodes["draft"].children["embedding"]
        emb = embedding(assembly.at(embedding_path).instance)
        row: Any = None
        caches = []

        def reset():
            nonlocal row, caches
            if row is not None:
                row.close()
            row = head.create(checkpoint) if mode == "execute" and not reference else None
            if row is not None:
                mx.eval(*head.states.arrays(row.state))
            caches = []
            for source in checkpoint.image.caches:
                cache = KVCache()
                cache.offset = source.offset
                cache.keys, cache.values = source.keys, source.values
                caches.append(cache)

        def upstream():
            hidden = program.combine(
                mx.concatenate(
                    [
                        program.normalize_embedding(emb(inputs.tokens)),
                        program.normalize_conditioning(inputs.conditioning["previous_hidden"]),
                    ],
                    axis=-1,
                )
            )
            for step, cache in zip(program.layers, caches, strict=True):
                hidden = step.layer(hidden, mask=create_attention_mask(hidden, cache), cache=cache)
            hidden = program.normalize_output(hidden)
            return program.project(hidden), hidden

        reset()
        expected = upstream()
        mx.eval(expected, *(c.state for c in caches))
        expected_kv = [c.state for c in caches]

        def execute():
            nonlocal row
            if mode == "restore":
                row = head.create(checkpoint)
                return ()
            if reference:
                return upstream()
            step = head.forward(
                row,
                inputs,
                ForwardRequest(
                    features=frozenset(("draft_hidden",)), committed_inputs=query_tokens
                ),
            )
            step.accept(query_tokens)
            step.complete()
            return step.output.logits, step.output.features["draft_hidden"]

        def complete(values):
            mx.eval(values, *(c.state for c in caches))
            if row is not None:
                row.complete_committed()
                mx.eval(*head.states.arrays(row.state))

        def validate(values):
            observed = (
                compare(values, expected, atol=0.002, rtol=0.002)
                if mode == "execute"
                else Observation(digest(history_tokens))
            )
            current = caches if reference else row.state.caches
            position = history_tokens + (query_tokens if mode == "execute" else 0)
            if any(c.offset != position for c in current):
                raise ValueError("MTP cache position differs from the requested boundary")
            wanted = (
                expected_kv if mode == "execute" else [c.state for c in checkpoint.image.caches]
            )
            for cache, expected_state in zip(current, wanted, strict=True):
                compare(
                    tuple(a for a in cache.state if a is not None),
                    tuple(a for a in expected_state if a is not None),
                )
            return Observation(observed.output_digest, observed.counters, {"fixture": provenance})

        try:
            run.measure(
                execute,
                prepare=reset,
                complete=complete,
                validate=validate,
                deterministic=True,
                dimension="EXEC" if mode == "execute" else "RESTORE",
            )
        finally:
            if row is not None:
                row.close()
    return run


def histories(engine, *, context_tokens=65536, lengths=(0, 4096, 16384), **record):
    return [
        benchmark(engine, context_tokens=context_tokens, history_tokens=n, **record)
        for n in lengths
    ]
