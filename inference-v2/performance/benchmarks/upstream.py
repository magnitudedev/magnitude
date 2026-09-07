"""Stock MLX-VLM controls over caller-loaded models and shared content fixtures."""

from typing import cast

from magnitude_engine.components import component
from performance.assembly import BoundAssembly, bind_operation
from performance.benchmarks.fixtures import record_inputs, tokens
from performance.facts import Configuration
from performance.records import Assembly, Observation, digest
from performance.runner import recording


def forward(
    model,
    *,
    artifact,
    context_tokens,
    measured_tokens=32,
    mode="generate",
    service_tokens=1,
    prefill_tokens=512,
    fixture="prose.moby-dick",
    prompt=None,
    continuation=None,
    eos_tokens=None,
    **record,
):
    import mlx.core as mx
    import numpy as np
    from mlx_vlm.models.cache import make_prompt_cache

    from magnitude_engine.models.state.native import _arrays, _detach

    language = getattr(model, "language_model", model)
    from performance.assembly import inspect_upstream

    assembly = inspect_upstream(language, artifact=artifact)
    graph = assembly.graph
    workload = {
        "context_tokens": context_tokens,
        "measured_tokens": measured_tokens,
        "mode": mode,
        "service_tokens": service_tokens,
        "prefill_tokens": prefill_tokens,
        "fixture": fixture,
    }
    with recording(
        assembly.at(graph.root),
        benchmark="model." + mode,
        workload=workload,
        boundary="upstream-forward-and-native-cache-ready",
        **record,
    ) as run:
        if (
            mode not in ("generate", "replay", "prefill")
            or min(measured_tokens, service_tokens, prefill_tokens) < 1
        ):
            raise ValueError("invalid upstream forward workload")
        provenance = {}
        if prompt is None:
            prepared = tokens(
                artifact,
                fixture=fixture,
                context_tokens=context_tokens,
                continuation_tokens=measured_tokens,
            )
            prompt, continuation, provenance = (
                prepared.prompt,
                prepared.continuation,
                prepared.provenance,
            )
        stopping = set(
            eos_tokens
            if eos_tokens is not None
            else cast(list[int], provenance.get("eos_tokens", []))
        )
        workload["eos_tokens"] = sorted(stopping)
        workload["stopping"] = "eos-or-limit" if stopping else "fixed-count"
        prompt, continuation = tuple(prompt), tuple(continuation or ())
        if not prompt or (mode != "generate" and len(continuation) < measured_tokens):
            raise ValueError("insufficient upstream inputs")
        record_inputs(run, prompt=prompt, continuation=continuation[:measured_tokens])
        history = prompt[:-1] if mode == "generate" else prompt
        caches = []
        saved = None

        def invoke(values):
            result = language(inputs=values, cache=caches)
            return result if isinstance(result, mx.array) else result.logits

        def complete(_=None):
            mx.eval([cache.state for cache in caches])
            mx.synchronize()

        def reset():
            nonlocal caches, saved
            complete()
            if saved is None:
                caches = make_prompt_cache(language)
                for start in range(0, len(history), prefill_tokens):
                    logits = invoke(mx.array([history[start : start + prefill_tokens]], mx.int32))
                    mx.eval(logits)
                    complete()
                saved = _detach(caches)
                mx.eval(*_arrays(saved))
            # Prefix construction is outside this measurement. Restore the same
            # completed native cache for each trial, including recurrent and ring
            # metadata, instead of repeating long-context prefill for every sample.
            caches = _detach(saved)
            complete()
            mx.reset_peak_memory()

        def execute():
            outputs = []
            if mode == "prefill":
                invoke(mx.array([continuation[:measured_tokens]], mx.int32))
                value = None
            else:
                current = mx.array([[prompt[-1]]], mx.int32)
                value = None
                for i in range(measured_tokens):
                    if mode == "replay":
                        current = mx.array([[continuation[i]]], mx.int32)
                    value = invoke(current)
                    if mode == "generate":
                        next_token = mx.argmax(value[0, -1]).astype(mx.int32)
                        mx.async_eval(next_token)
                        current = next_token.reshape(1, 1)
                        outputs.append(next_token)
                        if stopping and cast(int, next_token.item()) in stopping:
                            break
                    else:
                        mx.async_eval(value)
                    if (i + 1) % service_tokens == 0:
                        complete()
                mx.eval(value, outputs)
            return value, outputs

        def validate(result):
            value, outputs = result
            values = [int(token.item()) for token in outputs]
            if (
                mode == "generate"
                and len(outputs) != measured_tokens
                and not (values and values[-1] in stopping)
            ):
                raise ValueError("upstream produced incorrect output count")
            consumed = len(outputs) if mode == "generate" else measured_tokens
            for cache in caches:
                offset = getattr(cache, "offset", None)
                if isinstance(offset, int) and offset != len(history) + consumed:
                    raise ValueError("upstream cache did not reach the requested boundary")
            return Observation(
                digest(
                    np.asarray(value.astype(mx.float32)).tolist() if value is not None else consumed
                ),
                {"input_tokens": consumed, "output_tokens": len(outputs)},
                {"fixture": provenance, "tokens": values},
            )

        try:
            run.measure(
                execute, prepare=reset, complete=complete, validate=validate, deterministic=True
            )
        finally:
            complete()
            caches.clear()
    return run


def contexts(model, *, artifact, lengths=(4096, 16384, 65536), **options):
    return [forward(model, artifact=artifact, context_tokens=n, **options) for n in lengths]


def batch(
    model,
    processor,
    *,
    artifact,
    context_tokens,
    output_tokens=32,
    rows=2,
    prefill_tokens=512,
    waves=2,
    fixture="prose.moby-dick",
    prompts=None,
    **record,
):
    from time import perf_counter_ns

    import mlx.core as mx
    from mlx_vlm.utils import StoppingCriteria

    binding = bind_operation(
        batch_generator,
        batch_generator,
        parameters=Configuration(settings={"prefill_tokens": prefill_tokens, "max_batch": rows}),
    )
    from dataclasses import replace

    from performance.assembly import inspect_upstream

    target = inspect_upstream(getattr(model, "language_model", model), artifact=artifact)
    graph = binding.assembly.graph
    nodes = {**graph.nodes, **target.graph.nodes}
    nodes[graph.root] = replace(nodes[graph.root], children={"target": target.graph.root})
    assembly = BoundAssembly(
        Assembly(graph.root, nodes, "MLX-VLM batch generation", target.graph.artifacts),
        {**binding.assembly.objects, **target.objects},
        {**binding.assembly.sources, **target.sources},
    )
    workload = {
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "rows": rows,
        "waves": waves,
        "fixture": fixture,
        "statistics": {"TTFT": "max", "GAP": "max"},
    }
    with recording(
        assembly.at(graph.root),
        benchmark="upstream.batch",
        workload=workload,
        boundary="upstream-insert-through-completed-generation",
        **record,
    ) as run:
        if min(context_tokens, output_tokens, rows, prefill_tokens, waves) < 1:
            raise ValueError("invalid upstream batch workload")
        if prompts is None:
            prompts = [
                tokens(
                    artifact,
                    fixture=fixture,
                    context_tokens=context_tokens + i,
                    continuation_tokens=output_tokens,
                ).prompt
                for i in range(rows)
            ]
        if len(prompts) != rows:
            raise ValueError("one prompt per row is required")
        record_inputs(run, prompts=prompts)
        tokenizer = getattr(processor, "tokenizer", processor)
        previous_stopping = tokenizer.stopping_criteria
        tokenizer.stopping_criteria = StoppingCriteria([], tokenizer)

        def execute():
            outputs, intervals, first_times, gaps = [], [], [], []
            for _ in range(waves):
                start = perf_counter_ns()
                generator = batch_generator(
                    model.language_model,
                    processor,
                    max_tokens=output_tokens,
                    completion_batch_size=rows,
                    prefill_batch_size=rows,
                    prefill_step_size=prefill_tokens,
                    compute_logprobs=False,
                    greedy_sampling=True,
                )
                try:
                    kwargs = [
                        {
                            k: v
                            for k, v in model.get_input_embeddings(mx.array([prompt]), None)
                            .to_dict()
                            .items()
                            if v is not None
                        }
                        for prompt in prompts
                    ]
                    uids = generator.insert([list(p) for p in prompts], prompt_kwargs=kwargs)
                    values = {uid: [] for uid in uids}
                    first, last, finished = {}, {}, set()
                    for _ in range(sum(map(len, prompts)) + output_tokens * rows + rows + 10):
                        _, responses = generator.next()
                        now = perf_counter_ns() - start
                        for response in responses:
                            uid = response.uid
                            values[uid].append(response.token)
                            first.setdefault(uid, now)
                            if uid in last:
                                gaps.append(now - last[uid])
                            last[uid] = now
                            if response.finish_reason is not None:
                                if response.finish_reason != "length":
                                    raise ValueError("upstream ended before its output allowance")
                                finished.add(uid)
                        if len(finished) == rows:
                            break
                    else:
                        raise RuntimeError("upstream batch exceeded declared work bound")
                    mx.synchronize()
                    intervals.append(perf_counter_ns() - start)
                    first_times.extend(first[uid] for uid in uids)
                    outputs.extend(values[uid] for uid in uids)
                finally:
                    generator.close()
            return outputs, intervals, first_times, gaps

        def validate(result):
            outputs, intervals, first, gaps = result
            if len(outputs) != rows * waves or any(len(row) != output_tokens for row in outputs):
                raise ValueError("upstream batch did not complete declared work")
            metrics = {
                "RATE": sum(map(len, outputs)) / (sum(intervals) / 1e9),
                "TTFT": max(first) / 1e9,
            }
            if gaps:
                metrics["GAP"] = max(gaps) / 1e9
            return Observation(
                digest(outputs),
                {"output_tokens": sum(map(len, outputs))},
                {
                    "outputs": outputs,
                    "wave_intervals_ns": intervals,
                    "first_token_ns": first,
                    "gaps_ns": gaps,
                },
                metrics,
            )

        try:
            run.measure(
                execute,
                complete=lambda _: mx.synchronize(),
                validate=validate,
                deterministic=True,
                dimension=None,
            )
        finally:
            tokenizer.stopping_criteria = previous_stopping
    return run


@component("ENGINE:INFERENCE:VLM:BATCH_GENERATOR")
def batch_generator(*args, **kwargs):
    from mlx_vlm.generate.ar import BatchGenerator

    return BatchGenerator(*args, **kwargs)
