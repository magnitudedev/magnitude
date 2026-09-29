"""Complete request service, with publication timestamps and independent plain controls."""

from dataclasses import asdict
from time import perf_counter_ns

from performance.assembly import inspect_engine
from performance.benchmarks.fixtures import record_inputs, tokens
from performance.records import Observation, digest
from performance.runner import recording


def benchmark(
    residency,
    *,
    context_tokens,
    output_tokens=32,
    rows=1,
    waves=2,
    prefix_reuse=True,
    fixture="prose.moby-dick",
    prompts=None,
    validation="independent",
    **record,
):
    import mlx.core as mx

    from magnitude_engine.engine.delivery import Finished, Tokens
    from magnitude_engine.engine.requests import GenerationRequest
    from magnitude_engine.generation.methods.plain.runtime import PlainMethod
    from magnitude_engine.generation.runtime import GenerationRuntime
    from magnitude_engine.generation.sampling_policy import SamplingPolicy

    bound = inspect_engine(residency).at("engine")
    engine = residency.engine
    owner = engine.generation.model.owner
    sampling = SamplingPolicy(temperature=0)
    workload = {
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "rows": rows,
        "waves": waves,
        "prefix_reuse": prefix_reuse,
        "fixture": fixture,
        "validation": validation,
        "statistics": {"TTFT": "max", "GAP": "max"},
    }
    with recording(
        bound,
        benchmark="engine.waves",
        workload=workload,
        boundary="submission-through-publication-and-drained-delivery",
        **record,
    ) as run:
        if min(context_tokens, rows, output_tokens, waves) < 1 or validation not in (
            "independent",
            "workload",
        ):
            raise ValueError("invalid engine workload")
        provenance = []
        if prompts is None:
            prepared = [
                tokens(
                    residency.properties["target_path"],
                    fixture=fixture,
                    context_tokens=context_tokens + i,
                    continuation_tokens=output_tokens,
                )
                for i in range(rows)
            ]
            prompts, provenance = [p.prompt for p in prepared], [p.provenance for p in prepared]
        prompts = [tuple(p) for p in prompts]
        if len(prompts) != rows or any(not p for p in prompts):
            raise ValueError("provide a nonempty prompt for every row")
        record_inputs(run, prompts=prompts)
        workload["prompt_lengths"] = [len(p) for p in prompts]
        expected = []
        if validation == "independent":
            reference = GenerationRuntime(engine.generation.model, PlainMethod())
            for prompt in prompts:
                oracle = reference.create(prompt, sampling, output_tokens)
                try:
                    generated = []
                    while not oracle.finished:
                        generated.extend(oracle.step().tokens)
                    expected.append(generated)
                finally:
                    oracle.close()

        def reset():
            owner.backend.drain()
            engine.prefixes.close()
            engine.scheduler.reset()
            mx.reset_peak_memory()

        def execute():
            outputs, finishes, services, publications, intervals = [], [], [], [], []
            for wave in range(waves):
                start = perf_counter_ns()
                handles, originals = [], []
                values = [[] for _ in prompts]
                finished: list[Finished | None] = [None for _ in prompts]
                try:
                    for i, prompt in enumerate(prompts):
                        handle = engine.submit(
                            GenerationRequest(prompt, sampling, output_tokens),
                            output_capacity=residency.output_capacity,
                        )
                        handles.append(handle)
                        original = handle.delivery.publish
                        originals.append(original)

                        def publish(values, original=original, i=i, start=start, wave=wave):
                            original(values)
                            if values:
                                publications.append(
                                    {
                                        "wave": wave,
                                        "row": i,
                                        "tokens": len(values),
                                        "elapsed_ns": perf_counter_ns() - start,
                                    }
                                )

                        handle.delivery.publish = publish
                    work_limit = sum(map(len, prompts)) + rows * (output_tokens + 2)
                    for _ in range(work_limit):
                        services.extend(asdict(s) for s in engine.tick())
                        for i, handle in enumerate(handles):
                            if finished[i] is not None:
                                continue
                            while True:
                                try:
                                    event = handle.delivery.take(0)
                                except TimeoutError:
                                    break
                                if isinstance(event, Tokens):
                                    values[i].extend(event.values)
                                elif isinstance(event, Finished):
                                    finished[i] = event
                                    break
                        if all(f is not None for f in finished):
                            break
                    else:
                        raise RuntimeError("engine exceeded declared work bound")
                    intervals.append(perf_counter_ns() - start)
                    outputs.extend(values)
                    finishes.extend(finished)
                finally:
                    for handle, original in zip(handles, originals, strict=True):
                        handle.delivery.publish = original
                    if any(f is None for f in finished):
                        for handle in handles:
                            handle.cancel()
                        # Cancellation is resolved by the same production owner.
                        for _ in range(len(handles) + 1):
                            engine.tick()
            return outputs, finishes, services, publications, intervals

        def validate(result):
            outputs, finishes, services, publications, intervals = result
            if len(outputs) != rows * waves or any(len(o) != output_tokens for o in outputs):
                raise ValueError("engine did not complete every output allowance")
            if any(
                not isinstance(f, Finished)
                or f.reason != "length"
                or f.generated_tokens != output_tokens
                for f in finishes
            ):
                raise ValueError("engine did not finish every request normally")
            if validation == "independent" and any(
                o != expected[i % rows] for i, o in enumerate(outputs)
            ):
                raise ValueError("engine differs from independent plain generation")
            for i, finish in enumerate(finishes):
                cached = len(prompts[i % rows]) - 1 if i >= rows and prefix_reuse else 0
                if finish.cached_tokens != cached:
                    raise ValueError("actual retained-prefix behavior differs from workload")
            gaps = []
            for wave in range(waves):
                for i in range(rows):
                    times = [
                        p["elapsed_ns"] for p in publications if p["wave"] == wave and p["row"] == i
                    ]
                    gaps.extend(b - a for a, b in zip(times, times[1:], strict=False))
            metrics = {
                "RATE": sum(map(len, outputs)) / (sum(intervals) / 1e9),
                "TTFT": max(f.first_token_ns for f in finishes) / 1e9,
            }
            if gaps:
                metrics["GAP"] = max(gaps) / 1e9
            return Observation(
                digest(outputs),
                {
                    "requests": len(finishes),
                    "output_tokens": sum(map(len, outputs)),
                    "cached_tokens": sum(f.cached_tokens for f in finishes),
                },
                {
                    "outputs": outputs,
                    "requests": [asdict(f) for f in finishes],
                    "services": services,
                    "publications": publications,
                    "wave_intervals_ns": intervals,
                    "fixtures": provenance,
                },
                metrics,
            )

        run.measure(
            execute,
            prepare=reset,
            complete=lambda _: owner.backend.drain(),
            validate=validate,
            deterministic=True,
            dimension=None,
        )
    return run


def workloads(residency, *, contexts=(4096, 16384, 65536), **record):
    return [benchmark(residency, context_tokens=context, **record) for context in contexts]
