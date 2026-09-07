"""Plain and speculative service through the selected production generation runtime."""

import hashlib
import json
from contextlib import ExitStack

from performance.assembly import inspect_engine
from performance.benchmarks.fixtures import record_inputs, tokens
from performance.records import Observation
from performance.runner import recording


def benchmark(
    engine,
    *,
    context_tokens,
    output_tokens=32,
    rows=1,
    token_allowance=4,
    execution="shared",
    fixture="prose.moby-dick",
    validation="independent",
    prompts=None,
    profile=None,
    output=None,
    warmup=2,
    repetitions=7,
):
    import mlx.core as mx

    from magnitude_engine.generation.methods.plain.runtime import PlainMethod
    from magnitude_engine.generation.runtime import GenerationRuntime
    from magnitude_engine.generation.sampling_policy import SamplingPolicy

    if execution not in ("shared", "independent") or validation not in ("independent", "workload"):
        raise ValueError("invalid generation comparison contract")
    if min(context_tokens, output_tokens, rows, token_allowance) < 1:
        raise ValueError("generation work must be positive")
    bound = inspect_engine(engine).at("generation")
    generation = engine.engine.generation
    sampling = SamplingPolicy(temperature=0)
    workload = {
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "batch_size": rows,
        "token_allowance": token_allowance,
        "execution": execution,
        "fixture": fixture,
        "validation": validation,
    }
    with (
        recording(
            bound,
            benchmark="generation.advance",
            workload=workload,
            profile=profile,
            output=output,
            warmup=warmup,
            repetitions=repetitions,
        ) as run,
        ExitStack() as lifetime,
    ):
        prepared = []
        if prompts is None:
            prepared = [
                tokens(
                    engine.properties["target_path"],
                    fixture=fixture,
                    context_tokens=context_tokens + i,
                    continuation_tokens=output_tokens,
                )
                for i in range(rows)
            ]
            prompts = [p.prompt for p in prepared]
        if len(prompts) != rows or any(len(p) != context_tokens + i for i, p in enumerate(prompts)):
            raise ValueError("prompts differ from requested row lengths")
        record_inputs(run, prompts=prompts)
        workload["histories"] = [len(p) for p in prompts]
        checkpoints, expected, sequences = [], [], []
        reference = GenerationRuntime(generation.model, PlainMethod())
        for prompt in prompts:
            if validation == "independent":
                oracle = reference.create(prompt, sampling, output_tokens)
                try:
                    values = []
                    while not oracle.finished:
                        values.extend(oracle.step().tokens)
                    expected.append(values)
                finally:
                    oracle.close()
            seed = generation.create(prompt, sampling, output_tokens)
            try:
                checkpoint = seed.checkpoint()
                lifetime.callback(checkpoint.close)
                checkpoints.append(checkpoint)
            finally:
                seed.close()

        def reset():
            for sequence in sequences:
                sequence.close()
            sequences.clear()
            for prompt, checkpoint in zip(prompts, checkpoints, strict=True):
                sequences.append(
                    generation.create(prompt, sampling, output_tokens, checkpoint=checkpoint)
                )
            mx.reset_peak_memory()

        def execute():
            outputs, services = [[] for _ in sequences], []
            rounds = 0
            while any(not row.finished for row in sequences):
                rounds += 1
                if rounds > output_tokens * 4 + 32:
                    raise RuntimeError("generation made insufficient progress")
                indices = [i for i, row in enumerate(sequences) if not row.finished]
                if execution == "shared":
                    batches = generation.step_many(
                        tuple(sequences[i] for i in indices), (token_allowance,) * len(indices)
                    )
                    results = [s.outcome for s in batches]
                else:
                    results = [sequences[i].step(token_allowance) for i in indices]
                for index, result in zip(indices, results, strict=True):
                    if isinstance(result, BaseException):
                        raise result
                    if result is None:
                        continue
                    outputs[index].extend(result.tokens)
                    services.append(
                        {
                            "row": index,
                            "inputs": result.evaluated_inputs,
                            "outputs": len(result.tokens),
                            "proposed": result.proposed,
                            "accepted": result.accepted,
                            "forced": result.forced,
                        }
                    )
            return outputs, services

        def validate(result):
            outputs, services = result
            if any(len(row) != output_tokens for row in outputs) or not all(
                s.finished for s in sequences
            ):
                raise ValueError("generation did not complete requested output")
            if validation == "independent" and outputs != expected:
                raise ValueError("generation differs from independent plain execution")
            return Observation(
                hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
                {
                    "output_tokens": sum(map(len, outputs)),
                    "proposed_tokens": sum(s["proposed"] for s in services),
                    "accepted_tokens": sum(s["accepted"] for s in services),
                },
                {
                    "services": services,
                    "fixtures": [p.provenance for p in prepared],
                    "outputs": outputs,
                },
            )

        try:
            run.measure(
                execute,
                prepare=reset,
                complete=lambda _: generation.model.owner.backend.drain(),
                validate=validate,
                deterministic=True,
            )
        finally:
            for sequence in sequences:
                sequence.close()
    return run


def contexts(engine, *, lengths=(4096, 65536), **options):
    return [benchmark(engine, context_tokens=n, **options) for n in lengths]
