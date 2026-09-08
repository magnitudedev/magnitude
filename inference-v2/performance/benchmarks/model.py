"""Completed model runtime work on a restored prefix, including state transactions."""

import hashlib
from contextlib import contextmanager
from typing import cast

from performance.assembly import inspect_engine
from performance.benchmarks.fixtures import record_inputs, tokens
from performance.records import Observation
from performance.runner import recording


@contextmanager
def prefix(model, prompt, *, chunk_size=512):
    seed = model.create()
    checkpoint = None
    try:
        for start in range(0, len(prompt), chunk_size):
            model.prefill(seed, prompt[start : start + chunk_size])
        seed.complete_committed()
        checkpoint = seed.checkpoint()
    finally:
        seed.close()
    try:
        yield checkpoint
    finally:
        checkpoint.close()


def benchmark(
    engine,
    *,
    context_tokens,
    measured_tokens=32,
    rows=1,
    mode="replay",
    fixture="prose.moby-dick",
    prompt=None,
    continuation=None,
    eos_tokens=None,
    accepted_tokens=None,
    profile=None,
    output=None,
    warmup=2,
    repetitions=7,
):
    import mlx.core as mx

    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.runtime import ForwardRequest

    if mode not in ("replay", "prefill", "generate", "verify") or measured_tokens < 1:
        raise ValueError("invalid model execution mode")
    if rows < 1 or (rows > 1 and mode not in ("replay", "verify")):
        raise ValueError("multiple rows require fixed-input replay or verification")
    if accepted_tokens is not None and (
        mode != "verify" or not 0 <= accepted_tokens <= measured_tokens
    ):
        raise ValueError("accepted prefix requires verification and must fit its input")
    accepted = measured_tokens if accepted_tokens is None else accepted_tokens
    assembly = inspect_engine(engine)
    model = engine.engine.generation.model
    workload = {
        "context_tokens": context_tokens,
        "histories": [context_tokens] * rows,
        "batch_size": rows,
        "query_tokens": measured_tokens if mode in ("prefill", "verify") else 1,
        "measured_tokens": measured_tokens,
        "mode": mode,
        "fixture": fixture,
    }
    if mode == "verify":
        workload["accepted_tokens"] = accepted
    with recording(
        assembly.at("target"),
        benchmark="model." + mode,
        workload=workload,
        profile=profile,
        output=output,
        warmup=warmup,
        repetitions=repetitions,
    ) as run:
        provenance = {}
        if prompt is None:
            prepared = tokens(
                engine.properties["target_path"],
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
            raise ValueError("insufficient prompt or continuation inputs")
        record_inputs(run, prompt=prompt, continuation=continuation[:measured_tokens])
        history = prompt[:-1] if mode == "generate" else prompt
        workload["histories"], workload["context_tokens"] = [len(history)] * rows, len(history)
        with prefix(
            model, history, chunk_size=engine.engine.scheduler.prefill_tokens
        ) as checkpoint:
            sequences = []

            def close():
                for sequence in sequences:
                    sequence.close()
                sequences.clear()

            def reset():
                close()
                for _ in range(rows):
                    sequence = model.create(checkpoint)
                    sequences.append(sequence)
                    model.reserve(sequence, measured_tokens + 1)
                mx.reset_peak_memory()

            def forward(values, request, count):
                advances = (
                    (model.forward(sequences[0], values, request),)
                    if rows == 1
                    else model.forward_batch(
                        tuple(sequences), (ModelInputs.from_tokens(values),) * rows, request
                    )
                )
                for advance in advances:
                    advance.accept(count)
                    advance.complete()
                return mx.concatenate([advance.output.logits for advance in advances])

            def execute():
                generated, last = [], None
                if mode == "verify":
                    last = forward(continuation[:measured_tokens], ForwardRequest(), accepted)
                elif mode == "prefill":
                    model.prefill(sequences[0], continuation[:measured_tokens])
                else:
                    token = prompt[-1]
                    for i in range(measured_tokens):
                        value = token if mode == "generate" else continuation[i]
                        last = forward((value,), ForwardRequest(committed_inputs=1), 1)
                        if mode == "generate":
                            token = cast(int, mx.argmax(last[0, -1]).item())
                            generated.append(token)
                            if token in stopping:
                                break
                return last, generated

            def validate(result):
                last, generated = result
                for sequence in sequences:
                    saved = sequence.checkpoint()
                    try:
                        consumed = len(generated) if mode == "generate" else accepted
                        if saved.length != len(history) + consumed:
                            raise ValueError(
                                "model committed boundary differs from consumed inputs"
                            )
                    finally:
                        saved.close()
                h = hashlib.sha256(
                    bytes(memoryview(last.astype(mx.float32))) if last is not None else b""
                ).hexdigest()
                return Observation(
                    h,
                    {
                        "input_tokens": len(generated)
                        if mode == "generate"
                        else measured_tokens * rows,
                        "output_tokens": len(generated),
                        "mlx_peak_bytes": mx.get_peak_memory(),
                    },
                    {"fixture": provenance, "generated_tokens": generated},
                )

            try:
                run.measure(
                    execute,
                    prepare=reset,
                    complete=lambda _: model.owner.backend.drain(),
                    validate=validate,
                    deterministic=mode != "prefill",
                )
            finally:
                close()
    return run


def prose(engine, *, contexts=(4096, 16384, 65536), **options):
    return [
        benchmark(engine, context_tokens=n, fixture="prose.moby-dick", **options) for n in contexts
    ]


def tools(engine, *, contexts=(4096, 65536), **options):
    return [benchmark(engine, context_tokens=n, fixture="tools.bfcl", **options) for n in contexts]


def prefill_batch(
    engine,
    *,
    context_tokens=0,
    input_tokens=512,
    rows=4,
    execution="shared",
    fixture="prose.moby-dick",
    prompt=None,
    continuation=None,
    **record,
):
    import mlx.core as mx

    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.runtime import ForwardRequest
    from performance.benchmarks.numerics import compare

    bound = inspect_engine(engine).at("target")
    model = engine.engine.generation.model
    workload = {
        "histories": [context_tokens] * rows,
        "query_tokens": input_tokens,
        "batch_size": rows,
        "execution": execution,
        "fixture": fixture,
    }
    with recording(bound, benchmark="model.prefill_batch", workload=workload, **record) as run:
        if rows < 1 or input_tokens < 1 or execution not in ("shared", "independent"):
            raise ValueError("invalid prefill batch")
        if prompt is None:
            prepared = tokens(
                engine.properties["target_path"],
                fixture=fixture,
                context_tokens=max(1, context_tokens),
                continuation_tokens=input_tokens,
            )
            prompt, continuation = prepared.prompt[:context_tokens], prepared.continuation
        continuation = tuple(continuation or ())[:input_tokens]
        if len(prompt) != context_tokens or len(continuation) != input_tokens:
            raise ValueError("prefill inputs differ from operating point")
        record_inputs(run, prompt=tuple(prompt), continuation=continuation)
        with prefix(model, prompt) as checkpoint:
            sequences = []

            def close():
                for row in sequences:
                    row.close()
                sequences.clear()

            def reset():
                close()
                for _ in range(rows):
                    row = model.create(checkpoint)
                    sequences.append(row)
                    model.reserve(row, input_tokens + 1)

            def execute(shared):
                if shared:
                    if not model.can_batch(tuple(sequences)):
                        raise ValueError("model does not support the supplied shared batch")
                    advances = model.forward_batch(
                        tuple(sequences),
                        (ModelInputs.from_tokens(continuation),) * rows,
                        ForwardRequest(False, committed_inputs=input_tokens),
                    )
                    for step in advances:
                        step.accept(input_tokens)
                        step.complete()
                else:
                    for row in sequences:
                        model.prefill(row, continuation)

            def probe():
                outputs = []
                for row in sequences:
                    saved = row.checkpoint()
                    try:
                        if saved.length != context_tokens + input_tokens:
                            raise ValueError("prefill did not commit its complete input")
                    finally:
                        saved.close()
                    step = model.forward(
                        row, (continuation[-1],), ForwardRequest(committed_inputs=1)
                    )
                    step.accept(1)
                    step.complete()
                    outputs.append(step.output.logits)
                return tuple(outputs)

            try:
                reset()
                execute(False)
                expected = probe()
                mx.eval(expected)
                run.measure(
                    lambda: execute(execution == "shared"),
                    prepare=reset,
                    complete=lambda _: model.owner.backend.drain(),
                    validate=lambda _: compare(probe(), expected, atol=0.002, rtol=0.002),
                    deterministic=True,
                )
            finally:
                close()
    return run
