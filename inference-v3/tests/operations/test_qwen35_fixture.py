"""Real pinned 4B model against independently interpreted same-weight llama.cpp.

The reference JSON records the original encoded artifact, the FP32 oracle copy,
the independent decoder, and the llama.cpp executable. FP32 oracle storage is
only for isolating implementation arithmetic; v3 consumes the original GGUF.
"""

import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from magnitude_engine.blueprints import execution, models, operations
from magnitude_engine.blueprints import weights as containers
from magnitude_engine.composition import build
from magnitude_engine.composition.graph import dumps, loads
from magnitude_engine.generation.plain import (
    FinishReason,
    Generation,
    GenerationBatch,
    Options,
    Ready,
)
from magnitude_engine.models.qwen35.inputs import InputPlan, Inputs
from magnitude_engine.models.qwen35.runtime import ForwardRequest, LogitsSelection
from magnitude_engine.models.sequence import ModelRequest
from magnitude_engine.operations.sampling import SampleSelector
from magnitude_engine.platform.backend import Backend


@pytest.mark.model
@pytest.mark.device
def test_pinned_prefill_and_teacher_forced_decode():
    source = os.environ.get("MAGNITUDE_TEST_GGUF")
    reference_path = os.environ.get("MAGNITUDE_TEST_LLAMA_REFERENCE")
    if source is None or reference_path is None:
        pytest.skip("set MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_LLAMA_REFERENCE")
    path = Path(reference_path)
    record = json.loads(path.with_suffix(".json").read_text())
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record["reference_sha256"]
    rows, vocabulary = map(int, np.frombuffer(raw, "<i4", count=2))
    assert rows > 18
    assert len(raw) == 8 + rows * 4 + rows * vocabulary * 4
    tokens = tuple(map(int, np.frombuffer(raw, "<i4", count=rows, offset=8)))
    reference = np.frombuffer(raw, "<f4", offset=8 + rows * 4).reshape(rows, vocabulary)
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    with ExitStack() as cleanup:
        artifact = containers.GGUF(path=source)
        endpoint = execution.Device(backend=backend, budget_bytes=8 * 1024**3)
        arena = operations.Arena(context=endpoint)
        root = models.Qwen35Dense(
            description=models.Qwen35DenseDescription(format=artifact),
            operations=operations.Operations(
                weights=containers.Weights(format=artifact, context=endpoint), arena=arena
            ),
        )
        model = cleanup.enter_context(build(loads(dumps(root))))
        context = model.context
        assert model.artifact_identity == record["source_sha256"]
        assert model.geometry.vocabulary == vocabulary

        def run(state, inputs):
            forward = model._prepare_numerical(
                (ForwardRequest(state, Inputs.text(inputs, state.position), LogitsSelection.ALL),)
            )
            try:
                forward.submitted(context.submit(forward.commands))
                logits = np.frombuffer(forward.outputs[0].read_logits(), np.float32).reshape(
                    len(inputs), vocabulary
                )
                forward.outputs[0].commit()
                return logits
            finally:
                forward.close()

        def compare(actual, expected):
            # This fixture checks accumulated FP32 model error, not just one dot
            # product. Require both a bounded worst logit and a relative RMS
            # bound, and preserve the teacher-forced most likely continuation.
            error = actual.astype(np.float64) - expected
            rms_error = np.sqrt(np.mean(error * error, axis=1))
            reference_rms = np.sqrt(np.mean(expected.astype(np.float64) ** 2, axis=1))
            maximum = np.max(np.abs(error), axis=1)
            assert np.isfinite(actual).all()
            assert np.all(maximum < 0.003), maximum
            assert np.all(rms_error < 1e-4 * reference_rms), (rms_error, reference_rms)
            np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))

        state = model.states.create()
        compare(run(state, tokens), reference)
        assert state.position == rows
        state.close()
        state = model.states.create()
        compare(run(state, tokens[:16]), reference[:16])
        for position in range(16, rows):
            compare(run(state, tokens[position : position + 1]), reference[position : position + 1])
        assert state.position == rows
        state.close()
        selector = SampleSelector(context)
        cleanup.callback(selector.close)
        # Unequal histories share projection work and packed selected logits.
        # Both final rows still use the exact independently interpreted weights.
        sequences = tuple(model.create(InputPlan.text(tokens)) for _ in range(2))
        batch = model.prepare(
            tuple(
                ModelRequest(sequence, tokens[:count], LogitsSelection.NONE)
                for sequence, count in zip(sequences, (18, 16), strict=True)
            )
        )
        try:
            completion = context.submit(batch.commands)
            batch.submitted(completion)
            completion.wait()
            for advance in batch.advances:
                advance.commit()
        finally:
            batch.close()
        batch = model.prepare(
            tuple(
                ModelRequest(sequence, tokens[count:], LogitsSelection.LAST)
                for sequence, count in zip(sequences, (18, 16), strict=True)
            )
        )
        try:
            completion = context.submit(batch.commands)
            batch.submitted(completion)
            packed = np.frombuffer(
                context.read(batch.logits, after=completion), np.float32
            ).reshape(2, vocabulary)
            compare(packed, np.repeat(reference[-1:], 2, axis=0))
            batch.advances[0].commit()
        finally:
            batch.close()
        assert tuple(sequence.position for sequence in sequences) == (rows, 16)
        for sequence in sequences:
            sequence.close()

        generation = Generation(
            model.create(InputPlan.text(tokens)),
            tokens,
            selector,
            Options(max_tokens=1, output_capacity=1),
        )
        cleanup.callback(generation.close)
        while generation.finish_reason is None:
            ready = generation.ready(16)
            assert isinstance(ready, Ready)
            batch = GenerationBatch.prepare((ready,))
            completion = context.submit(batch.commands)
            batch.submitted(completion)
            completion.wait()
            batch.finish()
        output = generation.take(1)
        assert len(output) == 1 and output[0].token == int(reference[-1].argmax())
        assert generation.processed == rows and generation.published == 1
        assert generation.finish_reason == FinishReason.LENGTH
        model.close()
        cleanup.close()
        assert context.allocated_bytes == 0
