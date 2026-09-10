"""Plain generation over the real hybrid model and owned device submissions."""

import os
from contextlib import ExitStack

import numpy as np
import pytest
from test_qwen35_model import fixture, reference

from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.generation.plain import (
    FinishReason,
    Generation,
    GenerationBatch,
    Options,
    Ready,
    WaitReason,
    WorkKind,
)
from magnitude_engine.models.qwen35.artifact import inspect_dense
from magnitude_engine.models.qwen35.inputs import InputPlan
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.operations.factory import ResidentOperations
from magnitude_engine.operations.sampling import (
    SampleSelector,
    SelectionFailure,
    SelectionKind,
    UnselectableDistribution,
)
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.machine import open_context


@pytest.fixture
def runtime(tmp_path):
    path = tmp_path / "generation.gguf"
    weights = fixture(path)
    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")),
            64 * 1024**2,
            0,
        )
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        provider = ResidentOperations(artifact, context)
        cleanup.callback(provider.close)
        model = DenseRuntime(inspect_dense(artifact.directory, artifact.identity), provider)
        cleanup.callback(model.close)
        selector = SampleSelector(context)
        cleanup.callback(selector.close)
        yield model, selector, weights
        model.close()
        provider.close()
        assert context.allocated_bytes == 0


def execute(generation, allowance=32):
    ready = generation.ready(allowance)
    assert isinstance(ready, Ready)
    batch = GenerationBatch.prepare((ready,))
    assert generation.ready(allowance) == WaitReason.COMPLETION
    ticket = generation.sequence.context.submit(batch.commands)
    batch.submitted(ticket)
    ticket.wait()
    batch.finish()
    return batch.works[0].kind


@pytest.mark.device
def test_generation_chunk_credit_pending_input_checkpoint_and_eos(runtime):
    model, selector, weights = runtime
    prompt = (1, 3, 7, 8, 12)
    expected = []
    for _ in range(5):
        logits, _, _ = reference(weights, prompt + tuple(expected))
        expected.append(int(np.argmax(logits[-1])))
    generation = Generation(
        model.create(InputPlan.text(prompt)),
        prompt,
        selector,
        Options(max_tokens=5, output_capacity=2),
    )
    assert execute(generation, 2) == WorkKind.PREFILL
    assert generation.processed == 2 and generation.take(2) == ()
    assert execute(generation, 2) == WorkKind.PREFILL
    assert execute(generation, 2) == WorkKind.PREFILL
    assert generation.processed == len(prompt)
    assert generation.pending_input == expected[0]
    assert execute(generation) == WorkKind.DECODE
    assert generation.ready(1) == WaitReason.OUTPUT
    first = generation.take(1)
    assert [(value.index, value.token) for value in first] == [(0, expected[0])]
    checkpoint = generation.checkpoint()
    generation.close()
    generation = checkpoint.fork()
    checkpoint.close()
    assert generation.published == 1
    assert generation.processed == len(prompt) + 1
    assert generation.pending_input == expected[1]
    # The queued second token is retained, without replaying the published first.
    output = list(generation.take(2))
    while generation.finish_reason is None:
        execute(generation)
        output.extend(generation.take(2))
    assert generation.finish_reason == FinishReason.LENGTH
    assert generation.processed == len(prompt) + 4
    assert generation.pending_input is None
    assert [(item.index, item.token) for item in output] == list(enumerate(expected[1:], 1))
    generation.close()

    stop = Generation(
        model.create(InputPlan.text(prompt)),
        prompt,
        selector,
        Options(max_tokens=5, stop_tokens=frozenset((expected[0],))),
    )
    execute(stop)
    assert stop.finish_reason == FinishReason.STOP
    assert stop.take(1) == () and stop.processed == len(prompt)
    assert stop.ready(1) == WaitReason.FINISHED
    stop.close()


@pytest.mark.device
def test_cancel_submitted_work_preserves_peer_and_checkpoint_rng(runtime):
    model, selector, _ = runtime
    prompt = (1, 3, 7)
    options = Options(max_tokens=5, selection=SelectionKind.CATEGORICAL, seed=2**63 + 813)
    active = Generation(model.create(InputPlan.text(prompt)), prompt, selector, options)
    execute(active, 1)
    checkpoint = active.checkpoint()
    peer = checkpoint.fork()
    checkpoint.close()
    # The peer continuations now execute shared projections and one batched
    # selector, while acceptance and cancellation remain per request.
    first, second = active.ready(2), peer.ready(2)
    assert isinstance(first, Ready) and isinstance(second, Ready)
    batch = GenerationBatch.prepare((first, second))
    ticket = model.context.submit(batch.commands)
    batch.submitted(ticket)
    active.cancel()
    assert active.finish_reason == FinishReason.CANCELLED
    assert active.take(5) == ()
    ticket.wait()
    batch.finish()
    assert peer.processed == len(prompt)
    assert len(peer.take(1)) == 1
    checkpoint = peer.checkpoint()
    replay = checkpoint.fork()
    checkpoint.close()
    execute(peer)
    execute(replay)
    assert peer.take(1) == replay.take(1)  # Position-keyed sampling survives restore.
    active.close()
    peer.close()
    replay.close()


@pytest.mark.device
def test_prepared_abort_and_failed_selection_do_not_commit(runtime, monkeypatch):
    model, selector, weights = runtime
    prompt = (1, 3, 7)
    generation = Generation(
        model.create(InputPlan.text(prompt)),
        prompt,
        selector,
        Options(max_tokens=3),
    )
    ready = generation.ready(2)
    assert isinstance(ready, Ready)
    batch = GenerationBatch.prepare((ready,))
    batch.close()
    assert generation.processed == 0 and generation.pending is None
    peer = Generation(model.create(InputPlan.text(prompt)), prompt, selector, Options(max_tokens=1))
    ready, peer_ready = generation.ready(3), peer.ready(3)
    assert isinstance(ready, Ready) and isinstance(peer_ready, Ready)
    batch = GenerationBatch.prepare((ready, peer_ready))
    with pytest.raises(RuntimeError, match="reconciled"):
        generation.checkpoint()
    completion = model.context.submit(batch.commands)
    batch.submitted(completion)
    completion.wait()
    original_read = selector.read
    reads = 0

    def fail_first(*args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 1:
            return (UnselectableDistribution(reason=SelectionFailure.EMPTY),)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(selector, "read", fail_first)
    with pytest.raises(ValueError, match="cannot be selected"):
        batch.finish()
    assert generation.finish_reason == FinishReason.FAILED
    assert generation.processed == 0 and generation.take(3) == ()
    assert peer.processed == len(prompt) and peer.finish_reason == FinishReason.LENGTH
    expected = int(reference(weights, prompt)[0][-1].argmax())
    assert [item.token for item in peer.take(1)] == [expected]
    generation.close()
    peer.close()


@pytest.mark.device
def test_mixed_ready_batches_respect_output_credit_and_match_independent_generation(runtime):
    model, selector, weights = runtime
    prompts = ((1, 3, 7), tuple(index % 32 for index in range(17)))
    generations = tuple(
        Generation(
            model.create(InputPlan.text(prompt)),
            prompt,
            selector,
            Options(max_tokens=4, output_capacity=1),
        )
        for prompt in prompts
    )
    expected = []
    for prompt in prompts:
        values = []
        for _ in range(4):
            logits, _, _ = reference(weights, prompt + tuple(values))
            values.append(int(logits[-1].argmax()))
        expected.append(values)
    outputs = [[], []]
    kinds = []
    first_ready = tuple(generation.ready(4) for generation in generations)
    assert all(isinstance(item, Ready) for item in first_ready)
    with pytest.raises(ValueError, match="distinct"):
        GenerationBatch.prepare((first_ready[0], first_ready[0]))
    batch = GenerationBatch.prepare(first_ready)
    ticket = model.context.submit(batch.commands)
    batch.submitted(ticket)
    ticket.wait()
    batch.finish()
    assert generations[0].ready(4) == WaitReason.OUTPUT
    assert isinstance(generations[1].ready(4), Ready)
    with pytest.raises(ValueError, match="no longer ready"):
        GenerationBatch.prepare((first_ready[0],))
    # Readiness is inspected before preparation; a blocked output consumer has
    # no model work reserved, while its peer can continue filling context.
    for _ in range(32):
        ready = tuple(
            value for generation in generations if isinstance(value := generation.ready(4), Ready)
        )
        if ready:
            kinds.append(tuple(value.kind for value in ready))
            batch = GenerationBatch.prepare(ready)
            ticket = model.context.submit(batch.commands)
            batch.submitted(ticket)
            ticket.wait()
            batch.finish()
        for generation, collected in zip(generations, outputs, strict=True):
            collected.extend(item.token for item in generation.take(1))
        if all(generation.finish_reason is not None for generation in generations):
            break
    else:
        pytest.fail("generation did not finish in the bounded fixture")
    assert (WorkKind.DECODE, WorkKind.PREFILL) in kinds
    assert outputs == expected
    for generation in generations:
        generation.close()
