"""Actual hybrid execution through the single service/pressure lifecycle."""

import pytest
from test_generation import runtime  # noqa: F401
from test_qwen35_model import reference

from magnitude_engine.generation.plain import FinishReason, Options
from magnitude_engine.models.qwen35.inputs import InputPlan
from magnitude_engine.service.engine import Engine, Idle, Status, Submission
from magnitude_engine.service.policy import Limits


@pytest.mark.device
def test_service_shared_submission_cancel_and_output_credit(runtime):  # noqa: F811
    model, selector, weights = runtime
    engine = Engine(model, selector, Limits(prefill_tokens=8))
    prompt = (1, 3, 7)
    try:
        first, second = (
            engine.admit(
                model.input(InputPlan.text(prompt)), Options(max_tokens=2, output_capacity=1)
            )
            for _ in range(2)
        )
        submission = engine.step()
        assert isinstance(submission, Submission)
        assert submission.requests == (first, second) and submission.tokens == 6
        engine.cancel(first)
        assert engine.take(first, 2) == ()
        submission.completion.wait()
        assert isinstance(engine.step(), Idle)
        assert engine.snapshot(first).finish == FinishReason.CANCELLED
        assert engine.snapshot(second).status == Status.OUTPUT
        expected = int(reference(weights, prompt)[0][-1].argmax())
        output = engine.take(second, 1)
        assert [(o.index, o.token) for o in output] == [(0, expected)]
        next_step = engine.step()
        assert isinstance(next_step, Submission) and next_step.requests == (second,)
        next_step.completion.wait()
        assert isinstance(engine.step(), Idle)
        expected_next = int(reference(weights, (*prompt, expected))[0][-1].argmax())
        assert [(o.index, o.token) for o in engine.take(second, 1)] == [(1, expected_next)]
        assert engine.snapshot(second).finish == FinishReason.LENGTH
        assert (
            sum(r.service_ns for r in engine.requests.values())
            == engine.scheduler.completed_service_ns
        )
        engine.remove(first)
        engine.remove(second)
        assert engine.requests == {}
    finally:
        engine.close()


@pytest.mark.device
def test_capacity_eviction_replays_without_republishing_or_losing_output(runtime):  # noqa: F811
    model, selector, weights = runtime
    engine = Engine(model, selector, Limits(prefill_tokens=8))
    prompt = (1, 3, 7, 8, 12)
    budget = model.context.budget_bytes
    try:
        first = engine.admit(
            model.input(InputPlan.text(prompt)), Options(max_tokens=3, output_capacity=1)
        )
        step = engine.step()
        assert isinstance(step, Submission)
        step.completion.wait()
        assert isinstance(engine.step(), Idle)
        generation = engine.requests[first].generation
        assert generation is not None
        model.reclaim()
        # Exact ownership estimate excludes shared weights and includes the
        # backing released by discarding this state then reclaiming idle banks.
        predicted = model.reclaimable((generation.sequence,))
        assert predicted > 0
        checkpoint = generation.sequence.checkpoint()
        shared = checkpoint.fork()
        checkpoint.close()
        try:
            assert model.reclaimable((generation.sequence,)) == 0
            assert model.reclaimable((generation.sequence, shared)) == predicted
        finally:
            shared.close()
        before = model.context.allocated_bytes
        model.context.budget_bytes = before
        second = engine.admit(
            model.input(InputPlan.text((2, 4, 9))), Options(max_tokens=1, output_capacity=1)
        )
        step = engine.step()
        assert isinstance(step, Submission) and step.requests == (second,)
        assert not generation.resident
        assert engine.snapshot(first).preemptions == 1
        # Output credit is drained while numerical state is absent. Recovery
        # must consume this current cursor, not a stale saved output snapshot.
        collected = list(engine.take(first, 1))
        step.completion.wait()
        for _ in range(24):
            step = engine.step()
            if isinstance(step, Submission):
                step.completion.wait()
            collected.extend(engine.take(first, 1))
            engine.take(second, 1)
            if engine.snapshot(first).finish is not None:
                break
        else:
            pytest.fail("bounded recovery did not make useful progress")
        expected = []
        for _ in range(3):
            expected.append(int(reference(weights, (*prompt, *expected))[0][-1].argmax()))
        assert [(o.index, o.token) for o in collected] == list(enumerate(expected))
        assert engine.snapshot(first).finish == FinishReason.LENGTH
        assert engine.snapshot(first).preemptions == 1
        assert generation.sampled == expected
    finally:
        model.context.budget_bytes = budget
        engine.close()


@pytest.mark.device
def test_impossible_minimum_admission_reports_capacity_without_retrying(runtime):  # noqa: F811
    model, selector, _ = runtime
    engine = Engine(model, selector)
    budget = model.context.budget_bytes
    try:
        identity = engine.admit(model.input(InputPlan.text((1, 3))), Options(max_tokens=1))
        model.context.budget_bytes = 1
        result = engine.step()
        assert isinstance(result, Idle)
        snapshot = engine.snapshot(identity)
        assert snapshot.finish == FinishReason.FAILED
        assert snapshot.failure is not None and snapshot.failure.required_bytes is not None
        assert snapshot.failure.available_bytes is not None
        assert snapshot.failure.required_bytes > snapshot.failure.available_bytes
        again = engine.step()
        assert isinstance(again, Idle)
        assert engine.snapshot(identity).failure == snapshot.failure
        assert engine.scheduler.completed_service_ns == 0
        assert engine.requests[identity].source_closed
    finally:
        model.context.budget_bytes = budget
        engine.close()
