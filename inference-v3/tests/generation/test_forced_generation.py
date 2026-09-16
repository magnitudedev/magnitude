"""Forced transitions preserve pending input, addressed sampling and atomic acceptance."""

import pytest
from test_constrained_generation import Model, Sequence
from test_constraints import vocabulary

from engine.data import TokenId
from engine.generation.constraints import ConstraintState
from engine.generation.plain import (
    FinishReason,
    Generation,
    GenerationBatch,
    Options,
    WaitReason,
    WorkKind,
)
from engine.models.sequence import LogitsSelection


def generation(*, output=100, credit=100, context=100, quantum=32):
    model = Model()
    sequence = Sequence(model)
    sequence.context_limit = context
    request = Generation(
        sequence,
        (TokenId(10),),
        Options(
            max_tokens=output,
            output_capacity=credit,
            forced_quantum=quantum,
            stop_tokens=frozenset({TokenId(256), TokenId(257)}),
        ),
        constraint=ConstraintState(vocabulary(), 'root ::= "hello"'),
    )
    return model, request


def finish(request, allowance=32):
    ready = request.ready(allowance)
    GenerationBatch.prepare((ready,)).finish()
    return ready


def test_final_prefill_then_multi_token_decode_omit_readout_and_leave_last_token_pending():
    model, request = generation()
    first = request.ready(32)
    assert first.tokens == (10,) and first.forced == tuple(b"h")
    assert first.kind == WorkKind.PREFILL and first.selection == LogitsSelection.NONE
    assert request.generated == [] and request.constraint.position == 0
    batch = GenerationBatch.prepare((first,))
    assert request.generated == []
    assert model.requests[0].draw_words is None and model.requests[0].allowed_tokens is None
    batch.finish()
    assert request.processed == 1 and request.pending_input == ord("h")
    run = finish(request)
    assert run.tokens == tuple(b"hell") and run.forced == tuple(b"ello")
    assert run.kind == WorkKind.DECODE and run.selection == LogitsSelection.NONE
    assert request.processed == 5 and request.pending_input == ord("o")
    assert tuple(request.generated) == tuple(b"hello") and request.forced_tokens == 5
    eos = request.ready(32)
    assert eos.forced == () and eos.selection == LogitsSelection.LAST
    assert eos.sample_position == 5
    model.next_token = 256
    finish(request)
    assert request.finish_reason == FinishReason.STOP and request.constraint.stopped
    assert request.constraint.position == 6 and request.processed == 6
    assert tuple(item.token for item in request.take(100)) == tuple(b"hello")


@pytest.mark.parametrize(
    "limit,expected",
    [
        ({"output": 3}, FinishReason.LENGTH),
        ({"context": 3}, FinishReason.CONTEXT),
        ({"credit": 3}, None),
        ({"quantum": 2}, None),
    ],
)
def test_forced_run_bounds(limit, expected):
    _, request = generation(**limit)
    finish(request)
    run = finish(request)
    assert run.forced == tuple(b"el")
    assert request.generated == list(b"hel") and request.processed == 3
    assert request.finish_reason == expected
    if "credit" in limit:
        assert request.ready(32) == WaitReason.OUTPUT
        request.take(1)
        assert request.ready(32).forced == tuple(b"l")


def test_abandoned_and_failed_forced_proposals_leave_matcher_and_history_unchanged():
    model, request = generation()
    finish(request)
    proposal = request.ready(32)
    model.refuse = True
    with pytest.raises(RuntimeError, match="capacity"):
        GenerationBatch.prepare((proposal,))
    assert request.ready(32) == proposal
    assert request.generated == list(b"h") and request.constraint.position == 1
    model.refuse = False
    abandoned = GenerationBatch.prepare((proposal,))
    abandoned.close()
    assert request.ready(32) == proposal
    model.commit_fails = True
    with pytest.raises(RuntimeError, match="commit failure"):
        GenerationBatch.prepare((proposal,)).finish()
    assert request.generated == list(b"h") and request.constraint.position == 1
    assert request.processed == 1 and request.forced_tokens == 1


def test_recovery_and_fork_preserve_forced_history_without_double_consumption():
    model, request = generation(quantum=2)
    finish(request)
    finish(request)
    checkpoint = request.checkpoint()
    fork = checkpoint.fork()
    checkpoint.close()
    recovery = request.recovery()
    assert recovery.continuation.generated == tuple(b"hel")
    assert recovery.continuation.forced_tokens == 3
    assert recovery.continuation.forced_runs == ((1, 1), (2, 1))
    assert recovery.continuation.state_only_input_tokens == 3
    request.evict()
    request.restore(Sequence(model))
    while request.rebuilding:
        ready = finish(request, 1)
        assert ready.kind == WorkKind.REPLAY and ready.forced == ()
        assert request.constraint.position == 3 and request.forced_tokens == 3
    finish(request)
    assert request.generated == list(b"hello") and request.forced_tokens == 5
    assert fork.generated == list(b"hel") and fork.constraint.position == 3
    assert fork.forced_tokens == 3
    assert request.forced_runs == {1: 1, 2: 2} and fork.forced_runs == {1: 1, 2: 1}
    assert request.state_only_input_tokens == 5 and fork.state_only_input_tokens == 3
    request.close()
    fork.close()


def test_mixed_state_only_and_sampled_rows_commit_once_and_keep_addressed_positions():
    model, forced = generation()
    sampled = Generation(Sequence(model), (TokenId(10),), Options(max_tokens=10))
    first = forced.ready(32)
    second = sampled.ready(32)
    batch = GenerationBatch.prepare((first, second))
    assert model.requests[0].selection == LogitsSelection.NONE
    assert model.requests[1].selection == LogitsSelection.LAST
    batch.finish()
    assert forced.generated == list(b"h") and sampled.generated == list(b"y")
    assert forced.processed == sampled.processed == 1
    with pytest.raises(RuntimeError, match="completion"):
        batch.finish()
    assert forced.processed == sampled.processed == 1
    finish(forced)
    assert forced.ready(32).sample_position == 5


def test_same_next_sampling_address_with_forced_execution_disabled():
    model, ordinary = generation(quantum=0)
    for token in b"hello":
        model.next_token = token
        assert finish(ordinary).selection == LogitsSelection.LAST
    _, forced = generation()
    finish(forced)
    finish(forced)
    assert ordinary.generated == forced.generated
    assert ordinary.processed == forced.processed
    assert ordinary.ready(1).sample_position == forced.ready(32).sample_position == 5
    assert ordinary.forced_tokens == 0 and forced.forced_tokens == 5


def test_service_shrinks_actual_forced_shape_under_capacity_pressure():
    from types import SimpleNamespace

    import ops
    from engine.service.engine import Engine, Request
    from engine.service.policy import Limits, Phase, RequestId, Selection

    model, request = generation()
    finish(request)
    original_prepare = model.prepare
    attempts = []

    def prepare(requests):
        attempts.append(len(requests[0].tokens))
        if attempts[-1] > 2:
            raise ops.CapacityError(4, 2)
        return original_prepare(requests)

    model.prepare = prepare
    model.context = request.sequence.context
    model.reclaim = lambda: None
    engine = Engine(model, Limits(decode_tokens=32))
    identity = RequestId(0)
    engine.requests[identity] = Request(
        identity, SimpleNamespace(), request.options, 0, generation=request
    )
    selection = Selection(Phase.DECODE, (identity,), False)
    batch, selected, cost = engine._prepare(selection)
    assert attempts == [4, 4, 2]
    assert cost == 2 and selected.phase == Phase.DECODE
    assert request.generated == list(b"h") and request.constraint.position == 1
    batch.finish()
    assert request.generated == list(b"hel") and request.processed == 3


@pytest.mark.parametrize("submitted", [False, True])
def test_cancellation_discards_forced_proposal_without_advancing_accepted_state(submitted):
    _, request = generation()
    finish(request)
    assert tuple(item.token for item in request.take(100)) == tuple(b"h")
    proposal = request.ready(32)
    batch = GenerationBatch.prepare((proposal,)) if submitted else None
    if batch is not None:
        batch.completion.done = False
    request.cancel()
    assert request.finish_reason == FinishReason.CANCELLED
    assert request.constraint.position == 1 and request.generated == list(b"h")
    assert request.forced_tokens == 1 and request.take(100) == ()
    if batch is not None:
        batch.completion.done = True
        batch.finish()
        batch.close()
        assert request.constraint.position == 1 and request.generated == list(b"h")
    request.close()


def test_forced_decode_yields_to_prefill_and_peer_progress_with_actual_cost():
    from types import SimpleNamespace

    from engine.service.engine import Engine, Request, Submission
    from engine.service.policy import Limits, Phase, RequestId

    model, long = generation(quantum=4)
    long.constraint = ConstraintState(vocabulary(), 'root ::= "a"{40}')
    finish(long)
    peer = Generation(Sequence(model), (TokenId(10),), Options(max_tokens=4))
    model.context = long.sequence.context
    engine = Engine(model, Limits(max_batch=1, prefill_tokens=32, decode_tokens=4))
    for index, request in enumerate((long, peer)):
        identity = RequestId(index)
        engine.requests[identity] = Request(
            identity, SimpleNamespace(close=lambda: None), request.options, 0, generation=request
        )
    submissions = []
    for _ in range(20):
        action = engine.step()
        if isinstance(action, Submission):
            submissions.append(action)
            if action.requests == (RequestId(0),):
                assert action.phase == Phase.DECODE and action.tokens == 4
            assert action.completion.done
        if peer.finish_reason is not None:
            break
    assert submissions[0].requests == (RequestId(0),)
    assert submissions[0].phase == Phase.DECODE
    assert submissions[1].requests == (RequestId(1),)
    assert submissions[1].phase == Phase.PREFILL
    assert peer.generated == list(b"yyyy") and peer.finish_reason == FinishReason.LENGTH
    assert len(long.generated) < 40 and long.finish_reason is None
    assert engine.scheduler.completed_service_ns > 0
    engine.close()


def test_phase_cost_counts_preparation_when_device_work_can_overlap_it(monkeypatch):
    from types import SimpleNamespace

    from engine.service.engine import Engine, Request, Submission
    from engine.service.policy import RequestId

    clock = [0]
    monkeypatch.setattr("engine.service.engine.clock_ns", lambda: clock[0])
    model = Model()
    request = Generation(Sequence(model), (TokenId(10),), Options(max_tokens=1))
    model.context = request.sequence.context
    prepare = model.prepare

    def timed_prepare(requests):
        clock[0] += 100
        return prepare(requests)

    model.prepare = timed_prepare
    engine = Engine(model)
    identity = RequestId(0)
    engine.requests[identity] = Request(
        identity, SimpleNamespace(close=lambda: None), request.options, 0, generation=request
    )
    assert isinstance(engine.step(), Submission)
    clock[0] = 175
    engine.step()
    snapshot = engine.snapshot(identity)
    assert snapshot.preparation_ns == 100
    assert snapshot.service_ns == 75
    assert engine.scheduler.completed_service_ns == 175
    engine.close()
