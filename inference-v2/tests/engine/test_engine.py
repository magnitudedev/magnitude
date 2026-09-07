from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from magnitude_engine.engine.delivery import Finished, Tokens
from magnitude_engine.engine.prefixes.radix import Radix
from magnitude_engine.engine.prefixes.retention import LeastRecentlyUsed
from magnitude_engine.engine.requests import GenerationRequest
from magnitude_engine.engine.runtime import Engine
from magnitude_engine.engine.scheduler.time_shared import TimeShared
from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget


def setup(*, policy=None, retention=8):
    budget = MemoryBudget(4 << 20)
    calls = []
    failure = [None]

    def forward(tokens, caches):
        calls.append(tuple(tokens[0].tolist()))
        if failure[0] is not None:
            raise failure[0]
        values = tokens.astype(mx.float32)[:, None, :, None]
        caches[0].update_and_fetch(values, values)
        return -mx.square(mx.arange(7) - ((tokens + 1) % 7)[..., None]) * 2.0

    model = ModelRuntime(
        LibraryProgram(forward),
        LibraryStateStore(lambda: [KVCache()], budget, lambda n, q: 8192),
        ExecutionOwner(),
    )
    engine = Engine(
        GenerationRuntime(model, SuffixMethod(1, 3)),
        namespace=b"test:7:suffix",
        scheduler=policy or TimeShared(prefill_tokens=3, decode_tokens=4),
        prefixes=Radix(retention=LeastRecentlyUsed(retention, None)),
    )
    return engine, budget, calls, failure


def request(prompt=(0,), count=7, seed=7):
    return GenerationRequest(prompt, SamplingPolicy(temperature=0.7, seed=seed), count)


def available(handle):
    events = []
    while True:
        try:
            event = handle.delivery.take(0)
            events.append(event)
            if isinstance(event, Finished):
                return events
        except (TimeoutError, StopIteration):
            return events


def finish(engine, handle):
    events = []
    for _ in range(100):
        engine.tick()
        events.extend(available(handle))
        if any(isinstance(event, Finished) for event in events):
            return events
    raise AssertionError("request did not finish within the trace bound")


def tokens(events):
    return tuple(token for event in events if isinstance(event, Tokens) for token in event.values)


def test_engine_interleaves_prefill_decode_and_preserves_request_sampling():
    engine, budget, calls, _ = setup()
    long = engine.submit(request(tuple(range(7)) * 3, count=11), identity="long")
    short = engine.submit(request((1,), count=8), identity="short")
    assert not calls
    first = engine.tick()
    assert [(m.request_id, m.phase) for m in first] == [("short", "decode")]
    second = engine.tick()
    assert [(m.request_id, m.phase) for m in second] == [("long", "prefill")]
    assert second[0].input_tokens == 3
    assert short.delivery.credit < short.delivery.capacity
    output = finish(engine, long)
    other = finish(engine, short)
    direct = engine.generation.create(long.request.prompt, long.request.sampling, 11)
    expected = []
    while not direct.finished:
        expected.extend(direct.step(4).tokens)
    direct.close()
    assert tokens(output) == tuple(expected)
    assert len(tokens(other)) == 8
    assert output[-1].reason == other[-1].reason == "length"
    engine.close()
    engine.generation.model.owner.close()
    assert budget.snapshot().reserved == 0


def test_full_consumer_stops_only_its_row_and_cancellation_has_terminal_capacity():
    engine, budget, _, _ = setup(retention=0)
    stalled = engine.submit(request(count=20), identity="stalled", output_capacity=1)
    fast = engine.submit(request(count=5), identity="fast")
    engine.tick()
    assert stalled.delivery.credit == 0
    events = finish(engine, fast)
    assert len(tokens(events)) == 5
    assert stalled.delivery.finish is None
    assert engine.tick() == ()
    stalled.cancel()
    engine.tick()
    assert stalled.delivery.finish.reason == "cancelled"
    pending = available(stalled)
    assert len(tokens(pending)) == 1 and pending[-1].generated_tokens == 1
    engine.close()
    assert budget.snapshot().reserved == 0


def test_consumer_returns_credit_and_queued_cancellation_is_not_blocked_by_active_rows():
    engine, budget, _, _ = setup(policy=TimeShared(max_active=1, max_queued=2), retention=0)
    active = engine.submit(request(count=8), output_capacity=1)
    engine.tick()
    cancelled = engine.submit(request(count=9))
    waiting = engine.submit(request(count=3))
    with pytest.raises(OverflowError):
        engine.submit(request())
    cancelled.cancel()
    engine.tick()
    assert cancelled.delivery.finish.reason == "cancelled"
    assert waiting.delivery.finish is None
    first = active.delivery.take(0)
    assert isinstance(first, Tokens) and active.delivery.credit == 1
    assert engine.wake.is_set()
    rest = finish(engine, active)
    assert len(tokens([first, *rest])) == 8
    assert len(tokens(finish(engine, waiting))) == 3
    engine.close()
    assert budget.snapshot().reserved == 0


def test_complete_prefix_reuse_and_retention_eviction_are_detached_from_sampling():
    engine, budget, calls, _ = setup(retention=2)
    original = engine.submit(request(tuple(range(7)) * 2, count=5))
    original_events = finish(engine, original)
    before = len(calls)
    warm = engine.submit(request(original.request.prompt, count=5, seed=99))
    warm_events = finish(engine, warm)
    assert warm_events[-1].cached_tokens == len(original.request.prompt) - 1
    assert len(calls) - before <= 5  # no prompt forward on this compatible hit
    assert original_events[-1].cached_tokens == 0
    cold = engine.generation.create(warm.request.prompt, warm.request.sampling, 5)
    expected = []
    while not cold.finished:
        expected.extend(cold.step(4).tokens)
    cold.close()
    assert tokens(warm_events) == tuple(expected)
    assert len(engine.prefixes.eligible()) <= 2
    engine.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("error", [MemoryError("request pressure"), RuntimeError("device failure")])
def test_request_pressure_is_local_but_unexpected_execution_failure_terminates_every_request(error):
    engine, budget, _, failure = setup(policy=TimeShared(max_active=1), retention=0)
    active = engine.submit(request(count=8), output_capacity=1)
    engine.tick()
    active.delivery.take(0)
    queued = engine.submit(request(count=4))
    failure[0] = error
    if isinstance(error, MemoryError):
        engine.tick()
        assert active.delivery.finish.reason == "error" and queued.delivery.finish is None
        failure[0] = None
        assert len(tokens(finish(engine, queued))) == 4
    else:
        with pytest.raises(RuntimeError, match="device failure"):
            engine.tick()
        assert active.delivery.finish.reason == queued.delivery.finish.reason == "error"
        with pytest.raises(RuntimeError, match="unavailable"):
            engine.submit(request())
    engine.close()
    assert budget.snapshot().reserved == 0


def test_control_thread_submits_and_consumes_without_taking_execution_ownership():
    engine, budget, calls, _ = setup(retention=0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        handle = executor.submit(engine.submit, request(count=2)).result()
        assert not calls
        engine.tick()
        first = executor.submit(handle.delivery.take, 1).result()
        assert isinstance(first, Tokens)
        executor.submit(handle.cancel).result()
        engine.tick()
        terminal = executor.submit(handle.delivery.take, 1).result()
        assert isinstance(terminal, Finished) and terminal.reason == "cancelled"
    zero = engine.submit(request(count=0))
    engine.tick()
    assert zero.delivery.finish.reason == "length" and zero.delivery.finish.generated_tokens == 0
    engine.close()
    assert budget.snapshot().reserved == 0


def test_retention_allocation_pressure_is_a_cache_miss_and_does_not_fail_generation():
    engine, budget, _, _ = setup(retention=8)
    # The active cache fits; a detached checkpoint does not fit alongside it.
    budget.limit = 8192
    handle = engine.submit(request((1, 2, 3, 4), count=5))
    events = finish(engine, handle)
    assert len(tokens(events)) == 5 and events[-1].reason == "length"
    assert not engine.prefixes.eligible()
    engine.close()
    assert budget.snapshot().reserved == 0


def test_zero_output_request_completes_even_when_all_active_rows_are_backpressured():
    engine, budget, calls, _ = setup(policy=TimeShared(max_active=1), retention=0)
    blocked = engine.submit(request(count=10), output_capacity=1)
    engine.tick()
    before = len(calls)
    zero = engine.submit(request(count=0))
    assert engine.tick() == ()
    assert zero.delivery.finish.reason == "length" and len(calls) == before
    assert blocked.delivery.finish is None
    engine.close()
    assert blocked.delivery.finish.reason == "shutdown"
    assert budget.snapshot().reserved == 0



def test_first_token_closes_prompt_service_before_bounded_plain_decode():
    from magnitude_engine.generation.methods.plain.runtime import PlainMethod

    engine, budget, _, _ = setup(retention=0)
    engine.generation = GenerationRuntime(engine.generation.model, PlainMethod())
    handle = engine.submit(GenerationRequest((0,), SamplingPolicy(temperature=0), 7))
    first = engine.tick()
    assert len(first) == 1 and first[0].output_tokens == 1 and first[0].input_tokens == 2
    assert tokens(available(handle)) == (1,)
    second = engine.tick()
    assert len(second) == 1 and second[0].output_tokens == second[0].input_tokens == 4
    assert tokens(available(handle)) == (2, 3, 4, 5)
    rest = finish(engine, handle)
    assert tokens(rest) == (6, 0)
    assert rest[-1].first_decode_ns == first[0].elapsed_ns
    assert not engine.generation.model.owner._pending
    engine.close()
    assert budget.snapshot().reserved == 0


def test_memory_pressure_queues_in_order_until_an_active_request_releases_its_state():
    engine, budget, _, _ = setup(retention=0)
    budget.limit = 8192  # Exactly one native continuation.
    active = engine.submit(request(count=6), identity="active", output_capacity=1)
    oldest = engine.submit(request(count=3), identity="oldest")
    later = engine.submit(request(count=3), identity="later")
    assert [m.request_id for m in engine.tick()] == ["active"]
    assert tuple(engine._active) == ("active",)
    assert [h.identity for h in engine._queued] == ["oldest", "later"]
    assert oldest.delivery.finish is later.delivery.finish is None
    assert budget.snapshot().reserved == 8192
    active.cancel()
    assert [m.request_id for m in engine.tick()] == ["oldest"]
    assert len(tokens(finish(engine, oldest))) == 3
    assert len(tokens(finish(engine, later))) == 3
    engine.close()
    assert budget.snapshot().reserved == 0


def test_request_that_cannot_fit_alone_terminates_without_blocking_following_requests():
    engine, budget, _, _ = setup(retention=0)
    engine.generation.model.states.capacity = lambda n, q: 8192 if n < 10 else 16384
    budget.limit = 8192
    too_large = engine.submit(request((1,) * 10), identity="large")
    small = engine.submit(request(count=3), identity="small")
    assert [m.request_id for m in engine.tick()] == ["small"]
    assert too_large.delivery.finish.reason == "error"
    assert len(tokens(finish(engine, small))) == 3
    engine.close()
    assert budget.snapshot().reserved == 0


def test_phase_feedback_counts_completed_execution_once_and_excludes_retention(monkeypatch):
    from magnitude_engine.generation.runtime import (
        GenerationResult,
        GenerationService,
        PrefillService,
    )

    engine, budget, _, _ = setup(retention=0)
    now = [0]
    engine.clock = lambda: now[0]
    a = engine.submit(request(count=20), identity="a")
    b = engine.submit(request(count=20), identity="b")
    prompt = engine.submit(request((1,) * 20, count=3), identity="prompt")
    measured = []
    observe = engine.scheduler.observe

    def record(service):
        measured.append(service)
        observe(service)

    monkeypatch.setattr(engine.scheduler, "observe", record)

    def decode(sequences, allowances, *, clock, budget_ns):
        now[0] += 10_000_000
        return tuple(GenerationService(GenerationResult((1,), 0, 0, None, 1),
                                       10_000_000, len(sequences)) for _ in sequences)

    monkeypatch.setattr(engine.generation, "step_many", decode)
    first = engine.tick()
    assert sum(m.elapsed_ns for m in first) == 20_000_000
    assert measured[-1].elapsed_ns == 10_000_000
    assert engine.last_service is measured[-1]

    sequence = engine._active[prompt.identity].sequence

    def prefill(sequences, allowances, *, clock):
        now[0] += 40_000_000
        for row, allowance in zip(sequences, allowances, strict=True):
            row.prefilled += allowance
        return tuple(PrefillService(count, 40_000_000, len(sequences)) for count in allowances)

    monkeypatch.setattr(engine.generation, "prefill_many", prefill)
    engine.tick()
    assert measured[-1].phase == "prefill" and measured[-1].elapsed_ns == 40_000_000
    # Initial decode already contributed 10 ms. Another 30 ms pays the chunk.
    assert [engine.tick()[0].phase for _ in range(3)] == ["decode"] * 3
    assert engine.tick()[0].phase == "prefill"
    a.cancel()
    b.cancel()
    # With decode gone, prompt proceeds regardless of remaining debt.
    sequence.prefilled = sequence.prompt_length - 4
    monkeypatch.setattr(engine, "_retain", lambda _: now.__setitem__(0, now[0] + 1_000_000_000))
    assert engine.tick()[0].phase == "prefill"
    assert measured[-1].elapsed_ns == 40_000_000  # Retention is not prefill service.
    engine.close()
    assert budget.snapshot().reserved == 0


def test_unfinished_generation_service_reports_progress_without_publishing_a_token():
    engine, budget, _, _ = setup(retention=0)
    engine.scheduler.prefill_stall_seconds = 1e-9
    decoding = engine.submit(request((1,), count=4), identity='decoding')
    prompting = engine.submit(request((1,) * 8, count=2), identity='prompting')
    first = engine.tick()
    assert engine.last_service.phase == 'decode'
    assert len(first) == 1 and first[0].output_tokens == 0 and first[0].elapsed_ns > 0
    assert available(decoding) == []
    outputs = {decoding.identity: [], prompting.identity: []}
    for _ in range(100):
        engine.tick()
        for handle in (decoding, prompting):
            outputs[handle.identity].extend(available(handle))
        if all(handle.delivery.finish for handle in (decoding, prompting)):
            break
    assert len(tokens(outputs['decoding'])) == 4
    assert len(tokens(outputs['prompting'])) == 2
    engine.close()
    assert budget.snapshot().reserved == 0
