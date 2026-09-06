from magnitude_engine.engine.delivery import Delivery, Finished, PrefillProgress, Tokens
from tests.engine.test_engine import available, finish, request, setup, tokens


def test_progress_is_coalesced_and_does_not_consume_or_return_token_credit():
    wakes = []
    delivery = Delivery(1, lambda: wakes.append(True), progress=True)
    for done in range(1001):
        delivery.report(PrefillProgress(done, 1000, 0, done))
    assert delivery.credit == 1
    delivery.publish((42,))
    assert delivery.credit == 0
    terminal = Finished("length", 1001, 1, 0, 0, 0, 0, 1, 2)
    delivery.terminate(terminal)
    assert delivery.take(0) == PrefillProgress(1000, 1000, 0, 1000)
    assert not wakes
    assert delivery.take(0) == Tokens((42,))
    assert len(wakes) == 1
    assert delivery.take(0) == terminal


def test_opt_in_progress_reports_completed_chunks_and_warm_prefix_without_changing_output():
    engine, _, calls, _ = setup()
    try:
        handle = engine.submit(request((1, 2, 3, 4, 5, 6, 1, 2), count=4), progress=True)
        events = []
        while handle.delivery.finish is None:
            engine.tick()
            events.extend(available(handle))
        progress = [e for e in events if isinstance(e, PrefillProgress)]
        assert [p.completed_tokens for p in progress] == [3, 6, 7]
        assert all(p.total_tokens == 7 and p.cached_tokens == 0 for p in progress)
        assert progress[-1].elapsed_ns > 0
        cold_calls = len(calls)
        warm = engine.submit(handle.request, progress=True)
        warmed = finish(engine, warm)
        warm_progress = [e for e in warmed if isinstance(e, PrefillProgress)]
        assert warm_progress == [PrefillProgress(7, 7, 7, 0)]
        assert tokens(events) == tokens(warmed)
        assert len(calls) > cold_calls
        plain = engine.submit(handle.request)
        ordinary = finish(engine, plain)
        assert not any(isinstance(e, PrefillProgress) for e in ordinary)
        assert tokens(ordinary) == tokens(warmed)
    finally:
        engine.close()
