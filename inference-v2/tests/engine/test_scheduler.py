import pytest
from hypothesis import given
from hypothesis import strategies as st

from magnitude_engine.engine.scheduler.contracts import CompletedService, Runnable, Service
from magnitude_engine.engine.scheduler.time_shared import TimeShared

ROWS = (Runnable("prompt", 100_000, 64), Runnable("decode", 0, 64))


def test_preparation_counts_towards_service_share_without_distorting_token_rate():
    policy = TimeShared(prefill_stall_seconds=0.04)
    advance(policy)
    plan = policy.select(ROWS)
    assert plan.phase == "prefill" and plan.budget_ns == 40_000_000
    policy.observe(CompletedService("prefill", 90_000_000, 100, 80_000_000))
    assert policy._prefill_rate == 10000
    assert policy._decode_debt_ns == 80_000_000  # Includes the first decode round's credit.
    assert policy.select(ROWS).phase == "decode"


def advance(policy, rows=ROWS, *, decode_ms=10, prefill_ms=40):
    plan = policy.select(rows)
    assert plan is not None
    elapsed = (decode_ms if plan.phase == "decode" else prefill_ms) * 1_000_000
    count = sum(s.tokens for s in plan.services) if plan.phase == "prefill" else 0
    policy.observe(CompletedService(plan.phase, elapsed, count))
    return plan


def test_equal_share_repays_a_prompt_chunk_with_multiple_decode_rounds():
    policy = TimeShared()
    assert [advance(policy).phase for _ in range(12)] == [
        "decode",
        "prefill",
        "decode",
        "decode",
        "decode",
        "prefill",
        "decode",
        "decode",
        "decode",
        "decode",
        "prefill",
        "decode",
    ]


@given(
    durations=st.lists(st.integers(1, 100_000), min_size=100, max_size=200),
    percent=st.integers(10, 90),
)
def test_time_allocation_stays_within_one_indivisible_service(durations, percent):
    policy = TimeShared(decode_share=percent / 100)
    prefill_ns = decode_ns = 0
    max_prefill_ns = max_decode_ns = 0
    for elapsed in durations:
        plan = policy.select(ROWS)
        assert plan is not None
        policy.observe(CompletedService(plan.phase, elapsed))
        if plan.phase == "decode":
            decode_ns += elapsed
            max_decode_ns = max(max_decode_ns, elapsed)
        else:
            prefill_ns += elapsed
            max_prefill_ns = max(max_prefill_ns, elapsed)
        # Compare actual total service with the desired share. The error is
        # bounded by one indivisible operation, even as operation costs change.
        balance = prefill_ns * percent - decode_ns * (100 - percent)
        assert -max_decode_ns * (100 - percent) <= balance <= max_prefill_ns * percent
    assert prefill_ns > 0 and decode_ns > 0


def test_overshoot_credit_is_preserved_instead_of_rounding_each_chunk_up():
    policy = TimeShared()
    plans = [advance(policy, decode_ms=30, prefill_ms=20).phase for _ in range(10)]
    assert plans == [
        "decode",
        "prefill",
        "prefill",
        "decode",
        "prefill",
        "prefill",
        "decode",
        "prefill",
        "decode",
        "prefill",
    ]


def test_decode_credit_cannot_extend_a_prefill_interruption_indefinitely():
    policy = TimeShared(prefill_stall_seconds=0.04)
    # Decode is too coarse to match equal time shares while preserving the
    # interruption target. Responsiveness takes precedence over the exact ratio.
    for _ in range(10):
        assert advance(policy, decode_ms=100).phase == "decode"
        assert advance(policy).phase == "prefill"
    # When decode becomes cheap, old overservice cannot buy a long prompt burst.
    assert advance(policy, decode_ms=10).phase == "decode"
    assert advance(policy).phase == "prefill"
    assert [advance(policy).phase for _ in range(3)] == ["decode"] * 3


def test_consecutive_prompt_chunks_share_one_interruption_budget():
    policy = TimeShared(prefill_tokens=100, prefill_stall_seconds=0.04)
    assert advance(policy, decode_ms=100).phase == "decode"
    assert advance(policy, prefill_ms=30).phase == "prefill"
    plan = policy.select(ROWS)
    assert plan is not None and plan.phase == "prefill"
    # Learned 100 tokens / 30 ms, but only 10 ms remains in this interruption.
    assert plan.services[0].tokens == 33
    policy.observe(CompletedService("prefill", 10_000_000, 33))
    assert advance(policy).phase == "decode"


def test_no_debt_or_credit_survives_absent_or_output_blocked_work():
    policy = TimeShared()
    advance(policy)
    advance(policy, prefill_ms=1000)
    # Decoder cannot consume output: prompt work gets all available service.
    blocked = (ROWS[0], Runnable("decode", 0, 0))
    assert advance(policy, blocked).phase == "prefill"
    assert advance(policy).phase == "decode"
    assert advance(policy).phase == "prefill"
    # Decode alone cannot bank credit against a future arrival either.
    for _ in range(10):
        assert advance(policy, (ROWS[1],)).phase == "decode"
    assert advance(policy).phase == "decode"
    assert advance(policy).phase == "prefill"
    assert policy.select(()) is None
    assert policy.select((Runnable("blocked", 0, 0),)) is None


def test_prompt_order_chunk_duration_and_decode_output_allowances():
    policy = TimeShared(prefill_tokens=64, decode_tokens=4, prefill_stall_seconds=0.04)
    # Learn 100 tokens/s from completed prompt work, outside contention.
    warmup = policy.select((Runnable("warmup", 10, 1),))
    assert warmup is not None
    policy.observe(CompletedService("prefill", 100_000_000, 10))
    rows = (
        Runnable("oldest", 100, 8, "compatible"),
        Runnable("blocked", 0, 0),
        Runnable("a", 0, 2),
        Runnable("b", 0, 8),
        Runnable("later", 20, 8, "compatible"),
    )
    assert advance(policy, rows).services == (Service("a", 2), Service("b", 4))
    plan = policy.select(rows)
    assert plan is not None and plan.phase == "prefill"
    assert plan.services == (Service("oldest", 2), Service("later", 2))
    policy.observe(CompletedService("prefill", 40_000_000, 4))
    # No decode contention: the duration target no longer limits prompt throughput.
    assert advance(policy, (rows[0], rows[-1])).services == (
        Service("oldest", 32),
        Service("later", 20),
    )


@given(
    lengths=st.lists(st.integers(1, 10_000), min_size=1, max_size=64),
    budget=st.integers(1, 1024),
)
def test_prompt_batches_preserve_fifo_admission_and_aggregate_allowance(lengths, budget):
    policy = TimeShared(prefill_tokens=budget)
    rows = tuple(Runnable(str(i), length, 1, "compatible") for i, length in enumerate(lengths))
    plan = policy.select(rows)
    assert plan is not None and plan.phase == "prefill"
    assert sum(service.tokens for service in plan.services) <= budget
    assert tuple(service.identity for service in plan.services) == tuple(
        row.identity for row in rows[:budget]
    )
    for row, service in zip(rows, plan.services, strict=False):
        assert 1 <= service.tokens <= row.prefill_remaining
    # Equal lengths expose an equal-width physical batch; no padding is requested.
    if len(set(lengths)) == 1:
        assert len({service.tokens for service in plan.services}) == 1


def test_incompatible_prompts_do_not_fragment_the_oldest_prompts_allowance():
    policy = TimeShared(prefill_tokens=512)
    rows = (Runnable("first", 1024, 1), Runnable("second", 1024, 1))
    plan = policy.select(rows)
    assert plan is not None and plan.services == (Service("first", 512),)


def test_default_token_bound_does_not_impose_an_unrequested_duration_target():
    policy = TimeShared(prefill_tokens=64)
    initial = policy.select((Runnable("learn", 10, 1),))
    assert initial is not None
    policy.observe(CompletedService("prefill", 1_000_000_000, 10))
    decode = advance(policy)
    assert decode.budget_ns is None
    plan = policy.select(ROWS)
    assert plan is not None and plan.services == (Service("prompt", 64),)


def test_selection_requires_matching_completion_and_reset_clears_learning():
    policy = TimeShared(prefill_tokens=64)
    plan = policy.select(ROWS)
    assert plan is not None
    with pytest.raises(RuntimeError, match="complete scheduled"):
        policy.select(ROWS)
    with pytest.raises(RuntimeError, match="outstanding"):
        policy.reset()
    with pytest.raises(ValueError, match="match"):
        policy.observe(CompletedService("prefill", 1))
    with pytest.raises(ValueError, match="nonnegative"):
        policy.observe(CompletedService("decode", -1))
    policy.observe(CompletedService("decode", 10))
    advance(policy, prefill_ms=100_000)
    policy.reset()
    assert advance(policy).phase == "decode"
    assert advance(policy).services == (Service("prompt", 64),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decode_share": 0},
        {"decode_share": 1},
        {"decode_share": float("nan")},
        {"prefill_stall_seconds": float("inf")},
        {"prefill_tokens": 0},
        {"max_active": 65},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TimeShared(**kwargs)
