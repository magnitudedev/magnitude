import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod
from magnitude_engine.generation.proposals import Proposal
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget


def make_generation(method):
    budget = MemoryBudget(1 << 20)

    def call(tokens, caches):
        state = tokens.reshape(1, 1, -1, 1).astype(mx.float32)
        caches[0].update_and_fetch(state, state)
        next_ids = (tokens + 1) % 7
        return -mx.abs(mx.arange(7)[None, None, :] - next_ids[..., None]) * 2.0

    store = LibraryStateStore(lambda: [KVCache()], budget, lambda n, q: 8192)
    runtime = ModelRuntime(LibraryProgram(call), store, ExecutionOwner())
    return GenerationRuntime(runtime, method), budget


def collect(sequence, width):
    results = []
    while not sequence.finished:
        results.append(sequence.step(width + 1))
    return tuple(t for result in results for t in result.tokens), results


@pytest.mark.parametrize(
    "policy",
    [
        SamplingPolicy(temperature=0),
        SamplingPolicy(temperature=0.8, seed=42),
        SamplingPolicy(temperature=0.8, seed=11, repetition_penalty=1.3, presence_penalty=0.2),
    ],
)
def test_plain_speculative_position_sampling_and_penalties_agree(policy):
    prompt = tuple(range(7)) * 5
    plain, plain_budget = make_generation(PlainMethod())
    speculative, speculative_budget = make_generation(SuffixMethod(2, 6))
    a = plain.create(prompt, policy, 31, chunk_size=5)
    b = speculative.create(prompt, policy, 31, chunk_size=13)
    expected, _ = collect(a, 0)
    actual, rounds = collect(b, 6)
    assert actual == expected
    assert sum(r.proposed for r in rounds) > 0
    if policy.temperature == 0:
        assert sum(r.accepted for r in rounds) > 20
    a.close()
    b.close()
    assert plain_budget.snapshot().reserved == speculative_budget.snapshot().reserved == 0


def test_eos_and_output_budget_cut_inside_proposal():
    runtime, budget = make_generation(SuffixMethod(1, 3))
    row = runtime.create((0, 1, 2, 3, 4, 5, 6, 0), SamplingPolicy(temperature=0), 100, (3,))
    tokens, rounds = collect(row, 6)
    assert tokens == (1, 2, 3)
    assert rounds[-1].finish_reason == "stop"
    assert row.model.state.position == len(row.context) - 1
    row.close()
    short = runtime.create((0, 1, 2, 3, 4, 5, 6, 0), SamplingPolicy(temperature=0), 2)
    tokens, rounds = collect(short, 6)
    assert tokens == (1, 2)
    assert rounds[-1].finish_reason == "length"
    short.close()
    assert budget.snapshot().reserved == 0


def test_linked_checkpoint_reuses_state_but_not_request_sampling():
    runtime, budget = make_generation(SuffixMethod(1, 3))
    row = runtime.create(tuple(range(7)) * 3, SamplingPolicy(temperature=0), 9)
    row.step(5)
    checkpoint = row.checkpoint()
    prompt = tuple(row.context) + (1, 2, 3)
    row.close()
    policy = SamplingPolicy(temperature=1, seed=99, frequency_penalty=0.4)
    warm = runtime.create(prompt, policy, 17, checkpoint=checkpoint)
    cold = runtime.create(prompt, policy, 17)
    warm_tokens, _ = collect(warm, 5)
    cold_tokens, _ = collect(cold, 5)
    assert warm_tokens == cold_tokens
    assert checkpoint.length == len(checkpoint.tokens)
    with pytest.raises(ValueError, match="match"):
        runtime.create((6, 5, 4), policy, 2, checkpoint=checkpoint)
    warm.close()
    cold.close()
    checkpoint.close()
    assert budget.snapshot().reserved == 0


def test_plain_path_observes_bound_method_even_with_zero_proposal_budget():
    class ObservedSession:
        features = frozenset()
        prefill_features = frozenset()

        def __init__(self):
            self.prefilled = []
            self.observed = []

        def prefill(self, tokens, features):
            yield from ()
            self.prefilled.extend(tokens)

        def propose(self, context, limit):
            yield from ()
            return Proposal.from_tokens(())

        def observe(self, verification):
            self.observed.extend(verification.inputs[: verification.accepted_inputs])

        def close(self):
            pass

    session = ObservedSession()

    class Method:
        identity = "observer"

        def create(self, checkpoint=None, *, target):
            return session

    runtime, budget = make_generation(Method())
    row = runtime.create((1, 2, 3, 4), SamplingPolicy(temperature=0), 4, chunk_size=2)
    collect(row, 0)
    assert session.prefilled == [1, 2, 3]
    assert session.observed == [4, 5, 6, 0]
    row.close()
    assert budget.snapshot().reserved == 0


def test_incremental_prefill_yields_between_rows_and_restores_partial_prompt_checkpoint():
    runtime, budget = make_generation(SuffixMethod(1, 3))
    prompt = tuple(range(7)) * 4
    policy = SamplingPolicy(temperature=0.7, seed=111, frequency_penalty=0.1)
    long = runtime.prepare(prompt, policy, 9)
    assert budget.snapshot().reserved == 0
    assert long.prefill_remaining == len(prompt) - 1
    with pytest.raises(RuntimeError, match="prefill"):
        long.step(5)
    assert not long.failed
    assert long.prefill(5) == 5 and long.prefill_remaining == len(prompt) - 6
    checkpoint = long.checkpoint()
    assert checkpoint.tokens == prompt[:5]
    short = runtime.prepare((1, 2), SamplingPolicy(temperature=0), 2)
    assert short.prefill(99) == 1
    assert short.step().tokens == (3,)
    assert long.prefill_remaining == len(prompt) - 6
    restored = runtime.prepare(prompt, policy, 9, checkpoint=checkpoint)
    checkpoint.close()
    long.close()
    assert restored.prefill_remaining == len(prompt) - 6
    while restored.prefill_remaining:
        assert restored.prefill(3) <= 3
    assert restored.prefill(2) == 0
    cold = runtime.create(prompt, policy, 9)
    assert collect(restored, 3)[0] == collect(cold, 3)[0]
    short.close()
    restored.close()
    cold.close()
    runtime.model.owner.close()
    assert budget.snapshot().reserved == 0


def test_cancelled_partial_prefill_releases_state_and_invalid_allowance_does_not_mutate():
    runtime, budget = make_generation(PlainMethod())
    sequence = runtime.prepare((1, 2, 3, 4, 5), SamplingPolicy(temperature=0), 3)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="allowance"):
            sequence.prefill(invalid)
    assert sequence.prefilled == 0 and budget.snapshot().reserved == 0
    sequence.prefill(2)
    assert budget.snapshot().reserved > 0
    sequence.close()
    with pytest.raises(RuntimeError):
        sequence.prefill(2)
    runtime.model.owner.close()
    assert budget.snapshot().reserved == 0


def test_live_checkpoint_cannot_cross_target_residencies_even_with_the_same_method_name():
    first, first_budget = make_generation(PlainMethod())
    second, second_budget = make_generation(PlainMethod())
    row = first.create((1, 2, 3), SamplingPolicy(temperature=0), 2)
    checkpoint = row.checkpoint()
    with pytest.raises(ValueError, match="match"):
        second.prepare((1, 2, 3), SamplingPolicy(temperature=0), 2, checkpoint=checkpoint)
    assert second_budget.snapshot().reserved == 0
    restored = GenerationRuntime(first.model, PlainMethod()).prepare(
        (1, 2, 3),
        SamplingPolicy(temperature=0),
        2,
        checkpoint=checkpoint,
    )
    restored.close()
    checkpoint.close()
    row.close()
    assert first_budget.snapshot().reserved == 0


@pytest.mark.parametrize("policy", [
    SamplingPolicy(temperature=0),
    SamplingPolicy(temperature=0.7, seed=19),
    SamplingPolicy(temperature=0.7, seed=19, frequency_penalty=0.2),
])
@pytest.mark.parametrize("allowance", [2, 4, 9])
def test_causal_continuations_match_single_steps_with_bounded_pending_work(policy, allowance):
    runtime, budget = make_generation(PlainMethod())
    reference = runtime.create((1, 2, 3), policy, 19)
    expected, _ = collect(reference, 0)
    reference.close()
    row = runtime.create((1, 2, 3), policy, 19)
    actual = []
    while not row.finished:
        result = row.step(allowance)
        assert 1 <= len(result.tokens) <= allowance
        assert result.proposed == result.accepted == result.forced == 0
        assert row.model.pending is None  # All inputs are logically committed.
        pending = not row.finished and not policy.uses_history
        assert len(runtime.model.owner._pending) == int(pending)
        assert len(row.model._committed) == int(pending)
        assert row.model.state.position == len(row.context) - 1 + int(pending)
        actual.extend(result.tokens)
    assert tuple(actual) == expected
    checkpoint = row.checkpoint()
    prompt = tuple(row.context) + (2, 3)
    warm = runtime.create(prompt, policy, 7, checkpoint=checkpoint)
    cold = runtime.create(prompt, policy, 7)
    assert collect(warm, 3)[0] == collect(cold, 0)[0]
    warm.close()
    cold.close()
    checkpoint.close()
    row.close()
    assert budget.snapshot().reserved == 0


def test_final_carried_prediction_needs_no_additional_model_capacity(monkeypatch):
    runtime, budget = make_generation(PlainMethod())
    row = runtime.create((1, 2, 3), SamplingPolicy(temperature=0), 5)
    row.step(4)
    assert len(runtime.model.owner._pending) == 1

    def no_capacity(*args, **kwargs):
        raise MemoryError("no room for another forward")

    monkeypatch.setattr(runtime.model, "reserve", no_capacity)
    result = row.step(1)
    assert len(result.tokens) == 1 and result.evaluated_inputs == 0
    assert row.finished and not runtime.model.owner._pending
    row.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("stop", [1, 2, 3, 4])
def test_causal_eos_preserves_consumed_history_and_recurrent_checkpoint(stop):
    from tests.models.test_model_runtime import hybrid

    target, budget = hybrid()
    runtime = GenerationRuntime(target, PlainMethod())
    # Its recurrent sum gives 1, 2, 4, 8, 15 ...; cover first/interior/final EOS.
    oracle = runtime.create((0, 1), SamplingPolicy(temperature=0), 8)
    expected, _ = collect(oracle, 0)
    oracle.close()
    eos = expected[stop - 1]
    row = runtime.create((0, 1), SamplingPolicy(temperature=0), 8, (eos,))
    result = row.step(4)
    assert result.tokens == expected[:stop]
    assert result.finish_reason == "stop"
    assert result.evaluated_inputs == stop + 1
    boundary = len(row.context)
    assert row.target_position == row.model.state.position == boundary
    assert row.model.state.caches[0].offset == boundary
    assert not target.owner._pending
    checkpoint = row.checkpoint()
    prompt = tuple(row.context) + (1,)
    warm = runtime.create(prompt, SamplingPolicy(temperature=0), 4, checkpoint=checkpoint)
    cold = runtime.create(prompt, SamplingPolicy(temperature=0), 4)
    assert collect(warm, 3)[0] == collect(cold, 0)[0]
    warm.close()
    cold.close()
    checkpoint.close()
    row.close()
    assert budget.snapshot().reserved == 0


def test_causal_forward_failure_drains_sampling_and_releases_request():
    runtime, budget = make_generation(PlainMethod())
    row = runtime.create((1,), SamplingPolicy(temperature=0), 10)
    program = runtime.model.program
    original = program.call
    calls = 0

    def failing(tokens, cache):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("injected lookahead failure")
        return original(tokens, cache)

    program.call = failing
    with pytest.raises(ValueError, match="injected lookahead"):
        row.step(4)
    assert row.failed
    row.close()
    assert not runtime.model.owner._pending
    assert budget.snapshot().reserved == 0
