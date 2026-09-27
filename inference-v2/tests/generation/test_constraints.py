import llguidance as llg
import mlx.core as mx
import pytest
from tokenizers import Tokenizer, decoders, models

from magnitude_engine.generation.constraint_spec import ConstraintError, ConstraintSpec
from magnitude_engine.generation.guidance import GuidanceCompiler
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from tests.generation.test_generation_runtime import collect, make_generation
from tests.generation.test_mtp import setup


def compiler(vocabulary=7, capacity=2):
    vocabulary_map = {chr(97 + i): i for i in range(vocabulary - 1)}
    vocabulary_map["[EOS]"] = vocabulary - 1
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary_map, merges=[]))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["[EOS]"])
    return GuidanceCompiler(
        lambda: llg.LLTokenizer(tokenizer.to_str(), eos_token=vocabulary - 1), capacity
    )


def test_masks_hold_their_own_values_and_invalid_forks_cannot_poison_live_state():
    factory = compiler()
    spec = ConstraintSpec('start: "b" ("c" | "e") "d"')
    live = factory.create(spec)
    first = live.apply(mx.arange(7, dtype=mx.float32))
    fork = live.fork()
    assert fork.consume(1)
    second = fork.apply(mx.arange(7, dtype=mx.float32))
    assert not fork.consume(0)
    fork.close()
    assert mx.isfinite(first).tolist() == [False, True, False, False, False, False, False]
    assert mx.isfinite(second).tolist() == [False, False, True, False, True, False, False]
    assert live.forced() == (1,)
    assert live.consume(1) and live.consume(2) and live.consume(3)
    assert live.consume(6)
    assert mx.isfinite(live.apply(mx.zeros(7))).tolist() == [False] * 6 + [True]
    fresh = factory.create(spec)
    assert fresh.forced() == (1,)
    live.close()
    fresh.close()
    with pytest.raises(RuntimeError, match="closed"):
        fresh.forced()


@pytest.mark.parametrize("temperature", [0, 0.8])
def test_constrained_speculation_matches_plain_through_invalid_candidates_and_forced_spans(
    temperature,
):
    spec = ConstraintSpec('start: ("ab" | "ac" | "de")+ "f"')
    prompt = tuple(range(7)) * 4
    policy = SamplingPolicy(temperature=temperature, seed=92, frequency_penalty=0.2)
    plain, a_budget = make_generation(PlainMethod())
    draft, b_budget = make_generation(SuffixMethod(1, 3))
    plain.constraints = compiler()
    draft.constraints = compiler()
    a = plain.create(prompt, policy, 25, (6,), constraint=spec)
    b = draft.create(prompt, policy, 25, (6,), constraint=spec)
    expected, ar = collect(a, 0)
    actual, br = collect(b, 5)
    assert actual == expected
    assert sum(r.proposed for r in br) > 0
    assert sum(r.forced for r in ar) > 0
    assert sum(r.accepted for r in br) < sum(r.proposed for r in br)
    a.close()
    b.close()
    assert a_budget.snapshot().reserved == b_budget.snapshot().reserved == 0


def test_forced_mtp_blocks_skip_projection_observe_features_and_restore_fresh_matcher():
    target, method, budget, _, calls, pairs = setup()
    factory = compiler(128)
    runtime = GenerationRuntime(target, method, factory)
    spec = ConstraintSpec('start: "bcdef" ("g" | "h") "ijkl"')
    requests = []
    forward = target.program.forward

    def recorded(inputs, state, request, scope):
        requests.append((inputs.count, request.logits))
        return forward(inputs, state, request, scope)

    target.program.forward = recorded
    row = runtime.create((0,), SamplingPolicy(temperature=0), 20, (127,), constraint=spec)
    first = row.step(4)
    assert first.tokens == (1, 2, 3, 4) and first.forced == 4
    assert first.proposed == first.accepted == 0
    assert requests == [(4, False)] and not calls
    assert [token for token, _ in row.method.buffer] == [1, 2, 3]
    assert row.method.pending.value.item() == 3
    checkpoint = row.checkpoint()
    prompt = tuple(row.context)
    actual, rounds = collect(row, 4)
    assert actual == (5, 6, 8, 9, 10, 11, 127)
    assert any(r.proposed for r in rounds) and calls
    assert pairs[0].tolist() == [[[1, 0], [2, 1], [3, 2], [4, 3]]]
    assert pairs[1].tolist() == [[[5, 4]]]
    assert row.model.state.position == len(row.context) - 1
    restored = runtime.create(
        prompt, SamplingPolicy(temperature=0), 2, checkpoint=checkpoint, constraint=spec
    )
    assert restored.step(4).tokens == (1, 2)  # matcher is request-local, not cached
    restored.close()
    checkpoint.close()
    row.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


def test_forced_tokens_respect_stop_output_credit_and_compiler_bounds():
    runtime, budget = make_generation(PlainMethod())
    runtime.constraints = factory = compiler(capacity=1)
    spec = ConstraintSpec('start: "bcdef"')
    row = runtime.create((0,), SamplingPolicy(temperature=0), 100, (3,), constraint=spec)
    assert row.step(2).tokens == (1, 2)
    result = row.step(6)
    assert result.tokens == (3,) and result.finish_reason == "stop"
    row.close()
    factory.create(ConstraintSpec('start: "a"')).close()
    assert len(factory.prototypes) == 1
    assert factory.create(spec).forced() == (1, 2, 3, 4, 5)
    with pytest.raises(ConstraintError):
        runtime.prepare(
            (0,), SamplingPolicy(temperature=0), 1, constraint=ConstraintSpec("start: missing_rule")
        )
    assert budget.snapshot().reserved == 0
    with pytest.raises(ValueError):
        ConstraintSpec("x" * ((1 << 20) + 1))
