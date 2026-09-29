import mlx.core as mx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from magnitude_engine.generation.sampling import SequenceSampler, position_key
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.experts.grouping import grouped_ranges, support_missing


@settings(max_examples=70, deadline=None)
@given(
    st.lists(st.integers(0, 40), min_size=16, max_size=16),
    st.lists(st.booleans(), min_size=16, max_size=16),
    st.integers(1, 16),
)
def test_readiness_geometry_and_support_selection_match_reference(
    poc_module, counts, readiness, span
):
    oracle = poc_module("expert_streaming.prefill_plan")
    ids = np.repeat(np.arange(16), counts)
    ready = np.repeat(readiness, counts)
    expected = oracle.plan_expert_ranges(ids, span)
    actual = grouped_ranges(ids, span)
    if expected is None:
        assert actual is None
    else:
        assert [(g.start, g.end, g.first_expert, g.expert_limit) for g in actual] == [
            (g.begin, g.end, g.first_expert, g.stop_expert) for g in expected
        ]
    expected = oracle.support_missing_ranges(ids, ready, span, 16)
    actual = support_missing(ids, ready, target_span=span, storage_slots=16)
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert np.array_equal(actual[0], expected[0])
        assert np.array_equal(actual[1], expected[1])
        assert np.all(actual[0] ^ actual[1])


@pytest.mark.parametrize("seed", [-1, 0, 761, (1 << 64) - 1])
@pytest.mark.parametrize("penalties", [False, True])
def test_seed_mapping_and_sampled_tokens_match_reference(poc_module, seed, penalties):
    oracle = poc_module("scheduler.sampler")
    parameters = poc_module("scheduler.requests").SamplingParams
    policy = SamplingPolicy(
        seed=seed,
        temperature=0.7,
        top_k=23,
        min_p=0.05,
        top_p=0.9,
        repetition_penalty=1.1 if penalties else 1,
        presence_penalty=0.2 if penalties else 0,
        frequency_penalty=0.1 if penalties else 0,
    )
    new = SequenceSampler(policy)
    old = oracle.RowSampler.for_params(
        parameters(
            seed=seed,
            temperature=policy.temperature,
            top_k=policy.top_k,
            min_p=policy.min_p,
            top_p=policy.top_p,
            repetition_penalty=policy.repetition_penalty,
            presence_penalty=policy.presence_penalty,
            frequency_penalty=policy.frequency_penalty,
        )
    )
    history = (1, 3, 3, 7, 1)
    old.observe(list(history))
    new.observe(history)
    values = mx.random.normal((16, 32), key=mx.random.key(42))
    for offset in range(16):
        position = 2048 + offset
        assert mx.array_equal(position_key(seed, position), old.key_at(position)).item()
        assert (
            new.sample(values[offset], position).item()
            == oracle.sample_row(values[offset], old, position).item()
        )
