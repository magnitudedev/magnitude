import mlx.core as mx
import pytest

from magnitude_engine.generation.acceptance import accept_prefix
from magnitude_engine.generation.sampling import SequenceSampler, position_key
from magnitude_engine.generation.sampling_policy import SamplingPolicy


def test_seeded_draws_do_not_depend_on_grouping_or_speculative_read_order():
    sampler = SequenceSampler(SamplingPolicy(seed=182, temperature=0.8, top_p=0.9))
    logits = mx.random.normal((20, 32), key=mx.random.key(18))
    sequential = [sampler.sample(logits[i], 100 + i).item() for i in range(20)]
    reordered = {i: sampler.sample(logits[i], 100 + i).item() for i in reversed(range(20))}
    assert sequential == [reordered[i] for i in range(20)]
    assert len({tuple(position_key(182, i).tolist()) for i in range(100)}) == 100


def test_penalty_preview_does_not_commit_rejected_draft_history():
    sampler = SequenceSampler(
        SamplingPolicy(
            temperature=0,
            repetition_penalty=2,
            presence_penalty=1,
            frequency_penalty=0.25,
            history_window=3,
        )
    )
    sampler.observe((0, 1, 1))
    logits = mx.array([4.0, -2.0, 3.0])
    original = sampler.logits(logits).tolist()
    assert original == [0.75, -5.5, 3.0]
    preview = sampler.preview((2, 2))
    assert preview.logits(logits).tolist() == [4.0, -5.25, 0.0]
    assert sampler.logits(logits).tolist() == original


@pytest.mark.parametrize(
    "proposals,samples,stops,count,bonus",
    [
        ([1, 2, 3], [1, 2, 3, 4], (), 3, 4),
        ([1, 2, 3], [1, 9, 3, 4], (), 1, 9),
        ([1, 2, 3], [1, 2, 3, 4], (2,), 1, 2),
        ([], [7], (), 0, 7),
    ],
)
def test_acceptance_alignment_and_terminal_bonus(proposals, samples, stops, count, bonus):
    accepted = accept_prefix(mx.array(proposals, dtype=mx.int32), mx.array(samples), stops)
    assert accepted.count.item() == count
    assert accepted.bonus.item() == bonus
    assert accepted.target_inputs_committed.item() == count + 1


def test_top_k_above_vocabulary_is_safe_and_greedy_ignores_truncation():
    assert SequenceSampler(SamplingPolicy(top_k=100, seed=1)).sample(mx.array([0.0]), 0).item() == 0
    assert (
        SequenceSampler(SamplingPolicy(temperature=0, top_k=1))
        .sample(mx.array([-3.0, 9.0, 1.0]), 0)
        .item()
        == 1
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_greedy_order_and_tie_breaking_match_widened_logits(dtype):
    sampler = SequenceSampler(SamplingPolicy(temperature=0))
    values = mx.array([-mx.inf, -0.0, 0.0, 1.125, 1.125, -900], dtype=dtype)
    assert sampler.sample(values, 7).item() == mx.argmax(values.astype(mx.float32)).item() == 3
    logits = mx.random.normal((11, 257), key=mx.random.key(81)).astype(dtype)
    assert [sampler.sample(row, 9).item() for row in logits] == mx.argmax(
        logits.astype(mx.float32), axis=-1
    ).tolist()
