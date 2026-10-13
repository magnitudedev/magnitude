"""Quality metrics on synthetic logits: KL, top-1 agreement, contract summary and the paired
noise-aware judgement."""

import math

import compare_fake
import numpy as np
import pytest

from magnitude_compare import quality


def test_kl_and_agreement_match_a_direct_computation():
    generator = np.random.default_rng(1)
    truth = generator.normal(0, 2, (70, 50)).astype("<f4")  # more rows than one chunk
    test = (truth + generator.normal(0, 0.5, truth.shape)).astype("<f4")
    kl, agree = quality.compare_rows(truth, test)
    for row in (0, 33, 69):
        p = np.exp(truth[row].astype(np.float64)) / np.exp(truth[row].astype(np.float64)).sum()
        q = np.exp(test[row].astype(np.float64)) / np.exp(test[row].astype(np.float64)).sum()
        assert kl[row] == pytest.approx(float((p * np.log(p / q)).sum()), rel=1e-9)
        assert agree[row] == (truth[row].argmax() == test[row].argmax())
    same_kl, same_agree = quality.compare_rows(truth, truth)
    assert (same_kl == 0).all() and same_agree.all()


def test_non_finite_logits_fail():
    truth = np.zeros((2, 4), dtype="<f4")
    test = truth.copy()
    test[1, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        quality.compare_rows(truth, test)


def test_summary_trims_the_top_tenth_of_a_percent():
    kl = np.concatenate([np.full(1998, 0.01), [5.0, 7.0]])
    agree = np.array([True] * 1500 + [False] * 500)
    summary = quality.summarize(kl, agree)
    assert summary["positions"] == 2000
    assert summary["max_kl"] == 7.0
    assert summary["mean_kl"] == pytest.approx((1998 * 0.01 + 12.0) / 2000)
    assert summary["trimmed_kl"] == pytest.approx(0.01)  # the top 2 of 2000 dropped
    assert summary["p99_kl"] == pytest.approx(0.01)
    assert summary["median_kl"] == pytest.approx(0.01)
    assert summary["p90_kl"] == pytest.approx(0.01)
    assert summary["mean_kl_without_10_worst"] == pytest.approx(0.01)
    assert summary["positions_above_0_5_nats"] == 2
    assert summary["top1_agreement"] == 0.75
    small = quality.summarize(np.array([0.1, 0.2, 3.0]), np.array([True, True, True]))
    assert small["trimmed_kl"] == small["mean_kl"]  # under 1000 positions nothing is trimmed
    assert small["mean_kl_without_10_worst"] is None
    assert small["median_kl"] == pytest.approx(0.2)
    assert small["p90_kl"] == pytest.approx(2.44)
    assert small["positions_above_0_5_nats"] == 1


def positions(kl, agree=None) -> quality.Positions:
    rows = tuple({"fixture_id": "f", "position": index} for index in range(len(kl)))
    return quality.Positions(
        rows,
        np.asarray(kl, dtype=float),
        np.asarray(agree if agree is not None else [True] * len(kl)),
    )


def verdicts(changes) -> dict[str, str]:
    return {change.metric: change.verdict for change in changes}


def test_judge_improved_regressed_and_within_noise():
    generator = np.random.default_rng(2)
    base_kl = generator.exponential(0.01, 2000)
    base = positions(base_kl, generator.random(2000) < 0.9)
    worse = positions(base_kl * 1.5, base.agree)
    better = positions(base_kl * 0.5, base.agree)
    assert verdicts(quality.judge(base, worse, 0.02))["mean_kl"] == "regressed"
    assert verdicts(quality.judge(base, better, 0.02))["mean_kl"] == "improved"
    unchanged = verdicts(quality.judge(base, base, 0.02))
    assert set(unchanged.values()) == {"within noise"}
    # Zero-mean noise per position: even with no bar, the paired interval straddles zero.
    jitter = positions(base_kl * generator.normal(1, 0.05, 2000).clip(0), base.agree)
    assert verdicts(quality.judge(base, jitter, 0.0))["mean_kl"] == "within noise"


def test_judge_bar_suppresses_significant_but_small_changes():
    base_kl = np.random.default_rng(3).exponential(0.01, 2000)
    slightly = positions(base_kl * 1.01)
    changes = {
        change.metric: change for change in quality.judge(positions(base_kl), slightly, 0.02)
    }
    assert changes["mean_kl"].interval[0] > 0  # significant
    assert changes["mean_kl"].verdict == "within noise"  # but under the 2% bar
    assert changes["mean_kl"].relative == pytest.approx(0.01)
    assert verdicts(quality.judge(positions(base_kl), slightly, 0.005))["mean_kl"] == "regressed"


def test_judge_top1_higher_is_better_and_zero_base():
    kl = np.full(400, 0.01)
    base = positions(kl, [True] * 300 + [False] * 100)
    candidate = positions(kl, [True] * 380 + [False] * 20)
    assert verdicts(quality.judge(base, candidate, 0.02))["top1_agreement"] == "improved"
    perfect = positions(np.zeros(400))
    drifted = positions(np.full(400, 1e-3))
    change = {c.metric: c for c in quality.judge(perfect, drifted, 0.02)}["mean_kl"]
    assert change.verdict == "regressed" and math.isinf(change.relative)


def test_judge_requires_the_same_positions():
    with pytest.raises(quality.Incomparable):
        quality.judge(positions([0.1, 0.2]), positions([0.1, 0.2, 0.3]), 0.02)


def test_compare_outputs_requires_the_same_rendered_input_and_rows(tmp_path):
    rows = [{"fixture_id": "f", "position": position} for position in range(4)]
    values = compare_fake.truth(rows)
    for name, digest, written in (
        ("a", "d1", rows),
        ("b", "d1", rows),
        ("c", "d2", rows),
        ("d", "d1", rows[:3]),
    ):
        (tmp_path / name).mkdir()
        compare_fake.write_logits(
            tmp_path / name, "harness", digest, written, values[: len(written)]
        )
    same = quality.compare_outputs(tmp_path / "a", tmp_path / "b")
    assert (same.kl == 0).all() and same.rows == tuple(rows)
    with pytest.raises(quality.Incomparable, match="rendered_digest"):
        quality.compare_outputs(tmp_path / "a", tmp_path / "c")
    with pytest.raises(quality.Incomparable, match="rows"):
        quality.compare_outputs(tmp_path / "a", tmp_path / "d")


def test_worst_positions_are_diagnostic():
    found = quality.worst(positions([0.1, 0.5, 0.3], [True, False, True]), count=2)
    assert found == [
        {"fixture_id": "f", "position": 1, "kl": 0.5, "agree": False},
        {"fixture_id": "f", "position": 2, "kl": 0.3, "agree": True},
    ]
