"""Speed judgement against each side's noise, per-entry attribution and drafting."""

import pytest

from magnitude_compare import speed


def cell(rate: float, spread: float = 0.0, attention: float = 10.0, phase: str = "decode") -> dict:
    return {
        "phase": phase,
        "context": 4096,
        "concurrency": 1,
        "tokens_per_second": rate,
        "samples": [rate * (1 - spread), rate, rate, rate * (1 + spread)],
        "detail": {
            "attribution": {
                "entries": [
                    {"entry": "attention", "device_ms_per_step": attention},
                    {
                        "entry": "norm",
                        "device_ms_per_step": 0.1,
                    },  # under 2% of either side: not shown
                    {"entry": "matmul#1", "device_ms_per_step": 4.0},
                    {"entry": "matmul#2", "device_ms_per_step": 4.0},
                ]
            }
        },
    }


def samples(base: list[dict], candidate: list[dict]) -> list[speed.Sample]:
    return [
        *(speed.Sample("base", index, "tools", value) for index, value in enumerate(base)),
        *(
            speed.Sample("candidate", index, "tools", value)
            for index, value in enumerate(candidate)
        ),
    ]


def test_noise_widens_the_tolerance():
    # 4% slower, but the base's rounds spread 10%: within noise.
    [row] = speed.compare(
        samples([cell(100), cell(110), cell(105)], [cell(101), cell(100.8), cell(100.9)]), 0.0
    )
    assert row["verdict"] == "within noise"
    assert row["noise"] == pytest.approx(10 / 105)
    assert row["noisy"]
    # The same change on a quiet host is a regression.
    [row] = speed.compare(
        samples([cell(105), cell(105.1), cell(104.9)], [cell(101), cell(100.8), cell(100.9)]), 0.0
    )
    assert row["verdict"] == "regressed" and row["delta"] == pytest.approx(100.9 / 105 - 1)
    assert not row["noisy"]


def test_a_single_round_is_judged_against_its_sample_spread():
    [row] = speed.compare(samples([cell(100, spread=0.05)], [cell(104, spread=0.01)]), 0.0)
    assert row["noise"] == pytest.approx(speed.sample_spread([95, 100, 100, 105]))
    assert row["verdict"] == "within noise"
    [row] = speed.compare(samples([cell(100, spread=0.01)], [cell(104, spread=0.01)]), 0.0)
    assert row["verdict"] == "improved"


def test_bar_applies_above_noise():
    [row] = speed.compare(samples([cell(100)], [cell(103)]), 0.05)
    assert row["verdict"] == "within noise"


def test_attribution_names_the_moved_entries():
    [row] = speed.compare(samples([cell(100)], [cell(80, attention=12.5)]), 0.0)
    assert row["verdict"] == "regressed"
    assert [entry["entry"] for entry in row["entries"]] == ["attention", "matmul"]
    assert row["entries"][0] == {
        "entry": "attention",
        "base_ms": 10.0,
        "candidate_ms": 12.5,
        "delta": 0.25,
    }
    assert row["entries"][1]["base_ms"] == 8.0  # launches of one entry summed


def test_cells_without_attribution_are_judged_without_entries():
    plain = [{**cell(100), "detail": {"rows": 4094.0}}, {**cell(90), "detail": None}]
    [row] = speed.compare(samples([plain[0]], [plain[1]]), 0.0)
    assert row["verdict"] == "regressed" and row["entries"] == []


def test_entry_on_one_side_only_comes_first():
    candidate = cell(90)
    candidate["detail"]["attribution"]["entries"].append(
        {"entry": "fused", "device_ms_per_step": 3.0}
    )
    [row] = speed.compare(samples([cell(100)], [candidate]), 0.0)
    assert row["entries"][0] == {"entry": "fused", "only": "candidate"}


def test_cell_measured_on_one_side_is_missing():
    rows = speed.compare(samples([cell(100), cell(100, phase="prefill")], [cell(100)]), 0.0)
    assert {row["phase"]: row["verdict"] for row in rows} == {
        "decode": "within noise",
        "prefill": "missing",
    }
    assert rows[1]["only"] == "base"


def test_drafting_acceptance():
    drafting = [
        ("base", "tools", {"tokens_per_verify": 2.0}),
        ("base", "tools", {"tokens_per_verify": 2.0}),
        ("candidate", "tools", {"tokens_per_verify": 2.4}),
        ("candidate", "tools", {"tokens_per_verify": 2.4}),
        ("base", "code", {"tokens_per_verify": 2.0}),
    ]
    rows = {row["family"]: row for row in speed.compare_drafting(drafting, 0.0)}
    assert rows["tools"]["verdict"] == "improved" and rows["tools"]["delta"] == pytest.approx(0.2)
    assert rows["code"] == {"family": "code", "verdict": "missing", "only": "base"}
