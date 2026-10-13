import math

import pytest
import torch

from magnitude_references.sanity import Bounds, Gate


def confident(targets: torch.Tensor, vocabulary: int = 10) -> torch.Tensor:
    logits = torch.full((targets.shape[0], vocabulary), -10.0)
    logits[torch.arange(targets.shape[0]), targets] = 10.0
    return logits


def test_confident_correct_predictions_pass() -> None:
    gate = Gate()
    targets = torch.tensor([3, 1, 4, 1, 5])
    gate.add("a", confident(targets), targets)
    result = gate.result(Bounds())
    assert result["perplexity"] == pytest.approx(1.0, abs=1e-6)
    assert result["top1_accuracy"] == 1.0
    assert result["passed"] is True
    assert result["bounds"] == {"perplexity_max": 300.0, "top1_min": 0.10}
    assert result["fixtures"] == [{"fixture_id": "a", "perplexity": pytest.approx(1.0, abs=1e-6),
                                   "top1_accuracy": 1.0, "passed": True}]


def test_flat_distribution_fails_on_perplexity() -> None:
    """A near-flat distribution over a large vocabulary is the broken regime: perplexity ~ vocabulary."""
    gate = Gate()
    gate.add("a", torch.zeros(8, 1000), torch.arange(8))
    result = gate.result(Bounds())
    assert result["perplexity"] == pytest.approx(1000.0)
    assert result["passed"] is False


def test_top1_bound_fails_independently() -> None:
    """Low perplexity but the actual token never ranked first (two-way tie broken the wrong way)."""
    gate = Gate()
    logits = torch.full((4, 50), -30.0)
    logits[:, 0] = 1.0
    logits[:, 1] = 0.9
    gate.add("a", logits, torch.tensor([1, 1, 1, 1]))
    result = gate.result(Bounds())
    assert result["perplexity"] < 3
    assert result["top1_accuracy"] == 0.0
    assert result["passed"] is False


def test_one_broken_fixture_fails_the_reference() -> None:
    """Many sane rows would carry the aggregate past the bounds; the per-fixture gate still fails."""
    gate = Gate()
    targets = torch.arange(10).repeat(50)
    gate.add("sane", confident(targets), targets)
    gate.add("broken", torch.zeros(5, 10), torch.arange(5))
    result = gate.result(Bounds(perplexity_max=5.0, top1_min=0.5))
    assert result["perplexity"] < 5.0 and result["top1_accuracy"] > 0.5
    assert result["passed"] is False
    assert Gate.failing(result) == ["broken"]
    assert [fixture["passed"] for fixture in result["fixtures"]] == [True, False]


def test_accumulates_across_batches() -> None:
    whole, parts = Gate(), Gate()
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(12, 30, generator=generator)
    targets = torch.randint(0, 30, (12,), generator=generator)
    whole.add("a", logits, targets)
    parts.add("a", logits[:5], targets[:5])
    parts.add("a", logits[5:5], targets[5:5])
    parts.add("a", logits[5:], targets[5:])
    a, b = whole.result(Bounds()), parts.result(Bounds())
    assert math.isclose(a["perplexity"], b["perplexity"], rel_tol=1e-12)
    assert a["top1_accuracy"] == b["top1_accuracy"]


def test_no_scored_rows_is_an_error() -> None:
    gate = Gate()
    gate.add("a", torch.zeros(0, 4), torch.zeros(0, dtype=torch.int64))
    with pytest.raises(ValueError):
        gate.result(Bounds())
