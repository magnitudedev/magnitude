"""The released-code checks' own machinery: the float64 mode both implementations run under, the bar, and the
comparison with another producer's logits."""

import numpy as np
import pytest
import torch

from magnitude_references.checks import Float64, compare_logits, judge, without_float64
from magnitude_references.quality import compare_rows
from magnitude_references.model import DECODERS, submit


def test_float64_runs_float32_code_in_float64() -> None:
    third = torch.tensor([1.0]) / 3  # float32
    with Float64():
        widened = torch.tensor([1.0]).float() / 3
        upcast = torch.softmax(torch.zeros(4), -1, dtype=torch.float32)
        pooled = submit(DECODERS, lambda: torch.ones(2, dtype=torch.float32) / 3).result()
        with pytest.raises(AssertionError, match="in place"):
            third.mul_(2)
    assert widened.dtype == upcast.dtype == pooled.dtype == torch.float64
    assert float(widened) == 1 / 3 != float(third)


def test_the_bar_is_the_released_codes_own_float32_rounding() -> None:
    truth = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    released = truth.float() + torch.tensor([[0.0, 0.0, 1e-6]])
    near, far = truth + 1e-12, truth + torch.tensor([[0.0, 0.0, 1e-5]], dtype=torch.float64)
    choice = truth.argmax(-1)
    assert judge({"logits": (near, released, truth)}, {"argmax": (choice, choice)})["pass"]
    assert not judge({"logits": (far, released, truth)}, {"argmax": (choice, choice)})["pass"]
    assert not judge({"logits": (near, released, truth)}, {"argmax": (choice + 1, choice)})["pass"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="a device without float64 (MPS)")
def test_a_device_without_float64_is_unjudgeable_before_anything_runs() -> None:
    assert without_float64(torch.device("cpu")) is None
    refused = without_float64(torch.device("mps"))
    assert refused is not None and not refused["judgeable"] and not refused["pass"]


def test_another_producers_logits_compare_per_position() -> None:
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(2, 3, 5)).astype(np.float32)
    test = truth.copy()
    test[1, 2, truth[1, 2].argmax()] -= 10  # one position's top-1 moves
    positions, difference = compare_logits(truth, test)
    assert positions.rows == tuple({"prompt": p, "position": i} for p in range(2) for i in range(3))
    kl, agree = compare_rows(truth.reshape(-1, 5), test.reshape(-1, 5))
    assert np.array_equal(positions.kl, kl) and np.array_equal(positions.agree, agree)
    assert list(np.flatnonzero(positions.kl)) == [5] and not positions.agree[5]
    assert difference == pytest.approx(10 / np.abs(truth).max())
