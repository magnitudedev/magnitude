"""A candidate must not establish its own numerical acceptance budget."""

import numpy as np
import pytest

from performance.model_accuracy import compare


def test_paired_precision_preserves_every_metric_and_prediction():
    anchor = np.array([3.0, 1.0, -1.0, -2.0])
    control = anchor + np.array([0.04, -0.04, 0.03, -0.03])
    good = anchor + (control - anchor) * 0.5
    assert compare(anchor, control, good[None, :]).validation().passed
    assert not compare(anchor, control, np.stack([good, anchor + 0.5])).validation().passed
    shifted = compare(anchor, control, (anchor + 0.5)[None, :])
    assert shifted.candidates[0].anchor_to_candidate_kl < 1e-12
    assert not shifted.validation().passed
    assert (
        not compare(anchor, np.array([0.0, 5.0, -1.0, -2.0]), anchor[None, :]).validation().passed
    )


def test_paired_precision_rejects_invalid_and_uncalibrated_values():
    anchor = np.zeros(4)
    assert compare(anchor, anchor, anchor[None, :]).validation().passed
    assert not compare(anchor, anchor, np.full((1, 4), 0.002)).validation().passed
    for invalid in (np.full((1, 4), np.nan), np.full((1, 4), np.inf), np.zeros((1, 5))):
        with pytest.raises(ValueError):
            compare(anchor, anchor, invalid)
