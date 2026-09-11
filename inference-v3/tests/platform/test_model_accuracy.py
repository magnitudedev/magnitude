"""A faster candidate cannot establish its own numerical acceptance budget."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from magnitude_engine.kernels import precision
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.platform.host.measurement import exclusive_measurement
from magnitude_engine.weights.identity import ArtifactIdentity
from performance.model import LogitsReference, ModelCase, ModelWorkload, ReferenceIdentity
from performance.model_accuracy import compare


def test_precision_envelope_preserves_each_metric_and_each_peer():
    with exclusive_measurement():
        anchor = np.array([3.0, 1.0, -1.0, -2.0])
        control = anchor + np.array([0.04, -0.04, 0.03, -0.03])
        good = anchor + (control - anchor) * 0.5
        assessed = compare(anchor, control, good[None, :])
        assert assessed.validation().passed
        assert assessed.model_validate_json(assessed.model_dump_json()) == assessed
        assert not compare(anchor, control, np.stack([good, anchor + 0.5])).validation().passed
        # Distribution agreement alone cannot excuse shifted or inaccurate logits.
        shifted = compare(anchor, control, (anchor + 0.5)[None, :])
        assert shifted.candidates[0].anchor_to_candidate_kl < 1e-12
        assert not shifted.validation().passed
        # References cannot permit a changed winner even inside an error envelope.
        assert (
            not compare(anchor, np.array([0.0, 5.0, -1.0, -2.0]), anchor[None, :])
            .validation()
            .passed
        )


def test_precision_controls_handle_exact_and_invalid_values():
    with exclusive_measurement():
        anchor = np.zeros(4)
        assert compare(anchor, anchor, anchor[None, :]).validation().passed
        assert not compare(anchor, anchor, np.full((1, 4), 0.002)).validation().passed
        for invalid in (np.full((1, 4), np.nan), np.full((1, 4), np.inf), np.zeros((1, 5))):
            with pytest.raises(ValueError):
                compare(anchor, anchor, invalid)
        with pytest.raises(ValueError):
            compare(anchor, anchor, np.zeros((0, 4)))


def test_model_rejects_missing_or_mismatched_precision_anchor_before_execution(tmp_path):
    artifact = ArtifactIdentity("a" * 64)
    model = cast(
        DenseRuntime,
        SimpleNamespace(
            artifact_identity=artifact,
            precision=precision.MIXED_BF16,
            geometry=SimpleNamespace(context_limit=64, vocabulary=4),
        ),
    )

    def reference(name: str, tokens: tuple[int, ...], family: str):
        path: Path = tmp_path / name
        content = (
            np.array([len(tokens), 4, *tokens], dtype="<i4").tobytes()
            + np.zeros((len(tokens), 4), dtype="<f4").tobytes()
        )
        path.write_bytes(content)
        return LogitsReference(
            path=path,
            identity=ReferenceIdentity(hashlib.sha256(content).hexdigest()),
            artifact=artifact,
            precision=family,
        )

    with exclusive_measurement():
        control = reference("control.bin", (1, 2), "mixed_bf16")
        with pytest.raises(ValueError, match="explicit paired FP32"):
            ModelCase(model, ModelWorkload(reference=control))
        wrong_tokens = reference("anchor.bin", (1, 3), "reference_f32")
        with pytest.raises(ValueError, match="tokens or geometry"):
            ModelCase(model, ModelWorkload(reference=control, accuracy_anchor=wrong_tokens))
        with pytest.raises(ValueError, match="FP32 interpretation"):
            ModelCase(model, ModelWorkload(reference=control, accuracy_anchor=control))
