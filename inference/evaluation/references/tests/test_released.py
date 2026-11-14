"""The references against their families' released code and checkpoints (the `check` extra).

- Every synthetic variation case (`synthetic.CASES`, `DRAFT_CASES`): its reference against the
  family's released modeling code (`checks.judge`; a family with no float64 run of its released code is
  reported unjudgeable and skipped), and every draft case against the released draft code at its pinned
  revision. Network: template headers and pinned code.
- With `MAGNITUDE_ARTIFACTS` (a catalog acquisition object, as `mrig test --model` provides it):
  the model's target reference against the released modeling code, on its first
  `MAGNITUDE_REFERENCE_LAYERS` layers when set (a large model), and its catalog drafter against
  the released draft code; with `MAGNITUDE_CHECKPOINT` (the released safetensors directory), the
  target reference's weights against the checkpoint's (the GGUF mapping).
- Every real-weight slice (`SLICES`) present in the host's model store (`MAGNITUDE_MODELS`, as `mrig test` provides
  it): its reference against the released modeling code on the CPU.

They run on a device host: `mrig test --suite python [--model M] -- evaluation/references/tests/test_released.py`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="the references' `check` extra")

from magnitude_references.checks import check_hf, check_weights, device_of
from magnitude_references.families import open_reference
from magnitude_references.families.dflash import ZLAB_CODE, check_draft, open_draft
from magnitude_references.synthetic import CASES, DRAFT_CASES, check_families

ARTIFACTS = json.loads(Path(os.environ["MAGNITUDE_ARTIFACTS"]).read_text()) if "MAGNITUDE_ARTIFACTS" in os.environ else None
MODELS = Path(os.environ["MAGNITUDE_MODELS"]) if "MAGNITUDE_MODELS" in os.environ else None
# Real-weight slices a conversion writes into a host's model store, by their path in it, with the prompt length that
# exercises each one's mechanisms. GLM-5.3-Flash at 160 positions, three of transformers' 64-row KDA chunks, the last
# of them padded: slice A ({layer 0}: KDA, dense FP8 FFN, mHC, embedding and head) and slice B ({0-3}: adds NoPE MLA
# with its pooled indexer and FP8 MoE).
SLICES = {"glm-5.3-flash-A": ("zai-org/GLM-5.3-Flash-slice-A/GLM-5.3-Flash-slice-A.gguf", 160),
          "glm-5.3-flash-B": ("zai-org/GLM-5.3-Flash-slice-B/GLM-5.3-Flash-slice-B.gguf", 160)}


@pytest.fixture(scope="session")
def device():
    """The host's device, for a real model's checks."""
    return device_of("auto")


@pytest.fixture(scope="session")
def cpu():
    """The synthetic cases' device: they are small, and the CPU computes both implementations in float64 as well
    as float32 (`checks.judge`)."""
    return device_of("cpu")


@pytest.fixture(scope="session")
def zlab_source(tmp_path_factory) -> Path:
    """The Z-Lab DFlash/DFlash2 released code at its pinned revision."""
    repository, revision, path = ZLAB_CODE
    source = tmp_path_factory.mktemp("zlab") / Path(path).name
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repository}/{revision}/{path}", timeout=60) as response:
        source.write_bytes(response.read())
    return source


def judged(report: dict) -> None:
    """A check's report is a pass; one with no float64 truth is skipped as unjudgeable, never passed."""
    if not report["judgeable"]:
        pytest.skip(report["reason"])
    assert report["pass"], json.dumps(report, indent=1)


@pytest.mark.parametrize("name", [case.name for case in CASES] + [case.name for case in DRAFT_CASES])
def test_family_reference_matches_released_code(name, cpu, zlab_source):
    report, = (report for report in check_families({name}, cpu, zlab_source) if report["case"] == name)
    judged(report)


needs_model = pytest.mark.skipif(ARTIFACTS is None, reason="a catalog model: mrig test --model")


@needs_model
def test_model_reference_matches_released_code(device):
    layers = int(os.environ["MAGNITUDE_REFERENCE_LAYERS"]) if "MAGNITUDE_REFERENCE_LAYERS" in os.environ else None
    judged(check_hf(open_reference(Path(ARTIFACTS["target"]), device, True, layers), 3, 48, 20260927))


@needs_model
def test_model_drafter_matches_released_code(device, zlab_source):
    if ARTIFACTS.get("drafter") is None or ARTIFACTS["drafter"].get("path") is None:
        pytest.skip(f"{ARTIFACTS['model']} has no catalog drafter file")
    draft = open_draft(Path(ARTIFACTS["drafter"]["path"]), Path(ARTIFACTS["target"]), device)
    source = draft.released_code
    judged(check_draft(draft, 24, 20260927, None, source, zlab_source if source == "zlab" else None, None))


@needs_model
@pytest.mark.skipif("MAGNITUDE_CHECKPOINT" not in os.environ, reason="the released checkpoint: MAGNITUDE_CHECKPOINT")
def test_model_reference_weights_match_the_checkpoint(device):
    reference = open_reference(Path(ARTIFACTS["target"]), device, True)
    report = check_weights(reference, Path(os.environ["MAGNITUDE_CHECKPOINT"]))
    unmapped = [row for row in report["rows"] if "relative_error" not in row]
    assert not unmapped, json.dumps(unmapped, indent=1)
    # A mapping error shows as a relative error near 1; the rest is the file's quantization error.
    assert report["worst_relative_error"] < 0.5, json.dumps(report["rows"], indent=1)


@pytest.mark.parametrize("name", sorted(SLICES))
def test_slice_reference_matches_released_code(name, cpu):
    path, length = SLICES[name]
    if MODELS is None or not (MODELS / path).is_file():
        pytest.skip(f"{name} is not in this host's model store (MAGNITUDE_MODELS)")
    report = check_hf(open_reference(MODELS / path, cpu, True), 3, length, 20260927)
    print(json.dumps(report))  # the evidence, shown for a pass with -rP
    judged(report)
