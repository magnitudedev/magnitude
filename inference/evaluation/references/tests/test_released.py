"""The references against their families' released code and checkpoints (the `check` extra).

- Every synthetic variation case (`synthetic.CASES`, `DRAFT_CASES`): its reference against the
  family's released modeling code (transformers, or mlx_vlm for a family transformers lacks, run
  by `MAGNITUDE_MLX_PYTHON`, an interpreter that has it), and every draft case against the
  released draft code at its pinned revision. Network: template headers and pinned code.
- With `MAGNITUDE_ARTIFACTS` (a catalog acquisition object, as `mrig test --model` provides it):
  the model's target reference against the released modeling code, on its first
  `MAGNITUDE_REFERENCE_LAYERS` layers when set (a large model), and its catalog drafter against
  the released draft code; with `MAGNITUDE_CHECKPOINT` (the released safetensors directory), the
  target reference's weights against the checkpoint's (the GGUF mapping).

They run on a device host: `mrig test --suite python [--model M] -- evaluation/references/tests/test_released.py`.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="the references' `check` extra")

from magnitude_references.checks import check_hf, check_weights, device_of
from magnitude_references.families import REFERENCES, open_reference
from magnitude_references.families.dflash import ZLAB_CODE, check_draft, open_draft
from magnitude_references.gguf import Package
from magnitude_references.synthetic import CASES, DRAFT_CASES, check_families

MLX_PYTHON = Path(os.environ["MAGNITUDE_MLX_PYTHON"]) if "MAGNITUDE_MLX_PYTHON" in os.environ else None
ARTIFACTS = json.loads(Path(os.environ["MAGNITUDE_ARTIFACTS"]).read_text()) if "MAGNITUDE_ARTIFACTS" in os.environ else None
TARGETS = {case.name: case for case in CASES}


@pytest.fixture(scope="session")
def device():
    """The host's device, for a real model's checks."""
    return device_of("auto")


@pytest.fixture(scope="session")
def cpu():
    """The synthetic cases' device: they are small, and on the CPU both implementations compute
    in IEEE float32 with no device kernel's own accumulation order between them (MPS differs by
    1.6e-4 relative on gemma4-vk-512 where the CPU differs by 0)."""
    return device_of("cpu")


@pytest.fixture(scope="session")
def zlab_source(tmp_path_factory) -> Path:
    """The Z-Lab DFlash/DFlash2 released code at its pinned revision."""
    repository, revision, path = ZLAB_CODE
    source = tmp_path_factory.mktemp("zlab") / Path(path).name
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repository}/{revision}/{path}", timeout=60) as response:
        source.write_bytes(response.read())
    return source


def released(case) -> str:
    """The released code a target case's family is checked against."""
    with tempfile.TemporaryDirectory(prefix="magnitude-released-") as temporary:
        path = Path(temporary) / f"{case.name}.gguf"
        case.build(case).write(path)
        return getattr(REFERENCES[Package.local(path).architecture][1], "released", "transformers")


def checked(target) -> None:
    if released(target) == "mlx_vlm" and MLX_PYTHON is None:
        pytest.skip(f"{target.name} is checked against mlx_vlm: set MAGNITUDE_MLX_PYTHON")


@pytest.mark.parametrize("name", [case.name for case in CASES] + [case.name for case in DRAFT_CASES])
def test_family_reference_matches_released_code(name, cpu, zlab_source):
    draft = next((case for case in DRAFT_CASES if case.name == name), None)
    checked(TARGETS[draft.target if draft else name])
    reports = check_families({name}, cpu, zlab_source, MLX_PYTHON)
    failed = [report for report in reports if not report["pass"]]
    assert not failed, json.dumps(failed, indent=1)


needs_model = pytest.mark.skipif(ARTIFACTS is None, reason="a catalog model: mrig test --model")


@needs_model
def test_model_reference_matches_released_code(device):
    layers = int(os.environ["MAGNITUDE_REFERENCE_LAYERS"]) if "MAGNITUDE_REFERENCE_LAYERS" in os.environ else None
    report = check_hf(open_reference(Path(ARTIFACTS["target"]), device, True, layers), 3, 48, 20260927, MLX_PYTHON)
    assert report["pass"], json.dumps(report, indent=1)


@needs_model
def test_model_drafter_matches_released_code(device, zlab_source):
    if ARTIFACTS.get("drafter") is None or ARTIFACTS["drafter"].get("path") is None:
        pytest.skip(f"{ARTIFACTS['model']} has no catalog drafter file")
    draft = open_draft(Path(ARTIFACTS["drafter"]["path"]), Path(ARTIFACTS["target"]), device)
    source = draft.released_code
    report = check_draft(draft, 24, 20260927, None, source, zlab_source if source == "zlab" else None, None)
    assert report["pass"], json.dumps(report, indent=1)


@needs_model
@pytest.mark.skipif("MAGNITUDE_CHECKPOINT" not in os.environ, reason="the released checkpoint: MAGNITUDE_CHECKPOINT")
def test_model_reference_weights_match_the_checkpoint(device):
    reference = open_reference(Path(ARTIFACTS["target"]), device, True)
    report = check_weights(reference, Path(os.environ["MAGNITUDE_CHECKPOINT"]))
    unmapped = [row for row in report["rows"] if "relative_error" not in row]
    assert not unmapped, json.dumps(unmapped, indent=1)
    # A mapping error shows as a relative error near 1; the rest is the file's quantization error.
    assert report["worst_relative_error"] < 0.5, json.dumps(report["rows"], indent=1)
