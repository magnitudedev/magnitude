"""`magnitude-references kl`: a build against the cached reference, a pair of builds, a build
against a yardstick or the storage-resolution bar, and a slice judged without the sanity gate."""

import json

import logits_fake
import numpy as np
import pytest
from magnitude_references.digest import sha256_file
from test_cache import obtain, references_cli, rendered  # noqa: F401 (fixtures)

from magnitude_references import kl
from magnitude_references.cache import ReferenceRefused


def _logits(directory, rendered_file, values):
    directory.mkdir()
    (directory / "rendered.jsonl").write_text(rendered_file.read_text())
    lines = [json.loads(line) for line in rendered_file.read_text().splitlines()]
    rows = logits_fake.rows_of(lines)
    logits_fake.write_logits(directory, "harness", sha256_file(rendered_file), rows, values(rows))


def _acquired(tmp_path, sliced=False):
    """`tmp_path`'s acquisition object, of a target that is a `catalog-tool slice` output when `sliced`."""
    target = tmp_path / "model.gguf"
    target.write_bytes(b"GGUF")
    provenance = target.with_suffix(".json")
    if sliced:
        provenance.write_text(json.dumps({"main_layers": [0], "sha256": sha256_file(target)}))
    elif provenance.exists():
        provenance.unlink()
    (tmp_path / "artifacts.json").write_text(json.dumps({
        "model": logits_fake.MODEL, "target": str(target), "target_sha256": sha256_file(target),
        "drafter": None, "projector": None}))


@pytest.fixture(autouse=True)
def acquired(tmp_path):
    _acquired(tmp_path)


def test_a_pair_is_judged_against_the_reference_and_each_other(tmp_path, rendered, references_cli):  # noqa: F811
    built = obtain(rendered, tmp_path)
    lines = [json.loads(line) for line in rendered.read_text().splitlines()]
    rows = logits_fake.rows_of(lines)
    logits_fake.write_logits(built.directory, "reference", sha256_file(rendered), rows, logits_fake.truth(rows))
    noisy = np.random.default_rng(1).normal(0, 0.3, logits_fake.truth(rows).shape)
    _logits(tmp_path / "base", rendered, lambda r: logits_fake.truth(r) + noisy)
    _logits(tmp_path / "candidate", rendered, lambda r: logits_fake.truth(r) + noisy)
    summary = kl.run(tmp_path / "candidate", tmp_path / "artifacts.json", tmp_path / "build.json",
                     tmp_path / "cache", tmp_path / "out" / "kl.json", tmp_path / "base", None, None)
    assert summary["reference"]["cached"] is True
    assert summary["gate"] == {"mode": "model"}
    assert summary["builds_identical"] is True
    assert {change["verdict"] for change in summary["changes"]} == {"within noise"}
    assert summary["build"]["mean_kl"] > 0
    assert not list(tmp_path.glob("*/logits.f32"))  # judged logits are deleted
    assert json.loads((tmp_path / "out" / "kl.json").read_text())["build"] == summary["build"]


def _judge(tmp_path, mode, yardstick=None, candidate="candidate"):
    _acquired(tmp_path, sliced=mode == "slice")
    return kl.run(tmp_path / candidate, tmp_path / "artifacts.json", tmp_path / "build.json",
                  tmp_path / "cache", tmp_path / "out" / "kl.json", None, None, yardstick)


def _noise(scale):
    return np.random.default_rng(4).normal(0, scale, (2 * logits_fake.SCORED, logits_fake.VOCABULARY))


@pytest.fixture
def failed_slice(tmp_path, rendered, references_cli):  # noqa: F811
    """A slice: its reference fails the sanity gate, and its logits are the fake's truth, the
    same in every build."""
    _, behaviour = references_cli
    behaviour["passed"] = False
    _logits(tmp_path / "candidate", rendered, lambda r: logits_fake.truth(r) + _noise(0.3))
    return references_cli


def test_a_slice_is_judged_without_the_sanity_gate_and_refused_under_it(tmp_path, failed_slice):
    calls, _ = failed_slice
    with pytest.raises(ReferenceRefused, match="sanity gate"):
        _judge(tmp_path, "model")
    summary = _judge(tmp_path, "slice")
    assert summary["gate"] == {"mode": "slice"}
    assert summary["reference"]["sanity"]["passed"] is False  # recorded, not applied
    assert summary["reference"]["admission"]["agreement"] == "CPU-deterministic"
    assert summary["build"]["mean_kl"] > 0
    assert len(calls) == 1  # built once: the cached reference is judged as built


def test_a_slice_is_its_target_s_provenance_record(tmp_path):
    assert kl.mode_of(tmp_path / "artifacts.json") == "model"
    _acquired(tmp_path, sliced=True)
    assert kl.mode_of(tmp_path / "artifacts.json") == "slice"
    (tmp_path / "model.json").write_text(json.dumps({"sha256": "0" * 64}))  # a record of another file
    assert kl.mode_of(tmp_path / "artifacts.json") == "model"


def test_a_build_is_judged_against_a_yardstick_on_the_same_tokens(tmp_path, rendered, failed_slice):  # noqa: F811
    _logits(tmp_path / "level", rendered, lambda r: logits_fake.truth(r) + _noise(0.3))
    level = _judge(tmp_path, "slice", tmp_path / "level")
    assert level["yardstick"]["mean_kl"] == level["build"]["mean_kl"]
    assert level["within_yardstick"] is True
    assert (tmp_path / "level" / "logits.f32").is_file()  # the yardstick is the caller's
    _logits(tmp_path / "finer", rendered, lambda r: logits_fake.truth(r) + _noise(0.1))
    _logits(tmp_path / "drifted", rendered, lambda r: logits_fake.truth(r) + _noise(0.3))
    drifted = _judge(tmp_path, "slice", tmp_path / "finer", candidate="drifted")
    assert drifted["within_yardstick"] is False
    assert {change["metric"]: change["verdict"] for change in drifted["yardstick_changes"]}["mean_kl"] == "regressed"


def test_a_build_is_judged_against_the_storage_resolution_bar(tmp_path, rendered, references_cli):  # noqa: F811
    """The perturbed reference, built and cached beside the reference, is the bar: a build within 64 times its KL at
    every bounded quantile is within it, one at 16 times its logit noise (256 times its KL) is not."""
    calls, behaviour = references_cli
    noise = np.random.default_rng(7).normal(0, 0.01, (2 * logits_fake.SCORED, logits_fake.VOCABULARY))
    behaviour["variants"]["perturbed:3"] = lambda rows: logits_fake.truth(rows) + noise.astype(np.float32)

    def judged(scale, name):
        _logits(tmp_path / name, rendered, lambda r: logits_fake.truth(r) + (scale * noise).astype(np.float32))
        return kl.run(tmp_path / name, tmp_path / "artifacts.json", tmp_path / "build.json", tmp_path / "cache",
                      tmp_path / "out" / f"{name}.json", None, None, None, None, 3)

    near = judged(2, "near")
    bar = near["resolution"]
    assert bar["reference"]["implementation_digest"] != near["reference"]["implementation_digest"]
    assert bar["perturbed"]["positions"] == near["build"]["positions"] and bar["perturbed"]["median_kl"] > 0
    assert set(bar["bounds"]) == {"median_kl", "p90_kl", "p99_kl", "max_kl"} and bar["within"] is True
    assert all(bound["limit"] == 64 * bound["perturbed"] for bound in bar["bounds"].values())
    far = judged(16, "far")
    assert far["resolution"]["within"] is False and not any(b["within"] for b in far["resolution"]["bounds"].values())
    assert len(calls) == 2  # the reference and the perturbed reference, each built once on the CPU
    with pytest.raises(ValueError, match="perturbs the model's numerics"):
        kl.run(tmp_path / "far", tmp_path / "artifacts.json", tmp_path / "build.json", tmp_path / "cache",
               tmp_path / "out" / "x.json", None, None, None, "activations", 3)
