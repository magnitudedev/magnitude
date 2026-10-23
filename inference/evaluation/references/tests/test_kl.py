"""`magnitude-references kl`: a build against the cached reference, and a pair of builds."""

import json

import logits_fake
import numpy as np
from magnitude_references.digest import sha256_file
from test_cache import obtain, references_cli, rendered  # noqa: F401 (fixtures)

from magnitude_references import kl


def _logits(directory, rendered_file, values):
    directory.mkdir()
    (directory / "rendered.jsonl").write_text(rendered_file.read_text())
    lines = [json.loads(line) for line in rendered_file.read_text().splitlines()]
    rows = logits_fake.rows_of(lines)
    logits_fake.write_logits(directory, "harness", sha256_file(rendered_file), rows, values(rows))


def test_a_pair_is_judged_against_the_reference_and_each_other(tmp_path, rendered, references_cli):  # noqa: F811
    built = obtain(rendered, tmp_path)
    lines = [json.loads(line) for line in rendered.read_text().splitlines()]
    rows = logits_fake.rows_of(lines)
    logits_fake.write_logits(built.directory, "reference", sha256_file(rendered), rows, logits_fake.truth(rows))
    noisy = np.random.default_rng(1).normal(0, 0.3, logits_fake.truth(rows).shape)
    _logits(tmp_path / "base", rendered, lambda r: logits_fake.truth(r) + noisy)
    _logits(tmp_path / "candidate", rendered, lambda r: logits_fake.truth(r) + noisy)
    summary = kl.run(tmp_path / "candidate", tmp_path / "artifacts.json", tmp_path / "build.json",
                     tmp_path / "cache", tmp_path / "out" / "kl.json", tmp_path / "base", None, False)
    assert summary["reference"]["cached"] is True
    assert summary["builds_identical"] is True
    assert {change["verdict"] for change in summary["changes"]} == {"within noise"}
    assert summary["build"]["mean_kl"] > 0
    assert not list(tmp_path.glob("*/logits.f32"))  # judged logits are deleted
    assert json.loads((tmp_path / "out" / "kl.json").read_text())["build"] == summary["build"]
