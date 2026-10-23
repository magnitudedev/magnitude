"""The host's reference cache and the refusal of references that failed their sanity gate."""

import json
import subprocess
from pathlib import Path

import logits_fake
import pytest
from magnitude_references.digest import implementation, sha256_file

from magnitude_references import cache as reference


def sanity(passed: bool) -> dict:
    return {
        "perplexity": 2000.0 if not passed else 3.0,
        "top1_accuracy": 0.1 if not passed else 0.6,
        "bounds": {"perplexity_max": 300.0, "top1_min": 0.25},
        "passed": passed,
        "fixtures": [
            {"fixture_id": "f/0", "perplexity": 3.0, "top1_accuracy": 0.6, "passed": True},
            {"fixture_id": "f/1", "perplexity": 2000.0, "top1_accuracy": 0.1, "passed": passed},
        ],
    }


@pytest.fixture
def rendered(tmp_path):
    fixtures = tmp_path / "set.jsonl"
    fixtures.write_text("".join(json.dumps({"id": f"f/{index}"}) + "\n" for index in range(2)))
    path = tmp_path / "rendered.jsonl"
    path.write_text(
        "".join(
            json.dumps(line) + "\n" for line in logits_fake.rendered_lines(fixtures, 4096, False)
        )
    )
    return path


@pytest.fixture
def references_cli(monkeypatch):
    """A stand-in for `magnitude-references run`: writes a reference record with the given
    sanity result (exit 1 when it failed, as the real command does)."""
    calls = []
    behaviour = {"passed": True, "record": True}

    def run(command, stdout, stderr):
        calls.append(command)
        output = command[command.index("--output") + 1]
        rendered = command[command.index("--rendered") + 1]
        if behaviour["record"]:
            directory = Path(output)
            directory.mkdir(parents=True)
            (directory / "reference.json").write_text(
                json.dumps(
                    {
                        "model": logits_fake.MODEL,
                        "rendered_digest": sha256_file(rendered),
                        "family": "fake",
                        "implementation_digest": implementation(),
                        "dtype": "float32",
                        "sanity": sanity(behaviour["passed"]),
                    }
                )
            )
        return subprocess.CompletedProcess(
            command, 0 if behaviour["passed"] and behaviour["record"] else 1
        )

    monkeypatch.setattr(reference.subprocess, "run", run)
    return calls, behaviour


def obtain(rendered: Path, tmp_path: Path) -> reference.Reference:
    identity = tmp_path / "build-tools.json"
    return reference.obtain(
        rendered, tmp_path / "artifacts.json", identity, tmp_path / "cache", tmp_path / "log"
    )


def test_built_once_then_cached(tmp_path, rendered, references_cli):
    calls, _ = references_cli
    first = obtain(rendered, tmp_path)
    assert not first.cached and len(calls) == 1
    identity = calls[0][calls[0].index("--build-identity") + 1]
    assert identity == str(tmp_path / "build-tools.json")
    assert first.directory == reference.location(
        tmp_path / "cache", logits_fake.MODEL, sha256_file(rendered), implementation()
    )
    assert reference.check(first) is first
    second = obtain(rendered, tmp_path)
    assert second.cached and second.directory == first.directory and len(calls) == 1
    assert not list(first.directory.parent.glob("*.partial-*"))


def test_failed_sanity_is_cached_and_refused(tmp_path, rendered, references_cli):
    calls, behaviour = references_cli
    behaviour["passed"] = False
    with pytest.raises(reference.ReferenceRefused, match="failing fixtures f/1"):
        reference.check(obtain(rendered, tmp_path))
    again = obtain(rendered, tmp_path)
    assert again.cached and len(calls) == 1
    with pytest.raises(reference.ReferenceRefused):
        reference.check(again)


def test_run_without_a_record_fails(tmp_path, rendered, references_cli):
    _, behaviour = references_cli
    behaviour["record"] = False
    with pytest.raises(reference.ReferenceFailed, match="without a reference"):
        obtain(rendered, tmp_path)


def test_cached_record_of_another_identity_fails(tmp_path, rendered, references_cli):
    built = obtain(rendered, tmp_path)
    record = json.loads((built.directory / reference.RECORD).read_text())
    (built.directory / reference.RECORD).write_text(
        json.dumps({**record, "model": "other:gguf:q8"})
    )
    with pytest.raises(reference.ReferenceFailed, match="records model"):
        obtain(rendered, tmp_path)
