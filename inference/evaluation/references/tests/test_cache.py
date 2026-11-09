"""The host's reference cache, its admission of a CPU build, and the refusal of references that
failed their sanity gate."""

import json
import subprocess
from pathlib import Path

import logits_fake
import numpy as np
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
    sanity result (exit 1 when it failed, as the real command does), unless `missing`, and its logits
    (`logits`: the rows to values; the fake's truth by default; `variants`: a variant's own, by its
    name). Whether a build was stopped is `killed`."""
    calls = []
    behaviour = {"passed": True, "missing": False, "logits": logits_fake.truth, "variants": {}, "killed": False}

    class Popen:
        def __init__(self, command, stdout, stderr):
            calls.append(command)
            output = command[command.index("--output") + 1]
            rendered = command[command.index("--rendered") + 1]
            variant = command[command.index("--variant") + 1] if "--variant" in command else None
            self.code = 0 if behaviour["passed"] and not behaviour["missing"] else 1
            if behaviour["missing"]:
                return
            directory = Path(output)
            directory.mkdir(parents=True)
            (directory / "reference.json").write_text(
                json.dumps(
                    {
                        "model": logits_fake.MODEL,
                        "rendered_digest": sha256_file(rendered),
                        "family": "fake",
                        "implementation_digest": implementation(variant),
                        "dtype": "float32",
                        "sanity": sanity(behaviour["passed"]),
                    }
                )
            )
            lines = [json.loads(line) for line in Path(rendered).read_text().splitlines()]
            rows = logits_fake.rows_of(lines)
            logits_fake.write_logits(directory, "reference:fake", sha256_file(rendered), rows,
                                     behaviour["variants"].get(variant, behaviour["logits"])(rows))

        def wait(self):
            return self.code

        def kill(self):
            behaviour["killed"] = True

    monkeypatch.setattr(reference.subprocess, "Popen", Popen)
    return calls, behaviour


def obtain(rendered: Path, tmp_path: Path) -> reference.Reference:
    identity = tmp_path / "build-tools.json"
    return reference.obtain(
        rendered, tmp_path / "artifacts.json", identity, tmp_path / "cache", tmp_path / "log"
    )


def _devices(calls):
    return [command[command.index("--device") + 1] for command in calls]


def _leftovers(cache: Path) -> list:
    return sorted(path.name.split(".")[-1].split("-")[0] for path in cache.rglob("*")
                  if path.is_dir() and "." in path.name)


def test_built_admitted_once_then_cached(tmp_path, rendered, references_cli):
    calls, _ = references_cli
    first = obtain(rendered, tmp_path)
    assert not first.cached and _devices(calls) == ["cpu"]  # one CPU build, admitted alone
    assert {command[command.index("--build-identity") + 1] for command in calls} == {str(tmp_path / "build-tools.json")}
    assert first.directory == reference.location(
        tmp_path / "cache", logits_fake.MODEL, sha256_file(rendered), implementation()
    )
    assert first.record["admission"] == {"device": "cpu", "agreement": "CPU-deterministic"}
    assert reference.check(first) is first
    second = obtain(rendered, tmp_path)
    assert second.cached and second.directory == first.directory and len(calls) == 1
    assert not _leftovers(tmp_path / "cache")  # the partial build does not remain


def test_a_build_of_non_finite_logits_is_refused_and_kept(tmp_path, rendered, references_cli):
    _, behaviour = references_cli

    def logits(rows):
        values = logits_fake.truth(rows)
        values[3, 7] = np.inf
        return values

    behaviour["logits"] = logits
    with pytest.raises(reference.ReferenceRefused, match=r"\(built on cpu\) has non-finite logits"):
        obtain(rendered, tmp_path)
    directory = reference.location(tmp_path / "cache", logits_fake.MODEL, sha256_file(rendered), implementation())
    assert not directory.exists()  # nothing is admitted
    assert _leftovers(tmp_path / "cache") == ["partial"]  # the build is kept


@pytest.mark.parametrize("admission, reused", [
    (None, False),  # as a cache before admission left it
    ({"first_device": "mps", "check_device": "cpu", "agreement": "float32 rounding"}, False),
    ({"first_device": "cpu", "agreement": "CPU-deterministic"}, False),  # another record's shape
    ({"device": "cpu", "agreement": "CPU-deterministic"}, True),
])
def test_only_an_entry_admitted_as_a_cpu_build_is_reused(tmp_path, rendered, references_cli, admission, reused):
    calls, _ = references_cli
    built = obtain(rendered, tmp_path)
    record = json.loads((built.directory / reference.RECORD).read_text())
    if admission is None:
        del record["admission"]
    else:
        record["admission"] = admission
    (built.directory / reference.RECORD).write_text(json.dumps(record))
    again = obtain(rendered, tmp_path)
    assert again.cached is reused and len(calls) == (1 if reused else 2)
    assert again.record["admission"] == (admission if reused else {"device": "cpu", "agreement": "CPU-deterministic"})


def test_failed_sanity_is_cached_and_refused(tmp_path, rendered, references_cli):
    calls, behaviour = references_cli
    behaviour["passed"] = False
    with pytest.raises(reference.ReferenceRefused, match="failing fixtures f/1"):
        reference.check(obtain(rendered, tmp_path))
    again = obtain(rendered, tmp_path)
    assert again.cached and len(calls) == 1  # built once
    with pytest.raises(reference.ReferenceRefused):
        reference.check(again)


def test_run_without_a_record_fails_and_removes_the_build(tmp_path, rendered, references_cli):
    _, behaviour = references_cli
    behaviour["missing"] = True
    with pytest.raises(reference.ReferenceFailed, match="without a reference"):
        obtain(rendered, tmp_path)
    assert not _leftovers(tmp_path / "cache")


def test_an_interrupted_build_is_stopped_and_removed(tmp_path, rendered, references_cli, monkeypatch):
    _, behaviour = references_cli

    waited = reference.subprocess.Popen.wait

    def interrupted(self):  # the first wait is interrupted; the one after the kill returns
        monkeypatch.setattr(reference.subprocess.Popen, "wait", waited)
        raise KeyboardInterrupt

    monkeypatch.setattr(reference.subprocess.Popen, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        obtain(rendered, tmp_path)
    assert behaviour["killed"] and not _leftovers(tmp_path / "cache")


def test_cached_record_of_another_identity_fails(tmp_path, rendered, references_cli):
    built = obtain(rendered, tmp_path)
    record = json.loads((built.directory / reference.RECORD).read_text())
    (built.directory / reference.RECORD).write_text(
        json.dumps({**record, "model": "other:gguf:q8"})
    )
    with pytest.raises(reference.ReferenceFailed, match="records model"):
        obtain(rendered, tmp_path)
