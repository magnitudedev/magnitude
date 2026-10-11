"""The host runner end to end on a fake harness and fake references: schedule, judgement,
result record, report, refusals and rendering findings."""

import hashlib
import json
from pathlib import Path

import compare_fake
import pytest

from magnitude_remote.build import BuildIdentity

from magnitude_compare import runner
from magnitude_compare.plan import Build, Plan, Tuning
from magnitude_compare.reference import Reference
from magnitude_compare.steps import HostPaths

WORKTREE_IDENTITY = BuildIdentity("a" * 40, "b" * 64)
FIXTURES = [{"id": f"tools/fake/{index}", "family": "tools", "messages": []} for index in range(3)]
FIXTURES_TEXT = "".join(json.dumps(fixture) + "\n" for fixture in FIXTURES)
FIXTURES_DIGEST = hashlib.sha256(FIXTURES_TEXT.encode()).hexdigest()


def make_plan(**changes) -> Plan:
    settings = {
        "builds": {
            "base": Build("HEAD", BuildIdentity("a" * 40, None)),
            "candidate": Build("worktree", WORKTREE_IDENTITY),
        },
        "tools": WORKTREE_IDENTITY,
        "fixtures": "/controller/fixture-sets",
        "fixture_sets": {"tools": FIXTURES_DIGEST},
        "contexts": (4096,),
        "concurrency": (1, 2),
        "rounds": 2,
        "tuning": Tuning(5.0, "fresh"),
        "quality": True,
        "speed": True,
        "quality_bar": 0.02,
        "speed_bar": 0.0,
        "command": ("magnitude-compare", "run"),
    }
    return Plan(**{**settings, **changes})


def setup(tmp_path: Path, plan: Plan, base: dict, candidate: dict) -> HostPaths:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    plan.write(inputs / "plan.json")
    for side, build in plan.builds.items():
        (inputs / f"build-{side}.json").write_text(json.dumps(build.identity.to_json()))
    (inputs / "build-tools.json").write_text(json.dumps(plan.tools.to_json()))
    (inputs / "tools.jsonl").write_text(FIXTURES_TEXT)
    defaults = {"speed": 1.0, "noise": 0.05, "shift_tokens": False, "drafting": 2.0}
    compare_fake.install(tmp_path / "targets" / "base", {**defaults, **base})
    compare_fake.install(tmp_path / "targets" / "candidate", {**defaults, **candidate})
    (tmp_path / "artifacts.json").write_text(
        json.dumps(
            {
                "model": compare_fake.MODEL,
                "target": "/fake.gguf",
                "target_sha256": compare_fake.TARGET_SHA,
                "drafter": None,
                "projector": None,
            }
        )
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return HostPaths(
        inputs=str(inputs),
        run_dir=str(run_dir),
        artifacts=str(tmp_path / "artifacts.json"),
        root=str(tmp_path / "root"),
        targets={
            "base": str(tmp_path / "targets" / "base"),
            "candidate": str(tmp_path / "targets" / "candidate"),
        },
    )


@pytest.fixture
def references(tmp_path, monkeypatch):
    """Fake references: the true logits of the rendered input, with a configurable sanity result."""
    state = {"passed": True, "built": 0}

    def obtain(
        rendered: Path, artifacts: Path, build_identity: Path, cache: Path, log: Path
    ) -> Reference:
        assert json.loads(build_identity.read_text()) == WORKTREE_IDENTITY.to_json()
        lines = [json.loads(line) for line in rendered.read_text().splitlines()]
        digest = hashlib.sha256(rendered.read_bytes()).hexdigest()
        directory = tmp_path / "references" / digest
        directory.mkdir(parents=True, exist_ok=True)
        rows = compare_fake.rows_of(lines)
        compare_fake.write_logits(
            directory, "reference:fake", digest, rows, compare_fake.truth(rows)
        )
        state["built"] += 1
        sanity = {
            "perplexity": 3.0,
            "top1_accuracy": 0.6,
            "passed": state["passed"],
            "bounds": {"perplexity_max": 300.0, "top1_min": 0.25},
            "fixtures": [
                {
                    "fixture_id": line["fixture_id"],
                    "perplexity": 3.0,
                    "top1_accuracy": 0.6,
                    "passed": state["passed"],
                }
                for line in lines
            ],
        }
        record = {
            "model": compare_fake.MODEL,
            "rendered_digest": digest,
            "family": "fake",
            "implementation_digest": "c" * 64,
            "dtype": "float32",
            "sanity": sanity,
        }
        return Reference(directory, record, False)

    monkeypatch.setattr(runner, "obtain", obtain)
    return state


def result(paths: HostPaths) -> dict:
    return json.loads((Path(paths.run_dir) / runner.RESULT).read_text())


def test_identical_builds_pass(tmp_path, references):
    plan = make_plan()
    paths = setup(tmp_path, plan, {}, {})
    assert runner.Runner(plan, paths).run() == runner.EXIT["pass"]
    record = result(paths)
    assert record["tool"] == "compare" and record["schema"] == "evaluation-result/1"
    assert record["build"] == {"commit": "a" * 40, "dirty_digest": "b" * 64}
    assert record["sides"]["base"]["source"] == "HEAD"
    assert record["fixture_set"]["sets"] == [
        {
            "family": "tools",
            "path": "/controller/fixture-sets/tools.jsonl",
            "digest": FIXTURES_DIGEST,
        }
    ]
    comparison = record["measurements"]["comparison"]
    assert comparison["verdict"] == "pass" and not comparison["failures"]
    assert {row["verdict"] for row in comparison["speed"]} == {"within noise"}
    assert len(comparison["speed"]) == 4  # prefill and decode at concurrency 1 and 2
    [quality_row] = comparison["quality"]
    assert quality_row["builds_identical"] and quality_row["rendering"] == []
    assert {change["verdict"] for change in quality_row["changes"]} == {"within noise"}
    pairs = {entry["pair"] for entry in record["measurements"]["quality"]}
    assert pairs == {"base-reference", "candidate-reference", "candidate-base"}
    assert all(
        entry["positions"] == 3 * compare_fake.SCORED for entry in record["measurements"]["quality"]
    )
    # Speed rounds alternate, and only each side's first run tunes afresh.
    tuning = record["measurements"]["tuning"]
    assert [entry["run"] for entry in tuning[:4]] == [
        "speed/r0-tools-base",
        "speed/r0-tools-candidate",
        "speed/r1-tools-candidate",
        "speed/r1-tools-base",
    ]
    assert [entry["fresh"] for entry in tuning] == [True, True, False, False, False, False]
    trace = (Path(paths.run_dir) / "tuning.jsonl").read_text().splitlines()
    # Every run's `began` and `concluded`, and a searching load's `budget`.
    searching = sum(entry["fresh"] for entry in tuning)
    assert len(trace) == 2 * len(tuning) + searching
    assert json.loads(trace[0])["run"] == "speed/r0-tools-base"
    # The builds' logits are deleted once judged; the report is written.
    assert not list(Path(paths.run_dir).rglob("*.f32"))
    assert (Path(paths.run_dir) / runner.REPORT).read_text().startswith("# Compare: PASS")


def test_slower_noisier_candidate_regresses(tmp_path, references):
    plan = make_plan()
    paths = setup(tmp_path, plan, {}, {"speed": 0.8, "noise": 0.3, "drafting": 1.5})
    assert runner.Runner(plan, paths).run() == runner.EXIT["regressed"]
    comparison = result(paths)["measurements"]["comparison"]
    assert {row["verdict"] for row in comparison["speed"]} == {"regressed"}
    assert all(row["delta"] == pytest.approx(-0.2, abs=0.01) for row in comparison["speed"])
    # Attribution names the kernel that slowed down, launches of one entry summed.
    entries = comparison["speed"][0]["entries"]
    assert entries[0]["entry"] == "attention" and entries[0]["delta"] == pytest.approx(
        0.25, abs=1e-6
    )
    assert {entry["entry"] for entry in entries} == {"attention", "matmul"}
    assert comparison["drafting"][0]["verdict"] == "regressed"
    changes = {change["metric"]: change for change in comparison["quality"][0]["changes"]}
    assert changes["mean_kl"]["verdict"] == "regressed"
    assert changes["mean_kl"]["interval"][0] > 0
    assert not comparison["quality"][0]["builds_identical"]
    report = (Path(paths.run_dir) / runner.REPORT).read_text()
    assert report.startswith("# Compare: REGRESSED") and "`attention`" in report


def test_failed_reference_is_refused(tmp_path, references):
    references["passed"] = False
    plan = make_plan(speed=False)
    paths = setup(tmp_path, plan, {}, {})
    assert runner.Runner(plan, paths).run() == runner.EXIT["failed"]
    record = result(paths)
    comparison = record["measurements"]["comparison"]
    [failure] = comparison["failures"]
    assert failure["step"] == "quality/tools-4096/reference" and "sanity gate" in failure["error"]
    # Nothing is judged against the refused reference; the builds are still compared.
    assert {entry["pair"] for entry in record["measurements"]["quality"]} == {"candidate-base"}
    assert "changes" not in comparison["quality"][0]
    assert not comparison["quality"][0]["reference"]["sanity"]["passed"]


def test_speed_on_different_renderings_fails(tmp_path, references):
    plan = make_plan(quality=False, rounds=1)
    paths = setup(tmp_path, plan, {}, {"shift_tokens": True})
    assert runner.Runner(plan, paths).run() == runner.EXIT["failed"]
    comparison = result(paths)["measurements"]["comparison"]
    [differing] = comparison["speed_rendering"]
    assert differing["family"] == "tools" and differing["round"] == 0
    assert comparison["failures"][0]["step"] == "speed/r0-tools"
    assert "timed different tokens" in (Path(paths.run_dir) / runner.REPORT).read_text()


def test_rendering_difference_is_a_finding_and_quality_uses_the_base_tokens(tmp_path, references):
    plan = make_plan(speed=False)
    paths = setup(tmp_path, plan, {}, {"shift_tokens": True})
    assert runner.Runner(plan, paths).run() == runner.EXIT["pass"]
    record = result(paths)
    comparison = record["measurements"]["comparison"]
    row = comparison["quality"][0]
    assert [difference["tokens"]["first_difference"] for difference in row["rendering"]] == [5] * 3
    assert not comparison["failures"]
    # The candidate was measured on the base's exact tokens: every pair is comparable.
    assert {entry["pair"] for entry in record["measurements"]["quality"]} == {
        "base-reference",
        "candidate-reference",
        "candidate-base",
    }
    assert row["builds_identical"]
    command = json.loads(
        (Path(paths.run_dir) / "quality/tools-4096/candidate/result.json").read_text()
    )["command"]
    assert command[command.index("--rendered") + 1] == str(
        Path(paths.run_dir) / "quality/tools-4096/base/rendered.jsonl"
    )
    report = (Path(paths.run_dir) / runner.REPORT).read_text()
    assert "Finding: the candidate renders different tokens" in report


def test_failed_engine_run_is_recorded_and_the_rest_still_runs(tmp_path, references):
    plan = make_plan(quality=False, rounds=1)
    paths = setup(tmp_path, plan, {}, {})
    (Path(paths.targets["candidate"]) / "release" / "behaviour.json").write_text("not json")
    assert runner.Runner(plan, paths).run() == runner.EXIT["failed"]
    comparison = result(paths)["measurements"]["comparison"]
    assert comparison["failures"][0]["step"] == "speed/r0-tools-candidate"
    assert comparison["failures"][0]["log"] == "speed/r0-tools-candidate/log.txt"
    assert {row["verdict"] for row in comparison["speed"]} == {"missing"}


def test_replay_base_tuning(tmp_path, references):
    plan = make_plan(tuning=Tuning(5.0, "replay-base"), quality=False)
    paths = setup(tmp_path, plan, {}, {})
    assert runner.Runner(plan, paths).run() == runner.EXIT["pass"]
    commands = [
        json.loads((Path(paths.run_dir) / "speed" / name / "result.json").read_text())["command"]
        for name in ("r0-tools-base", "r0-tools-candidate", "r1-tools-base")
    ]
    pins = str(Path(paths.run_dir) / "tuning-pins.json")
    assert commands[0][-3:] == ["--fresh-tuning", "--tuning-record", pins]
    assert commands[1][-2:] == ["--tuning-replay", pins]
    assert commands[2][-2:] == ["--tuning-replay", pins]


def test_a_shipped_fixture_set_other_than_planned_is_refused(tmp_path, references):
    plan = make_plan()
    paths = setup(tmp_path, plan, {}, {})
    (Path(paths.fixtures("tools"))).write_text(FIXTURES_TEXT + "\n")
    with pytest.raises(ValueError, match="tools fixture set on this host"):
        runner.Runner(plan, paths).run()
