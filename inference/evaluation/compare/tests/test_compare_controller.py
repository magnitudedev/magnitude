"""The controlling side: the remote job, the inputs, the dry run and the exit codes."""

import hashlib
import json
from pathlib import Path

import pytest
from magnitude_remote.build import BuildIdentity
from magnitude_remote.flow import Tree

from magnitude_compare import cli, controller, steps
from magnitude_compare.plan import Build, Plan, Tuning
from test_compare_runner import make_plan


def ref_plan(**changes) -> Plan:
    return make_plan(**changes)  # base HEAD (a ref), candidate the working tree


def test_job_builds_both_harnesses_and_runs_the_host_runner():
    trees = (Tree("compare-base", Path("/local/base"), ("Cargo.toml",)),)
    job = controller.job(ref_plan(), "m:gguf:q4", trees, None)
    assert (
        job.tool == "compare"
        and job.model == "m:gguf:q4"
        and job.trees == trees
        and job.gate is None
    )
    assert job.env == (("UV_PROJECT_ENVIRONMENT", "{root}/venv"),)
    sync, base, candidate = job.build
    assert sync[:3] == ("uv", "sync", "--frozen")
    assert base == (
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        "{tree:compare-base}/Cargo.toml",
        "-p",
        "magnitude-harness",
        "--target-dir",
        "{root}/targets/compare-base",
    )
    assert candidate == (
        "cargo",
        "build",
        "--release",
        "-p",
        "magnitude-harness",
        "--target-dir",
        "{target}",
    )
    command = list(job.command)
    assert command[command.index("host") - 1] == "magnitude-compare"
    assert command[command.index("--base-target") + 1] == "{root}/targets/compare-base"
    assert command[command.index("--candidate-target") + 1] == "{target}"
    assert command[command.index("--inputs") + 1] == "{tree:compare-inputs}"


def test_replay_builds_with_pinned_tuning():
    job = controller.job(ref_plan(tuning=Tuning(30, "replay-base")), "m:gguf:q4", (), None)
    assert all(("--features", "pinned-tuning") == command[-4:-2] for command in job.build[1:])


def test_two_refs_each_get_a_tree_and_target():
    plan = ref_plan(
        builds={
            "base": Build("v1", BuildIdentity("a" * 40, None)),
            "candidate": Build("v2", BuildIdentity("b" * 40, None)),
        }
    )
    paths = controller.host_paths(plan)
    assert paths.targets == {
        "base": "{root}/targets/compare-base",
        "candidate": "{root}/targets/compare-candidate",
    }


def test_prepare_writes_the_inputs(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(
        controller.builds,
        "materialize",
        lambda side, build, root: built.append(side) or Tree(f"compare-{side}", root / side, ()),
    )
    sets = tmp_path / "fixture-sets"
    sets.mkdir()
    for family in ("tools", "code", "retrieval"):
        (sets / f"{family}.jsonl").write_text(f'{{"id": "{family}/0"}}\n')
    plan = ref_plan(
        fixtures=str(sets), fixture_sets=controller.fixture_sets(sets, ("tools", "code"))
    )
    prepared = controller.prepare(plan, tmp_path)
    assert built == ["base"]
    inputs = prepared.trees[-1]
    assert inputs.name == "compare-inputs" and inputs.local == tmp_path / "trees" / "compare-inputs"
    assert set(inputs.paths) == {
        "plan.json",
        "build-base.json",
        "build-candidate.json",
        "build-tools.json",
        "tools.jsonl",
        "code.jsonl",
    }
    assert Plan.read(inputs.local / "plan.json") == plan
    assert (inputs.local / "code.jsonl").read_bytes() == (sets / "code.jsonl").read_bytes()
    assert json.loads((inputs.local / "build-candidate.json").read_text()) == {
        "commit": "a" * 40,
        "dirty_digest": "b" * 64,
    }
    # A set edited after planning is refused rather than shipped under the planned digest.
    (sets / "code.jsonl").write_text('{"id": "code/1"}\n')
    with pytest.raises(controller.FixturesUnavailable, match="changed after it was planned"):
        controller.prepare(plan, tmp_path)


def test_fixture_sets_are_prebuilt_files_recorded_by_digest(tmp_path):
    (tmp_path / "tools.jsonl").write_bytes(b"x\n")
    assert controller.fixture_sets(tmp_path, ("tools",)) == {
        "tools": hashlib.sha256(b"x\n").hexdigest()
    }
    with pytest.raises(controller.FixturesUnavailable, match="no fixture sets for code"):
        controller.fixture_sets(tmp_path, ("tools", "code"))


def test_dry_run_prints_jobs_and_host_schedule(capsys, tmp_path):
    (tmp_path / "tools.jsonl").write_bytes(b"x\n")
    code = cli.main(
        [
            "run",
            "--fixtures",
            str(tmp_path),
            "--base",
            "HEAD",
            "--model",
            "a:gguf:q4,b:gguf:q8",
            "--family",
            "tools",
            "--contexts",
            "4096",
            "--rounds",
            "2",
            "--host",
            "gpu-pool",
            "--dry-run",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert output.count("job: tool compare") == 2 and "on gpu-pool" in output
    assert "speed/r1-tools-candidate:" in output and "quality/tools-4096/base:" in output
    plan = json.loads(output[output.index("{") : output.index("\n\njob:")])
    assert plan["fixture_sets"] == {"tools": hashlib.sha256(b"x\n").hexdigest()}
    assert plan["fixtures"] == str(tmp_path.resolve())
    assert (
        plan["builds"]["base"]["source"] == "HEAD"
        and plan["builds"]["candidate"]["source"] == "worktree"
    )


def test_cli_rejects_unknown_families_and_identical_sides():
    with pytest.raises(SystemExit):
        cli.main(["run", "--base", "HEAD", "--model", "m", "--family", "poetry", "--dry-run"])
    with pytest.raises(SystemExit, match="same source"):
        cli.main(["run", "--base", "worktree", "--model", "m", "--family", "tools", "--dry-run"])


def test_exit_code_severity():
    assert cli._worse(0, 2) == 2
    assert cli._worse(2, 1) == 1
    assert cli._worse(1, 0) == 1
    assert cli._worse(0, 137) == 1


def test_schedule_without_speed_tunes_on_the_first_quality_runs():
    plan = ref_plan(speed=False, tuning=Tuning(10, "replay-base"))
    paths = controller.host_paths(plan)
    scheduled = steps.schedule(plan, paths)
    assert scheduled.speed == ()
    [cell] = scheduled.quality
    first, second = (cell.runs[side].argv for side in ("base", "candidate"))
    assert first[-3:] == ("--fresh-tuning", "--tuning-record", "{run_dir}/tuning-pins.json")
    assert second[-2:] == ("--tuning-replay", "{run_dir}/tuning-pins.json")
    # The candidate measures the base's exact tokens and renders on its own for the finding.
    rendered = second[second.index("--rendered") + 1]
    assert rendered == "{run_dir}/quality/tools-4096/base/rendered.jsonl"
    assert "--context" not in second
    assert cell.render.argv[1] == "render" and "--cache-dir" not in cell.render.argv
