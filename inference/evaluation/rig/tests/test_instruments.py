"""Selectors fail before anything queues; instruments plan what they say; `ab` alternates and
judges only beyond what both sides resolved."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest

from magnitude_rig import ab, cli, engine, selectors, store
from magnitude_rig.build import BuildIdentity
from magnitude_rig.host import memory
from magnitude_rig.instruments import REGISTRY, WORKSPACE, JobResult, Measure, device, harness
from magnitude_rig.instruments import step as step_instrument
from magnitude_rig.instruments.resolution import median_interval


def parse(*argv: str) -> argparse.Namespace:
    return cli.parser().parse_args(list(argv))


COMMON = ("--backend", "cuda", "--purpose", "p", "--estimate", "2m")
MODEL = "qwen3.5-4b:gguf:q4"  # a catalog model


def test_durations():
    assert [selectors.seconds(t) for t in ("45", "45s", "2m", "1h30m", "2m30s", "0.5s")] == [45, 45, 120, 5400, 150, 0.5]
    for bad in ("", "soon", "2x", "m"):
        with pytest.raises(argparse.ArgumentTypeError):
            selectors.seconds(bad)


def test_purpose_and_estimate_are_required_for_every_run(capsys):
    for name in REGISTRY:
        with pytest.raises(SystemExit):
            parse(name, "--backend", "cpu")
        assert "--purpose" in capsys.readouterr().err


def test_a_model_must_be_in_the_catalog():
    assert selectors.model(MODEL) == MODEL
    with pytest.raises(argparse.ArgumentTypeError, match="choose from"):
        selectors.model(f"{MODEL.split(':')[0]}:nope")


def test_test_builds_without_running_then_runs_its_executables_from_the_output_on_the_device():
    arguments = parse("test", *COMMON, "--", "-p", "magnitude-kernels", "--test", "routed", "--", "general")
    plan = REGISTRY["test"].plan(arguments, None, "alpha")
    (build,) = plan.builds  # it names its target, so no documentation tests
    assert build.argv == ("cargo", "test", "--message-format=short", "--no-run", "-p", "magnitude-kernels", "--test", "routed")
    ((_, job),) = plan.jobs
    assert job.argv == ("{python}", "-m", "magnitude_rig.host", "tests", "--list", "{output}/tests.json", "--", "general")
    # A cargo suite has the release catalog's headers, so no test of it lacks its inputs; the
    # host package is on PYTHONPATH from the job's `record`, not set again.
    assert plan.device == "device" and job.env == (("MAGNITUDE_CATALOG_HEADERS", "{catalog}/headers"),)
    # Each executable gets cargo test's name filter first, then what follows `--`, as cargo passes them.
    for given, passed in ((("-p", "x", "--test", "routed", "general", "--", "--exact"), ("general", "--exact")),
                          (("--lib", "general", "-p", "x"), ("general",)),
                          (("--features=a", "-F", "b", "--test", "--lib", "general"), ("general",))):
        ((_, filtered),) = REGISTRY["test"].plan(parse("test", *COMMON, "--", *given), None, "alpha").jobs
        assert filtered.argv[filtered.argv.index("--") + 1:] == passed
    assert not job.checkout
    # One that runs documentation tests (names no target, or `--doc`) runs cargo in its checkout,
    # which it keeps for the job.
    for given in (("-p", "x"), ("-p", "x", "--doc"), ("-p", "x", "--lib", "--doc")):
        ((_, whole),) = REGISTRY["test"].plan(parse("test", *COMMON, "--", *given, "--", "f"), None, "alpha").jobs
        assert whole.argv == ("cargo", "test", *given, "--", "f") and whole.checkout
    ((_, named),) = REGISTRY["test"].plan(parse("test", *COMMON, "--", "-p", "x", "--test=routed"), None, "alpha").jobs
    assert not named.checkout
    assert plan.acquires == engine.Headers(None)
    vulkan = parse("test", "--backend", "vulkan", "--purpose", "p", "--estimate", "1m", "--", "-p", "x")
    assert REGISTRY["test"].plan(vulkan, None, "alpha").jobs[0][1].env[:2] == (
        ("SEISMIC_TEST_BACKEND", "vulkan"), ("MAGNITUDE_CATALOG_HEADERS", "{catalog}/headers"))
    # One that reads a model's files has the model, not the catalog.
    model = REGISTRY["test"].plan(parse("test", *COMMON, "--model", MODEL, "--", "-p", "x"), None, "alpha")
    assert model.acquires == engine.Weights(MODEL)
    assert "MAGNITUDE_CATALOG_HEADERS" not in dict(model.jobs[0][1].env)
    timed = parse("test", *COMMON, "--timed", "--", "-p", "x")
    assert REGISTRY["test"].plan(timed, None, "alpha").device == "timed"


def test_python_tests_run_in_the_checkouts_environment_with_the_model_they_check():
    model = MODEL
    arguments = parse("test", *COMMON, "--suite", "python", "--model", model, "--",
                      "evaluation/references/tests/test_released.py", "-k", "qwen")
    plan = REGISTRY["test"].plan(arguments, None, "alpha")
    assert "--all-extras" in plan.builds[0].argv
    job = plan.jobs[0][1]
    assert job.argv[0] == "{venv}/bin/pytest" and job.argv[-3:] == ("evaluation/references/tests/test_released.py", "-k", "qwen")
    assert dict(job.env)["MAGNITUDE_ARTIFACTS"] == "{artifacts}" and plan.acquires == engine.Weights(model)


def test_every_environment_reinstalls_the_workspace_packages_from_the_tree_it_builds():
    """uv keys a local package's built wheel by its project file alone, so a non-editable sync
    reinstalls every workspace package, rebuilt from the tree, rather than a wheel another tree
    built; the packages are the workspace's members."""
    import tomllib
    workspace = Path(__file__).resolve().parents[2]
    members = tomllib.loads((workspace / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]
    assert set(WORKSPACE) == {tomllib.loads((workspace / member / "pyproject.toml").read_text())["project"]["name"]
                              for member in members}
    syncs = [REGISTRY["test"].plan(parse("test", *COMMON, "--suite", "python", "--", "x"), None, "alpha").builds[0],
             REGISTRY["kl"].plan(parse("kl", "--backend", "metal", "--model", MODEL, "--context", "4096", "--purpose", "p",
                                       "--estimate", "1m"), None, "alpha").builds[1]]
    for step in syncs:
        assert step.argv[:4] == ("uv", "sync", "--frozen", "--no-editable")
        assert {step.argv[at + 1] for at, argument in enumerate(step.argv) if argument == "--reinstall-package"} \
            == set(WORKSPACE)


def test_python_test_answer_counts_pytests_summary(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("FAILED references/tests/test_released.py::test_a[x] - AssertionError\n"
                   "1 failed, 12 passed, 3 skipped in 4.20s\n")
    arguments = parse("test", *COMMON, "--suite", "python", "--", "x")
    answer = REGISTRY["test"].answer(arguments, {"test": JobResult(tmp_path, log, 1)})
    assert answer.lines[0] == "FAIL 12 passed, 1 failed, 3 skipped (cuda)"
    assert answer.lines[1] == "failed: references/tests/test_released.py::test_a[x]"


def test_test_answer_counts_and_names_failures(tmp_path):
    log = tmp_path / "job.log"
    log.write_text(
        "running 3 tests\n---- routed::a stdout ----\n[stderr] thread 'routed::a' panicked at src/x.rs:1:1:\n"
        "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured\n"
    )
    answer = REGISTRY["test"].answer(parse("test", *COMMON, "--", "-p", "x"), {"test": JobResult(tmp_path, log, 101)})
    assert not answer.ok
    assert answer.lines[0] == "FAIL 2 passed, 1 failed, 0 ignored (cuda)"
    assert answer.lines[1] == "failed: routed::a"


def test_a_test_without_the_libtest_harness_reports_its_exit_and_its_own_output(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("[stderr]      Running tests/residency_lifecycle.rs\nG-D residency lifecycle: all scenarios passed\n")
    answer = REGISTRY["test"].answer(parse("test", *COMMON, "--", "-p", "x"), {"test": JobResult(tmp_path, log, 0)})
    assert answer.ok and answer.lines == ("PASS exit 0, no test counts (cuda)",
                                          "  G-D residency lifecycle: all scenarios passed")


def test_script_ships_the_file_and_runs_it_by_its_interpreter(tmp_path):
    script = tmp_path / "probe.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    arguments = parse("script", *COMMON, str(script), "--hold", "timed", "--prepare", "cargo build -p x")
    plan = REGISTRY["script"].plan(arguments, None, "alpha")
    assert plan.device == "timed" and plan.builds[0].argv == ("cargo", "build", "-p", "x")
    job = plan.jobs[0][1]
    assert job.argv[:2] == ("{python}", "-c") and dict(job.env)["MRIG_RUN"] == "{run_dir}"
    assert plan.files == (script,) and job.argv[-1] == "{inputs}/probe.sh"


def test_large_script_uses_staged_input_without_command_line_payload(tmp_path):
    script = tmp_path / "large.py"
    script.write_text(f"#!{sys.executable}\n#" + "x" * 262144 + "\nprint('staged script ran')\n")
    arguments = parse("script", *COMMON, str(script))
    plan = REGISTRY["script"].plan(arguments, None, "alpha")
    job = plan.jobs[0][1]
    run = tmp_path / "run"
    argv = engine._substitute(job.argv, {"python": sys.executable, "inputs": str(tmp_path)})
    assert max(map(len, argv)) < 4096
    result = subprocess.run(argv, env={"MRIG_RUN": str(run)}, text=True, capture_output=True, check=True)
    assert result.stdout == "staged script ran\n"
    assert (run / "script").read_bytes() == script.read_bytes()


def test_the_median_interval_is_distribution_free():
    middle, half = median_interval([float(v) for v in range(1, 101)])
    assert middle == 50.5 and 8 <= half <= 12
    assert median_interval([1.0, 3.0]) == (2.0, 1.0)


class _Measuring:
    """A stand-in instrument whose job results are measures written as JSON."""

    name = "fake"
    comparable = True
    pairing = None
    alternate = None

    @staticmethod
    def answer(arguments, results):
        from magnitude_rig.instruments import Answer

        values = json.loads((results["m"].directory / "m.json").read_text())
        return Answer(("",), True, {name: Measure(v, "ms", r, True) for name, (v, r) in values.items()})


def test_ab_alternates_rounds_base_first_then_candidate_first():
    plans = {side: type("P", (), {"jobs": (("m", engine.Step("m", ("x",))),)})() for side in ab.SIDES}
    arguments = argparse.Namespace(rounds=3)
    jobs = ab._jobs(_Measuring, arguments, plans)
    assert [job.name for job in jobs] == [
        "base-r0-m", "candidate-r0-m", "candidate-r1-m", "base-r1-m", "base-r2-m", "candidate-r2-m",
    ]


def _results(tmp_path: Path, values: dict[str, list[dict]]) -> dict[str, JobResult]:
    results = {}
    for side, rounds in values.items():
        for index, measures in enumerate(rounds):
            directory = tmp_path / f"{side}-r{index}-m"
            directory.mkdir()
            (directory / "m.json").write_text(json.dumps(measures))
            results[directory.name] = JobResult(directory, directory / "log", 0)
    return results


def test_ab_counts_a_change_only_beyond_both_sides_noise(tmp_path):
    arguments = argparse.Namespace(rounds=2, base="main", candidate="worktree")
    results = _results(tmp_path, {
        "base": [{"step": [10.0, 0.1], "small": [5.0, 0.2]}, {"step": [10.2, 0.1], "small": [5.1, 0.2]}],
        "candidate": [{"step": [9.0, 0.1], "small": [5.05, 0.2]}, {"step": [9.1, 0.1], "small": [5.0, 0.2]}],
    })
    answer = ab.compare(_Measuring, arguments, results)
    rows = {line.split()[0]: line for line in answer.lines[1:]}
    assert rows["step"].endswith("better") and rows["small"].endswith("below resolution")
    assert answer.ok


def test_step_reports_a_memory_admission_refusal_with_its_reading(tmp_path):
    record = {"moment": "admission refused", "store": "host RAM", "constraint": "host RAM",
              "headroom_bytes": 24_000_000_000, "planning_bytes": 5_153_960_755,
              "distress": "the system is swapping", "wired_bytes": 26_600_000_000,
              "compressed_bytes": 212_000_000,
              "windows": [{"ms": 1000, "compressed_pages": 0, "decompressed_pages": 3, "swapped_out_pages": 120,
                           "swapped_in_pages": 0},
                          {"ms": 1001, "compressed_pages": 0, "decompressed_pages": 0, "swapped_out_pages": 0,
                           "swapped_in_pages": 0}]}
    log = tmp_path / "log"
    log.write_text("[stderr] magnitude-harness: prefill context=65536 concurrency=1\n"
                   f"[stderr] magnitude-harness: memory {json.dumps(record)}\n"
                   "[stderr] magnitude-harness: the system is swapping (...)\n")
    arguments = argparse.Namespace(context=65536, sequences=1, weight_cap=None, untimed=False, resident=None, streamed_only=False, tuning_defaults=False, tuning_record=None, tuning_replay=None)
    answer = REGISTRY["step"].answer(arguments, {"step": JobResult(tmp_path, log, 1)})
    assert not answer.ok
    assert answer.lines[0] == "refused by memory admission (host RAM): the system is swapping"
    assert answer.lines[1].startswith("  headroom 24.00 GB · planning reserve 5.15 GB · wired 26.60 GB")
    assert answer.lines[2].startswith("  2 displacement windows over 2.0 s · swapped out 120 pages in 1 of them")


def _step_record(directory: Path, decode_detail: dict, decode_extra: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps({"measurements": {"speed": [
        {"phase": "prefill", "detail": {"rows": 4096, "samples_ms": [600.0, 610.0, 590.0]}},
        {"phase": "decode", "detail": decode_detail, **(decode_extra or {})}]}}))


def test_step_answers_a_record_whose_harness_lacks_optional_sections(tmp_path):
    """Each side runs its own harness: a build whose record lacks a section's fields (an older
    model, attribution without shares, no cold prefill) still answers, saying what it lacks."""
    steps = [{"interval_ms": 15.0} for _ in range(8)]
    resident = {"steps": steps, "window": {"wall_ms_per_step": 15.0}, "selected_sha256": "a",
                "attribution": {"entries": [{"entry": "e", "device_ms_per_step": 1.0, "launches_per_step": 2.0}]}}
    _step_record(tmp_path / "step", resident)
    streamed_detail = {"steps": steps, "window": {"wall_ms_per_step": 16.0}}
    _step_record(tmp_path / "streamed-7", streamed_detail, {"model": {"predicted_tok_s": 60.0, "all_hit": {}}})
    arguments = argparse.Namespace(context=4096, sequences=1, weight_cap=[7], untimed=False, resident=None, streamed_only=False, tuning_defaults=False, tuning_record=None, tuning_replay=None)
    log = tmp_path / "log"
    results = {name: JobResult(tmp_path / name, log, 0) for name in ("step", "streamed-7")}
    results["measure"] = JobResult(tmp_path, log, 0)
    answer = REGISTRY["step"].answer(arguments, results)
    assert answer.ok
    assert answer.lines[1].endswith("tok/s"), "no cold prefill recorded, none shown"
    assert answer.lines[2] == "  graphs not recorded by this build's harness (no window.graphs)"
    assert answer.lines[3] == "  attribution not recorded by this build's harness (no share)"
    streamed = next(line for line in answer.lines if line.startswith("streamed 7"))
    assert streamed.endswith("tokens not recorded") and "0.938× resident" in streamed
    assert any(line.startswith("  model not recorded by this build's harness (no miss_shares.staged") for line in answer.lines)
    assert "streamed-7 mean_step_ms" in answer.measures and "entry e" not in answer.measures


def test_a_step_cell_whose_tuning_stopped_at_its_budget_is_void(tmp_path):
    _timed_result(tmp_path / "step", 13.0, 12.5, "aa")
    (tmp_path / "step" / "tuning.jsonl").write_text(json.dumps(
        {"event": "budget", "budget_seconds": 60.0, "used_seconds": 57.4, "refused": [{"unit": "u", "class": "measure", "reason": "seconds"}, {"unit": "v", "class": "form", "reason": "bytes"}]}) + "\n")
    arguments = argparse.Namespace(context=4096, sequences=1, weight_cap=None, untimed=False, resident=None, streamed_only=False, tuning_defaults=False, tuning_record=None, tuning_replay=None)
    answer = REGISTRY["step"].answer(arguments, {"step": JobResult(tmp_path / "step", tmp_path / "log", 0)})
    assert not answer.ok
    assert answer.lines[-1] == (
        "  resident tuning incomplete: 2 pieces of work refused (57 of 60 s): "
        "u (measure, seconds); v (form, bytes) · VOID"
    )


def test_a_pinned_cell_is_never_void_and_reports_what_the_pin_lacked(tmp_path):
    _timed_result(tmp_path / "step", 13.0, 12.5, "aa")
    (tmp_path / "step" / "tuning.jsonl").write_text("\n".join(json.dumps(event) for event in (
        {"event": "budget", "budget_seconds": 60.0, "used_seconds": 57.4, "refused": [{"unit": "u", "class": "measure", "reason": "seconds"}]},
        {"event": "pinned", "entry": "a", "choice": "replayed"},
        {"event": "pinned", "entry": "b", "choice": "not_replayed"},
    )) + "\n")
    base = dict(context=4096, sequences=1, weight_cap=None, untimed=False, resident=None, streamed_only=False,
                tuning_defaults=False, tuning_record=None)
    answer = REGISTRY["step"].answer(argparse.Namespace(**base, tuning_replay=Path("metal.json")),
                                     {"step": JobResult(tmp_path / "step", tmp_path / "log", 0)})
    assert answer.ok
    assert answer.lines[-1] == "  resident tuning pinned: 1 units held, 1 not in the pin ran at default: b"


def test_a_recording_keeps_its_configurations_on_this_machine(tmp_path):
    _timed_result(tmp_path / "step", 13.0, 12.5, "aa")
    (tmp_path / "tuning-pins.json").write_text("[]")
    kept = tmp_path / "kept"
    arguments = argparse.Namespace(context=4096, sequences=1, weight_cap=None, untimed=False, resident=None,
                                   streamed_only=False, tuning_defaults=False, tuning_record=kept, tuning_replay=None)
    REGISTRY["step"].answer(arguments, {"step": JobResult(tmp_path / "step", tmp_path / "log", 0)})
    assert kept.read_text() == "[]"


def test_step_names_the_measurement_a_builds_record_lacks(tmp_path):
    _step_record(tmp_path, {"median_step": {}})
    arguments = argparse.Namespace(context=4096, sequences=1, weight_cap=None, untimed=False, resident=None, streamed_only=False, tuning_defaults=False, tuning_record=None, tuning_replay=None)
    answer = REGISTRY["step"].answer(arguments, {"step": JobResult(tmp_path, tmp_path / "log", 0)})
    assert not answer.ok and answer.lines == ("this build's harness records no detail.steps",)
    # A step timed by its own wall time, not from the completion before it, is no interval.
    _step_record(tmp_path, {"steps": [{"interval_ms": 15.0}, {"wall_ms": 14.0}]})
    answer = REGISTRY["step"].answer(arguments, {"step": JobResult(tmp_path, tmp_path / "log", 0)})
    assert not answer.ok and answer.lines == ("this build's harness records no decode step interval_ms",)


def test_step_answers_the_median_step_its_interval_launches_and_entries(tmp_path):
    steps = [{"interval_ms": 15.0 + 0.01 * (index % 5)} for index in range(64)]
    entries = [
        {"entry": "dense_expand#1", "launches_per_step": 22.3, "device_ms_per_step": 3.2, "share": 0.3},
        {"entry": "attention_decode_k8v4", "launches_per_step": 5.3, "device_ms_per_step": 0.2, "share": 0.02},
    ]
    decode = {"phase": "decode", "tokens_per_second": 66.0, "samples": [],
              "detail": {"steps": steps, "median_step": {},
                         "attribution": {"steps": 3, "entries": entries}}}
    (tmp_path / "result.json").write_text(json.dumps({"measurements": {"speed": [
        {"phase": "prefill", "detail": {"rows": 4096, "cold_ms": 900.0, "samples_ms": [600.0, 610.0, 590.0]}},
        decode]}}))
    arguments = argparse.Namespace(context=4096, sequences=1, weight_cap=None, untimed=False, resident=None, streamed_only=False, tuning_defaults=False, tuning_record=None, tuning_replay=None)
    answer = REGISTRY["step"].answer(arguments, {"step": JobResult(tmp_path, tmp_path / "log", 0)})
    assert answer.lines[0].startswith("step @4096 · 15.020 ms") and answer.lines[0].endswith("28 launches")
    assert answer.lines[1].startswith("prefill @4096 · 600.0 ms (±10.0") and "6827 tok/s" in answer.lines[1]
    assert "dense_expand#1" in answer.lines[4]
    assert answer.measures["prefill_ms"].value == 600.0
    assert answer.measures["step_ms"].value == 15.02 and answer.measures["step_ms"].lower_is_better


def test_kernel_shows_configurations_as_changes_from_the_default_and_the_search(tmp_path):
    def configuration(**params):
        return {"statics": {}, "params": params}

    def measured(*times):
        return {"Measured": {"points": [{"point": f"p{i}", "median_seconds": t, "deviation_seconds": 1e-7}
                                        for i, t in enumerate(times)], "confirmed": []}}

    default, fast = configuration(A=1, B=2), configuration(A=1, B=4)
    record = {"points": [{"label": "p0"}, {"label": "p1"}], "served": [default], "overall": fast,
              "configurations": [{"configuration": default, "outcome": measured(20e-6, 30e-6)},
                                 {"configuration": fast, "outcome": measured(10e-6, 40e-6)}]}
    (tmp_path / "survey").mkdir()
    (tmp_path / "survey" / "e.json").write_text(json.dumps({"entry": "e", "bindings": "A=bf16", "result": record}))
    (tmp_path / "survey" / "replay.md").write_text(
        "### e [A=bf16] {}\n\n2 admissible, 2 measured, allowance 1.00 s (4 evaluations); true best [1]\n\n"
        "| objective | evaluations | within 1% | within 2% | within 5% | median excess | p95 excess | evaluated | n95 (2%) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| keyed (production) | 2 | 90.0% | 95.0% | 100.0% | 0.0% | 3.0% | 2.0 | 2 |\n"
        "| separate (previous) | 2 | 80.0% | 85.0% | 100.0% | 0.0% | 4.0% | 2.0 | 2 |\n\n"
        "Weighted per-point gap: 12.5%\n"
    )
    log = tmp_path / "log"
    log.write_text("")
    answer = REGISTRY["kernel"].answer(argparse.Namespace(entry="e"), {"kernel": JobResult(tmp_path, log, 0)})
    assert answer.lines[2] == "  tuner's choice: B=4"
    assert answer.lines[4].split()[-1] == "B=4" and answer.lines[5].split()[-1] == "default"
    assert answer.lines[6] == "  search on e [A=bf16] {}"
    assert answer.lines[7].split()[:7] == ["2", "evaluations:", "within", "1%", "90.0%,", "2%", "95.0%,"]
    assert answer.lines[8] == "    one configuration for all points: per-point gap: 12.5%"
    assert answer.measures["e A=bf16 p0"].value == pytest.approx(10.0)


def test_assess_breakdown_sums_each_entrys_launches_and_time(tmp_path):
    log = tmp_path / "log"
    log.write_text("\n".join([
        '{"model": "m"}',
        "[stderr] magnitude-assess:   at 25000: 900.0 µs per step = 1.0 × (...)",
        "[stderr] magnitude-assess:     dense_expand        10.00 µs bytes     20.00 µs compute    1.00 µs fixed 2 launches",
        "[stderr] magnitude-assess:     dense_expand        30.00 µs bytes      5.00 µs compute    0.50 µs fixed 1 launches",
        "[stderr] magnitude-assess:     attention_decode     4.00 µs bytes      1.00 µs compute    2.00 µs fixed 1 launches",
    ]) + "\n")
    arguments = argparse.Namespace(breakdown=True, catalog=False)
    answer = REGISTRY["assess"].answer(arguments, {"assess": JobResult(tmp_path, log, 0)})
    assert answer.lines[0].startswith("{")
    at = next(index for index, line in enumerate(answer.lines) if line.startswith("step at depth 25000:"))
    rows = {line.split()[0]: line.split()[1:] for line in answer.lines[at + 2:]}
    assert rows["dense_expand"] == ["3", "51.5"] and rows["attention_decode"] == ["1", "6.0"]


def test_ab_step_runs_every_round_on_the_bases_first_tuning():
    arguments = parse("ab", "step", "--backend", "vulkan", "--model", MODEL, "--context", "4096",
                      "--base", "main", "--rounds", "2", "--purpose", "p", "--estimate", "5m")
    plans = {side: REGISTRY["step"].plan(arguments, None, "alpha") for side in ab.SIDES}
    jobs = ab._jobs(REGISTRY["step"], arguments, plans)
    pinning = [[a for a in job.step.argv if a.startswith("--tuning-")] for job in jobs]
    assert jobs[0].name == "base-r0-step"
    assert pinning == [["--tuning-record"], ["--tuning-replay"], ["--tuning-replay"], ["--tuning-replay"]]
    argv = jobs[0].step.argv
    assert argv[argv.index("--device") + 1] == "vulkan"  # the harness runs on the run's backend
    own = argparse.Namespace(**{**vars(arguments), "tuning": "own"})
    assert all(not [a for a in job.step.argv if a.startswith("--tuning-")] for job in ab._jobs(REGISTRY["step"], own, plans))


def test_every_checkout_on_a_host_shares_its_kernel_cache():
    from types import SimpleNamespace
    from magnitude_rig.connection import Remote
    remote = Remote("h", False, False, "Linux", "aarch64", "python3", "/r", "/m", "/r/agent/x.zip",
                    {}, ("cpu",), True)
    plan = engine.Plan(None, engine.Work(sides=()), remote, "run", {}, ())
    sides = [SimpleNamespace(label=label) for label in ("base", "candidate")]
    tokens = [plan.tokens(side, None) for side in sides]
    assert tokens[0]["checkout"] != tokens[1]["checkout"]
    assert tokens[0]["engine_cache"] == tokens[1]["engine_cache"] == "/r/cache/engine"
    arguments = parse("step", "--backend", "metal", "--model", MODEL, "--context", "4096", "--purpose", "p",
                      "--estimate", "5m")
    argv = dict(REGISTRY["step"].plan(arguments, None, "h").jobs)["step"].argv
    assert argv[argv.index("--cache-dir") + 1] == "{engine_cache}/kernels"


def test_a_streamed_step_measures_then_times_resident_then_each_cap_from_storage():
    arguments = parse("step", "--backend", "metal", "--model", MODEL, "--context", "4096", "--purpose", "p",
                      "--estimate", "5m", "--weight-cap", "13041030544", "--weight-cap", "21775704464")
    plan = REGISTRY["step"].plan(arguments, None, "alpha")
    assert [name for name, _ in plan.jobs] == ["measure", "step", "streamed-13041030544", "streamed-21775704464"]
    measure, resident, *streamed = (step for _, step in plan.jobs)
    # The product's measurement job for its own time: no time of the rig's.
    assert "--seconds" not in measure.argv and not measure.fresh
    assert not resident.fresh and "--weight-cap" not in resident.argv
    for step, cap in zip(streamed, arguments.weight_cap):
        assert step.fresh and step.argv[step.argv.index("--weight-cap") + 1] == str(cap)
        assert step.argv[step.argv.index("--resident-decode") + 1] == "{runs}/step/result.json"
    # Under ab, each round's streamed steps are predicted from that round's resident step.
    jobs = REGISTRY["step"].alternate(arguments, "candidate", 1)
    step = dict(jobs)["streamed-13041030544"]
    assert step.argv[step.argv.index("--resident-decode") + 1] == "{runs}/candidate-r1-step/result.json"


def test_a_streamed_step_takes_its_resident_cell_only_from_a_record_of_the_same_cell(tmp_path, monkeypatch):
    """`--resident RUN` serves the streamed cells from RUN's recorded resident step, untimed again,
    on the host, model, backend and build it was measured at; any other refuses, naming why."""
    monkeypatch.setattr(store, "results", lambda: tmp_path / "results")
    fixture = tmp_path / "prose-repeat.jsonl"
    fixture.write_text("{}\n")
    build = BuildIdentity("c" * 40, None)
    run = tmp_path / "results" / "earlier"
    _timed_result(run / "step", 13.0, 12.5, "aa")
    record = json.loads((run / "step" / "result.json").read_text())
    record.update(host={"alias": "alpha"}, model=MODEL, build=build.to_json(),
                  fixture_set={"digest": hashlib.sha256(fixture.read_bytes()).hexdigest()},
                  command=["magnitude-harness", "time", "--device", "cuda", "--contexts", "4096",
                           "--concurrency", "1", "--lookahead", "on"])
    (run / "step" / "result.json").write_text(json.dumps(record))
    (run / "rig.json").write_text(json.dumps({"codes": {"step": 0}}))
    arguments = parse("step", *COMMON, "--model", MODEL, "--context", "4096", "--weight-cap", "7", "--resident", "earlier")
    arguments.fixture = fixture
    source = SimpleNamespace(identity=build)
    plan = REGISTRY["step"].plan(arguments, source, "alpha")
    assert [name for name, _ in plan.jobs] == ["measure", "streamed-7"]
    streamed = dict(plan.jobs)["streamed-7"]
    assert streamed.argv[streamed.argv.index("--resident-decode") + 1] == "{inputs}/result.json"
    assert plan.files == (fixture, run / "step" / "result.json")
    _timed_result(tmp_path / "cap", 20.0, 25.0, "aa", {"unavailable": "a test"})
    answer = REGISTRY["step"].answer(arguments, {"measure": JobResult(tmp_path, tmp_path / "log", 0),
                                                 "streamed-7": JobResult(tmp_path / "cap", tmp_path / "log", 0)})
    assert answer.lines[0] == "resident step recorded by run earlier, not timed again"
    assert answer.lines[1].startswith("step @4096 · 13.000 ms") and answer.measures["step_ms"].value == 13.0
    assert any(line.startswith("streamed 7 · mean 25.000 ms · 40.0 tok/s · 0.500× resident") for line in answer.lines)
    with pytest.raises(engine.Refused, match="another build"):
        REGISTRY["step"].plan(arguments, SimpleNamespace(identity=BuildIdentity("d" * 40, None)), "alpha")
    with pytest.raises(engine.Refused, match="host 'alpha', not 'beta'"):
        REGISTRY["step"].plan(arguments, source, "beta")
    metal = argparse.Namespace(**{**vars(arguments), "backend": "metal"})
    with pytest.raises(engine.Refused, match="backend 'cuda', not 'metal'"):
        REGISTRY["step"].plan(metal, source, "alpha")


def _timed_result(directory: Path, step_ms: float, mean_ms: float, selected: str,
                  model: dict | None = None, cached: tuple[int, int] = (0, 0)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    decode = {"phase": "decode", "detail": {"steps": [{"interval_ms": step_ms}] * 8, "selected_sha256": selected,
                                            "window": {"wall_ms_per_step": mean_ms}}}
    if model is not None:
        decode["model"] = model
        before, after = cached
        decode["detail"]["expert_pages"] = {"pages": 100, "cached_before": before, "cached_after": after}
        decode["cache_served"] = before + after > 0
    (directory / "result.json").write_text(json.dumps({"measurements": {"speed": [
        {"phase": "prefill", "detail": {"rows": 4096, "cold_ms": 900.0, "samples_ms": [600.0] * 3}}, decode]}}))
    # Its load's tuning, searched to completion.
    (directory / "tuning.jsonl").write_text(json.dumps(
        {"event": "budget", "budget_seconds": 60.0, "used_seconds": 40.0, "refused": []}) + "\n")


def test_a_streamed_step_answers_against_resident_and_voids_a_cache_served_one(tmp_path):
    # Two layers of one record form: one tier line (read in place, and beside a prompt group),
    # one file landing line and one CPU site line, each printed once.
    tier = {"served": {"fixed_seconds": 0.0, "record_seconds": 76e-6},
            "shared": {"fixed_seconds": 0.0, "record_seconds": 150e-6}}
    file = {"landing": {"fixed_seconds": 120e-6, "record_seconds": 80e-6}}
    site = {"device_record_seconds": 60e-6, "memory_record_seconds": 30e-6, "features": 512,
            "share": {"Features": {"extent": 384, "request": {"fixed_seconds": 90e-6, "record_seconds": 40e-6}}}}
    figures = {"tier": [tier, tier], "file": [file, file], "site": [site, None]}
    model = {"predicted_tok_s": 52.0, "ratio": 0.98, "misses_per_step": 12.0, "all_hit_steps": 2,
             "window_steps": 7, "figures": {**figures, "overhead_ms": 0.5},
             "miss_shares": {"staged": 0.5, "reading": 0.25, "unanticipated": 0.25}}
    _timed_result(tmp_path / "step", 13.0, 12.5, "aa")
    _timed_result(tmp_path / "cap", 20.0, 25.0, "aa", model)
    log = tmp_path / "cap.log"
    event = lambda fields: "[stderr] magnitude-harness: event " + json.dumps(fields)
    log.write_text("\n".join([
        event({"message": "expert bank growth decision", "holding": "Host", "outcome": "begun"}),
        event({"message": "memory grown", "holding": "Host", "from": 0, "to": 42, "grown": 2860515328,
               "micros": 1250000}),
        event({"message": "expert bank growth decision", "holding": "Host", "outcome": "not paying"}),
        event({"message": "expert landings 9: read 1.0 ms (mean 0.111)"}),
    ]) + "\n")
    measure = tmp_path / "measure.log"
    measure.write_text("[stderr] magnitude-assess: not measured: the reads of v: expired\n")
    arguments = argparse.Namespace(context=4096, sequences=1, weight_cap=[7], untimed=False, resident=None, streamed_only=False, tuning_defaults=False, tuning_record=None, tuning_replay=None)
    results = {"step": JobResult(tmp_path / "step", tmp_path / "log", 0),
               "measure": JobResult(tmp_path, measure, 0),
               "streamed-7": JobResult(tmp_path / "cap", log, 0)}
    answer = REGISTRY["step"].answer(arguments, results)
    assert answer.lines[3] == "magnitude-assess: not measured: the reads of v: expired"
    # Throughput is the window's mean step; the median is beside it.
    assert answer.lines[4].startswith("streamed 7 · mean 25.000 ms · 40.0 tok/s · 0.500× resident")
    assert "median 20.000 ms" in answer.lines[4] and answer.lines[4].endswith("same tokens")
    assert "measured/model 0.980" in answer.lines[5] and answer.lines[5].endswith("all-hit 1.040× resident")
    # The lines its misses are served by, each distinct one once.
    assert answer.lines[6] == "  tier read in place 0.0 µs + 76.0 µs/record · beside a prompt group 0.0 µs + 150.0 µs/record"
    assert answer.lines[7] == "  file landings past their read 120.0 µs + 80.0 µs/record"
    assert answer.lines[8] == ("  CPU site: device 60.0 µs/record, memory 30.0 µs/record · split 384 of 512 features, "
                               "its CPU request 90.0 µs + 40.0 µs/record")
    # What its bank did: each decision's outcome counted, each resize, and the landings' account.
    assert answer.lines[9] == "  growth decisions Host begun ×1, Host not paying ×1"
    assert answer.lines[10] == "  Host grown 0 → 42 (2.86 GB) in 1.25 s"
    assert answer.lines[11] == "  landings 9: read 1.0 ms (mean 0.111)"
    assert answer.ok and answer.measures["streamed-7 mean_step_ms"].value == 25.0
    assert "  expert pages cached 0 before, 0 after, of 100" in answer.lines
    _timed_result(tmp_path / "cap", 20.0, 25.0, "bb", model, cached=(0, 3))
    answer = REGISTRY["step"].answer(arguments, results)
    assert not answer.ok and any(line.endswith("VOID: its reads were served from the file cache")
                                 for line in answer.lines)
    # Without an all-hit step σ is unmeasured: the model's step on C alone. The estimate is
    # assessment's.
    _timed_result(tmp_path / "cap", 20.0, 25.0, "aa", {**model, "predicted_tok_s": 41.0, "ratio": 0.976,
                                                        "all_hit_steps": 0,
                                                        "figures": {**figures, "tier": [None, None],
                                                                    "overhead_ms": None}})
    record = json.loads((tmp_path / "cap" / "result.json").read_text())
    record["measurements"]["speed"][1]["estimate"] = {"slots": 128, "tok_s": 42.0, "ratio": 0.952,
                                                      "record": {"decode_steps": 44}}
    (tmp_path / "cap" / "result.json").write_text(json.dumps(record))
    answer = REGISTRY["step"].answer(arguments, results)
    assert answer.lines[5].startswith("  model 41.0 tok/s · measured/model 0.976 · 12.0 misses/step")
    assert answer.lines[5].endswith("0/7 all-hit steps, no all-hit step, σ unmeasured: the model's step on C alone")
    assert answer.lines[6] == "  tier none: the bank has no host tier"
    assert answer.lines[12] == "  bar not recorded by this build's harness"
    assert answer.lines[13] == ("  estimate 42.0 tok/s at 128 slots · measured/estimate 0.952 "
                               "(from a record of 44 decode steps)")
    # The bar: the share resident, the measured step, the ceiling and the frontier × resident; the
    # three miss counts; the gaps by term and the routers.
    record["measurements"]["speed"][1]["bar"] = {
        "form": "discrete",
        "step_ms": {"measured": 25.0, "model": 24.4, "bar": 10.0},
        "tok_s": {"measured": 40.0, "model": 41.0, "bar": 100.0},
        "bar_ms": {"resident": 9.0, "link": 0.0, "cpu": 0.5, "memory": 0.25, "file": 0.0, "ahead": 0.0,
                   "handshake": 0.25},
        "window": {"capacity_per_step": 23.3, "compulsory_per_step": 1.0},
        "frontier_ms": 12.0,
        "misses_per_step": {"belady": 24.3, "ranking": 42.0, "live": 75.8, "design": 17.7, "implementation": 33.8},
        "gaps": {"implementation_ms": 0.6, "design_ms": {"route": None, "exposed": 14.4, "total": 14.4}},
        "router": {"absent": "the resident cell's trace has no per-entry times (--attribution-steps 0)"},
    }
    record["measurements"]["speed"][1]["detail"]["expert_bank"] = {"slots": 64, "experts": 256}
    (tmp_path / "cap" / "result.json").write_text(json.dumps(record))
    answer = REGISTRY["step"].answer(arguments, results)
    assert list(answer.lines[12:16]) == [
        ("  discrete: 25.0% of experts resident · measured 0.360× resident (25.000 ms) · ceiling 0.900× resident "
         "(10.000 ms) · frontier 0.750× resident (12.000 ms)"),
        ("  window misses/step: Bélády 24.3 (compulsory 1.0) · ranking 42.0 · live 75.8 "
         "(design +17.7, implementation +33.8)"),
        ("  gap ms/step: implementation (measured − model) +0.600 · design (model − bar) +14.400: "
         "route unmeasured, exposed +14.400"),
        "  router per-entry times absent: the resident cell's trace has no per-entry times (--attribution-steps 0)",
    ]
    # A unified cell adds its ceiling's terms and gap split, keyed as the harness's bar keys them.
    record["measurements"]["speed"][1]["bar"]["unified"] = {
        "overlap": 0.982, "ceiling_ms": 10.0, "schedule_ms": 9.5, "sustained_ms": 9.8,
        "gap_split_ms": {"ranking_and_schedule": 4.0, "misses_implementation": 1.0, "read_path_below_raw": 0.5,
                         "exposure": 9.5},
    }
    (tmp_path / "cap" / "result.json").write_text(json.dumps(record))
    answer = REGISTRY["step"].answer(arguments, results)
    assert answer.lines[14] == ("  unified: η 0.982 · schedule 9.500 ms · sustained 9.800 ms · gap ms/step: ranking and schedule "
                                "+4.000, misses implementation +1.000, read path below raw +0.500, exposure +9.500")
    del record["measurements"]["speed"][1]["bar"]["unified"]
    # Another build's window (its compulsory share unrecorded) is named, not read.
    record["measurements"]["speed"][1]["bar"]["window"] = {"capacity_per_step": 23.3}
    (tmp_path / "cap" / "result.json").write_text(json.dumps(record))
    answer = REGISTRY["step"].answer(arguments, results)
    assert answer.lines[12] == "  bar not recorded by this build's harness (no window.compulsory_per_step)"


def test_a_streamed_only_step_times_no_resident_cell_and_answers_absolute_terms_alone(tmp_path, monkeypatch):
    """`--streamed-only` plans no resident step and prices no streamed one against one; its answer
    is the cells' absolute terms (steps, device busy and idle, boundaries' held and queued time,
    the estimate), never a ratio to resident; it takes no recorded resident cell either."""
    common = ("step", "--backend", "vulkan", "--model", MODEL, "--context", "4096", "--purpose", "p",
              "--estimate", "5m", "--weight-cap", "7", "--streamed-only")
    plan = REGISTRY["step"].plan(parse(*common), None, "alpha")
    assert [name for name, _ in plan.jobs] == ["measure", "streamed-7"]
    streamed = dict(plan.jobs)["streamed-7"]
    assert streamed.fresh and "--resident-decode" not in streamed.argv
    with pytest.raises(SystemExit, match="give --weight-cap"):
        REGISTRY["step"].plan(parse(*common[:-3], "--streamed-only"), None, "alpha")
    with pytest.raises(SystemExit, match="drop --resident"):
        REGISTRY["step"].plan(parse(*common, "--resident", "20261101T000000Z-step-000001"), None, "alpha")

    # A record a resident run would price: a model and a bar the streamed-only answer never reads.
    bar = {"form": "discrete", "step_ms": {"measured": 25.0, "model": 24.4, "bar": 10.0}, "bar_ms": {"resident": 9.0}}
    model = {"predicted_tok_s": 52.0, "ratio": 0.98, "misses_per_step": 12.0, "all_hit_steps": 2, "window_steps": 7,
             "figures": {"overhead_ms": 0.5}, "miss_shares": {"staged": 0.5, "reading": 0.25, "unanticipated": 0.25}}
    _timed_result(tmp_path / "cap", 20.0, 25.0, "aa", model)
    record = json.loads((tmp_path / "cap" / "result.json").read_text())
    decode = record["measurements"]["speed"][1]
    decode["detail"]["window"].update({
        "device_busy_ms_per_step": 18.5, "device_idle_ms_per_step": 6.5,
        "idle": {"between_runs": {"ms_per_step": 4.5, "host_ms_per_step": 4.0, "queued_ms_per_step": 0.5,
                                  "by_submission": []},
                 "at_boundaries": {"per_step": 4.0, "ms_per_step": 2.0, "held_ms_per_step": 0.25,
                                   "awaiting_data_ms_per_step": 1.0,
                                   "awaiting_data_by_park_ms_per_step": {"tier wait": 0.75, "sited combine": 0.25},
                                   "answer_latency_ms_per_step": 0.5,
                                   "awaiting_signal_ms_per_step": 0.0, "starting_ms_per_step": 0.25,
                                   "submitting_ms_per_step": 0.125}}})
    decode.update(bar=bar, estimate={"slots": 128, "tok_s": 42.0, "ratio": 0.952, "record": {"decode_steps": 44}})
    (tmp_path / "cap" / "result.json").write_text(json.dumps(record))
    measure = tmp_path / "measure.log"
    measure.write_text("")
    arguments = parse(*common)
    results = {"measure": JobResult(tmp_path, measure, 0), "streamed-7": JobResult(tmp_path / "cap", tmp_path / "log", 0)}
    answer = REGISTRY["step"].answer(arguments, results)
    assert answer.ok and answer.lines[0] == "streamed only: no resident cell timed; absolute streamed terms"
    assert answer.lines[1].startswith("streamed 7 · mean 25.000 ms · 40.0 tok/s · median 20.000 ms")
    assert answer.lines[1].endswith("tokens not compared (no resident cell)")
    assert answer.lines[2] == ("  device busy 18.500 ms, idle 6.500 ms a step: between runs 4.500 "
                               "(host 4.000, queued 0.500), at boundaries 2.000")
    assert answer.lines[3] == ("  idle at 4.0 boundaries a step: held 0.250, awaiting data 1.000 (tier wait 0.750, "
                               "sited combine 0.250), answer latency "
                               "0.500, awaiting signal 0.000, starting 0.250 (submitting 0.125) ms")
    assert answer.lines[4] == "  model not priced: streamed only, no resident cell"
    assert any(line.startswith("  estimate 42.0 tok/s at 128 slots") for line in answer.lines)
    assert not any("resident" in line and "×" in line for line in answer.lines)
    assert set(answer.measures) == {"streamed-7 mean_step_ms", "streamed-7 step_ms", "streamed-7 prefill_ms"}

    # A ratio to a resident cell in its answer is an invariant violated, never shown.
    monkeypatch.setattr(step_instrument, "_bank", lambda result: ["  0.500× resident"])
    with pytest.raises(AssertionError, match="never timed"):
        REGISTRY["step"].answer(arguments, results)


def test_kl_with_weight_caps_passes_equal_logits_and_says_where_others_part(tmp_path):
    arguments = parse("kl", "--backend", "metal", "--model", MODEL, "--context", "512", "--purpose", "p",
                      "--estimate", "5m", "--weight-cap", "7", "--weight-cap", "9")
    plan = REGISTRY["kl"].plan(arguments, None, "alpha")
    assert [name for name, _ in plan.jobs] == ["logits", "streamed-7", "streamed-9", "compare-streamed-7",
                                               "compare-streamed-9"]
    assert "magnitude-references" not in " ".join(" ".join(step.argv) for _, step in plan.jobs), "no reference is read"
    jobs = dict(plan.jobs)
    resident = jobs["logits"]
    assert "--tuning-defaults" in resident.argv and not resident.fresh and "--weight-cap" not in resident.argv
    for cap in (7, 9):
        step = jobs[f"streamed-{cap}"]
        assert step.fresh and "--tuning-defaults" in step.argv
        assert step.argv[step.argv.index("--rendered") + 1] == "{runs}/logits/rendered.jsonl"
    # On the host: each comparison writes where the logits part, removes the candidate's logits,
    # and keeps the base's until the last comparison has read them.
    rows = [{"fixture_id": "f", "position": position} for position in range(3)]
    loads = (("logits", [[0, 1], [2, 3], [4, 5]], None),
             ("streamed-7", [[0, 1], [2, 3], [4, 5]], {"least_layer_misses": 2, "misses": 9, "sited": 0}),
             ("streamed-9", [[0, 1], [2, 3.5], [4, 5]], {"least_layer_misses": 1, "misses": 5, "sited": 0}))
    for name, values, experts in loads:
        (tmp_path / name).mkdir()
        numpy.array(values, dtype="<f4").tofile(tmp_path / name / "logits.f32")
        (tmp_path / name / "logits.json").write_text(json.dumps({"vocabulary": 2, "rows": rows, "experts": experts}))
    for cap in (7, 9):
        name = f"compare-streamed-{cap}"
        (tmp_path / name).mkdir()
        tokens = {"{venv}/bin/python": sys.executable, "{run_dir}": str(tmp_path / name), "{runs}": str(tmp_path)}
        argv = [next((part.replace(k, v) for k, v in tokens.items() if k in part), part) for part in jobs[name].argv]
        subprocess.run(argv, check=True)
        assert not (tmp_path / f"streamed-{cap}" / "logits.f32").exists()
        assert (tmp_path / "logits" / "logits.f32").exists() == (cap == 7)
    results = {name: JobResult(tmp_path / name, tmp_path / "log", 0)
               for name in ("compare-streamed-7", "compare-streamed-9")}
    assert json.loads((tmp_path / "compare-streamed-7" / "compare.json").read_text())["base_experts"] is None
    answer = REGISTRY["kl"].answer(arguments, results)
    assert answer.lines[0].endswith("against uncapped, which held its experts whole")
    assert answer.lines[1] == "streamed 7: equal logits (3 rows); fewest misses at a streamed layer 2, 0 sited"
    # The differing row's KL against its base: softmax([2, 3]) against softmax([2, 3.5]).
    p, q = (numpy.exp(row) / numpy.exp(row).sum() for row in (numpy.array([2., 3.]), numpy.array([2., 3.5])))
    kl = numpy.array([0., float((p * numpy.log(p / q)).sum()), 0.])
    assert answer.lines[2] == ("streamed 9: logits differ at 1 of 3 rows, first f@1, largest 0.5; "
                               f"KL against its base mean {kl.mean():.3g}, p99 {numpy.percentile(kl, 99):.3g}, "
                               f"max {kl.max():.3g}, top-1 100.00%; fewest misses at a streamed layer 1, 0 sited")
    assert not answer.ok
    # Equal logits from a load that never missed at some streamed layer did not ask the question.
    summary = json.loads((tmp_path / "compare-streamed-9" / "compare.json").read_text())
    (tmp_path / "compare-streamed-9" / "compare.json").write_text(json.dumps(
        {**summary, "differing": 0, "first": None, "experts": {"least_layer_misses": 0, "misses": 3, "sited": 0}}))
    assert not REGISTRY["kl"].answer(arguments, results).ok
    (tmp_path / "compare-streamed-9" / "compare.json").write_text(json.dumps(
        {**summary, "differing": 0, "first": None, "experts": {"least_layer_misses": 1, "misses": 3, "sited": 0}}))
    assert REGISTRY["kl"].answer(arguments, results).ok
    # On a discrete device, the base and each streamed load beside its own also run under a
    # profile approving no CPU site, so every miss on either side is the device's: that load must
    # equal the base with its site serving nothing, and the other's site must have served misses.
    discrete = parse("kl", "--backend", "cuda", "--model", MODEL, "--context", "512", "--purpose", "p",
                     "--estimate", "5m", "--weight-cap", "7", "--without-site")
    jobs = dict(REGISTRY["kl"].plan(discrete, None, "alpha").jobs)
    assert list(jobs) == ["profile", "logits", "streamed-7", "unsited-7", "compare-streamed-7", "compare-unsited-7"]
    assert all("--cache-dir" in jobs[name].argv for name in ("streamed-7", "unsited-7")), "both keep the kernel cache"
    assert "--numerical-profile" not in jobs["streamed-7"].argv
    for name in ("logits", "unsited-7"):
        argv = jobs[name].argv
        assert argv[argv.index("--numerical-profile") + 1] == "{runs}/profile/unsited.json"
    profile = jobs["profile"].argv
    subprocess.run([sys.executable if part == "{python}" else part.replace("{run_dir}", str(tmp_path)) for part in profile],
                   check=True)
    assert json.loads((tmp_path / "unsited.json").read_text()) == {"cpu_sites": False}
    compared = lambda differing, sited, base=None: {"rows": 3, "differing": differing, "largest": 0.5 * differing,
                                         "first": rows[1] if differing else None,
                                         "experts": {"least_layer_misses": 2, "misses": 9, "sited": sited},
                                         "base_experts": base,
                                         "nonfinite": {"base": 0, "candidate": 0, "first": None},
                                         "kl": {"mean": 0.01, "p99": 0.02, "max": 0.03}, "top1": 1.0}
    for name, summary in (("compare-streamed-7", compared(1, 4)), ("compare-unsited-7", compared(0, 0))):
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / "compare.json").write_text(json.dumps(summary))
    results = {name: JobResult(tmp_path / name, tmp_path / "log", 0) for name in ("compare-streamed-7", "compare-unsited-7")}
    answer = REGISTRY["kl"].answer(discrete, results)
    # The sited load's logits are reported against its base, not held to equality: its row's
    # verdict is the load approving no CPU site's.
    assert answer.lines[1] == ("streamed 7: its CPU site served misses in arithmetic admitted against α, so this is "
                               "not an equality check (the load approving no CPU site judges it): logits differ at "
                               "1 of 3 rows, first f@1, largest 0.5; KL against its base mean 0.01, p99 0.02, max "
                               "0.03, top-1 100.00%; fewest misses at a streamed layer 2, 4 sited")
    assert answer.ok and answer.lines[2] == ("streamed 7 approving no CPU site: equal logits (3 rows); "
                                             "fewest misses at a streamed layer 2, 0 sited")
    (tmp_path / "compare-unsited-7" / "compare.json").write_text(json.dumps(compared(1, 0)))
    assert not REGISTRY["kl"].answer(discrete, results).ok, "the load approving no CPU site differs"
    (tmp_path / "compare-unsited-7" / "compare.json").write_text(json.dumps(compared(0, 0)))
    (tmp_path / "compare-streamed-7" / "compare.json").write_text(json.dumps(compared(0, 0)))
    answer = REGISTRY["kl"].answer(discrete, results)
    assert not answer.ok, "an approved site that served nothing was not admitted"
    assert answer.lines[1] == ("streamed 7: equal logits (3 rows); its CPU site served no miss, so the site went untested; "
                               "fewest misses at a streamed layer 2, 0 sited")
    # A base that streams says so, and one approving no CPU site is still the reference the load
    # approving none must equal: every miss on either side is the device's.
    streaming = {"least_layer_misses": 3, "misses": 11, "sited": 0}
    for name, sited in (("compare-streamed-7", 4), ("compare-unsited-7", 0)):
        (tmp_path / name / "compare.json").write_text(json.dumps(compared(0, sited, streaming)))
    answer = REGISTRY["kl"].answer(discrete, results)
    assert answer.ok and answer.lines[0].endswith(
        "against uncapped, which streamed, fewest misses at a streamed layer 3, 0 sited")
    # A site serving misses under a profile approving none is an invariant violated, the base's
    # as the load's.
    for name, summary in (("compare-unsited-7", compared(0, 1)), ("compare-streamed-7", compared(0, 4, {**streaming, "sited": 2}))):
        (tmp_path / name / "compare.json").write_text(json.dumps(summary))
        with pytest.raises(AssertionError, match="approving none"):
            REGISTRY["kl"].answer(discrete, results)
        (tmp_path / name / "compare.json").write_text(json.dumps(compared(0, 4 if name == "compare-streamed-7" else 0)))
    # Without --without-site, a sited load's row is unjudged.
    sited = parse("kl", "--backend", "cuda", "--model", MODEL, "--context", "512", "--purpose", "p",
                  "--estimate", "5m", "--weight-cap", "7")
    (tmp_path / "compare-streamed-7" / "compare.json").write_text(json.dumps(compared(0, 4)))
    answer = REGISTRY["kl"].answer(sited, {"compare-streamed-7": results["compare-streamed-7"]})
    assert not answer.ok and answer.lines[1] == (
        "streamed 7: its CPU site served misses in arithmetic admitted against α, so this is not an equality "
        "check (unjudged: --without-site judges it): equal logits (3 rows); fewest misses at a streamed layer 2, 4 sited")
    # Against a base whose own site served misses, no load is held to equality either, even one
    # whose own site served none.
    (tmp_path / "compare-streamed-7" / "compare.json").write_text(json.dumps(
        compared(1, 0, {"least_layer_misses": 3, "misses": 11, "sited": 5})))
    answer = REGISTRY["kl"].answer(sited, {"compare-streamed-7": results["compare-streamed-7"]})
    assert not answer.ok and answer.lines[0].endswith("which streamed, fewest misses at a streamed layer 3, 5 sited")
    assert answer.lines[1] == (
        "streamed 7: its base's CPU site served misses in arithmetic admitted against α, so this is not an equality "
        "check (unjudged: --without-site judges it): logits differ at 1 of 3 rows, first f@1, largest 0.5; KL against "
        "its base mean 0.01, p99 0.02, max 0.03, top-1 100.00%; fewest misses at a streamed layer 2, 0 sited")
    # A load that held its experts whole streamed nothing, and says nothing of misses.
    (tmp_path / "compare-streamed-7" / "compare.json").write_text(json.dumps({**compared(0, 0), "experts": None}))
    answer = REGISTRY["kl"].answer(sited, {"compare-streamed-7": results["compare-streamed-7"]})
    assert not answer.ok and answer.lines[1] == "streamed 7: held its experts whole, so it streamed nothing"
    # Under --between the first streamed load is the base, and no uncapped load runs.
    between = parse("kl", "--backend", "metal", "--model", MODEL, "--context", "512", "--purpose", "p",
                    "--estimate", "5m", "--weight-cap", "7", "--weight-cap", "9", "--between")
    jobs = dict(REGISTRY["kl"].plan(between, None, "alpha").jobs)
    assert list(jobs) == ["streamed-7", "streamed-9", "compare-streamed-9"]
    assert jobs["streamed-9"].argv[jobs["streamed-9"].argv.index("--rendered") + 1] == "{runs}/streamed-7/rendered.jsonl"
    assert "--weight-cap" in jobs["streamed-7"].argv and jobs["streamed-7"].fresh, "the base is itself streamed"
    # The second cap holds where its logits equal the first's and it missed at every streamed layer.
    (tmp_path / "between").mkdir()
    unfinished = {"base": 0, "candidate": 0, "first": None}
    first_load = {"least_layer_misses": 2, "misses": 7, "sited": 0}
    (tmp_path / "between" / "compare.json").write_text(json.dumps(
        {"rows": 3, "differing": 0, "largest": 0.0, "first": None, "experts": {"least_layer_misses": 4, "misses": 9, "sited": 0},
         "base_experts": first_load, "nonfinite": unfinished, "kl": {"mean": 0.0, "p99": 0.0, "max": 0.0}, "top1": 1.0}))
    results = {"compare-streamed-9": JobResult(tmp_path / "between", tmp_path / "log", 0)}
    answer = REGISTRY["kl"].answer(between, results)
    assert answer.ok and answer.lines[0].endswith("against 7, which streamed, fewest misses at a streamed layer 2, 0 sited")
    assert answer.lines[1] == "streamed 9: equal logits (3 rows); fewest misses at a streamed layer 4, 0 sited"
    (tmp_path / "between" / "compare.json").write_text(json.dumps(
        {"rows": 3, "differing": 1, "largest": 0.5, "first": rows[1], "experts": {"least_layer_misses": 4, "misses": 9, "sited": 0},
         "base_experts": first_load, "nonfinite": unfinished, "kl": {"mean": 0.01, "p99": 0.02, "max": 0.03}, "top1": 1.0}))
    assert not REGISTRY["kl"].answer(between, results).ok
    # --repeat runs its passes on one load under one cap: neither comparison of loads takes it.
    for refused in (("--weight-cap", "9", "--between"), ("--without-site",)):
        repeated = parse("kl", "--backend", "cuda", "--model", MODEL, "--context", "512", "--purpose", "p",
                         "--estimate", "5m", "--weight-cap", "7", "--repeat", "2", *refused)
        with pytest.raises(SystemExit, match="--repeat"):
            REGISTRY["kl"].plan(repeated, None, "alpha")


def test_kl_fails_a_streamed_load_whose_logits_are_not_finite(tmp_path):
    """A NaN row is neither equal logits nor a difference: it fails by name, whichever load holds
    it, even where both loads hold the same bits."""
    arguments = parse("kl", "--backend", "metal", "--model", MODEL, "--context", "512", "--purpose", "p",
                      "--estimate", "5m", "--weight-cap", "7")
    jobs = dict(REGISTRY["kl"].plan(arguments, None, "alpha").jobs)
    rows = [{"fixture_id": "f", "position": position} for position in range(3)]
    nan = float("nan")
    for base, candidate, counts in (([[0, 1], [2, 3], [4, 5]], [[0, 1], [nan, nan], [4, 5]], (0, 1)),
                                    ([[0, 1], [2, nan], [4, 5]], [[0, 1], [2, nan], [4, 5]], (1, 1))):
        for name, values, experts in (("logits", base, None),
                                      ("streamed-7", candidate, {"least_layer_misses": 2, "misses": 9, "sited": 0})):
            (tmp_path / name).mkdir(exist_ok=True)
            numpy.array(values, dtype="<f4").tofile(tmp_path / name / "logits.f32")
            (tmp_path / name / "logits.json").write_text(json.dumps({"vocabulary": 2, "rows": rows, "experts": experts}))
        (tmp_path / "compare").mkdir(exist_ok=True)
        tokens = {"{venv}/bin/python": sys.executable, "{run_dir}": str(tmp_path / "compare"), "{runs}": str(tmp_path)}
        argv = [next((part.replace(k, v) for k, v in tokens.items() if k in part), part)
                for part in jobs["compare-streamed-7"].argv]
        subprocess.run(argv, check=True)
        summary = json.loads((tmp_path / "compare" / "compare.json").read_text())
        assert summary["largest"] == 0.0 and summary["kl"]["max"] == 0.0, "only finite rows are measured"
        answer = REGISTRY["kl"].answer(arguments, {"compare-streamed-7": JobResult(tmp_path / "compare", tmp_path / "log", 0)})
        assert not answer.ok
        assert answer.lines[1] == (f"streamed 7: non-finite logits at {counts[0]} rows of its base and {counts[1]} of its "
                                   "own of 3, first f@1; fewest misses at a streamed layer 2, 0 sited")


def test_bench_takes_its_measurement_options_and_builds_our_binaries_apart():
    arguments = parse("bench", "--backend", "vulkan", "--model", MODEL,
                      "--purpose", "p", "--estimate", "5m", "--mode", "tools", "--context", "4k",
                      "--engine", "magnitude", "--engine", "llama.cpp", "--passes", "2")
    plan = REGISTRY["bench"].plan(arguments, None, "alpha")
    jobs = dict(plan.jobs)
    assert list(jobs) == ["measure", "bench"]
    argv = jobs["bench"].argv
    assert argv[argv.index("--magnitude-device") + 1] == "vulkan"
    assert argv[argv.index("--passes") + 1] == "2" and argv.count("--engine") == 2
    assert argv[argv.index("--magnitude-binary") + 1] == "{target}/release/magnitude-engine"
    assert argv[argv.index("--harness") + 1] == "{target}/release/magnitude-harness"
    # The product's measurer prices the run's own fresh cache, which the engine then serves from.
    assert argv[argv.index("--magnitude-cache-dir") + 1] == "{runs}/kernels"
    assert jobs["measure"].argv == ("{target}/release/magnitude-assess", "--device", "vulkan",
                                    "--kernels", "{runs}/kernels", "--topology")
    # The served engine and the harness never share a cargo invocation (feature unification).
    cargo = [step.argv for step in plan.builds if step.argv[0] == "cargo"]
    assert cargo == [("cargo", "build", "--release", "-p", "magnitude-engine-cli"),
                     ("cargo", "build", "--release", "-p", "magnitude-engine", "--bin", "magnitude-assess"),
                     ("cargo", "build", "--release", "-p", "magnitude-harness")]
    # A cache the run names is priced and served as given.
    named = parse("bench", "--backend", "cuda", "--model", MODEL, "--purpose", "p",
                  "--estimate", "5m", "--mode", "tools", "--context", "4k", "--magnitude-cache-dir", "/c")
    jobs = dict(REGISTRY["bench"].plan(named, None, "alpha").jobs)
    assert jobs["measure"].argv[jobs["measure"].argv.index("--kernels") + 1] == "/c"
    assert jobs["bench"].argv.count("--magnitude-cache-dir") == 1
    assert jobs["bench"].argv[jobs["bench"].argv.index("--magnitude-cache-dir") + 1] == "/c"
    # Without our engine, nothing of ours is built, priced or passed.
    other = parse("bench", "--backend", "cuda", "--model", MODEL, "--purpose", "p",
                  "--estimate", "5m", "--mode", "tools", "--context", "4k", "--engine", "mlx-vlm")
    plan = REGISTRY["bench"].plan(other, None, "alpha")
    argv = dict(plan.jobs)["bench"].argv
    assert [name for name, _ in plan.jobs] == ["bench"]
    assert "--harness" not in argv and "--magnitude-binary" not in argv and "--magnitude-cache-dir" not in argv


def test_a_bench_whose_machine_measurement_failed_has_no_answer(tmp_path):
    (tmp_path / "report.md").write_text("# Serving bench\n")
    (tmp_path / "measure.log").write_text("magnitude-assess: no device\n")
    bench = JobResult(tmp_path, tmp_path / "bench.log", 0)
    answer = REGISTRY["bench"].answer(None, {"measure": JobResult(tmp_path, tmp_path / "measure.log", 1), "bench": bench})
    assert not answer.ok and "served unpriced" in answer.lines[0] and "no device" in answer.lines[-1]
    assert REGISTRY["bench"].answer(None, {"measure": JobResult(tmp_path, tmp_path / "measure.log", 0), "bench": bench}).ok
    assert REGISTRY["bench"].answer(None, {"bench": bench}).ok  # another engine alone


def test_kl_judges_a_selection_against_the_defaults_on_the_same_tokens(tmp_path):
    trace = tmp_path / "run" / "tuning.jsonl"
    trace.parent.mkdir()
    trace.write_text("")
    model = MODEL
    arguments = parse("kl", *COMMON, "--model", model, "--context", "4096", "--selection", str(trace.parent),
                      "--unit", "dense_expand")
    plan = REGISTRY["kl"].plan(arguments, None, "alpha")
    jobs = dict(plan.jobs)
    assert "--tuning-defaults" in jobs["defaults"].argv
    selected = jobs["selected"].argv
    assert selected[selected.index("--rendered") + 1] == "{runs}/defaults/rendered.jsonl"
    assert selected[selected.index("--tuning-select") + 1] == "{inputs}/tuning.jsonl"
    assert selected[selected.index("--tuning-select-unit") + 1] == "dense_expand"
    assert jobs["kl"].argv[jobs["kl"].argv.index("--base") + 1] == "{runs}/defaults"
    assert trace in plan.files
    with pytest.raises(SystemExit):
        parse("kl", *COMMON, "--model", model, "--context", "4096", "--selection", str(tmp_path / "none"))


def test_draft_answers_the_drafters_acceptance_exactly(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"measurements": {"drafting": {
        "proposed": 64, "accepted": 40, "tokens_per_verify": 2.25, "matched_fixture": 30, "anchors": 32}}}))
    arguments = argparse.Namespace(model="m:v", fixture=Path("prose-repeat.jsonl"), context=4096, attribute=False,
                                   weight_cap=None)
    answer = REGISTRY["draft"].answer(arguments, {"draft": JobResult(tmp_path, tmp_path / "log", 0)})
    assert answer.lines[1].startswith("2.250 tokens per verification · accepted 40 of 64 proposals (62.5%)")
    assert answer.measures["tokens_per_verify"].resolution is None


def test_draft_compares_each_streamed_load_with_the_uncapped_anchor_by_anchor(tmp_path):
    def run(name: str, rows: list[dict], code: int = 0) -> JobResult:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps({"measurements": {"drafting": {
            "proposed": 4, "accepted": 2, "tokens_per_verify": 2.0, "anchors": 2}}}))
        (directory / "draft.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return JobResult(directory, directory / "log", code)

    def row(anchor: int, target: list[int]) -> dict:
        return {"fixture_id": "f", "anchor": anchor, "proposals": [1, 2], "target": target}

    uncapped = [row(3, [1, 2, 9]), row(7, [1, 5, 9])]
    arguments = argparse.Namespace(model="m:v", fixture=Path("prose-repeat.jsonl"), context=4096, attribute=False,
                                   weight_cap=[10, 20])
    answer = REGISTRY["draft"].answer(arguments, {
        "draft": run("draft", uncapped),
        "measure": run("measure", []),
        harness.streamed(10): run("same", uncapped),
        harness.streamed(20): run("other", [row(3, [1, 2, 9]), row(7, [1, 6, 9])]),
    })
    assert "streamed 10: same proposals and target tokens at all 2 anchors" in answer.lines
    assert any(line.startswith("streamed 20: 1 of 2 anchors differ, first f @7") for line in answer.lines)
    assert not answer.ok, "a cap that differs fails the answer"
    alone = argparse.Namespace(**{**vars(arguments), "weight_cap": [10]})

    def alone_answer(name: str, streamed: JobResult, measured: int = 0):
        return REGISTRY["draft"].answer(alone, {"draft": run(f"{name}-draft", uncapped),
                                                "measure": run(f"{name}-measure", [], measured),
                                                harness.streamed(10): streamed})

    assert alone_answer("same", run("same-again", uncapped)).ok
    assert not alone_answer("failed", run("failed", uncapped, 1)).ok, "a streamed job that failed fails it"
    assert not alone_answer("unmeasured", run("unmeasured", uncapped), 1).ok, "a failed measurement job fails it"
    assert not alone_answer("moved", run("moved", [row(3, [1, 2, 9]), row(8, [1, 5, 9])])).ok, "other anchors fail it"
    nothing = REGISTRY["draft"].answer(alone, {"draft": run("empty-draft", []), "measure": run("empty-measure", []),
                                               harness.streamed(10): run("empty", [])})
    assert not nothing.ok, "no anchor compared is no pass"
    plan = REGISTRY["draft"].plan(parse("draft", *COMMON, "--model", MODEL, "--context", "4096", "--weight-cap", "10"), None, "alpha")
    assert [name for name, _ in plan.jobs] == ["draft", "measure", harness.streamed(10)]


def test_assess_takes_a_model_or_the_catalog():
    model = MODEL
    with pytest.raises(SystemExit, match="--model or --catalog"):
        REGISTRY["assess"].plan(parse("assess", *COMMON), None, "alpha")
    with pytest.raises(SystemExit, match="--model or --catalog"):
        REGISTRY["assess"].plan(parse("assess", *COMMON, "--model", model, "--catalog"), None, "alpha")
    catalog = REGISTRY["assess"].plan(parse("assess", *COMMON, "--catalog"), None, "alpha")
    assert catalog.acquires == engine.Headers(None)
    assert [name for name, _ in catalog.jobs] == ["measure", "catalog"]
    assert "{catalog}/model-planner-inputs.bundle" in dict(catalog.jobs)["catalog"].argv


def test_device_adds_the_drivers_facts_on_vulkan():
    jobs = lambda backend: [name for name, _ in REGISTRY["device"].plan(
        parse("device", "--backend", backend, "--purpose", "p", "--estimate", "1m"), None, "alpha").jobs]
    assert jobs("vulkan") == ["host", "topology", "vulkan"] and jobs("metal") == ["host", "topology"]


def record_topology(results: Path, run: str, host: str, backend: str, memory: str, *, failed=None) -> None:
    """A `mrig device` run of `host` in the results store, its engine's topology stating its
    `backend` GPU's memory as `memory` (`Host` or `Own`) beside the host CPU's."""
    directory = results / run
    (directory / "logs").mkdir(parents=True)
    (directory / "rig.json").write_text(json.dumps({"host": host, "failed": failed, "codes": {"host": 0, "topology": 0}}))
    cpu, gpu = {"Host": {"machine": "m"}}, {"Uuid": [1, 2]}
    topology = {"physical": [{"identity": cpu, "memory": "Host"}, {"identity": gpu, "memory": memory}],
                "executors": [{"selector": "HostCpu", "physical": cpu},
                              {"selector": {backend.capitalize(): {"uuid": [1, 2]}}, "physical": gpu}]}
    (directory / "logs" / "job-topology.log").write_text(
        "rig: $ magnitude-assess --topology\n[stderr] magnitude-assess: a GPU\n"
        + json.dumps({"topology": topology, "executor": f"{backend}:0102"}) + "\n")


def test_memorys_answer_expects_the_results_of_the_memory_its_plan_measured(tmp_path):
    def ran(kind: str, written: str) -> bool:
        directory = tmp_path / kind / "memory"
        directory.mkdir(parents=True)
        (directory / written).write_text("[]")
        argv = ["python3", "-m", "magnitude_rig.host", "record", "--", "python3", "-m", "magnitude_rig.host.memory",
                "--backend", "cuda", "--memory", kind, "--scratch", str(directory)]
        (directory.parent / "plan.json").write_text(json.dumps({"ops": [{"op": "run", "argv": argv, "job": "memory"}]}))
        log = directory.parent / "memory.log"
        log.write_text("")
        arguments = parse("memory", *COMMON, "--model", MODEL)
        return REGISTRY["memory"].answer(arguments, {"memory": JobResult(directory, log, 0)}).ok

    assert ran("unified", memory.RESULTS["unified"])
    assert not ran("dedicated", memory.RESULTS["unified"])  # the other kind's file is not its answer


def test_memory_follows_the_devices_memory_as_its_hosts_recorded_topology_states_it(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "results", lambda: tmp_path)
    record_topology(tmp_path, "20261020T000000Z-device-000001", "gb10", "cuda", "Own")  # superseded
    record_topology(tmp_path, "20261028T000000Z-device-000002", "gb10", "cuda", "Host")
    record_topology(tmp_path, "20261029T000000Z-device-000003", "gb10", "cuda", "Own", failed={"phase": "x"})
    record_topology(tmp_path, "20261028T000000Z-device-000004", "rtx", "cuda", "Own")
    record_topology(tmp_path, "20261028T000000Z-device-000005", "mac", "metal", "Host")
    assert device.memory("gb10", "cuda") == "unified" and device.memory("rtx", "cuda") == "dedicated"
    assert device.memory("mac", "metal") == "unified" and device.memory("mac", "cpu") == "unified"
    with pytest.raises(engine.Refused, match="run mrig device --host bare first"):
        device.memory("bare", "cuda")
    with pytest.raises(engine.Refused, match="cuda devices of memory none"):
        device.memory("mac", "cuda")

    def plan(host, backend, *options):
        return REGISTRY["memory"].plan(parse("memory", "--backend", backend, "--model", MODEL, *options,
                                             "--purpose", "p", "--estimate", "1m"), None, host)

    rtx, mac, gb10 = plan("rtx", "cuda"), plan("mac", "metal"), plan("gb10", "cuda")
    assert [step.name for step in rtx.builds] == ["engine", "assess"] and rtx.device == "timed"
    for unified in (mac, gb10):  # a unified CUDA device measures as Metal does
        assert unified.builds[0].argv[-1] == "magnitude-harness" and unified.device == "device"
        argv = dict(unified.jobs)["memory"].argv
        assert argv[argv.index("--memory") + 1] == "unified" and not dict(unified.jobs)["memory"].fresh
    streamed = plan("gb10", "cuda", "--weight-cap", "7")
    assert [name for name, _ in streamed.jobs] == ["measure", "memory"]
    job = dict(streamed.jobs)["memory"]
    assert job.fresh and job.argv[job.argv.index("--weight-cap") + 1] == "7"
    with pytest.raises(SystemExit, match="unified-memory"):
        plan("rtx", "cuda", "--weight-cap", "7")
    with pytest.raises(engine.Refused, match="run mrig device --host bare first"):
        plan("bare", "cuda")
    # The host's own judge of its options refuses what it would refuse run alone.
    with pytest.raises(SystemExit, match="it takes --residency"):
        plan("gb10", "cuda", "--weight-cap", "7", "--pressure", "5", "--pressure-store", "host")
    with pytest.raises(SystemExit, match="loads nothing"):
        plan("mac", "metal", "--weight-cap", "7", "--recommit", "1")
    # A good run whose topology log holds no topology document refuses; it never escapes.
    record_topology(tmp_path, "20261030T000000Z-device-000006", "mac", "metal", "Host")
    (tmp_path / "20261030T000000Z-device-000006" / "logs" / "job-topology.log").write_text("rig: $ assess\n")
    with pytest.raises(engine.Refused, match="recorded no topology"):
        device.memory("mac", "metal")


def test_the_host_costs_fit_covers_every_load_at_the_least_overcharge():
    # Own memory exactly 100 MB + 2 kB per vocabulary entry for three loads; a fourth needs 10 MB more.
    loads = [{"vocabulary": v, "kernels": k, "own_bytes": 100_000_000 + 2_000 * v}
             for v, k in ((32_000, 40), (150_000, 60), (250_000, 50))]
    loads.append({"vocabulary": 64_000, "kernels": 80, "own_bytes": 110_000_000 + 2_000 * 64_000})
    runtime, per_token, per_kernel = memory.fit(loads)
    charge = lambda load: runtime + per_token * load["vocabulary"] + per_kernel * load["kernels"]
    assert all(charge(load) >= load["own_bytes"] for load in loads)
    # No worse than one feasible choice: 90 MB, 2 kB per entry, 0.25 MB per kernel over-charges 7.5 MB.
    assert sum(charge(load) - load["own_bytes"] for load in loads) <= 7_500_000


def test_a_fixture_set_is_built_and_stored_by_its_content(tmp_path, monkeypatch):
    monkeypatch.setattr(selectors, "FIXTURES", tmp_path)
    path = selectors.fixture("retrieval")
    assert path.name == "retrieval.jsonl" and path.parent.parent == tmp_path
    assert selectors.fixture("retrieval") == path
    with pytest.raises(argparse.ArgumentTypeError):
        selectors.fixture("nope")


RECORDS = Path(__file__).parent / "records"


def test_step_prints_a_prompt_groups_bar_its_terms_ceilings_and_achieved_moves():
    """An RTX 3090's recorded CUDA prompt group as a harness records it now: its
    bar, the bar's read, device and link terms at the ceilings, the ceilings, and what the run's
    own convert and tier copy achieved (that run's attribution); the ceilings and the link term
    here stand for a measured link of 12 GB/s."""
    group = json.loads((RECORDS / "prefill-group.json").read_text())
    group.update({
        "bar_ms": 1100.0, "ratio": 1100.0 / group["measured_ms"],
        "bar_terms": {"read_ms": 0.0, "resident_ms": 1088.2922105000007, "convert_ms": 0.0, "link_ms": 1100.0},
        "ceilings": {"convert_us_per_record": [5.0, 6.0], "link_bytes_per_second": 12e9},
        "achieved": {"convert_ms": 22.687606811516048, "copy_ms": 1147.5439758302032, "copy_us_per_record": 1147.5439758302032e3 / 7160},
    })
    assert step_instrument._group(group) == [
        "  1 prompt groups of 6 steps, 6 in all · measured 2364.7 ms · bar 1100.0 ms · bar/measured 0.465",
        "  bar terms: read 0.0 ms · device 1088.3 ms (resident 1088.3 + convert 0.0) · link 1100.0 ms",
        "  ceilings: convert 5.0–6.0 µs/record · link 12.00 GB/s",
        "  achieved: convert 22.7 ms · copy 1147.5 ms (160.3 µs/record)",
        "  union per layer and group 204–256 (mean 227.2) · conveyed 0 · pulled 7160 · seeded 0 · waited for experts 23.6 ms",
    ]
    # Without a bar, why; on unified memory, no link; without an attributed prefill, why.
    group.update({"bar_ms": {"unavailable": "no resident"}, "ratio": None, "bar_terms": None,
                  "ceilings": {"convert_us_per_record": [5.0], "link_bytes_per_second": None},
                  "achieved": {"unavailable": "no prefill was attributed"}})
    lines = step_instrument._group(group)
    assert lines[0].endswith("measured 2364.7 ms · no bar (no resident)")
    assert lines[1:3] == ["  ceilings: convert 5.0–5.0 µs/record · link none (unified memory)",
                          "  achieved: unavailable (no prefill was attributed)"]


def test_step_names_what_a_recorded_prompt_group_lacks():
    """The same group as that run's harness recorded it, before its bar's terms, ceilings and
    achieved moves: the printout names them, and reads nothing in their place."""
    group = json.loads((RECORDS / "prefill-group.json").read_text())
    assert step_instrument._group(group) == [
        "  prompt groups not recorded by this build's harness (no bar_terms, ceilings, achieved)"]


def test_a_served_run_installs_the_catalog_beside_the_model_and_bounds_only_what_it_is_given():
    arguments = argparse.Namespace(scenario="stop", context=4096, weight_cap=None, plain=True, model="m:gguf:q4",
                                   fixture=Path("set.jsonl"), backend="cuda")
    plan = REGISTRY["served"].plan(arguments, None, "host")
    assert plan.acquires == engine.Weights("m:gguf:q4", headers=True)
    assert plan.device == "device"
    (name, job), (_, answers), (_, decoded), (_, judge) = plan.jobs
    assert name == "served" and job.fresh
    # Its answers are teacher-forced through the whole model, computed again as the service
    # computes them on the same tokens, then judged.
    assert answers.argv[answers.argv.index("--fixtures") + 1] == "{runs}/served/answers.jsonl"
    assert "--weight-cap" not in answers.argv and "--tuning-defaults" in answers.argv
    assert decoded.argv[decoded.argv.index("--rendered") + 1] == "{runs}/answers/rendered.jsonl"
    assert "--tuning-defaults" not in decoded.argv and decoded.argv[decoded.argv.index("--batch") + 1] == "1"
    assert judge.argv[-3:] == ("{runs}/answers", "{runs}/decoded", "{run_dir}/judged.json")
    assert job.argv[job.argv.index("--context") + 1] == "4096"
    assert "--weight-cap" not in job.argv and "--plain" in job.argv
    assert "CUDA_VISIBLE_DEVICES" not in dict(job.env)


def test_a_served_run_on_vulkan_hides_cuda_and_fails_when_the_service_selects_another_backend(tmp_path):
    arguments = argparse.Namespace(scenario="stop", context=None, weight_cap=None, plain=False, model="m:gguf:q4",
                                   fixture=Path("set.jsonl"), backend="vulkan")
    (_, job), *_ = REGISTRY["served"].plan(arguments, None, "host").jobs
    assert dict(job.env)["CUDA_VISIBLE_DEVICES"] == ""
    (tmp_path / "served.json").write_text(json.dumps({"report": ["ready"]}))
    (tmp_path / "service-capped.log").write_text('\x1b[2mserving device selected\x1b[0m backend="cuda"\n')
    result = JobResult(tmp_path, tmp_path / "job.log", 0)
    answer = REGISTRY["served"].answer(arguments, {"served": result})
    assert not answer.ok and "did not select vulkan" in answer.lines[0]


def _judged_answers(tmp_path, rows: list[list[float]], targets: list[int], served: dict,
                    deltas: dict[tuple[int, int], float] | None = None) -> Path:
    """A served run's one answer `a`, its reference logits `rows` predicting `targets`, and the
    service-shaped computation's, off the reference by `deltas` at (row, token), by default 0.1
    at the first row's top."""
    from magnitude_rig.host import served as host
    from magnitude_rig.instruments import served as instrument
    for directory in ("served", "answers", "decoded", "judge"):
        (tmp_path / directory).mkdir(parents=True)
    fixture = {"id": "a", "sizing": {"unit": "fixed"}, "messages": [], "served": served}
    (tmp_path / "served" / host.ANSWERS).write_text(json.dumps(fixture) + "\n")
    (tmp_path / "served" / host.RESULTS).write_text(json.dumps({"report": [], "unjudged": []}))
    prompt = 3
    (tmp_path / "answers" / "rendered.jsonl").write_text(json.dumps({"fixture_id": "a", "tokens": [0] * prompt + targets}) + "\n")
    header = {"vocabulary": len(rows[0]), "rows": [{"fixture_id": "a", "position": prompt - 1 + index} for index in range(len(rows))]}
    decoded = numpy.asarray(rows, dtype="<f4")
    for (row, token), delta in (deltas if deltas is not None else {(0, int(decoded[0].argmax())): 0.1}).items():
        decoded[row, token] += delta
    for directory, logits in (("answers", numpy.asarray(rows, dtype="<f4")), ("decoded", decoded)):
        (tmp_path / directory / "logits.json").write_text(json.dumps(header))
        logits.tofile(tmp_path / directory / "logits.f32")
    subprocess.run([sys.executable, "-c", instrument.JUDGE, *(str(tmp_path / name) for name in ("answers", "decoded")),
                    str(tmp_path / "judge" / "judged.json")], check=True)
    assert not (tmp_path / "answers" / "logits.f32").exists(), "the judge leaves the logits on the host"
    return tmp_path


def test_a_served_answer_may_part_only_within_the_measured_move_of_a_competing_margin(tmp_path):
    """Each answered token is held to its row's top within B, the largest move of a margin below a
    row's top measured between the reference and the service-shaped computation among the tokens
    competing with it: with the first row's top 0.1 off and a token 0.05 below it, B = 0.1, so a
    flip 0.087 nats below the top and the answer past it pass, one 0.12 below fails, and so does a
    token far below its row's top and a non-finite row."""
    from magnitude_rig.instruments.served import judged
    top = lambda token: [0.0 if index == token else -9.0 for index in range(4)]
    near = lambda margin: [-margin, 0.0, -9.0, -9.0]  # token 0, `margin` below the top
    run = _judged_answers(tmp_path / "flip", [near(0.05), near(0.087), top(2)], [1, 0, 2], {"tokens": 3, "finish": "length"})
    lines, passed = judged(run / "served", run / "judge")
    assert passed and lines[1:] == ["a: 2/3 tokens their row's top, worst 0.087 nats at token 1"]
    assert lines[0].startswith("largest move between the reference and the service-shaped computation of a margin below "
                               "a row's top, among the tokens competing with it, 0.1000 (at a position 2, token 0: 0.0500")
    run = _judged_answers(tmp_path / "wide", [near(0.05), near(0.12), top(2)], [1, 0, 2], {"tokens": 3, "finish": "length"})
    lines, passed = judged(run / "served", run / "judge")
    assert not passed and lines[1] == "a: 2/3 tokens their row's top, worst 0.120 nats at token 1, 1 past 0.1000 FAILED"
    run = _judged_answers(tmp_path / "garbage", [near(0.05), top(1), top(1)], [1, 3, 2], {"tokens": 3, "finish": "length"})
    lines, passed = judged(run / "served", run / "judge")
    assert not passed and lines[1] == "a: 1/3 tokens their row's top, worst 9.000 nats at token 1, 2 past 0.1000 FAILED"
    run = _judged_answers(tmp_path / "nan", [near(0.05), [float("nan")] * 4], [1, 2], {"tokens": 3, "finish": "stop"})
    lines, passed = judged(run / "served", run / "judge")
    assert not passed and lines[1] == "a: 1/2 tokens their row's top, worst 0.000 nats, 1 non-finite FAILED"


def test_a_served_corruption_nats_below_the_top_fails_whatever_the_tail_differs_by(tmp_path):
    """The bound is measured over the tokens competing with each row's top, not the tail: a tail
    logit 20 nats below its top that differs by 3.0 (as LFM2.5's do) leaves the bound at the 0.1 a
    competing token's margin moved, so an answer corrupted 1.5 nats below its row's top fails. Over
    the whole vocabulary the bound would be twice 3.0, which that corruption sits within: such a
    bar cannot fail it."""
    from magnitude_rig.instruments.served import judged
    rows = [[0.0, -20.0, -0.05, -9.0], [-1.5, 0.0, -9.0, -9.0], [-9.0, -9.0, 0.0, -9.0]]
    run = _judged_answers(tmp_path, rows, [0, 0, 2], {"tokens": 3, "finish": "length"}, {(0, 2): 0.1, (0, 1): 3.0})
    lines, passed = judged(run / "served", run / "judge")
    assert lines[0].startswith("largest move between the reference and the service-shaped computation of a margin below "
                               "a row's top, among the tokens competing with it, 0.1000 (at a position 2, token 2")
    assert not passed and lines[1] == "a: 2/3 tokens their row's top, worst 1.500 nats at token 1, 1 past 0.1000 FAILED"


def test_a_served_judgement_without_a_measured_difference_bounds_nothing(tmp_path):
    """A service-shaped computation equal to the reference bit for bit moves no margin: the bound
    would fail every honest near tie, so the judgement fails as unsound."""
    from magnitude_rig.instruments.served import judged
    run = _judged_answers(tmp_path, [[0.0, -9.0]], [0], {"tokens": 1, "finish": "length"}, deltas={})
    lines, passed = judged(run / "served", run / "judge")
    assert not passed and lines[0].startswith("the service-shaped computation moved no margin of a token competing with its row's top")


def test_a_served_answer_the_harness_tokenized_otherwise_is_unjudged(tmp_path):
    """An answer whose rows are not the service's tokens is not judged, and fails; a stopped answer
    may count its end of turn among them."""
    from magnitude_rig.instruments.served import judged
    rows = [[0.0, -0.05], [0.0, -9.0]]
    run = _judged_answers(tmp_path / "length", rows, [0, 0], {"tokens": 3, "finish": "length"})
    lines, passed = judged(run / "served", run / "judge")
    assert not passed and lines[1].startswith("a: unjudged, the harness's tokens of it are not the service's (2 rows, 3")
    run = _judged_answers(tmp_path / "stop", rows, [0, 0], {"tokens": 3, "finish": "stop"})
    assert judged(run / "served", run / "judge")[1]


def test_a_served_answer_that_only_reasoned_is_kept_for_judgement(tmp_path):
    """A model whose reasoning cannot be turned off answers in its reasoning: that answer is
    judged as any other, and only one that generated nothing is unjudged."""
    from magnitude_rig.host.served import ANSWERS, RESULTS, Run
    run = Run.__new__(Run)
    run.directory, run.scratch, run.report, run.record, run.answers = tmp_path, tmp_path / "scratch", [], {}, []
    asked = [{"role": "user", "content": "q"}]
    served = {"usage": {"completion_tokens": 2}, "finish": "length"}
    run.answered("thought", asked, served, "", "so")
    run.answered("silent", asked, served, "", "")
    run.finish()
    kept = [json.loads(line) for line in (tmp_path / ANSWERS).read_text().splitlines()]
    assert [fixture["id"] for fixture in kept] == ["thought"]
    assert kept[0]["messages"][-1] == {"role": "assistant", "content": "", "scored": True, "reasoning_content": "so"}
    assert json.loads((tmp_path / RESULTS).read_text())["unjudged"] == ["silent"]


def test_a_served_run_counts_overlapping_workers_from_the_services_own_records():
    """Workers counted from the service's start and exit records, in their order: a start before
    the last worker's exit is two at once, however briefly; one after it is one."""
    from magnitude_rig.host.served import most_workers
    line = "2026-11-05T09:38:58.138001Z  INFO tokio-rt-worker magnitude_service_server::residency::worker: "
    started = lambda pid: f"{line}inference worker started worker.pid={pid}\n"
    exited = lambda pid: f"{line}inference worker exited worker.pid={pid} status=signal: 9 (SIGKILL)\n"
    relayed = f"{line}inference worker worker.pid=7 message=inference worker ended: Shutdown\n"
    assert most_workers(started(7) + relayed + exited(7) + started(8) + exited(8)) == 1
    assert most_workers(started(7) + started(8) + exited(7) + exited(8)) == 2
    assert most_workers("") == 0


def test_a_header_export_names_a_models_components_by_its_whole_reference(tmp_path):
    from magnitude_rig.host.__main__ import _header_artifacts
    file = {"repository": "owner/repo", "revision": "r"}
    (tmp_path / "index.json").write_text(json.dumps({"components": [
        {"model": "m:gguf:q5", "role": "target", "path": "q5.gguf", **file},
        {"model": "m:gguf:q4", "role": "target", "path": "q4.gguf", **file},
        {"model": "m:gguf:q4", "role": "draft", "path": "draft.gguf", **file},
    ]}))
    output = tmp_path / "artifacts.json"
    ask = lambda model: _header_artifacts(argparse.Namespace(export=tmp_path, model=model, output=output))
    assert ask("m:gguf:q4") == 0
    artifacts = json.loads(output.read_text())
    assert Path(artifacts["target"]).name == "q4.gguf" and Path(artifacts["drafter"]["path"]).name == "draft.gguf"
    assert ask("m:gguf:q6") == 1
