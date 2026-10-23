"""Selectors fail before anything queues; instruments plan what they say; `ab` alternates and
judges only beyond what both sides resolved."""

import argparse
import json
from pathlib import Path

import pytest

from magnitude_rig import ab, cli, engine, selectors
from magnitude_rig.instruments import REGISTRY, JobResult, Measure
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


def test_test_builds_without_running_then_runs_on_the_device():
    arguments = parse("test", *COMMON, "--", "-p", "magnitude-kernels", "--test", "routed", "--", "general")
    plan = REGISTRY["test"].plan(arguments, None)
    (build,) = plan.builds
    assert build.argv == ("cargo", "test", "--message-format=short", "--no-run", "-p", "magnitude-kernels", "--test", "routed")
    ((_, job),) = plan.jobs
    assert job.argv == ("cargo", "test", "-p", "magnitude-kernels", "--test", "routed", "--", "general")
    assert plan.device == "device" and job.env == ()
    vulkan = parse("test", "--backend", "vulkan", "--purpose", "p", "--estimate", "1m", "--", "-p", "x")
    assert REGISTRY["test"].plan(vulkan, None).jobs[0][1].env == (("SEISMIC_TEST_BACKEND", "vulkan"),)


def test_python_tests_run_in_the_checkouts_environment_with_the_model_they_check():
    model = MODEL
    arguments = parse("test", *COMMON, "--suite", "python", "--model", model, "--",
                      "evaluation/references/tests/test_released.py", "-k", "qwen")
    plan = REGISTRY["test"].plan(arguments, None)
    assert "--all-extras" in plan.builds[0].argv
    job = plan.jobs[0][1]
    assert job.argv[0] == "{venv}/bin/pytest" and job.argv[-3:] == ("evaluation/references/tests/test_released.py", "-k", "qwen")
    assert dict(job.env)["MAGNITUDE_ARTIFACTS"] == "{artifacts}" and plan.acquires == engine.Weights(model)


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
    plan = REGISTRY["script"].plan(arguments, None)
    assert plan.device == "timed" and plan.builds[0].argv == ("cargo", "build", "-p", "x")
    job = plan.jobs[0][1]
    assert job.argv[:2] == ("{python}", "-c") and dict(job.env)["MRIG_RUN"] == "{run_dir}"


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
    answer = ab.compare(_Measuring, arguments, results, (("m", None),))
    rows = {line.split()[0]: line for line in answer.lines[1:]}
    assert rows["step"].endswith("better") and rows["small"].endswith("below resolution")
    assert answer.ok


def test_step_answers_the_median_step_its_interval_launches_and_entries(tmp_path):
    steps = [{"wall_ms": 15.0 + 0.01 * (index % 5)} for index in range(64)]
    entries = [
        {"entry": "dense_expand#1", "launches_per_step": 22.3, "device_ms_per_step": 3.2, "share": 0.3},
        {"entry": "attention_decode_k8v4", "launches_per_step": 5.3, "device_ms_per_step": 0.2, "share": 0.02},
    ]
    decode = {"phase": "decode", "tokens_per_second": 66.0, "samples": [],
              "detail": {"steps": steps, "median_step": {"wall_ms": 15.02},
                         "attribution": {"steps": 3, "entries": entries}}}
    (tmp_path / "result.json").write_text(json.dumps({"measurements": {"speed": [
        {"phase": "prefill", "detail": {"rows": 4096, "cold_ms": 900.0, "samples_ms": [600.0, 610.0, 590.0]}},
        decode]}}))
    arguments = argparse.Namespace(context=4096, sequences=1)
    answer = REGISTRY["step"].answer(arguments, {"step": JobResult(tmp_path, tmp_path / "log", 0)})
    assert answer.lines[0].startswith("step @4096 · 15.020 ms") and answer.lines[0].endswith("28 launches")
    assert answer.lines[1].startswith("prefill @4096 · 600.0 ms (±10.0") and "6827 tok/s" in answer.lines[1]
    assert "dense_expand#1" in answer.lines[3]
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


def test_ab_step_runs_every_round_on_the_bases_first_tuning(tmp_path):
    fixture = tmp_path / "prose-repeat.jsonl"
    arguments = argparse.Namespace(fixture=fixture, model="m:v", backend="vulkan", context=4096, steps=64,
                                   prefill_samples=6, attribution=0, lookahead="on", tuning="replay-base",
                                   rounds=2, sequences=1)
    plans = {side: REGISTRY["step"].plan(arguments, None) for side in ab.SIDES}
    jobs = ab._jobs(REGISTRY["step"], arguments, plans)
    pinning = [[a for a in job.step.argv if a.startswith("--tuning-")] for job in jobs]
    assert jobs[0].name == "base-r0-step"
    assert pinning == [["--tuning-record"], ["--tuning-replay"], ["--tuning-replay"], ["--tuning-replay"]]
    argv = jobs[0].step.argv
    assert argv[argv.index("--device") + 1] == "vulkan"  # the harness runs on the run's backend
    own = argparse.Namespace(**{**vars(arguments), "tuning": "own"})
    assert all(not [a for a in job.step.argv if a.startswith("--tuning-")] for job in ab._jobs(REGISTRY["step"], own, plans))


def test_bench_takes_its_measurement_options_and_builds_our_binaries_apart():
    arguments = parse("bench", "--backend", "vulkan", "--model", MODEL,
                      "--purpose", "p", "--estimate", "5m", "--mode", "tools", "--context", "4k",
                      "--engine", "magnitude", "--engine", "llama.cpp", "--passes", "2")
    plan = REGISTRY["bench"].plan(arguments, None)
    argv = plan.jobs[0][1].argv
    assert argv[argv.index("--magnitude-device") + 1] == "vulkan"
    assert argv[argv.index("--passes") + 1] == "2" and argv.count("--engine") == 2
    assert argv[argv.index("--magnitude-binary") + 1] == "{target}/release/magnitude-engine"
    assert argv[argv.index("--harness") + 1] == "{target}/release/magnitude-harness"
    assert argv[argv.index("--magnitude-kernel-cache-root") + 1] == "{root}/kernel-cache"
    # The served engine and the harness never share a cargo invocation (feature unification).
    cargo = [step.argv for step in plan.builds if step.argv[0] == "cargo"]
    assert cargo == [("cargo", "build", "--release", "-p", "magnitude-engine-cli"),
                     ("cargo", "build", "--release", "-p", "magnitude-harness")]
    given = parse("bench", "--backend", "cuda", "--model", MODEL, "--purpose", "p",
                  "--estimate", "5m", "--mode", "tools", "--context", "4k", "--engine", "mlx-vlm",
                  "--magnitude-cache-dir", "/c")
    argv = REGISTRY["bench"].plan(given, None).jobs[0][1].argv
    assert "--harness" not in argv and "--magnitude-binary" not in argv and "--magnitude-kernel-cache-root" not in argv


def test_kl_judges_a_selection_against_the_defaults_on_the_same_tokens(tmp_path):
    trace = tmp_path / "run" / "tuning.jsonl"
    trace.parent.mkdir()
    trace.write_text("")
    model = MODEL
    arguments = parse("kl", *COMMON, "--model", model, "--context", "4096", "--selection", str(trace.parent),
                      "--unit", "dense_expand")
    plan = REGISTRY["kl"].plan(arguments, None)
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
    arguments = argparse.Namespace(model="m:v", fixture=Path("prose-repeat.jsonl"), context=4096, attribute=False)
    answer = REGISTRY["draft"].answer(arguments, {"draft": JobResult(tmp_path, tmp_path / "log", 0)})
    assert answer.lines[1].startswith("2.250 tokens per verification · accepted 40 of 64 proposals (62.5%)")
    assert answer.measures["tokens_per_verify"].resolution is None


def test_assess_takes_a_model_or_the_catalog():
    model = MODEL
    with pytest.raises(SystemExit, match="--model or --catalog"):
        REGISTRY["assess"].plan(parse("assess", *COMMON), None)
    with pytest.raises(SystemExit, match="--model or --catalog"):
        REGISTRY["assess"].plan(parse("assess", *COMMON, "--model", model, "--catalog"), None)
    catalog = REGISTRY["assess"].plan(parse("assess", *COMMON, "--catalog"), None)
    assert catalog.acquires == engine.Headers(None)
    assert [name for name, _ in catalog.jobs] == ["measure", "catalog"]
    assert "{catalog}/model-planner-inputs.bundle" in dict(catalog.jobs)["catalog"].argv


def test_device_adds_the_drivers_facts_on_vulkan_and_memory_needs_a_dedicated_gpu():
    jobs = lambda backend: [name for name, _ in REGISTRY["device"].plan(
        parse("device", "--backend", backend, "--purpose", "p", "--estimate", "1m"), None).jobs]
    assert jobs("vulkan") == ["host", "topology", "vulkan"] and jobs("metal") == ["host", "topology"]
    model = MODEL
    with pytest.raises(SystemExit, match="dedicated GPU"):
        REGISTRY["memory"].plan(parse("memory", "--backend", "metal", "--model", model, "--purpose", "p",
                                      "--estimate", "1m"), None)


def test_a_fixture_set_is_built_and_stored_by_its_content(tmp_path, monkeypatch):
    monkeypatch.setattr(selectors, "FIXTURES", tmp_path)
    path = selectors.fixture("retrieval")
    assert path.name == "retrieval.jsonl" and path.parent.parent == tmp_path
    assert selectors.fixture("retrieval") == path
    with pytest.raises(argparse.ArgumentTypeError):
        selectors.fixture("nope")
