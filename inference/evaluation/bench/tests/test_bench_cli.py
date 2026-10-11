from pathlib import Path

import pytest
from magnitude_bench import cli
from magnitude_bench.models import catalog_reference, targets


def test_contexts_and_sections():
    assert cli.contexts("16k,4k,4096") == (4096, 16384)
    with pytest.raises(ValueError):
        cli.contexts("0")
    assert cli.choices("all", ("a", "b")) == ("a", "b")
    with pytest.raises(ValueError):
        cli.choices("context,nope", ("context",))


def test_models_come_only_from_the_catalog():
    assert catalog_reference("qwen3.5-4b:gguf:q4") == "qwen3.5-4b:gguf:q4"
    for reference in ("qwen3.5-4b", "qwen3.5-4b:gguf:q3", "not-a-model:gguf:q4"):
        with pytest.raises(ValueError):
            catalog_reference(reference)


def test_gguf_engines_serve_the_catalog_target_and_others_name_their_form(acquisition):
    selected = targets(
        ["magnitude", "llama.cpp", "mlx-vlm"],
        {"mlx-vlm": "hf:mlx-community/Qwen3.5-4B-4bit@" + "a" * 40},
        acquisition,
    )
    assert [t.reference for t in selected[:2]] == [str(acquisition.target)] * 2
    assert all(t.sha256 == acquisition.target_sha256 for t in selected[:2])
    assert selected[2].reference.startswith("hf:mlx-community/")
    with pytest.raises(ValueError, match="takes no --artifact"):
        targets(["llama.cpp"], {"llama.cpp": "/x.gguf"}, acquisition)
    with pytest.raises(ValueError, match="needs --artifact omlx"):
        targets(["omlx"], {}, acquisition)
    with pytest.raises(ValueError, match="not selected"):
        targets(["magnitude"], {"omlx": "/m"}, acquisition)
    with pytest.raises(ValueError, match="Hub snapshots"):
        targets(["mlx-vlm"], {"mlx-vlm": "hf:owner/repo@main"}, acquisition)


def test_run_hands_the_job_its_measurement_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = [
        "--mode", "tools", "--model", "qwen3.5-4b:gguf:q4", "--context", "4k",
        "--host", "gpu-pool", "--no-gate", "--llama-binary", "bin/llama-server",
        "--engine", "llama.cpp", "--engine=magnitude",
    ]
    remote = cli.job_arguments(argv, local=False, engines=["llama.cpp", "magnitude"])
    assert "--host" not in remote and "gpu-pool" not in remote and "--no-gate" not in remote
    assert remote[remote.index("--llama-binary") + 1] == "bin/llama-server"
    assert remote[remote.index("--magnitude-binary") + 1] == "{target}/release/magnitude-engine"
    assert remote[remote.index("--harness") + 1] == "{target}/release/magnitude-harness"
    assert remote[remote.index("--magnitude-kernel-cache-root") + 1] == "{root}/kernel-cache"
    named = cli.job_arguments(
        [*argv, "--magnitude-cache-dir", "/scratch/cache"], local=False,
        engines=["llama.cpp", "magnitude"],
    )
    assert "--magnitude-kernel-cache-root" not in named
    local = cli.job_arguments(
        [*argv, "--harness=custom/harness"], local=True, engines=["llama.cpp", "magnitude"]
    )
    assert local[local.index("--llama-binary") + 1] == str(tmp_path / "bin/llama-server")
    assert local[local.index("--harness") + 1] == str(tmp_path / "custom/harness")
    assert "--engine=magnitude" in local
    llama = cli.job_arguments(argv, local=True, engines=["llama.cpp"])
    assert "--magnitude-binary" not in llama and "--harness" in llama
    assert "--magnitude-kernel-cache-root" not in llama
    assert "--harness" not in cli.job_arguments(argv, local=True, engines=["mlx-vlm"])


def test_run_builds_one_flow_job(monkeypatch):
    from magnitude_remote import flow

    jobs = []

    def execute(job, host):
        jobs.append((job, host))
        return flow.Outcome("run-id", host, Path("/results/run-id"), 0)

    monkeypatch.setattr(flow, "execute", execute)
    code = cli.main([
        "run", "--mode", "prose-repeat", "--model", "qwen3.5-4b:gguf:q4", "--context", "16k",
        "--host", "gpu-pool", "--tuning-seconds", "60", "--fresh-tuning",
    ])
    assert code == 0
    job, host = jobs[0]
    assert host == "gpu-pool" and job.tool == "bench" and job.model == "qwen3.5-4b:gguf:q4"
    assert job.build == (
        ("cargo", "build", "--release", "-p", "magnitude-engine-cli", "-p", "magnitude-harness"),
    )
    assert job.gate is not None
    command = list(job.command)
    assert command[: len(cli.JOB)] == list(cli.JOB)
    assert command[command.index("--artifacts") + 1] == "{artifacts}"
    assert command[command.index("--output") + 1] == "{run_dir}"
    assert "--fresh-tuning" in command and command[command.index("--tuning-seconds") + 1] == "60"
    assert command[command.index("--build-identity") + 1] == "{build_file}"


def test_comparison_runs_build_only_what_they_use(monkeypatch):
    from magnitude_remote import flow

    jobs = []
    monkeypatch.setattr(
        flow, "execute",
        lambda job, host: jobs.append(job) or flow.Outcome("r", None, Path("/r"), 0),
    )
    assert cli.main([
        "run", "--mode", "tools", "--model", "qwen3.5-4b:gguf:q4", "--context", "4k",
        "--engine", "llama.cpp", "--no-gate",
    ]) == 0
    assert jobs[0].build == (("cargo", "build", "--release", "-p", "magnitude-harness"),)
    assert jobs[0].gate is None and "--magnitude-binary" not in jobs[0].command
    assert cli.main([
        "run", "--mode", "tools", "--model", "qwen3.5-4b:gguf:q4", "--context", "4k",
        "--engine", "mlx-vlm", "--artifact", "mlx-vlm=/m", "--no-gate",
    ]) == 0
    assert jobs[1].build == () and "--harness" not in jobs[1].command


def test_a_given_engine_binary_is_measured_as_is(monkeypatch):
    from magnitude_remote import flow

    jobs = []
    monkeypatch.setattr(
        flow, "execute",
        lambda job, host: jobs.append(job) or flow.Outcome("r", None, Path("/r"), 0),
    )
    build = '{"commit": "b2f79f00f", "dirty_digest": null}'
    assert cli.main([
        "run", "--mode", "prose-repeat", "--model", "qwen3.5-4b:gguf:q4", "--context", "4k",
        "--magnitude-binary", "{root}/origin/target/release/magnitude-engine",
        "--magnitude-build", build, "--no-tuning-trace", "--host", "host-a",
    ]) == 0
    # Only the harness (for rendering) is built; the given engine is not rebuilt.
    assert jobs[0].build == (("cargo", "build", "--release", "-p", "magnitude-harness"),)
    command = list(jobs[0].command)
    assert command[command.index("--magnitude-binary") + 1] == (
        "{root}/origin/target/release/magnitude-engine"
    )
    assert command[command.index("--magnitude-build") + 1] == build
    assert "--no-tuning-trace" in command
    args = cli.parser().parse_args(command[len(cli.JOB) - 1:] + ["--build-identity", "build.json"])
    options = cli.engine_options(args).magnitude
    assert options.build.commit == "b2f79f00f" and options.tuning_trace is False


def test_invalid_selection_fails_before_any_job(monkeypatch, capsys):
    from magnitude_remote import flow

    monkeypatch.setattr(flow, "execute", lambda job, host: pytest.fail("no job may start"))
    assert cli.main([
        "run", "--mode", "tools", "--model", "qwen3.5-4b:gguf:q3", "--context", "4k",
    ]) == 2
    assert "no variant" in capsys.readouterr().err
