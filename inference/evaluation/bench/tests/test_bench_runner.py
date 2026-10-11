import asyncio
import contextlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import magnitude_fixtures
import psutil
import pytest
from bench_support import character_lengths
from magnitude_bench import report, runner
from magnitude_bench.adapters.base import Adapter
from magnitude_bench.client import Observation
from magnitude_bench.models import Target
from magnitude_bench.options import EngineOptions, Watchdog
from magnitude_bench.store import RunStore

BUILD = {"commit": "a" * 40, "dirty_digest": None}


class FixtureAdapter(Adapter):
    def argv(self, port, context, parallel, directory):
        return [sys.executable, str(Path(__file__).with_name("bench_engine.py")), str(port)]

    def verify_ready(self, data, context, parallel):
        assert data["status"] == "ready"

    @asynccontextmanager
    async def counter(self):
        yield character_lengths


class GgufAdapter(FixtureAdapter):
    @property
    def host_rendered(self):
        return True

    def counter(self):
        raise AssertionError("GGUF engines are counted by the harness")


@pytest.fixture
def fake_harness(tmp_path):
    script = tmp_path / "magnitude-harness"
    harness = Path(__file__).with_name("bench_harness.py")
    script.write_text(f"#!/bin/sh\nexec {sys.executable} {harness} \"$@\"\n")
    script.chmod(0o755)
    return script


@pytest.fixture
def fake_runtime(monkeypatch, fixtures):
    # Test runs may overlap (other checkouts' suites); the machine lock guards real runs only.
    monkeypatch.setattr(runner, "machine_lock", contextlib.nullcontext)
    monkeypatch.setattr(magnitude_fixtures, "build", lambda mode, count: fixtures[:count])
    monkeypatch.setitem(runner.ADAPTERS, "mlx-vlm", FixtureAdapter)


def measure(tmp_path, acquisition, artifacts, reference, sections=("single", "parallel"),
            options=None):
    target = Target(engine="mlx-vlm", reference=str(reference))
    return runner.run(
        RunStore(tmp_path / "run"),
        "tools",
        [target],
        sections,
        (1500,),
        acquisition,
        artifacts,
        BUILD,
        options or EngineOptions(),
        ["magnitude-bench", "job", "--mode", "tools"],
        lambda _: None,
    )


async def test_real_process_full_run_and_contract_record(
    tmp_path, artifact_path, acquisition, artifacts, fake_runtime
):
    result = await measure(tmp_path, acquisition, artifacts, artifact_path)
    assert result["status"] == "completed", result
    # single + four parallel requests, on each of the two balanced passes
    assert result["completed"] == result["planned"] == 10
    path = Path(result["path"])
    assert "Serving bench" in (path / "report.md").read_text()
    record = json.loads((path / "result.json").read_text())
    assert record["schema"] == "evaluation-result/1" and record["tool"] == "bench"
    assert record["build"] == BUILD and record["model"] == acquisition.model
    assert record["artifacts"] == json.loads(artifacts.read_text())
    fixture_set = record["fixture_set"]
    assert (path / fixture_set["path"]).is_file() and len(fixture_set["digest"]) == 64
    assert record["rendered_digest"] is None  # Sized by a comparison engine's own counter.
    serving = record["measurements"]["serving"]
    assert len(serving) == 10 and all(row["valid"] for row in serving)
    parallel = next(row for row in serving if row["section"] == "parallel")
    assert parallel["context"] == 1500 and parallel["concurrency"] == 4
    assert parallel["prefill_tps"] == 500 and parallel["decode_tps"] == 100
    assert parallel["draft_n"] == 3 and parallel["draft_n_accepted"] == 2
    assert "drafting" not in record["measurements"] and "tuning" not in record["measurements"]
    requests = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    sized = next(r for r in requests if r["section"] == "parallel")
    assert character_lengths_of(sized) < 1500  # The serving prompt stops before the answer.
    counts = json.loads((path / "mlx-vlm-prompt-counts.json").read_text())
    assert set(counts) == {r["id"] for r in requests}
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "stopped" for event in events) == 2
    assert (path / "memory.jsonl").is_file()


def character_lengths_of(request: dict) -> int:
    body = {"messages": request["messages"], "tools": request["tools"]}
    return len(json.dumps(body))


async def test_gguf_engines_are_sized_and_counted_by_the_harness(
    tmp_path, acquisition, artifacts, fake_runtime, fake_harness, monkeypatch
):
    from magnitude_bench.options import MagnitudeOptions

    monkeypatch.setitem(runner.ADAPTERS, "llama.cpp", GgufAdapter)
    target = Target(engine="llama.cpp", reference=str(acquisition.target), sha256="0" * 64)
    monkeypatch.setattr(
        runner, "prepare_artifact",
        lambda target, store: _artifact(acquisition),
    )
    result = await runner.run(
        RunStore(tmp_path / "run"), "tools", [target], ("context",), (1500,), acquisition,
        artifacts, BUILD, EngineOptions(magnitude=MagnitudeOptions(harness=fake_harness)),
        ["magnitude-bench"], lambda _: None,
    )
    assert result["status"] == "completed", result
    path = Path(result["path"])
    plan = json.loads((path / "plan.json").read_text())
    assert plan["sized_by"] == "magnitude-harness"
    assert plan["rendering"]["harness"] == str(fake_harness)
    record = json.loads((path / "result.json").read_text())
    assert len(record["rendered_digest"]) == 64
    counts = json.loads((path / "llama.cpp-prompt-counts.json").read_text())
    requests = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    assert counts == {
        r["id"]: len(json.dumps({"messages": r["messages"], "tools": r["tools"]}))
        for r in requests
    }


async def _artifact(acquisition):
    from magnitude_bench.models import Artifact, ArtifactFile

    stat = acquisition.target.stat()
    return Artifact(
        reference=str(acquisition.target), path=acquisition.target, kind="gguf",
        context_limit=262144, metadata={},
        files=(ArtifactFile(path="target.gguf", size=stat.st_size, sha256="0" * 64,
                            mtime_ns=stat.st_mtime_ns),),
    )


async def test_preparation_failure_still_has_report_and_record(
    tmp_path, acquisition, artifacts, fake_runtime
):
    result = await measure(tmp_path, acquisition, artifacts, tmp_path / "missing")
    assert result["status"] == "failed"
    assert result["completed"] == 0
    assert "command failed" in result["error"]
    path = Path(result["path"])
    assert "model directory" in next((path / "logs").glob("*-artifact.log")).read_text()
    assert (path / "report.md").is_file()
    record = json.loads((path / "result.json").read_text())
    assert record["measurements"] == {"serving": []}


async def test_cancel_keeps_completed_result_and_retires_child(
    tmp_path, artifact_path, acquisition, artifacts, fake_runtime, monkeypatch
):
    existing_children = {child.pid for child in psutil.Process().children(recursive=True)}
    entered = asyncio.Event()
    original = runner.measure

    async def controlled(client, endpoint, model, request, event, **kwargs):
        if request.id.startswith("parallel"):
            entered.set()
            await asyncio.Event().wait()
        return await original(client, endpoint, model, request, event, **kwargs)

    monkeypatch.setattr(runner, "measure", controlled)
    task = asyncio.create_task(measure(tmp_path, acquisition, artifacts, artifact_path))
    await asyncio.wait_for(entered.wait(), 10)
    children = [
        child
        for child in psutil.Process().children(recursive=True)
        if child.pid not in existing_children
    ]
    assert children
    task.cancel()
    result = await task
    assert result["status"] == "cancelled"
    path = Path(result["path"])
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    assert any(r["phase"] == "measured" and r["observation"]["outcome"] == "valid" for r in rows)
    assert any(r["observation"]["outcome"] == "cancelled" for r in rows)
    assert all(not child.is_running() for child in children)
    assert (path / "report.md").is_file()
    serving = json.loads((path / "result.json").read_text())["measurements"]["serving"]
    assert {row["outcome"] for row in serving} == {"valid", "cancelled"}


def test_reports_exclude_truncation_but_keep_context_semantic_invalidity():
    evidence = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        "timings": {"prompt_n": 10, "prompt_ms": 10, "predicted_n": 2, "predicted_ms": 10},
    }
    rows = [
        {
            "phase": "measured",
            "target": "test",
            "section": section,
            "family": "tools",
            "checkpoint": 1024,
            "timing_basis": "native-model-service",
            "observation": Observation(
                request_id=f"{section}-{outcome}",
                outcome=outcome,
                terminal=evidence,
                ttft_ms=12,
                completed_ms=30,
            ).model_dump(),
        }
        for section in ("context", "session")
        for outcome in ("valid", "invalid", "truncated")
    ]
    summary = report.summarize(rows, "failed", "command", "id", 6)
    assert [r["eligible"] for r in summary["rows"]] == [2, 1]
    assert summary["outcomes"]["truncated"] == 2


async def test_sigterm_reaches_managed_cleanup(monkeypatch):
    import os
    import signal

    from magnitude_bench import cli

    cleaned = []

    async def pending():
        asyncio.get_running_loop().call_later(0.01, os.kill, os.getpid(), signal.SIGTERM)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaned.append(True)
            return {"status": "cancelled"}

    assert (await cli.managed(pending()))["status"] == "cancelled"
    assert cleaned == [True]


async def test_owner_loss_retires_supervised_engine(tmp_path):
    import os
    import signal

    from magnitude_bench.adapters import supervise

    record = tmp_path / "owned.pid"
    child_code = (
        "import os,pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )
    owner_code = (
        "import os,subprocess,sys,time; "
        "subprocess.Popen([sys.executable,sys.argv[1],str(os.getpid()),sys.executable,'-c',"
        "sys.argv[3],sys.argv[2]],start_new_session=True); time.sleep(60)"
    )
    owner = await asyncio.create_subprocess_exec(
        sys.executable, "-c", owner_code, supervise.__file__, str(record), child_code
    )
    pid = None
    try:
        async with asyncio.timeout(5):
            while not record.exists():
                await asyncio.sleep(0.01)
        pid = int(record.read_text())
        os.kill(owner.pid, signal.SIGKILL)
        await owner.wait()
        async with asyncio.timeout(5):
            while psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                await asyncio.sleep(0.05)
    finally:
        if owner.returncode is None:
            owner.kill()
            await owner.wait()
        if pid and psutil.pid_exists(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_report_separates_concurrency_points():
    rows = [
        {
            "phase": "measured",
            "target": "test",
            "section": "concurrency",
            "family": "tools",
            "checkpoint": 1024,
            "concurrency": n,
            "timing_basis": "native-model-service",
            "observation": {"outcome": "truncated"},
        }
        for n in (1, 2, 4, 8)
    ]
    summary = report.summarize(rows, "failed", "command", "id", 4)
    assert [row["concurrency"] for row in summary["rows"]] == [1, 2, 4, 8]


async def test_engine_crash_keeps_prior_results_and_marks_target_failure(
    tmp_path, artifact_path, acquisition, artifacts, fake_runtime, monkeypatch
):
    import os
    import signal

    original = runner.measure
    killed = False

    async def crash(client, endpoint, model, request, event, **kwargs):
        nonlocal killed
        if request.id.startswith("parallel"):
            if not killed:
                killed = True
                child = next(
                    process
                    for process in psutil.Process().children(recursive=True)
                    if process.cmdline()[1].endswith("bench_engine.py")
                )
                os.kill(child.pid, signal.SIGKILL)
            await asyncio.Event().wait()
        return await original(client, endpoint, model, request, event, **kwargs)

    monkeypatch.setattr(runner, "measure", crash)
    result = await asyncio.wait_for(measure(tmp_path, acquisition, artifacts, artifact_path), 10)
    assert result["status"] == "failed"
    assert "exited during execution" in result["error"]
    assert result["outcomes"] == {"valid": 1, "target-failure": 4}
    assert (Path(result["path"]) / "report.md").is_file()


@pytest.mark.parametrize(
    ("watchdog", "reason"),
    [
        (Watchdog(stall_seconds=0.5), "no progress"),
        (Watchdog(request_seconds=0.5), "exceeded 0.5s"),
    ],
)
async def test_watchdog_retires_a_stalled_engine_and_fails_the_run(
    tmp_path, artifact_path, acquisition, artifacts, fake_runtime, monkeypatch, watchdog, reason
):
    original = runner.measure

    async def stall(client, endpoint, model, request, event, **kwargs):
        if request.id.startswith("parallel"):
            await asyncio.Event().wait()
        return await original(client, endpoint, model, request, event, **kwargs)

    monkeypatch.setattr(runner, "measure", stall)
    result = await asyncio.wait_for(
        measure(tmp_path, acquisition, artifacts, artifact_path,
                options=EngineOptions(watchdog=watchdog)),
        10,
    )
    assert result["status"] == "failed"
    assert "watchdog" in result["error"] and reason in result["error"]
    events = (Path(result["path"]) / "events.jsonl").read_text()
    assert '"event":"watchdog"' in events
