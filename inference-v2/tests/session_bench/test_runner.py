import asyncio
import json
import sys
from pathlib import Path

import psutil
import pytest

from session_bench import report, runner
from session_bench.client import Observation
from session_bench.engines.base import Adapter
from session_bench.models import Target
from session_bench.results import RunStore, inspect_run


class FixtureAdapter(Adapter):
    pids = []

    def argv(self, port, context, parallel, directory):
        return [sys.executable, str(Path(__file__).with_name("engine_fixture.py")), str(port)]

    def verify_ready(self, data, context, parallel):
        assert data["status"] == "ready"

    async def prompt_counts(self, plan):
        return {r.id: 10 for r in plan.prepared_requests}


@pytest.fixture
def fake_runtime(monkeypatch, interaction):
    async def corpus(root, categories):
        return [interaction], "test-corpus"

    monkeypatch.setattr(runner.corpus, "prepare", corpus)
    monkeypatch.setitem(runner.ADAPTERS, "magnitude", FixtureAdapter)


async def test_real_process_full_run_and_readable_evidence(tmp_path, artifact_path, fake_runtime):
    target = Target(engine="magnitude", reference=str(artifact_path))
    result = await runner.run(
        tmp_path,
        [target],
        ("single", "parallel"),
        (1024,),
        ("simple-python",),
        1,
        None,
        lambda _: None,
    )
    assert result["status"] == "completed", result
    assert result["completed"] == result["planned"] == 10
    path = Path(result["path"])
    assert inspect_run(path)["status"] == "completed"
    assert str(artifact_path) in (path / "command.txt").read_text()
    assert "Session bench" in (path / "report.md").read_text()
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    assert sum(event["event"] == "stopped" for event in events) == 2
    assert (path / "memory.jsonl").is_file()


async def test_preparation_failure_still_has_report(tmp_path, fake_runtime):
    target = Target(engine="magnitude", reference=str(tmp_path / "missing"))
    result = await runner.run(
        tmp_path, [target], ("single",), (1024,), ("simple-python",), 1, None, lambda _: None
    )
    assert result["status"] == "failed"
    assert result["completed"] == 0
    assert "command failed" in result["error"]
    assert (
        "model directory"
        in next((Path(result["path"]) / "logs").glob("*-artifact.log")).read_text()
    )
    assert (Path(result["path"]) / "report.md").is_file()


async def test_cancel_keeps_completed_result_and_retires_child(
    tmp_path, artifact_path, fake_runtime, monkeypatch
):
    entered = asyncio.Event()
    original = runner.measure

    async def controlled(client, endpoint, model, request, event, **kwargs):
        if request.id.startswith("parallel"):
            entered.set()
            await asyncio.Event().wait()
        return await original(client, endpoint, model, request, event, **kwargs)

    monkeypatch.setattr(runner, "measure", controlled)
    target = Target(engine="magnitude", reference=str(artifact_path))
    task = asyncio.create_task(
        runner.run(
            tmp_path,
            [target],
            ("single", "parallel"),
            (1024,),
            ("simple-python",),
            1,
            None,
            lambda _: None,
        )
    )
    await asyncio.wait_for(entered.wait(), 10)
    children = psutil.Process().children(recursive=True)
    task.cancel()
    result = await task
    assert result["status"] == "cancelled"
    path = Path(result["path"])
    rows = [json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()]
    assert any(r["phase"] == "measured" and r["observation"]["outcome"] == "valid" for r in rows)
    assert any(r["observation"]["outcome"] == "cancelled" for r in rows)
    assert all(not child.is_running() for child in children)
    assert (path / "report.md").is_file()


def test_reports_exclude_truncation_but_keep_context_semantic_invalidity():
    evidence = {
        "usage": {"prompt_tokens": 10},
        "timings": {"prompt_n": 10, "prompt_ms": 10, "predicted_n": 2, "predicted_ms": 10},
    }
    rows = [
        {
            "phase": "measured",
            "target": "test",
            "section": section,
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


def test_new_evidence_directory_never_overwrites_old(tmp_path):
    first = RunStore(tmp_path, "one", {})
    second = RunStore(tmp_path, "two", {})
    assert first.path != second.path
    assert "one" in (first.path / "command.txt").read_text()


def test_inspection_does_not_mistake_reused_pid_or_unknown_schema_for_success(tmp_path):
    store = RunStore(tmp_path, "command", {})
    assert inspect_run(store.path)["status"] == "running"
    path = store.path / "run.json"
    value = json.loads(path.read_text())
    value["process_started_at"] -= 1
    path.write_text(json.dumps(value))
    assert inspect_run(store.path)["status"] == "interrupted"
    (store.path / "summary.json").write_text('{"format":999,"status":"completed"}')
    assert inspect_run(store.path)["status"] == "unsupported-format"


async def test_sigterm_reaches_managed_cleanup(monkeypatch):
    import os
    import signal

    from session_bench import cli

    cleaned = []

    async def pending():
        asyncio.get_running_loop().call_later(0.01, os.kill, os.getpid(), signal.SIGTERM)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaned.append(True)
            return {"status": "cancelled"}

    monkeypatch.setattr(cli, "run", pending)
    assert (await cli.managed_run())["status"] == "cancelled"
    assert cleaned == [True]


async def test_owner_loss_retires_supervised_engine(tmp_path):
    import os
    import signal

    from session_bench.engines import supervise

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
    tmp_path, artifact_path, fake_runtime, monkeypatch
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
                    if process.cmdline()[1].endswith("engine_fixture.py")
                )
                os.kill(child.pid, signal.SIGKILL)
            await asyncio.Event().wait()
        return await original(client, endpoint, model, request, event, **kwargs)

    monkeypatch.setattr(runner, "measure", crash)
    target = Target(engine="magnitude", reference=str(artifact_path))
    result = await asyncio.wait_for(
        runner.run(
            tmp_path,
            [target],
            ("single", "parallel"),
            (1024,),
            ("simple-python",),
            1,
            None,
            lambda _: None,
        ),
        10,
    )
    assert result["status"] == "failed"
    assert "exited during execution" in result["error"]
    assert result["outcomes"] == {"valid": 1, "target-failure": 4}
    assert (Path(result["path"]) / "report.md").is_file()
