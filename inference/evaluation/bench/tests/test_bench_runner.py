import asyncio
import contextlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
import pytest
from bench_support import TREE, body_length, character_lengths, template_paths
from magnitude_bench import report, runner
from magnitude_bench.adapters.base import Adapter
from magnitude_bench.adapters.llama_cpp import LlamaCpp
from magnitude_bench.adapters.native import Native
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
def fake_runtime(monkeypatch):
    # Test runs may overlap (other checkouts' suites); the machine lock guards real runs only.
    monkeypatch.setattr(runner, "machine_lock", contextlib.nullcontext)
    monkeypatch.setitem(runner.ADAPTERS, "mlx-vlm", FixtureAdapter)


def measure(tmp_path, acquisition, artifacts, reference, sections=("context", "parallel"),
            options=None):
    target = Target(engine="mlx-vlm", reference=str(reference))
    return runner.run(
        RunStore(tmp_path / "run"),
        template_paths(tmp_path / "templates", 2),
        [target],
        sections,
        acquisition,
        artifacts,
        BUILD,
        options or EngineOptions(inference=TREE),
        ["magnitude-bench", "job", "--template", "tools-2k.json"],
        lambda _: None,
    )


async def test_real_process_full_run_and_contract_record(
    tmp_path, artifact_path, acquisition, artifacts, fake_runtime
):
    result = await measure(tmp_path, acquisition, artifacts, artifact_path)
    assert result["status"] == "completed", result
    # one context request + four parallel requests, on each of the two balanced passes
    assert result["completed"] == result["planned"] == 10
    path = Path(result["path"])
    assert "Serving bench" in (path / "report.md").read_text()
    record = json.loads((path / "result.json").read_text())
    assert record["schema"] == "evaluation-result/1" and record["tool"] == "bench"
    assert record["build"] == BUILD and record["model"] == acquisition.model
    assert record["artifacts"] == json.loads(artifacts.read_text())
    (template,) = record["templates"]
    assert template["template"] == "tools:2k" and template["context"] == 2048
    assert (path / template["path"]).is_file() and len(template["digest"]) == 64
    assert record["rendered_digest"] is None  # Sized by a comparison engine's own counter.
    serving = record["measurements"]["serving"]
    assert len(serving) == 10 and all(row["valid"] for row in serving)
    parallel = next(row for row in serving if row["section"] == "parallel")
    assert parallel["context"] == 2048 and parallel["concurrency"] == 4
    assert parallel["template"] == "tools:2k" and parallel["prompt_sha256"] is None
    assert parallel["prefill_tps"] == 500 and parallel["decode_tps"] == 100
    assert parallel["draft_n"] == 3 and parallel["draft_n_accepted"] == 2
    assert "drafting" not in record["measurements"] and "tuning" not in record["measurements"]
    requests = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    sized = next(r for r in requests if r["section"] == "parallel")
    assert character_lengths_of(sized) < 2048  # The serving prompt stops before the answer.
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
        RunStore(tmp_path / "run"), template_paths(tmp_path / "templates", 2, 4), [target],
        ("context",), acquisition, artifacts, BUILD, EngineOptions(inference=TREE, magnitude=MagnitudeOptions.of(TREE, harness=fake_harness)),
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
    assert counts == {r["id"]: body_length(r["messages"], r["tools"]) for r in requests}
    # Every lane is sized as the harness sized it, and carries its rendered digest.
    rendered = {(line["template"], line["lane"]): line for line in
                map(json.loads, (path / "rendered" / "rendered.jsonl").read_text().splitlines())}
    assert {r["template"] for r in requests} == {"tools:2k", "tools:4k"}
    assert all(r["prompt_sha256"] == rendered[(r["template"], r["lane"])]["prompt_sha256"]
               for r in requests)


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
            "template": "tools:1k",
            "key": "tools",
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


def test_the_expert_bank_is_followed_over_each_pass():
    def sample(at, slots, experts=None, tables=None, host=None, label="block-0", target="magnitude"):
        """An observation as the engine's census serializes it: each holding's own last resize."""
        def wire(resize):  # a BankResize: Shrank by `by` to `to`, or Grew by `by` from `from`
            if resize is None:
                return None
            if resize["kind"] == "Grew":
                return {"Grew": {"from": resize["from"], "by": resize["to"] - resize["from"],
                                 "at_unix_ms": resize["at_unix_ms"]}}
            return {"Shrank": {"to": resize["to"], "by": resize["from"] - resize["to"],
                               "at_unix_ms": resize["at_unix_ms"]}}

        bank = None if slots is None else {
            "experts": 256, "slots": slots, "floor": 26, "ceiling": 256, "stores": [], "disk_bytes": 0,
            "cover": {"begun": 1, "reading": False},
            "host": {"slots": 1, "ceiling": 256, "bytes": 1, "last_resize": wire(host)},
            "tables": [{"records": 3, "floor": 1, "ceiling": 4, "bytes": 3, "last_resize": wire(resize)}
                       for resize in (tables or [None])],
            "last_resize": wire(experts),
        }
        return {"target": target, "label": label, "at": at,
                "observation": {"census": {"serving": {"group_places": 1, "group_capacity": 8, "experts": bank}}}}

    def resize(at, before, after, kind):
        """A resize as the summary reads it: its kind, its size before and after, and when."""
        return {"kind": kind, "from": before, "to": after, "at_unix_ms": at}

    grown, hosted = resize(1, 64, 96, "Grew"), resize(2, 0, 1, "Grew")
    shrunk = resize(4, 96, 80, "Shrank")
    table = resize(5, 2, 3, "Grew")
    samples = [
        sample(2.0, 96, grown, host=hosted), sample(1.0, 64), sample(3.0, 96, grown, host=hosted),
        sample(4.0, 80, shrunk, host=hosted),
        sample(5.0, 80, shrunk, tables=[table], host=hosted),  # a table's, latest of all
        sample(6.0, None, label="block-1"),  # a load without a bank
        # The adapter's own observation, as it reads its log: it states no bank.
        {"target": "llama.cpp", "label": "block-0", "at": 1.0,
         "observation": {"state": {"kv_bytes": 1 << 20, "recurrent_bytes": 0}}},
    ]
    banks_of = {"magnitude": lambda observation: Native.expert_bank(None, observation),
                "llama.cpp": lambda observation: LlamaCpp.expert_bank(None, observation)}
    (bank,) = report.expert_banks(samples, banks_of)
    assert bank == {"target": "magnitude", "label": "block-0", "first_slots": 64, "last_slots": 80,
                    "min_slots": 64, "max_slots": 96, "floor": 26, "ceiling": 256,
                    "resizes": {"experts": {"Grew": 1, "Shrank": 1}, "host": {"Grew": 1},
                                "table 0": {"Grew": 1}},
                    "latest_resize": {"holding": "table 0", **table}}
    shown = report.markdown({**report.summarize([], "completed", "c", "id", 0), "key": "tools",
                             "expert_banks": [bank]})
    assert ("| magnitude | block-0 | 64 → 80 | 64–96 | 26 | 256 | experts Grew 1, Shrank 1; host Grew 1; "
            "table 0 Grew 1 | table 0 Grew 2 → 3 |") in shown


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
            "template": "tools:1k",
            "key": "tools",
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
                options=EngineOptions(inference=TREE, watchdog=watchdog)),
        10,
    )
    assert result["status"] == "failed"
    assert "watchdog" in result["error"] and reason in result["error"]
    events = (Path(result["path"]) / "events.jsonl").read_text()
    assert '"event":"watchdog"' in events


def test_tuning_health_report_and_alerts(tmp_path):
    def candidate(points: dict[str, float], admitted: bool = True):
        return {
            "admitted": admitted,
            "outcome": {
                "Measured": {
                    "points": [{"point": label, "median_seconds": seconds} for label, seconds in points.items()]
                }
            },
        }

    def finished(entry: str, bindings: str, qualification: str, stop: str | None,
                 weights: dict[str, float], candidates: list[dict]):
        return {
            "event": "finished",
            "entry": entry,
            "bindings": bindings,
            "qualification": qualification,
            "stop": stop,
            "points": [{"label": label, "weight": weight} for label, weight in weights.items()],
            "candidates": candidates,
        }

    # Empty trace -> None and zero impact on markdown.
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert report.tuning_health(empty) is None
    base = {**report.summarize([], "completed", "c", "id", 0), "key": "tools"}
    assert "## Tuning health" not in report.markdown(base)

    # Healthy trace: 10% fallback (1/10, not > 10%) and no >1.25x gap -> no stdout alert.
    healthy_events = [
        {"event": "began", "schema": "tuning-trace/3", "work": "search", "budget_seconds": 60.0},
        *[
            finished(f"ok_{i}", "M=1", "qualified", "Exhausted", {"m1": 10.0},
                     [candidate({"m1": 1e-4}), candidate({"m1": 0.9e-4})])
            for i in range(9)
        ],
        finished("fb_0", "M=1", "unqualified", "Exhausted", {"m1": 5.0},
                 [candidate({"m1": 2e-4})]),
        {"event": "concluded", "units": 10, "canonical_fallbacks": 1},
    ]
    healthy_trace = tmp_path / "healthy.jsonl"
    healthy_trace.write_text("\n".join(json.dumps(e) for e in healthy_events) + "\n")
    healthy = report.tuning_health(healthy_trace)
    assert healthy is not None
    assert not healthy["high_fallback"]
    assert healthy["worst_default_gaps"] == []
    assert report.tuning_health_alert(healthy) == []
    healthy_md = report.markdown({**base, "tuning_health": healthy})
    assert "## Tuning health" in healthy_md
    assert "WARNING: High tuning fallback rate" not in healthy_md
    assert "Qualified: 9/10" in healthy_md

    # Unhealthy trace: prefers `search` load over subsequent `load`; checks weighted ratios,
    # rejects non-admitted candidates, and ranks top 5 gaps and slowest fallbacks.
    unhealthy_events = [
        {"event": "began", "schema": "tuning-trace/3", "work": "search", "budget_seconds": 60.0},
        # 2.5x gap (weighted); also has a faster unadmitted candidate that must be ignored.
        finished("routed_output", "M=2..9", "qualified", "Expired", {"m2": 1.0, "m8": 3.0}, [
            candidate({"m2": 2.0e-4, "m8": 2.0e-4}),
            candidate({"m2": 0.8e-4, "m8": 0.8e-4}, admitted=True),
            candidate({"m2": 0.2e-4, "m8": 0.2e-4}, admitted=False),
        ]),
        # Faster on tiny-weight m256, much slower on heavy m512 -> NOT a gap; ranked as fallback.
        finished("stream_mix_down", "M=65..", "unqualified", "Exhausted", {"m256": 10.0, "m512": 190.0}, [
            candidate({"m256": 5.0e-4, "m512": 4.0e-4}),
            candidate({"m256": 2.0e-4, "m512": 10.0e-4}),
        ]),
        # Expired fallback with higher weighted cost.
        finished("readout_rows", "M=1..2", "unqualified_budget", "Expired", {"m1": 50.0}, [
            candidate({"m1": 2.0e-3}),
        ]),
        {"event": "concluded", "units": 3, "canonical_fallbacks": 2},
        # Subsequent `load` (stored) without the expired candidate: must not shadow `search`.
        {"event": "began", "schema": "tuning-trace/3", "work": "load", "budget_seconds": 60.0},
        finished("routed_output", "M=2..9", "qualified", "Exhausted", {"m2": 1.0}, []),
        {"event": "concluded", "units": 1, "canonical_fallbacks": 0},
    ]
    unhealthy_trace = tmp_path / "unhealthy.jsonl"
    unhealthy_trace.write_text("\n".join(json.dumps(e) for e in unhealthy_events) + "\n")
    health = report.tuning_health(unhealthy_trace)
    assert health is not None
    assert health["work"] == "search"
    assert (health["qualified"], health["canonical_fallbacks"], health["total_units"], health["expired"]) == (1, 2, 3, 2)
    assert health["high_fallback"] is True
    assert len(health["worst_default_gaps"]) == 1
    gap = health["worst_default_gaps"][0]
    assert gap["entry"] == "routed_output"
    assert gap["ratio"] == pytest.approx(2.5)
    assert [row["entry"] for row in health["slowest_fallbacks"]] == ["readout_rows", "stream_mix_down"]
    assert health["slowest_fallbacks"][0]["reason"] == "Expired"
    assert health["slowest_fallbacks"][0]["weighted_ms"] == pytest.approx(100.0)

    md = report.markdown({**base, "tuning_health": health})
    assert "WARNING: High tuning fallback rate" in md
    assert "2/3 (66.7%)" in md
    assert "`routed_output [M=2..9]`" in md
    assert "`readout_rows [M=1..2]`" in md
    alert = report.tuning_health_alert(health)
    assert len(alert) == 2
    assert "WARNING 2/3 (66.7%)" in alert[0]
    assert "routed_output [M=2..9]" in alert[1] and "2.50×" in alert[1]
