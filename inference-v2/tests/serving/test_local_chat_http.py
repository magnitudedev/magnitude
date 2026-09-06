"""Opt-in real model gate through the maintained Python session benchmark client."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from threading import Event
from time import monotonic

import httpx
import psutil
import pytest

from session_bench.client import measure
from session_bench.sessions import ExpectedCall, Request


@pytest.mark.model
def test_real_qwen_mtp_http_tool_call_passes_session_benchmark_transport(tmp_path):
    target = os.environ.get("MAGNITUDE_TEST_MTP_TARGET")
    head = os.environ.get("MAGNITUDE_TEST_MTP_HEAD")
    if target is None or head is None:
        pytest.skip("set MAGNITUDE_TEST_MTP_TARGET and MAGNITUDE_TEST_MTP_HEAD to local artifacts")
    run_http_gate(tmp_path, target, head)


@pytest.mark.model
def test_real_resident_upstream_http_tool_call_passes_session_benchmark_transport(tmp_path):
    target = os.environ.get("MAGNITUDE_TEST_UPSTREAM_TARGET")
    if target is None:
        pytest.skip("set MAGNITUDE_TEST_UPSTREAM_TARGET to a local artifact")
    run_http_gate(tmp_path, target, None)


def run_http_gate(tmp_path, target, head):
    evidence = Path(os.environ.get("MAGNITUDE_TEST_HTTP_EVIDENCE", str(tmp_path / "http.json")))
    assert not evidence.exists(), "refusing to overwrite existing HTTP evidence"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"http://127.0.0.1:{port}"
    logfile = tmp_path / "server.log"
    with logfile.open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "magnitude_engine.serving",
                "--target",
                target,
                *([] if head is None else ["--head", head]),
                "--model",
                "qwen-http-gate",
                "--context-tokens",
                "34816",
                "--memory-bytes",
                str(48 << 30),
                "--output-capacity",
                "8",
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        workers = []
        try:
            deadline = monotonic() + 80
            with httpx.Client(base_url=address, timeout=1) as client:
                while True:
                    assert server.poll() is None, logfile.read_text()
                    assert monotonic() < deadline, logfile.read_text()
                    try:
                        response = client.get("/health")
                        if response.status_code == 200:
                            readiness = response.json()
                            break
                    except httpx.RequestError:
                        pass
                    Event().wait(0.1)
            workers = psutil.Process(server.pid).children()
            assert len(workers) == 1 and "magnitude_engine.worker" in workers[0].cmdline()
            assert readiness["speculative_backend"] == ("none" if head is None else "mtp")
            if head is None:
                assert readiness["program_implementation"].startswith("mlx_vlm.")
            request = Request(
                id="qwen-http-tool-gate", section="single", session="http-gate",
                checkpoint=0, fixture_id="weather",
                messages=[{"role": "user", "content":
                           "Get the current weather in San Francisco in celsius."}],
                tools=[{"type": "function", "function": {
                    "name": "get_weather", "description": "Look up current weather in a city.",
                    "parameters": {"type": "object", "properties": {
                        "city": {"type": "string", "enum": ["San Francisco"]},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    }, "required": ["city", "unit"], "additionalProperties": False},
                }}],
                expected=[ExpectedCall(name="get_weather", arguments={
                    "city": ["San Francisco"], "unit": ["celsius"],
                })],
            )
            events = []

            async def submit():
                async with httpx.AsyncClient(timeout=60) as client:
                    return await asyncio.wait_for(measure(
                        client, address, "qwen-http-gate", request, events.append,
                    ), timeout=65)

            result = asyncio.run(submit())
            evidence.write_text(json.dumps({
                "target": target, "head": head, "readiness": readiness,
                "request": request.model_dump(), "result": result.model_dump(),
                "events": events,
            }, indent=2))
            assert result.outcome == "valid", result
            assert result.finish_reason == "tool_calls"
            assert result.terminal is not None
            assert result.terminal["timings"].get("speculative_backend") == (
                "mtp" if head is not None else None
            )
            print({"evidence": str(evidence), "tool_calls": result.tool_calls,
                   "terminal": result.terminal, "server_pid": server.pid})
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=3)
    # Uvicorn deliberately re-raises a received termination signal after its lifespan
    # cleanup. Process return code alone is not proof that the model worker retired.
    assert server.returncode in (0, -signal.SIGTERM), logfile.read_text()
    gone, alive = psutil.wait_procs(workers, timeout=3)
    assert len(gone) == 1 and not alive, "HTTP shutdown left its model worker alive"
