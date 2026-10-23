"""The engine end to end on local hosts (`conftest.rig`): placement, phases and holds, builds
in the checkout's own target, jobs recorded and fetched, failures with their logs, and the
report a run writes when asked; and how a caller follows a run to its end or its estimate."""

import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from magnitude_rig import cli, engine, results
from magnitude_rig.connection import AgentSession, connect


def request(source, *, builds=(), jobs=(), device="device", estimate=120.0, host="pool"):
    side = engine.Side("source", source, tuple(builds))
    return engine.Request(
        instrument="test", purpose="settles a test", estimate=estimate, backend="cpu",
        sides=(side,), jobs=tuple(engine.Job("source", name, step) for name, step in jobs),
        device=device, host=host,
    )


PY = sys.executable


def test_a_run_builds_in_its_checkout_runs_recorded_and_fetches(rig):
    hosts, source, tmp_path = rig
    build = engine.Step("build", (PY, "-c", "import os, sys; os.makedirs(sys.argv[1], exist_ok=True); "
                                            "open(sys.argv[1] + '/built', 'w').write('x')", "{target}"))
    job = engine.Step("job", (PY, "-c", "import os, sys; print(open('data.txt').read().strip()); "
                                        "open(os.path.join(sys.argv[1], 'result.txt'), 'w').write(sys.argv[2])",
                              "{run_dir}", "{target}"))
    lines = []
    outcome = engine.execute(request(source, builds=[build], jobs=[("job", job)]), engine.new_run_id("test"), hosts=hosts, announce=lines.append)
    assert outcome.codes == {"job": 0}
    assert lines[0].startswith("mrig test · cpu → alpha · estimate 2m00s · holders: none · target: none")
    checkout = tmp_path / outcome.host / "checkouts" / "test-checkout"
    assert (checkout / "target" / "built").exists() and (checkout / "src" / "data.txt").exists()
    assert (outcome.results / "job" / "result.txt").read_text() == str(checkout / "target")
    assert (outcome.results / "job" / "host.json").exists()  # recorded with the host's facts
    assert "the source" in (outcome.results / "logs" / "job-job.log").read_text()
    record = json.loads((outcome.results / "rig.json").read_text())
    assert [p["phase"] for p in record["phases"]] == ["connect", "queue-mirror-source", "sync-source", "queue-build", "build-source", "queue-device", "job-job", "fetch"]
    assert record["purpose"] == "settles a test" and record["failed"] is None
    assert int((checkout / "size").read_text()) > 0  # recorded after the build from cold
    assert not (tmp_path / outcome.host / "runs" / outcome.run_id).exists()


def test_placement_takes_the_member_that_frees_first(rig):
    hosts, source, _ = rig
    busy = AgentSession(connect(hosts.host("alpha")), who="other", run="busy", purpose="p", estimate=3600)
    try:
        busy.hold("device", lambda blocking: None)
        job = engine.Step("job", (PY, "-c", "print('ok')"))
        lines = []
        outcome = engine.execute(request(source, jobs=[("job", job)]), engine.new_run_id("test"), hosts=hosts, announce=lines.append)
        assert outcome.host == "beta"
    finally:
        busy.close()


def test_a_failed_build_names_its_phase_and_keeps_its_log(rig):
    hosts, source, _ = rig
    build = engine.Step("build", (PY, "-c", "import sys; print('src/lib.rs:1:1: error: no'); sys.exit(3)"))
    with pytest.raises(engine.RunFailed) as failed:
        engine.execute(request(source, builds=[build], jobs=[("job", engine.Step("job", (PY, "-c", "")))]),
                       engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert failed.value.phase == "build-source"
    log = failed.value.outcome.results / "logs" / "build-source.log"
    assert "error: no" in log.read_text()
    assert json.loads((failed.value.outcome.results / "rig.json").read_text())["failed"]["phase"] == "build-source"


def test_a_run_writes_its_report_when_asked(rig):
    hosts, source, _ = rig
    job = engine.Step("job", (PY, "-c", "import time; print('working', flush=True); time.sleep(4)"))
    run_id = engine.new_run_id("test")
    report = engine.RESULTS / run_id / engine.REPORT

    def ask():
        while not (engine.RESULTS / run_id / "logs" / "job-job.log").exists():
            time.sleep(0.1)
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGUSR1)

    threading.Thread(target=ask, daemon=True).start()
    engine.execute(request(source, jobs=[("job", job)], estimate=2.0), run_id, hosts=hosts, announce=lambda line: None)
    text = report.read_text()
    assert "estimate 2s" in text and "phases:" in text and "job-job" in text
    assert "'working'" in text and "mrig stop" in text


def test_a_job_failure_is_its_exit_code_not_a_failed_run(rig):
    hosts, source, _ = rig
    job = engine.Step("job", (PY, "-c", "import sys; sys.exit(5)"))
    outcome = engine.execute(request(source, jobs=[("job", job)]), engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert outcome.codes == {"job": 5}


def test_this_machine_is_a_target_like_any_other(rig):
    hosts, source, tmp_path = rig
    (tmp_path / "bin" / "ssh").write_text("#!/bin/sh\nexit 99\n")  # this machine needs no ssh
    job = engine.Step("job", (PY, "-c", "print(open('data.txt').read().strip())"))
    outcome = engine.execute(request(source, jobs=[("job", job)], host="local"), engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert outcome.host == "local" and outcome.codes == {"job": 0}
    assert (tmp_path / "local" / "checkouts" / source.key / "src" / "data.txt").exists()
    assert "the source" in (outcome.results / "logs" / "job-job.log").read_text()


FAKE_RUN = """
import os, signal, sys, time
results = sys.argv[1]
def report(signum, frame):
    open(os.path.join(results, "report.txt"), "w").write("mrig: fake at 3s, estimate 1s\\n  phases: job-job 3s\\n")
signal.signal(signal.SIGUSR1, report)
print("mrig fake · cpu → local", flush=True)
time.sleep(float(sys.argv[2]))
print("the answer", flush=True)
open(os.path.join(results, "mrig.code"), "w").write("0")
"""


def _fake_run(tmp_path, seconds: float):
    directory = tmp_path / "20261023T000000Z-fake-000000"
    directory.mkdir()
    with open(directory / engine.TRANSCRIPT, "w") as transcript:
        child = subprocess.Popen([PY, "-c", FAKE_RUN, str(directory), str(seconds)], stdout=transcript)
    return directory, child


def test_a_run_past_its_estimate_returns_its_report_and_what_to_do(tmp_path, capsys):
    directory, child = _fake_run(tmp_path, 30)
    time.sleep(0.5)
    code = results.follow(directory.name, directory, child.pid, 1)
    child.kill()
    output = capsys.readouterr().out
    assert code == results.OVERRUN
    assert "mrig fake" in output and "phases: job-job 3s" in output
    assert f"mrig stop {directory.name}" in output and "--justification" in output
    assert "passed this wait's estimate (1s)" in output


def test_a_run_that_ends_first_returns_its_answer_and_code(tmp_path, capsys):
    directory, child = _fake_run(tmp_path, 1)
    code = results.follow(directory.name, directory, child.pid, 30)
    child.wait()
    assert code == 0 and "the answer" in capsys.readouterr().out


def test_waiting_records_its_estimate_and_justification(rig, capsys):
    directory = engine.RESULTS / "20261023T000000Z-fake-000001"
    directory.mkdir(parents=True)
    (directory / engine.TRANSCRIPT).write_text("the answer\n")
    (directory / engine.CODE).write_text("2")
    code = cli.main(["wait", "fake-000001", "--estimate", "10m", "--justification", "a cold build of 40 crates"])
    assert code == 2 and "the answer" in capsys.readouterr().out
    assert engine.waits(directory)[0]["justification"] == "a cold build of 40 crates"
    assert engine.waits(directory)[0]["estimate_seconds"] == 600
