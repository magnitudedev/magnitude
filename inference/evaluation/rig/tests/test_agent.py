"""The host agent, run as real subprocesses: which holds conflict, arrival order, liveness,
the hold order, eviction and the queue estimates other runs see."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

AGENT = Path(__file__).parents[1] / "src" / "magnitude_rig" / "host" / "agent.py"


class Session:
    def __init__(self, root: Path, run: str, *, estimate=60.0, heartbeat_timeout=30.0, agent=AGENT):
        self.process = subprocess.Popen(
            [sys.executable, str(agent), "session", "--root", str(root), "--who", "tester",
             "--run", run, "--purpose", f"purpose of {run}", "--estimate", str(estimate),
             "--heartbeat-timeout", str(heartbeat_timeout)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self.events: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            self.events.put(json.loads(line))
        self.events.put(None)

    def send(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

    def expect(self, event: str, timeout=10.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            found = self.events.get(timeout=max(0.01, deadline - time.monotonic()))
            assert found is not None, f"agent exited before {event}"
            if found["event"] == event:
                return found

    def hold(self, kind: str, timeout=10.0) -> None:
        self.send(op="hold", kind=kind)
        self.expect("held", timeout)

    def blocked(self, kind: str) -> list:
        self.send(op="hold", kind=kind)
        return self.expect("waiting")["blocking"]

    def end(self) -> int:
        self.send(op="end")
        assert self.expect("stopped")["reason"] == "ended"
        return self.process.wait(10)


def status(root: Path, checkout: str | None = None) -> dict:
    argv = [sys.executable, str(AGENT), "status", "--root", str(root)]
    if checkout:
        argv += ["--checkout", checkout]
    return json.loads(subprocess.run(argv, capture_output=True, text=True, check=True).stdout)


def test_builds_share_the_host_and_devices_take_turns(tmp_path):
    first, second = Session(tmp_path, "first"), Session(tmp_path, "second")
    first.hold("build")
    second.hold("build")
    for session in (first, second):
        session.send(op="release", kind="build")
        session.expect("released")
    first.hold("device")
    assert [b["run"] for b in second.blocked("device")] == ["first"]
    assert first.end() == 0
    second.expect("held")
    assert second.end() == 0


def test_a_measurement_waits_for_builds_and_new_builds_wait_behind_it(tmp_path):
    builder, measurer, later = Session(tmp_path, "builder"), Session(tmp_path, "measurer"), Session(tmp_path, "later")
    builder.hold("build")
    assert [b["run"] for b in measurer.blocked("timed")] == ["builder"]
    # Arrival order: a build requested after the waiting measurement waits behind it.
    assert [b["run"] for b in later.blocked("build")] == ["measurer"]
    builder.end()
    measurer.expect("held")
    measurer.end()
    later.expect("held")
    later.end()


def test_runs_of_one_checkout_share_it_and_mirror_it_one_at_a_time(tmp_path):
    first, second, other = Session(tmp_path, "first"), Session(tmp_path, "second"), Session(tmp_path, "other")
    first.hold("checkout:a")
    second.hold("checkout:a")
    assert (tmp_path / "checkouts" / "a" / "used").exists()
    first.hold("mirror:a")
    other.hold("mirror:b")
    assert [b["run"] for b in second.blocked("mirror:a")] == ["first"]
    first.end()
    second.expect("held")
    second.end()
    other.end()


def test_device_holds_take_the_hosts_device_lock(tmp_path):
    session = Session(tmp_path, "run")
    session.hold("device")
    probe = subprocess.run(
        [sys.executable, "-c",
         "import fcntl, sys; f = open(sys.argv[1], 'a+b'); "
         "fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)", str(tmp_path / "lock")],
        capture_output=True,
    )
    assert probe.returncode != 0
    session.end()


def test_a_device_is_never_requested_while_holding_build(tmp_path):
    session = Session(tmp_path, "run")
    session.hold("build")
    session.send(op="hold", kind="timed")
    assert "while holding" in session.expect("error")["message"]


def test_a_dead_session_holds_nothing(tmp_path):
    dead, waiting = Session(tmp_path, "dead"), Session(tmp_path, "waiting")
    dead.hold("device")
    assert [b["run"] for b in waiting.blocked("device")] == ["dead"]
    dead.process.kill()
    dead.process.wait()
    waiting.expect("held")
    assert [e["run"] for e in status(tmp_path)["entries"]] == ["waiting"]
    waiting.end()


def test_a_hold_of_another_protocol_conflicts_with_everything(tmp_path):
    other = tmp_path / "agent-next.py"
    current = int(re.search(r"\nPROTOCOL = (\d+)\n", AGENT.read_text())[1])
    other.write_text(AGENT.read_text().replace(f"\nPROTOCOL = {current}\n", f"\nPROTOCOL = {current + 1}\n"))
    root = tmp_path / "root"
    newer, ours = Session(root, "newer", agent=other), Session(root, "ours")
    newer.hold("build")
    assert [b["run"] for b in ours.blocked("build")] == ["newer"]
    newer.end()
    ours.expect("held")
    ours.end()


@pytest.mark.parametrize("ending", ["heartbeat", "eof"])
def test_controller_loss_kills_work_and_withdraws_holds(tmp_path, ending):
    session = Session(tmp_path, "run", heartbeat_timeout=2.0)
    session.hold("device")
    pid_file = tmp_path / "child.pid"
    script = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    session.send(op="run", argv=[sys.executable, "-c", script], cwd=str(tmp_path), env={}, input=None)
    deadline = time.monotonic() + 10
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    if ending == "eof":
        session.process.stdin.close()
    stopped = session.expect("stopped", timeout=15)
    assert stopped["reason"].startswith("controller closed" if ending == "eof" else "no heartbeat")
    assert session.process.wait(10) == 2
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 15
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, "work outlived its session"
        time.sleep(0.05)
    assert status(tmp_path)["entries"] == []


def test_kill_ends_the_running_command_and_the_session_goes_on(tmp_path):
    session = Session(tmp_path, "run")
    session.send(op="run", argv=[sys.executable, "-c", "import time; time.sleep(600)"], cwd=str(tmp_path), env={}, input=None)
    time.sleep(0.5)
    session.send(op="kill")
    assert session.expect("exited")["code"] != 0
    session.expect("killed")
    session.send(op="run", argv=[sys.executable, "-c", "print('after')"], cwd=str(tmp_path), env={}, input=None)
    assert session.expect("output")["text"] == "after\n"
    session.end()


def test_command_output_and_exit_code(tmp_path):
    session = Session(tmp_path, "run")
    script = "import sys; print(sys.stdin.read().upper()); sys.exit(7)"
    session.send(op="run", argv=[sys.executable, "-c", script], cwd=str(tmp_path), env={}, input="hi")
    assert session.expect("output")["text"] == "HI\n"
    assert session.expect("exited")["code"] == 7
    session.end()


def test_status_estimates_the_wait_from_the_holders_own_estimates(tmp_path):
    holder = Session(tmp_path, "holder", estimate=600)
    holder.hold("device")
    report = status(tmp_path, checkout="mine")
    (entry,) = report["entries"]
    assert entry["purpose"] == "purpose of holder" and entry["state"] == "held"
    assert 590 < report["waits"]["device"] <= 600 and report["waits"]["build"] == 0
    assert report["checkout"] == {"present": False, "size": 0}
    holder.end()


def test_eviction_frees_the_largest_target_from_the_least_recently_used(tmp_path):
    base = tmp_path / "checkouts"
    for index, key in enumerate(("old", "held", "new")):
        (base / key / "target").mkdir(parents=True)
        (base / key / "used").write_text("x")
        os.utime(base / key / "used", (1000 + index, 1000 + index))
    (base / "new" / "size").write_text(str(10**18))  # never fits: evict all that may go
    holding, evicting = Session(tmp_path, "holding"), Session(tmp_path, "evicting")
    holding.hold("checkout:held")
    evicting.send(op="evict", keep="new")
    assert evicting.expect("evicted")["removed"] == ["old"]
    assert sorted(p.name for p in base.iterdir()) == ["held", "new"]
    holding.end()
    evicting.end()


def test_prune_keeps_only_manifest(tmp_path):
    for path in ("keep/a.txt", "keep/sub/b.txt", "drop/c.txt", "keep/d.txt", "e.txt"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(path)
    (tmp_path / "empty").mkdir()
    (tmp_path / "keep" / "link").symlink_to(tmp_path / "drop")
    manifest = "keep/a.txt\nkeep/sub/b.txt\ne.txt\n"
    result = subprocess.run(
        [sys.executable, str(AGENT), "prune", "--dir", str(tmp_path)],
        input=manifest, capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == {"removed_files": 3}
    remaining = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert remaining == ["e.txt", "keep", "keep/a.txt", "keep/sub", "keep/sub/b.txt"]
