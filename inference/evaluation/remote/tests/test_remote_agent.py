"""The lock agent, run as a real subprocess: lock, holder, heartbeat loss, EOF, max hold."""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

AGENT = Path(__file__).parents[1] / "src" / "magnitude_remote" / "host" / "agent.py"


class Agent:
    def __init__(self, root: Path, *, run="run-1", max_hold=60.0, heartbeat_timeout=2.0):
        self.process = subprocess.Popen(
            [sys.executable, str(AGENT), "hold", "--root", str(root), "--who", "tester",
             "--run", run, "--max-hold", str(max_hold), "--heartbeat-timeout", str(heartbeat_timeout)],
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

    def expect(self, kind: str, timeout=10.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            event = self.events.get(timeout=max(0.01, deadline - time.monotonic()))
            assert event is not None, f"agent exited before {kind}"
            if event["event"] == kind:
                return event

    def finish(self, timeout=10.0) -> int:
        return self.process.wait(timeout)


def status(root: Path) -> dict:
    output = subprocess.run(
        [sys.executable, str(AGENT), "status", "--root", str(root)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(output)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_lock_holder_and_command_output(tmp_path):
    agent = Agent(tmp_path)
    locked = agent.expect("locked")
    holder = json.loads((tmp_path / "holder.json").read_text())
    assert holder == locked["holder"]
    assert holder["who"] == "tester" and holder["run"] == "run-1" and holder["since"]
    assert status(tmp_path)["locked"] is True
    script = "import sys; print(sys.stdin.read().upper()); sys.exit(7)"
    agent.send(op="run", argv=[sys.executable, "-c", script], cwd=str(tmp_path), env={}, input="hi")
    assert agent.expect("output")["text"] == "HI\n"
    assert agent.expect("exited")["code"] == 7
    agent.send(op="release")
    assert agent.expect("stopped")["reason"] == "released"
    assert agent.finish() == 0
    assert not (tmp_path / "holder.json").exists()
    assert status(tmp_path) == {
        "root": str(tmp_path), "locked": False, "holder": None, "stale_holder": None, "queue": []
    }
    assert os.listdir(tmp_path / "queue") == []


def test_second_agent_waits_and_sees_holder(tmp_path):
    first = Agent(tmp_path, run="first")
    first.expect("locked")
    second = Agent(tmp_path, run="second")
    waiting = second.expect("waiting")
    assert waiting["holder"]["run"] == "first"
    first.send(op="release")
    assert first.finish() == 0
    assert second.expect("locked")["holder"]["run"] == "second"
    second.send(op="release")
    assert second.finish() == 0


def _start_sleeper(agent: Agent, tmp_path: Path) -> int:
    pid_file = tmp_path / "child.pid"
    script = (
        "import os, subprocess, sys, time; "
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
        f"open({str(pid_file)!r}, 'w').write(f'{{os.getpid()}} {{grandchild.pid}}'); time.sleep(600)"
    )
    agent.send(op="run", argv=[sys.executable, "-c", script], cwd=str(tmp_path), env={}, input=None)
    deadline = time.monotonic() + 10
    while not pid_file.exists() or len(pid_file.read_text().split()) != 2:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    return [int(p) for p in pid_file.read_text().split()]


def _assert_dead(pids, timeout=15.0):
    deadline = time.monotonic() + timeout
    while any(alive(pid) for pid in pids):
        assert time.monotonic() < deadline, "work outlived the lock"
        time.sleep(0.05)


@pytest.mark.parametrize("ending", ["heartbeat", "eof"])
def test_controller_loss_kills_work_and_releases_lock(tmp_path, ending):
    agent = Agent(tmp_path, heartbeat_timeout=2.0)
    agent.expect("locked")
    pids = _start_sleeper(agent, tmp_path)
    if ending == "eof":
        agent.process.stdin.close()
        expected = "controller closed the session"
    else:
        expected = "no heartbeat"
    stopped = agent.expect("stopped", timeout=15)
    assert stopped["reason"].startswith(expected)
    assert agent.finish() == 2
    _assert_dead(pids)
    assert not (tmp_path / "holder.json").exists()
    assert status(tmp_path)["locked"] is False


def test_heartbeats_keep_the_lock(tmp_path):
    agent = Agent(tmp_path, heartbeat_timeout=1.5)
    agent.expect("locked")
    for _ in range(4):
        time.sleep(0.8)
        agent.send(op="heartbeat")
    assert status(tmp_path)["locked"] is True
    agent.send(op="release")
    assert agent.finish() == 0


def test_max_hold_backstop(tmp_path):
    agent = Agent(tmp_path, max_hold=1.0, heartbeat_timeout=60)
    agent.expect("locked")
    pids = _start_sleeper(agent, tmp_path)
    assert agent.expect("stopped", timeout=15)["reason"].startswith("maximum hold time")
    assert agent.finish() == 2
    _assert_dead(pids)
    assert status(tmp_path)["locked"] is False


def test_killed_agent_releases_lock_and_reports_stale_holder(tmp_path):
    agent = Agent(tmp_path)
    agent.expect("locked")
    agent.process.kill()
    agent.process.wait()
    report = status(tmp_path)
    assert report["locked"] is False
    assert report["stale_holder"]["run"] == "run-1"


def test_waiters_take_the_lock_in_arrival_order(tmp_path):
    first = Agent(tmp_path, run="first", heartbeat_timeout=60)
    first.expect("locked")
    second = Agent(tmp_path, run="second", heartbeat_timeout=60)
    assert second.expect("waiting")["ahead"] == 0
    third = Agent(tmp_path, run="third", heartbeat_timeout=60)
    assert third.expect("waiting")["ahead"] == 1
    assert [w["run"] for w in status(tmp_path)["queue"]] == ["second", "third"]
    first.send(op="release")
    assert first.finish() == 0
    assert second.expect("locked")["holder"]["run"] == "second"
    assert third.expect("waiting")["ahead"] == 0
    second.send(op="release")
    assert second.finish() == 0
    assert third.expect("locked")["holder"]["run"] == "third"
    third.send(op="release")
    assert third.finish() == 0


def test_dead_waiters_never_block_the_queue(tmp_path):
    first = Agent(tmp_path, run="first", heartbeat_timeout=60)
    first.expect("locked")
    crashed = Agent(tmp_path, run="crashed", heartbeat_timeout=60)
    crashed.expect("waiting")
    later = Agent(tmp_path, run="later", heartbeat_timeout=60)
    assert later.expect("waiting")["ahead"] == 1
    crashed.process.kill()  # a waiter that dies without withdrawing its ticket
    crashed.process.wait()
    # A stalled waiter: a live process whose ticket has not been touched for too long.
    stale = tmp_path / "queue" / "000000000000-1.json"
    stale.write_text(json.dumps({"who": "x", "run": "stalled", "pid": os.getpid()}))
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    assert later.expect("waiting")["ahead"] == 0
    first.send(op="release")
    assert first.finish() == 0
    assert later.expect("locked")["holder"]["run"] == "later"
    assert os.listdir(tmp_path / "queue") == []
    later.send(op="release")
    assert later.finish() == 0


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
