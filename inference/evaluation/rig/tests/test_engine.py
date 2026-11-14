"""The engine end to end on local hosts (`conftest.rig`): placement, phases and holds, builds
in the checkout's own target, jobs recorded and fetched, failures with their logs, and the
report a run writes when asked; and how a caller follows a run to its end or its estimate."""

import argparse
import contextlib
import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from conftest import link_down
from waiting import until_exists
from magnitude_rig import cli, engine, results, sources, state, store, sync
from magnitude_rig.build import identify
from magnitude_rig.connection import (
    CommandFailed, HostRun, HostUnavailable, _json_line, connect, host_status, run_state,
)
from magnitude_rig.host import agent
from magnitude_rig.instruments import Answer


def request(source, *, builds=(), jobs=(), device="device", estimate=120.0, host="pool", files=()):
    work = engine.Work(
        sides=(engine.Side("source", source, tuple(builds)),),
        jobs=tuple(engine.Job("source", name, step) for name, step in jobs), device=device, files=files,
    )
    return engine.Request(instrument="test", purpose="settles a test", estimate=estimate, backend="cpu",
                          work=lambda alias: work, host=host)


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
    assert "holders: none · target: no free checkout (cold build) · room " in lines[0]
    assert lines[0].startswith(f"mrig test · cpu → {outcome.host} · staging source test-branch@ccccccccc (clean) from "
                               f"{tmp_path / 'test-worktree'} · estimate 2m00s")
    staged = json.loads((outcome.results / "rig.json").read_text())["sources"]["source"]
    assert staged["branch"] == "test-branch" and staged["described"] == "test-branch@ccccccccc (clean)"
    assert staged["tree"] == str(tmp_path / "test-worktree") and staged["commit"] == "c" * 40
    checkout = tmp_path / outcome.host / "checkouts" / "slot-0"
    assert (checkout / "target" / "built").exists() and (checkout / "src" / "data.txt").exists()
    # The job read the build's output, not the checkout, and the output went with the run.
    output = Path((outcome.results / "job" / "result.txt").read_text()).parent
    assert output.parent == tmp_path / outcome.host / "outputs" and not output.exists()
    assert (outcome.results / "job" / "host.json").exists()  # recorded with the host's facts
    assert "the source" in (outcome.results / "logs" / "job-job.log").read_text()
    record = json.loads((outcome.results / "rig.json").read_text())
    assert [p["phase"] for p in record["phases"]] == ["connect", "stage", "queue-install-source", "install-source", "queue-build", "build-source", "queue-device", "job-job", "fetch"]
    assert record["purpose"] == "settles a test" and record["failed"] is None
    assert int((checkout / "size").read_text()) > 0  # recorded after the build
    (measured,) = (tmp_path / outcome.host / "rig" / "builds").glob("*.json")
    assert json.loads(measured.read_text())["size"] > 0  # what the cold build added: its size
    engine.complete(outcome.run_id, hosts)
    assert not (tmp_path / outcome.host / "runs" / outcome.run_id).exists()


def test_a_run_builds_in_the_checkout_nearest_its_source(rig):
    """The checkout a source last built is found again; a source new to the host builds in the
    one built last, and gets its own tree there."""
    hosts, source, tmp_path = rig
    show = engine.Step("job", (PY, "-c", "print(open('data.txt').read().strip())"))
    lines = []
    for _ in range(2):
        engine.execute(request(source, jobs=[("job", show)], host="alpha"), engine.new_run_id("test"), hosts=hosts,
                       announce=lines.append)
    assert "target: checkout slot-0, last built from this source · room " in lines[1]
    (source.directory / "data.txt").write_text("another source\n")
    other = dataclasses.replace(source, key="other-checkout")
    outcome = engine.execute(request(other, jobs=[("job", show)], host="alpha"), engine.new_run_id("test"),
                             hosts=hosts, announce=lines.append)
    assert "target: checkout slot-0, last built from test-checkout (rebuilds what differs) · room " in lines[2]
    assert "another source" in (outcome.results / "logs" / "job-job.log").read_text()
    assert sorted(p.name for p in (tmp_path / "alpha" / "checkouts").iterdir()) == ["slot-0"]


def test_a_run_reads_what_its_caller_staged_and_never_waits_on_it(rig):
    """Everything the caller provides is on the host before the run starts: no operation of the
    plan waits on the caller, the jobs read the request's files, and the fetch leaves the
    stages on the host to go with the run's directory."""
    hosts, source, tmp_path = rig
    (tmp_path / "prompt.txt").write_text("the input\n")
    job = engine.Step("job", (PY, "-c", "import sys; print(open(sys.argv[1]).read().strip())", "{inputs}/prompt.txt"))
    run = request(source, jobs=[("job", job)], host="alpha", files=(tmp_path / "prompt.txt",))
    work = run.work("alpha")
    candidate = engine.Candidate(connect(hosts.host("alpha")), {}, work,
                                 {engine.build_key(work, side): 0 for side in work.sides}, 0.0, (60.0,))
    ops = engine.plan_for(run, candidate, "planned").ops
    assert {op["op"] for op in ops} == {"phase", "checkout", "hold", "release", "evict", "run", "built"}
    outcome = engine.execute(run, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert outcome.codes == {"job": 0} and "the input" in (outcome.results / "logs" / "job-job.log").read_text()
    assert (outcome.results / "inputs" / "prompt.txt").exists()
    assert not list(outcome.results.glob("stage-*"))


def test_a_timed_run_gates_with_its_hosts_temperature_setting(rig):
    """A timed run plans the thermal gate before its jobs, passing whether its host has
    temperature sensors."""
    hosts, source, _ = rig
    run = request(source, jobs=[("job", engine.Step("job", (PY, "-c", "pass")))], device="timed", host="alpha")
    work = run.work("alpha")
    candidate = engine.Candidate(connect(hosts.host("alpha")), {}, work,
                                 {engine.build_key(work, side): 0 for side in work.sides}, 0.0, (60.0,))
    ops = engine.plan_for(run, candidate, "planned").ops
    (gate,) = [op for op in ops if op["op"] == "run" and "gate" in op["argv"]]
    argv = list(gate["argv"])
    assert argv[argv.index("--temperatures") + 1] == str(candidate.remote.temperatures)


def test_a_side_releases_its_checkout_as_its_build_ends_unless_a_job_of_it_reads_the_checkout(rig):
    """A released side's jobs run from its output, without cargo's target or the recording
    wrapper, and naming its checkout is a planning error; a job that reads the checkout keeps it."""
    hosts, source, _ = rig

    def ops(job: engine.Step) -> list:
        run = request(source, jobs=[("job", job)], host="alpha")
        work = run.work("alpha")
        candidate = engine.Candidate(connect(hosts.host("alpha")), {}, work,
                                     {engine.build_key(work, side): 0 for side in work.sides}, 0.0, (60.0,))
        return engine.plan_for(run, candidate, "planned").ops

    released = ops(engine.Step("job", (PY, "{target}/tool")))
    (built,) = [op for op in released if op["op"] == "built"]
    (job,) = [op for op in released if op.get("job") == "job"]
    assert built["release"] and job["cwd"] == "{output:source}/src" and "{output:source}/target/tool" in job["argv"]
    assert engine.COMPILER not in job["env"] and "CARGO_TARGET_DIR" not in job["env"]
    with pytest.raises(ValueError, match="names {checkout}"):
        ops(engine.Step("job", (PY, "{checkout}/x")))
    kept = ops(engine.Step("job", (PY, "{checkout}/x"), checkout=True))
    (built,) = [op for op in kept if op["op"] == "built"]
    (job,) = [op for op in kept if op.get("job") == "job"]
    assert not built["release"] and "{checkout:source}/x" in job["argv"]
    assert job["env"]["CARGO_TARGET_DIR"] == "{checkout:source}/target"


def test_placement_plans_each_candidate_and_waits_for_the_hold_of_its_own_work(rig):
    """The work is planned per host: a host it cannot be planned for is refused with why, and
    each candidate is ranked by the hold its own work waits for."""
    hosts, source, _ = rig
    job = engine.Step("job", (PY, "-c", "print('ok')"))

    def work(device: str) -> engine.Work:
        return engine.Work(sides=(engine.Side("source", source, ()),), jobs=(engine.Job("source", "job", job),),
                           device=device)

    def refusing(alias: str) -> engine.Work:
        raise engine.Refused(f"no fact of {alias} is recorded")

    with pytest.raises(engine.HostError, match="alpha: no fact of alpha is recorded"):
        engine.place(engine.Request("test", "p", 60.0, "cpu", refusing, host="pool"), hosts, "run")
    busy = [_holding(connect(hosts.host(alias)), f"busy-{alias}", "device") for alias in ("alpha", "beta")]
    try:
        # Both devices are held: on alpha the work waits for no device (its builds alone), on
        # beta for the device, so it starts first on alpha.
        by_host = {"alpha": work(None), "beta": work("device")}
        request = engine.Request("test", "p", 60.0, "cpu", lambda alias: by_host[alias], host="pool")
        placed = engine.place(request, hosts, "run")
        assert placed.remote.alias == "alpha" and placed.work is by_host["alpha"]
    finally:
        for holder in busy:
            holder.stop()


def _staged(hosts, alias: str, run: str) -> agent.Session:
    """A live run on `alias` that has no entry yet: staged, its build still to come."""
    return agent.Session(agent.Rig(hosts.host(alias).root), {
        "who": "other", "run": run, "purpose": "p", "estimate": 600.0, "started": time.time(),
        "protocol": agent.PROTOCOL,
    })


def test_runs_placed_together_spread_across_equal_hosts_and_a_pin_holds(rig):
    """Three runs placed onto three equal idle hosts, each staged before the next is placed,
    land on three hosts: each sees the builds the others owe. A run pinned to a host goes there
    however busy it is."""
    hosts, source, _ = rig
    staged = []
    try:
        for index in range(3):
            run = f"run-{index}"
            placed = engine.place(request(source, host=None), hosts, run)
            staged.append((placed.remote.alias, _staged(hosts, placed.remote.alias, run)))
        assert sorted(alias for alias, _ in staged) == ["alpha", "beta", "local"]
        assert engine.place(request(source, host=staged[0][0]), hosts, "pinned").remote.alias == staged[0][0]
    finally:
        for _, session in staged:
            session.run_lock.close()


def _candidate(alias: str, *, checkouts: list, building: list, builds=(1200.0,), room_wait=0.0):
    status = {"checkouts": checkouts, "building": building,
              "waits": {"build": 0.0, "device": 0.0, "timed": 0.0},
              "device_lock_held_elsewhere": None, "entries": [],
              "free_bytes": 13e9, "room": {"available": 24e9, "owed": 0, "ending": []}}
    return engine.Candidate(types.SimpleNamespace(alias=alias), status, engine.Work(sides=(), device="device"),
                            {}, room_wait, builds)


def test_a_warm_checkout_on_a_busy_host_wins_only_while_it_outweighs_the_contention():
    """A warm build shares the CPU with each owed build for as long as it lasts itself: beside
    two cold builds it still ends before a cold build alone, until its own seconds times three
    pass a cold build's."""
    warm = {"name": "slot-0", "source": "mine", "size": 0, "warmth": agent.SAME, "shared": False}
    idle = _candidate("idle", checkouts=[None], building=[])
    busy = _candidate("busy", checkouts=[warm], building=[1200.0, 1200.0], builds=(60.0,))
    assert engine._start(busy) == 180 and engine._start(idle) == 1200
    assert min([busy, idle], key=engine._start) is busy
    slow = _candidate("busy", checkouts=[warm], building=[1200.0, 1200.0], builds=(500.0,))
    assert engine._start(slow) == 1500
    assert min([slow, idle], key=engine._start) is idle


def test_the_start_line_shows_the_queue_and_each_run_against_its_estimate():
    """Every run ahead is shown, held or queued, with its time left or how far it is past its
    estimate; a pinned run says which idle hosts its `--host` passed over."""
    run = lambda name, state, **timing: {"kind": "device", "who": "w", "run": name, "state": state,
                                         "estimate": 540.0, "queued": 0.0, **timing}
    busy = _candidate("m4-pro-a", checkouts=[None], building=[])
    busy.status["entries"] = [run("long", "held", remaining=1020.0, over=480.0),
                              run("next", "waiting", remaining=300.0, over=0.0),
                              {"kind": "checkout:slot-0", "run": "long", "state": "held"}]
    line = engine.launch_line(request(None, host="m4-pro-a"), busy, ["m4-pro-b"])
    assert ("holders: device long, 8m00s past its 9m00s estimate; "
            "queued device next, 5m00s left of its 9m00s estimate · ") in line
    assert line.endswith("· pinned by --host m4-pro-a to a busy host while m4-pro-b is idle")
    assert "pinned" not in engine.launch_line(request(None, host="m4-pro-a"), busy, [])


def test_a_run_pinned_to_a_busy_host_finds_the_idle_hosts_it_passes_over(rig):
    """`--host` still decides where the run goes; when it names a host that is not idle, the
    capable idle hosts outside it are found for the start line, and none when it is idle or the
    run is placed freely."""
    hosts, source, _ = rig
    busy = _holding(connect(hosts.host("alpha")), "busy-alpha", "device")
    try:
        pinned = request(source, host="alpha")
        placed = engine.place(pinned, hosts, "run")
        assert placed.remote.alias == "alpha"
        assert sorted(engine.idle_elsewhere(pinned, hosts, placed, "run")) == ["beta", "local"]
        free = request(source, host=None)
        assert engine.idle_elsewhere(free, hosts, engine.place(free, hosts, "run"), "run") == []
        idle = request(source, host="beta")
        assert engine.idle_elsewhere(idle, hosts, engine.place(idle, hosts, "run"), "run") == []
    finally:
        busy.stop()


def test_placement_prefers_a_host_with_room_for_the_build():
    """A host whose build must wait for live runs to free room starts it that much later, and
    one where even their ending would not make room never does."""
    roomy = _candidate("roomy", checkouts=[None], building=[])
    full = _candidate("full", checkouts=[None], building=[], room_wait=600.0)
    never = _candidate("never", checkouts=[None], building=[], room_wait=None)
    assert engine._start(full) == 1800 and engine._start(never) == float("inf")
    assert min([full, never, roomy], key=engine._start) is roomy


def test_a_build_needs_its_measured_size_and_a_host_that_cannot_fit_it_ranks_last():
    """A build's need is its own host's measurement of that build, else the largest any host
    measured of it, else of any build: the host measuring none waits by the largest; a host
    whose room even its live runs' ending could not make ranks after every one that can."""
    work = engine.Work(sides=(engine.Side("source", None, (engine.Step("harness", ("cargo", "build", "--release")),)),))
    key = engine.build_key(work, work.sides[0])
    other = engine.Work(sides=(engine.Side("source", None, (engine.Step("check", ("cargo", "check")),)),))
    assert engine.build_key(other, other.sides[0]) != key  # another target or profile, another build
    measured = lambda size, largest: {"measured": {key: {} if size is None else {"size": size}}, "largest": largest}
    assert engine.fallbacks([(work, measured(9e9, 9e9)), (work, measured(None, 30e9)), (work, measured(14e9, 14e9))]) \
        == {key: 14e9}
    assert engine.fallbacks([(work, measured(None, 30e9)), (work, measured(None, 5e9))]) == {key: 30e9}
    assert engine.fallbacks([(work, measured(None, 0))]) == {key: 0}
    # 21 GB free beside a live run that owes 4 GB and frees 12 GB as it ends in 10 minutes.
    room = {"available": 21e9, "owed": 4e9, "ending": [[600.0, 12e9, 4e9]]}
    assert engine.room_wait(room, 14e9) == 0.0
    assert engine.room_wait(room, 20e9) == 600.0
    assert engine.room_wait(room, 40e9) is None
    fits = _candidate("fits", checkouts=[None], building=[], room_wait=engine.room_wait(room, 20e9))
    never = _candidate("never", checkouts=[None], building=[], room_wait=engine.room_wait(room, 40e9))
    fits.status["waits"]["build"] = 7200.0  # however busy the host that fits
    assert min([never, fits], key=engine._start) is fits
    # The start line counts the rig's own checkouts as room: 13 GB free and 11 GB to reclaim.
    assert engine._room(fits.status) == "room 13.0 GB free + 11.0 GB of the rig's own to reclaim"


def test_a_build_unmeasured_on_a_host_takes_its_size_from_hosts_of_its_platform_alone():
    """A Mac that had not measured a `cargo check` must not take a Linux host's size, of other
    targets, and be refused for room it has: a host borrows only its own platform's."""
    work = engine.Work(sides=(engine.Side("source", None, (engine.Step("check", ("cargo", "check")),)),))
    key = engine.build_key(work, work.sides[0])
    host = lambda system, machine: types.SimpleNamespace(system=system, machine=machine)
    status = lambda size, largest: {"measured": {key: {} if size is None else {"size": size}}, "largest": largest}
    known = engine.platform_fallbacks([
        (host("Darwin", "arm64"), status(None, 7.6e9), work),  # measured none of it
        (host("Darwin", "arm64"), status(6.1e9, 6.1e9), work),
        (host("Linux", "x86_64"), status(17.2e9, 17.2e9), work),
    ])
    assert known == {("Darwin", "arm64"): {key: 6.1e9}, ("Linux", "x86_64"): {key: 17.2e9}}


def test_a_build_is_priced_by_the_hosts_measured_builds_and_a_derived_estimate_by_its_start():
    """A build is incremental where the checkout it would claim built it at an ancestor of its
    commit, else cold, and priced by this host's latest build of that class of it, else the
    slowest of that class elsewhere, an incremental one unmeasured as a cold one, else the
    slowest build here. An estimate derived at placement counts the queue ahead and the build."""
    head, parent = sources.commit("HEAD"), sources.commit("HEAD~1")
    unrelated = "f" * 40  # no commit of this repository
    key = "harness"
    build = lambda seconds, start, built: {"seconds": seconds, "from": start, "to": head, "built": built}
    history = [build(1500, None, False), build(900, unrelated, True), build(200, parent, True),
               build(1400, parent, False)]  # cold, near-cold, incremental, first build of it there
    host = lambda builds, checkout: {"checkouts": [checkout], "measured": {key: {"builds": builds}}, "slowest": 1800}
    claims = lambda commit, built: {"commit": commit, "built": built, "shared": False}
    here = host(history, claims(parent, True))
    assert engine.build_seconds(here, 0, key, head, []) == 200  # incremental: the latest incremental
    for cold in (claims(parent, False), claims(unrelated, True), claims(None, True), None):
        assert engine.build_seconds(host(history, cold), 0, key, head, []) == 1400  # the latest cold
    others = [host([build(1300, None, False)], None), host([build(1700, unrelated, True)], None)]
    assert engine.build_seconds(host([], claims(parent, True)), 0, key, head, others) == 1700
    assert engine.build_seconds(host([build(1500, None, False)], claims(parent, True)), 0, key, head, []) == 1500
    assert engine.build_seconds(host([], None), 0, key, head, []) == 1800
    # A checkout live runs of the tree and build hold: what their build still owes, then fresh.
    assert engine.build_seconds(host(history, {"commit": parent, "built": True, "shared": True, "owed": 30}),
                                0, key, head, []) == 30

    placed = _candidate("rtx", checkouts=[None], building=[], builds=(1400.0,))
    placed.status["waits"]["device"] = 2000.0  # a run holding the device for 2000 s more
    derived = engine.Request(instrument="gate", purpose="p", estimate=lambda alias, start: start + 300.0,
                             backend="cuda", work=None)
    assert engine.placed_estimate(derived, placed) == 2000.0 + 300.0
    placed.status["waits"]["device"] = 0.0
    assert engine.placed_estimate(derived, placed) == 1400.0 + 300.0
    assert engine.placed_estimate(dataclasses.replace(derived, estimate=60.0), placed) == 60.0


def test_a_start_whose_reply_was_lost_is_followed_not_started_again(rig, monkeypatch):
    """The first start's runner owns the run but its reply never arrives: the retried start
    finds that runner and the caller follows it. Nothing is staged again, and the job runs once."""
    hosts, source, tmp_path = rig
    monkeypatch.setattr(engine, "BACKOFF_FIRST", 0.1)
    real, lost, staged = HostRun.staging, [], []

    @contextlib.contextmanager
    def losing(cls, remote, run):
        with real(remote, run) as held:
            if isinstance(held, HostRun) or lost:
                yield held
                return

            def start(plan):
                lost.append(held(plan))  # the runner owns the run; its reply is lost
                raise engine.HostUnavailable("alpha: unreachable (the reply was lost)")

            yield start

    stage = engine.Run.stage

    def counted(run, plan, stop):
        staged.append(run.run_id)
        stage(run, plan, stop)

    monkeypatch.setattr(HostRun, "staging", classmethod(losing))
    monkeypatch.setattr(engine.Run, "stage", counted)
    ran = tmp_path / "ran"
    job = engine.Step("job", (PY, "-c", "import sys; open(sys.argv[1], 'a').write('ran\\n')", str(ran)))
    outcome = engine.execute(request(source, jobs=[("job", job)], host="alpha"), engine.new_run_id("test"),
                             hosts=hosts, announce=lambda line: None)
    assert outcome.codes == {"job": 0} and ran.read_text() == "ran\n" and len(staged) == 1
    assert len(lost) == 1 and "the reply was lost" in (outcome.results / "logs" / "caller.log").read_text()
    engine.complete(outcome.run_id, hosts)
    assert not (tmp_path / "alpha" / "runs" / outcome.run_id).exists() and state.Launched.gone() == []


def test_a_run_stopped_while_staging_never_starts(rig, monkeypatch):
    """SIGTERM while the caller stages: the host's hold is withdrawn, no runner starts, and the
    run is recorded as stopped in its stage."""
    hosts, source, tmp_path = rig
    stage = engine.Run.stage

    def stopped(run, plan, stop):
        os.kill(os.getpid(), signal.SIGTERM)  # the caller's stop arrives mid-stage
        stage(run, plan, stop)

    monkeypatch.setattr(engine.Run, "stage", stopped)
    job = engine.Step("job", (PY, "-c", "print('never')"))
    run_id = engine.new_run_id("test")
    with pytest.raises(engine.RunFailed) as failed:
        engine.execute(request(source, jobs=[("job", job)], host="alpha"), run_id, hosts=hosts,
                       announce=lambda line: None)
    assert failed.value.phase == "stage" and "before it started" in failed.value.detail
    directory = tmp_path / "alpha" / "runs" / run_id
    assert not (directory / "plan.json").exists() and not (directory / "runner.json").exists()
    record = json.loads((failed.value.outcome.results / "rig.json").read_text())
    assert [p["phase"] for p in record["phases"]] == ["connect", "stage"]
    assert state.Launched.gone() == []


def test_placement_takes_the_member_that_frees_first(rig):
    hosts, source, _ = rig
    busy = _holding(connect(hosts.host("alpha")), "busy", "device")
    try:
        job = engine.Step("job", (PY, "-c", "print('ok')"))
        lines = []
        outcome = engine.execute(request(source, jobs=[("job", job)]), engine.new_run_id("test"), hosts=hosts, announce=lines.append)
        assert outcome.host == "beta"
    finally:
        busy.stop()


def _ended(holder: HostRun) -> None:
    """Stop another run's runner and wait until it has ended, its holds released."""
    holder.stop()
    for _ in holder.follow(0):
        pass


def _session(hosts, alias: str, run: str) -> agent.Session:
    """Run `run`'s session on `alias`, with no entry yet."""
    return agent.Session(agent.Rig(hosts.host(alias).root), {
        "who": "other", "run": run, "purpose": "p", "estimate": 600.0, "started": time.time(),
        "protocol": agent.PROTOCOL,
    })


def _queued(hosts, alias: str, run: str) -> agent.Session:
    """Run `run` queued on `alias` for a build slot: its request is entered, none held."""
    session = _session(hosts, alias, run)
    assert session.request("build") == []
    return session


def test_a_queued_run_sees_ahead_of_it_what_its_host_would_grant_first(rig):
    """What is ahead of a queued run is what the host's grant rule puts before its request
    (`Session.blocking`): whatever another run holds and every request entered before its own,
    though that run entered after it; never a request entered after its own."""
    hosts, _, _ = rig
    sessions = [_session(hosts, "alpha", run) for run in ("mine", "holder", "requester", "behind")]
    mine, holder, requester, behind = sessions
    try:
        assert mine.claim("s", "mine-tree", "b", 0, "c0", 60)[2] == []  # its first entry
        assert holder.claim("s", "holder-tree", "b", 0, "c0", 60)[2] == []
        assert holder.request("timed") == [] and holder.blocking("timed") == []
        assert requester.request("build") == []
        assert mine.request("build") == []
        assert behind.request("timed") == []
        seen = agent.report(hosts.host("alpha").root, [], "mine")
        assert {e["run"] for e in seen["entries"]} == {"holder", "requester"}
        # Of those, what conflicts with its build is what keeps its build from it.
        assert ({(e["run"], e["kind"]) for e in mine.blocking("build")}
                == {(e["run"], e["kind"]) for e in seen["entries"] if agent.conflicts(e["kind"], "build")}
                == {("holder", "timed")})
        assert seen["waits"]["build"] == pytest.approx(600, abs=30)  # the holder's timed
        assert seen["waits"]["timed"] == pytest.approx(1200, abs=30)  # and the requester's build
        assert len(seen["room"]["ending"]) == 1  # the holder's checkout
        assert len(seen["building"]) == 1  # the holder's build of it
    finally:
        for session in sessions:
            session.release_all()
            session.run_lock.close()


def test_a_run_waiting_for_room_and_its_install_sees_ahead_of_it_what_comes_before_its_install(rig):
    """A run waiting for room (`make_room`) waits for its install hold beside it: that hold's
    request decides what is ahead of it, as it decides what keeps the hold from it."""
    hosts, _, _ = rig
    sessions = [_session(hosts, "alpha", run) for run in ("mine", "earlier", "later")]
    mine, earlier, later = sessions
    try:
        assert mine.request("room") == []
        assert earlier.request("timed") == []
        assert mine.request("install") == []
        assert later.request("timed") == []
        seen = agent.report(hosts.host("alpha").root, [], "mine")
        assert ({(e["run"], e["kind"]) for e in seen["entries"]}
                == {(e["run"], e["kind"]) for e in mine.blocking("install")} == {("earlier", "timed")})
    finally:
        for session in sessions:
            session.release_all()
            session.run_lock.close()


def test_a_queued_run_moves_to_the_candidate_that_would_start_it_sooner_by_more_than_the_margin(rig, monkeypatch):
    hosts, source, _ = rig
    busy = {alias: _holding(connect(hosts.host(alias)), f"busy-{alias}", "timed") for alias in ("alpha", "beta")}
    queued = _queued(hosts, "alpha", "mine")
    try:
        asked = request(source)
        # Both hosts hold a timed run: nothing is earlier elsewhere (the run's own request on alpha is
        # not counted against alpha).
        assert engine.replaced(asked, hosts, "mine", "alpha") is None
        _ended(busy.pop("beta"))
        moved = engine.replaced(asked, hosts, "mine", "alpha")
        assert moved is not None and moved.remote.alias == "beta"
        # Gaining less than the margin leaves it queued where it is.
        monkeypatch.setattr(engine, "MOVE_MARGIN", 7200.0)
        assert engine.replaced(asked, hosts, "mine", "alpha") is None
    finally:
        queued.run_lock.close()
        for holder in busy.values():
            holder.stop()


def test_policy_and_pins_still_decide_which_hosts_a_queued_run_may_move_to(rig):
    hosts, source, _ = rig
    busy = _holding(connect(hosts.host("alpha")), "busy-alpha", "timed")
    queued = _queued(hosts, "alpha", "mine")
    try:
        job = engine.Step("job", (PY, "-c", "print('ok')"))
        work = request(source, jobs=[("job", job)]).work("alpha")

        def excluding_beta(alias: str) -> engine.Work:
            if alias == "beta":
                raise engine.Refused("beta is out of use")
            return work

        excluded = engine.Request("test", "p", 60.0, "cpu", excluding_beta, host="pool")
        assert engine.replaced(excluded, hosts, "mine", "alpha") is None  # beta is idle but not a candidate
        assert engine.replaced(request(source, host="alpha"), hosts, "mine", "alpha") is None  # pinned
        assert engine.replaced(request(source, host="pool"), hosts, "mine", "alpha").remote.alias == "beta"
    finally:
        queued.run_lock.close()
        busy.stop()


def test_placement_connects_a_healthy_host_once_and_one_whose_read_failed_again(rig, monkeypatch):
    """What a kept connection's host can run and whether it is set up are read with each status."""
    hosts, source, _ = rig
    monkeypatch.setattr(engine, "_connected", {})
    connected, connecting = [], engine.connect

    def counted(config):
        connected.append(config.alias)
        return connecting(config)

    monkeypatch.setattr(engine, "connect", counted)
    asked = request(source)
    engine.place(asked, hosts, "run")
    engine.place(asked, hosts, "run")
    assert sorted(connected) == ["alpha", "beta"]
    read, failing = engine.host_status, ["alpha"]

    def alpha_fails_once(remote, *arguments):
        if remote.alias in failing:
            failing.remove(remote.alias)
            raise CommandFailed("alpha: the agent replied no JSON: ''")
        return read(remote, *arguments)

    monkeypatch.setattr(engine, "host_status", alpha_fails_once)
    connected.clear()
    assert engine.place(asked, hosts, "run").remote.alias == "beta"  # alpha unreadable this round
    engine.place(asked, hosts, "run")
    assert connected == ["alpha"]
    ready = Path(hosts.host("alpha").root) / "rig" / "ready.json"
    ready.rename(ready.with_suffix(".away"))  # `mrig setup` found it no longer fit
    connected.clear()
    candidates, skipped, unreadable = engine._candidates(asked, (hosts.host("alpha"),), "run")
    assert candidates == [] and unreadable == set() and "not set up" in skipped[0] and connected == []


def test_a_reply_that_does_not_parse_is_the_agents_failure_and_never_ends_placing_again(monkeypatch):
    for reply in ("", "Traceback (most recent call last):\n  KeyError: 'x'\n"):
        with pytest.raises(CommandFailed):
            _json_line(reply)
    followed, rounds, notes = threading.Event(), [], []

    def placing(*arguments):
        rounds.append(arguments)
        if len(rounds) == 1:  # every host's status failed so
            raise CommandFailed("no cpu host among pool: alpha: the agent replied no JSON: ''")
        if len(rounds) == 3:
            followed.set()
            return None
        return types.SimpleNamespace(work=types.SimpleNamespace(sides=()))

    def stop(queued=False):
        raise CommandFailed("alpha: `stop` exited 1: the agent replied no JSON: ''")

    monkeypatch.setattr(engine, "REPLACE_SECONDS", 0.0)
    monkeypatch.setattr(engine, "replaced", placing)
    queued = types.SimpleNamespace(remote=types.SimpleNamespace(windows=False, alias="alpha"), built=False,
                                   queued=True, run_id="queued", host=types.SimpleNamespace(stop=stop),
                                   note=notes.append, destination=None)
    engine._replace_when(queued, None, None, followed)  # returns: neither failure ended it
    assert len(rounds) == 3 and len(notes) == 1 and "stop to move it failed" in notes[0]


def test_a_home_host_that_cannot_be_read_keeps_the_run_and_one_no_longer_fit_lets_it_go(rig, monkeypatch):
    hosts, source, _ = rig
    busy = _holding(connect(hosts.host("alpha")), "busy-alpha", "timed")
    queued = _queued(hosts, "alpha", "mine")
    try:
        work = request(source).work("alpha")

        def excluding_alpha(alias: str) -> engine.Work:
            if alias == "alpha":
                raise engine.Refused("alpha is out of use")
            return work

        refused = engine.Request("test", "p", 60.0, "cpu", excluding_alpha, host="pool")
        assert engine.replaced(refused, hosts, "mine", "alpha").remote.alias == "beta"
        read = engine.host_status

        def alpha_unreadable(remote, *arguments):
            if remote.alias == "alpha":
                raise HostUnavailable("alpha: unreachable")
            return read(remote, *arguments)

        monkeypatch.setattr(engine, "host_status", alpha_unreadable)
        assert engine.replaced(request(source), hosts, "mine", "alpha") is None
    finally:
        queued.run_lock.close()
        busy.stop()


def test_a_run_granted_its_build_or_on_windows_is_never_stopped_to_move(rig, monkeypatch):
    hosts, source, _ = rig
    remote = connect(hosts.host("alpha"))
    holder = _holding(remote, "holder", "build")
    built = _holding(remote, "built", "build", released=True)
    try:
        deadline = time.monotonic() + 10
        while any(e["run"] == "built" for e in host_status(remote)["entries"]):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        # One holds its build, the other holds nothing now: each was granted it, so its runner
        # takes the stop to move as void and runs on.
        assert holder.stop(queued=True)["stopped"] and built.stop(queued=True)["stopped"]
        time.sleep(1.0)
        assert run_state(remote, "holder") == run_state(remote, "built") == "running"
        # A run the caller has seen granted its build, or on a Windows host, is not placed again at all.
        monkeypatch.setattr(engine, "REPLACE_SECONDS", 0.0)
        monkeypatch.setattr(engine, "replaced", lambda *args: pytest.fail("placed again"))
        for windows, built_seen in ((False, True), (True, False)):
            queued = types.SimpleNamespace(remote=types.SimpleNamespace(windows=windows, alias="alpha"),
                                           built=built_seen, queued=True, run_id="queued")
            engine._replace_when(queued, None, None, threading.Event())
        building = engine.Run.__new__(engine.Run)
        building.phase, building.built = "queue-build", False
        assert building.queued
        building.enter = engine.Run.enter.__get__(building)
        building.phases, building.phase_started = [], time.monotonic()
        building.state = types.SimpleNamespace(update=lambda **fields: None)
        building.enter("build-source")
        assert building.built and not building.queued
    finally:
        holder.stop()
        built.stop()


def _committed(source: sources.Source) -> sources.Source:
    """`source` as the worktree: its directory committed, its identity read from it."""
    for arguments in (("init", "-q"), ("add", "-A"), ("-c", "user.name=rig", "-c", "user.email=rig@example.invalid",
                                                      "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                                                      "commit", "-qm", "source")):
        subprocess.run(["git", "-C", str(source.directory), *arguments], check=True, capture_output=True)
    return dataclasses.replace(source, name=sources.WORKTREE, identity=identify(source.directory))


def test_a_worktree_is_current_while_unedited_and_readable_and_a_fixed_source_always_is(rig):
    _, source, tmp_path = rig
    worktree = _committed(source)
    assert worktree.current()
    assert not dataclasses.replace(worktree, directory=tmp_path / "gone").current()  # cannot be read
    (worktree.directory / "data.txt").write_text("edited\n")
    assert not worktree.current()
    assert source.current()  # fixed for the run, as a ref is


@pytest.mark.parametrize("trouble", [None, "reply_lost", "fetch_fails", "edited", "edited_after_check"])
def test_a_run_queued_behind_a_timed_run_goes_where_its_jobs_start_sooner(rig, monkeypatch, trouble):
    """Whether it moved is its runner's end's to say: a stop whose reply is lost moves it too. A
    run whose directory where it was could not be removed stays launched there, to be collected.
    One whose worktree was edited while it queued stays: staged again, it would build the edit.
    One whose worktree is edited once it was found unedited, as it is stopped to move, fails at
    its stage there, which finds the edit."""
    hosts, source, _ = rig
    source = _committed(source)
    monkeypatch.setattr(engine, "REPLACE_SECONDS", 0.3)
    reply_lost = trouble == "reply_lost"
    if trouble == "fetch_fails":
        def fetched_refused(self, wait=False):
            raise CommandFailed(f"{self.remote.alias}: `fetched` exited 1: run {self.run} is still running")

        monkeypatch.setattr(HostRun, "fetched", fetched_refused)
    if reply_lost:
        stop, lost = HostRun.stop, []

        def stop_reply_lost(self, queued=False):
            reply = stop(self, queued)
            if queued and not lost:
                lost.append(reply)
                raise HostUnavailable("the stop's reply was lost")
            return reply

        monkeypatch.setattr(HostRun, "stop", stop_reply_lost)
    if trouble == "edited_after_check":
        stop_to_move = HostRun.stop

        def stop_then_edit(self, queued=False):
            reply = stop_to_move(self, queued)
            if queued:
                (source.directory / "data.txt").write_text("edited after it was found unedited\n")
            return reply

        monkeypatch.setattr(HostRun, "stop", stop_then_edit)
    busy = {alias: _holding(connect(hosts.host(alias)), f"busy-{alias}", "timed") for alias in ("alpha", "beta")}
    lines = []
    try:
        job = engine.Step("job", (PY, "-c", "print('ok')"))
        run_id = engine.new_run_id("test")

        def free_the_other_host():  # once the run queues for a hold where it was placed
            deadline = time.monotonic() + 60
            while not any(r["run"] == run_id and r["phase"].startswith("queue-") for r in state.live_runs()):
                assert time.monotonic() < deadline, "the run never queued"
                time.sleep(0.1)
            placed = next(alias for alias in busy if f"→ {alias} ·" in lines[0])
            if trouble == "edited":
                (source.directory / "data.txt").write_text("edited while it queued\n")
            _ended(busy.pop("beta" if placed == "alpha" else "alpha"))
            if trouble == "edited":
                time.sleep(2.0)  # rounds of placing again, each finding the other host free
                _ended(busy.pop(placed))

        freeing = threading.Thread(target=free_the_other_host)
        freeing.start()
        if trouble == "fetch_fails":
            with pytest.raises(CommandFailed):
                engine.execute(request(source, jobs=[("job", job)]), run_id, hosts=hosts, announce=lines.append)
            freeing.join()
            placed = next(alias for alias in ("alpha", "beta") if f"→ {alias} ·" in lines[0])
            assert state.Launched.read(run_id)["host"] == placed
            return
        if trouble == "edited_after_check":
            with pytest.raises(engine.RunFailed) as failed:
                engine.execute(request(source, jobs=[("job", job)]), run_id, hosts=hosts, announce=lines.append)
            freeing.join()
            placed = next(alias for alias in ("alpha", "beta") if f"→ {alias} ·" in lines[0])
            assert failed.value.phase == "stage" and "source changed while it was staged" in failed.value.detail
            assert failed.value.outcome.host != placed and failed.value.outcome.codes == {}
            assert any("its jobs start sooner there" in line for line in lines)
            return
        outcome = engine.execute(request(source, jobs=[("job", job)]), run_id, hosts=hosts, announce=lines.append)
        freeing.join()
        placed = next(alias for alias in ("alpha", "beta") if f"→ {alias} ·" in lines[0])
        if trouble == "edited":
            assert outcome.host == placed and outcome.codes == {"job": 0}
            assert not any("its jobs start sooner there" in line for line in lines)
            assert "not moved: its worktree has changed" in (outcome.results / "logs" / "caller.log").read_text()
            return
        assert outcome.host != placed and outcome.codes == {"job": 0}
        assert any("its jobs start sooner there" in line for line in lines)
        assert not reply_lost or lost == [{"stopped": True}]
        assert not (Path(hosts.host(placed).root) / "runs" / run_id).exists()  # nothing of it is left behind
    finally:
        for holder in busy.values():
            holder.stop()


def test_a_lost_connection_loses_nothing_of_the_run(rig, monkeypatch):
    """The link drops twice while the job runs: the caller follows again from the last event it
    saw, and the run's answer is whole."""
    hosts, source, _ = rig
    monkeypatch.setattr(engine, "BACKOFF_FIRST", 0.1)
    real = HostRun.follow
    drops = []

    def dropping(self, after):
        for event in real(self, after):
            yield event
            if event["event"] == "output" and len(drops) < 2:
                drops.append(event["seq"])
                raise engine.HostUnavailable("the link dropped")

    monkeypatch.setattr(HostRun, "follow", dropping)
    job = engine.Step("job", (PY, "-c", "import time\nfor i in range(3):\n    print(i, flush=True); time.sleep(0.3)"))
    outcome = engine.execute(request(source, jobs=[("job", job)]), engine.new_run_id("test"), hosts=hosts,
                             announce=lambda line: None)
    assert len(drops) == 2 and outcome.codes == {"job": 0}
    assert "0\n1\n2\n" in (outcome.results / "logs" / "job-job.log").read_text()
    assert "connection lost" in (outcome.results / "logs" / "caller.log").read_text()


RESULT = engine.Step("job", (PY, "-c", "import json, os, sys; print('measured')\n"
                                       "json.dump({'value': 7}, open(os.path.join(sys.argv[1], 'result.json'), 'w'))",
                              "{run_dir}"))


def _carried_out(rig, monkeypatch, run_id: str, *, jobs=(("job", RESULT),), files=()) -> int:
    """`run_id` carried out as a run's own process does: run, answered from its result, coded."""
    hosts, source, _ = rig
    monkeypatch.setattr(engine.Hosts, "load", classmethod(lambda cls: hosts))
    arguments = type("Arguments", (), {"argv": ["test"], "purpose": "settles a test", "estimate": 120.0})()

    def answer(found):
        value = json.loads((found["job"].directory / "result.json").read_text())["value"]
        return Answer((f"value {value}",), True)

    (store.results() / run_id).mkdir()
    return results.carry_out(arguments, request(source, jobs=list(jobs), host="alpha", files=files), answer, run_id)


def test_an_ended_run_holds_nothing_and_is_answered_before_the_rest_of_it_is_fetched(rig, monkeypatch, capsys):
    """Once the jobs end the host's holds are free and the answer is fetched, recorded and coded,
    the run no longer in flight; only then is the rest (the runner's own records) fetched and the
    directory removed from the host. The request's files are the caller's own, never fetched."""
    hosts, _, tmp_path = rig
    alpha = connect(hosts.host("alpha"))
    run_id = engine.new_run_id("test")
    results_dir, directory = store.results() / run_id, tmp_path / "alpha" / "runs" / run_id
    (tmp_path / "prompt.txt").write_text("the input\n")
    fetch_answer, fetch, seen = sync.fetch_answer, sync.fetch, {}

    def answering(*arguments):
        seen["holds at the answer"] = host_status(alpha)["entries"]
        fetch_answer(*arguments)

    def resting(*arguments):
        seen["coded"] = (results_dir / engine.CODE).read_text()
        seen["in flight"] = state.run_state(run_id)
        seen["records here"] = sorted(p.name for p in results_dir.iterdir() if p.name in sync.RECORDS)
        seen["holds at the rest"] = host_status(alpha)["entries"]
        fetch(*arguments)

    monkeypatch.setattr(sync, "fetch_answer", answering)
    monkeypatch.setattr(sync, "fetch", resting)
    assert _carried_out(rig, monkeypatch, run_id, files=(tmp_path / "prompt.txt",)) == 0
    assert "value 7" in capsys.readouterr().out
    assert seen == {"holds at the answer": [], "coded": "0", "in flight": None, "records here": [],
                    "holds at the rest": []}
    assert all((results_dir / name).exists() for name in sync.RECORDS)
    assert (results_dir / "inputs" / "prompt.txt").read_text() == "the input\n"
    assert not directory.exists() and not (state.Launched.DIRECTORY / f"{run_id}.json").exists()


def test_a_run_answered_before_its_caller_died_is_fetched_whole_and_never_answered_again(rig, monkeypatch, capsys):
    """The caller answered its run and died before fetching the rest: the next collector fetches
    the rest and prunes the host, and the answer stays the one the caller gave."""
    tmp_path = rig[2]
    run_id = engine.new_run_id("test")

    def dying(run_id, hosts=None):
        state.RunState.remove(run_id)  # as the caller's process would, dying
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "complete", dying)
    with pytest.raises(KeyboardInterrupt):
        _carried_out(rig, monkeypatch, run_id)
    transcript = (store.results() / run_id / engine.TRANSCRIPT)
    assert not transcript.exists() and (tmp_path / "alpha" / "runs" / run_id).exists()
    assert cli._collect(sys.stderr, rest=True) == 0  # answered already: fetched alone
    assert not (tmp_path / "alpha" / "runs" / run_id).exists() and state.Launched.gone() == []
    assert (store.results() / run_id / "events.jsonl").exists() and not transcript.exists()
    assert (store.results() / run_id / engine.CODE).read_text() == "0"
    assert json.loads((store.results() / run_id / "rig.json").read_text())["purpose"] == "settles a test"
    capsys.readouterr()


def test_a_run_that_ends_while_its_link_is_down_is_finished_from_the_hosts_end(rig, monkeypatch):
    """The link goes down while the job runs and stays down until the runner has ended; the
    caller follows again from the host's end without its output, then the fetch drops too (an
    rsync that exits 1, as macOS's does for anything): the run is answered whole, its output
    from the fetched logs."""
    hosts, source, tmp_path = rig
    monkeypatch.setattr(engine, "BACKOFF_FIRST", 0.1)
    down, follow, fetch_answer, dropped = link_down(tmp_path), HostRun.follow, sync.fetch_answer, []
    run_id = engine.new_run_id("test")
    ended = tmp_path / "alpha" / "runs" / run_id / "ended.json"

    def up_once_ended():
        while not ended.exists():
            time.sleep(0.05)
        time.sleep(0.5)  # the caller retries against the down link meanwhile
        down.unlink()

    def dropping(self, after):
        for event in follow(self, after):
            yield event
            if event["event"] == "phase" and event["phase"] == "job-job" and not dropped:
                dropped.append("follow")
                down.touch()
                threading.Thread(target=up_once_ended, daemon=True).start()
                raise HostUnavailable("alpha: unreachable (the link dropped)")

    def dropping_fetch(*arguments):
        if dropped == ["follow"]:
            dropped.append("fetch")
            down.touch()
            threading.Timer(0.5, down.unlink).start()
        fetch_answer(*arguments)

    monkeypatch.setattr(HostRun, "follow", dropping)
    monkeypatch.setattr(sync, "fetch_answer", dropping_fetch)
    job = engine.Step("job", (PY, "-c", "import json, os, sys, time\nfor i in range(3):\n"
                                        "    print('line', i, flush=True); time.sleep(0.3)\n"
                                        "json.dump({'value': 7}, open(os.path.join(sys.argv[1], 'result.json'), 'w'))",
                              "{run_dir}"))
    outcome = engine.execute(request(source, jobs=[("job", job)], host="alpha"), run_id, hosts=hosts,
                             announce=lambda line: None)
    assert dropped == ["follow", "fetch"] and outcome.codes == {"job": 0}
    assert json.loads((outcome.results / "job" / "result.json").read_text())["value"] == 7
    assert "line 0\nline 1\nline 2\n" in (outcome.results / "logs" / "job-job.log").read_text()
    notes = (outcome.results / "logs" / "caller.log").read_text()
    assert "connection lost" in notes and "fetch failed" in notes


def test_an_rsync_that_fails_is_a_lost_connection_only_when_its_host_does_not_answer(rig):
    hosts, _, tmp_path = rig
    alpha = connect(hosts.host("alpha"))
    with pytest.raises(CommandFailed):  # the host answers: the run's directory is not there
        sync.fetch_answer(alpha, alpha.path("runs", "none"), tmp_path / "fetched")
    link_down(tmp_path).touch()
    with pytest.raises(HostUnavailable):
        sync.fetch_answer(alpha, alpha.path("runs", "none"), tmp_path / "fetched")


def _orphaned(rig, monkeypatch, *, ended: bool) -> str:
    """A run on alpha whose caller died once its job started; with `ended`, once its runner
    has ended alone."""
    hosts, source, tmp_path = rig
    monkeypatch.setattr(cli.Hosts, "load", classmethod(lambda cls: hosts))

    def dying(run):
        for event in run.host.follow(run.sequence):
            run.receive(event)
            if event["event"] == "phase" and event["phase"] == "job-script":
                state.RunState.remove(run.run_id)  # as the caller's process would, dying
                raise KeyboardInterrupt

    monkeypatch.setattr(engine, "_follow", dying)
    command = ("script", "--backend", "cpu", "--host", "alpha", "--purpose", "answered later",
               "--estimate", "1m", "--hold", "build", str(tmp_path / "unused.sh"))
    job = engine.Step("script", (PY, "-c", "import time; time.sleep(1); print('the collected answer')"))
    run_id = engine.new_run_id("script")
    with pytest.raises(KeyboardInterrupt):
        engine.execute(request(source, jobs=[("script", job)], host="alpha"), run_id, hosts=hosts,
                       announce=lambda line: None, command=command)
    (tmp_path / "unused.sh").write_text("")  # the script instrument plans from the file it names
    deadline = time.monotonic() + 30
    while ended and not (tmp_path / "alpha" / "runs" / run_id / "ended.json").exists():
        assert time.monotonic() < deadline
        time.sleep(0.2)
    return run_id


def test_a_run_whose_caller_died_is_answered_by_the_next_command(rig, monkeypatch, capsys):
    """The caller dies once the job has started; the runner finishes alone, and the next mrig
    command fetches the run and answers it from its results with the run's own command."""
    tmp_path = rig[2]
    run_id = _orphaned(rig, monkeypatch, ended=False)
    deadline = time.monotonic() + 30
    while not cli._collect(sys.stderr, rest=True):  # until the runner has ended
        assert time.monotonic() < deadline
        time.sleep(0.2)
    results_dir = store.results() / run_id
    transcript = (results_dir / engine.TRANSCRIPT).read_text()
    assert "purpose: answered later" in transcript and "the collected answer" in transcript
    assert (results_dir / engine.CODE).read_text() == "0"
    assert "the collected answer" in capsys.readouterr().err
    record = json.loads((results_dir / "rig.json").read_text())
    assert record["codes"] == {"script": 0} and record["failed"] is None
    assert not (tmp_path / "alpha" / "runs" / run_id).exists() and state.Launched.gone() == []


def test_a_collected_run_is_answered_once_whatever_this_checkout_can_resolve(rig, monkeypatch, capsys):
    """Launched records are the machine's: a run is answered from its fetched results without
    opening any source, and one whose command this checkout's mrig cannot read (an option it
    lacks, a ref that no longer resolves, which `--build` resolves as it parses) is answered by
    its record. None is left owed to fail every later command."""
    answered, gone, unreadable = (_orphaned(rig, monkeypatch, ended=True) for _ in range(3))
    monkeypatch.setattr(cli.sources, "opened", lambda name: pytest.fail("a collected run opened a source"))

    def recommand(run_id, change):
        path = state.Launched.DIRECTORY / f"{run_id}.json"
        launched = json.loads(path.read_text())
        path.write_text(json.dumps({**launched, "command": change(launched["command"])}))

    recommand(gone, lambda command: [*command[:1], "--build", "no-such-ref", *command[1:]])
    recommand(unreadable, lambda command: [*command[:1], "--no-such-option", *command[1:]])
    assert cli._collect(sys.stderr, rest=True) == 3 and cli._collect(sys.stderr, rest=True) == 0
    assert "the collected answer" in (store.results() / answered / engine.TRANSCRIPT).read_text()
    assert (store.results() / answered / engine.CODE).read_text() == "0"
    for run_id in (gone, unreadable):
        assert "cannot read" in (store.results() / run_id / engine.TRANSCRIPT).read_text()
        assert (store.results() / run_id / engine.CODE).read_text() == "1"
    assert state.Launched.gone() == []
    capsys.readouterr()


def test_an_orphaned_run_is_collected_by_one_command_while_others_skip_it(rig, monkeypatch):
    """A second command collecting while the first fetches the run finds it claimed and skips
    it; the first collects it once."""
    hosts = rig[0]
    run_id = _orphaned(rig, monkeypatch, ended=True)
    fetch, during, answered = engine.sync.fetch_answer, [], []

    def racing(*arguments):
        during.append(engine.collect(hosts, answered.append, rest=True)[0])  # another command starting meanwhile
        fetch(*arguments)

    monkeypatch.setattr(engine.sync, "fetch_answer", racing)
    assert [launched["run"] for launched in engine.collect(hosts, answered.append, rest=True)[0]] == [run_id]
    assert during == [[]] and [launched["run"] for launched in answered] == [run_id]
    assert state.Launched.gone() == []
    assert json.loads((store.results() / run_id / "rig.json").read_text())["codes"] == {"script": 0}


def test_a_failed_collection_is_reported_and_the_run_stays_owed(rig, monkeypatch, capsys):
    """A fetch that fails is shown on stderr with the run, its host and the error; the command
    goes on, and the run stays on the host for a later command to collect."""
    hosts, _, tmp_path = rig
    run_id = _orphaned(rig, monkeypatch, ended=True)
    fetch = engine.sync.fetch_answer

    def vanishing(*arguments):
        raise CommandFailed("rsync: some files vanished (exit 23)")

    monkeypatch.setattr(engine.sync, "fetch_answer", vanishing)
    answered = []
    assert engine.collect(hosts, answered.append, rest=True)[0] == [] and answered == []
    err = capsys.readouterr().err
    assert run_id in err and "alpha" in err and "CommandFailed: rsync: some files vanished" in err
    assert [launched["run"] for launched in state.Launched.gone()] == [run_id]
    monkeypatch.setattr(engine.sync, "fetch_answer", fetch)
    assert [launched["run"] for launched in engine.collect(hosts, answered.append, rest=True)[0]] == [run_id]
    assert not (tmp_path / "alpha" / "runs" / run_id).exists()


def test_a_run_whose_runner_was_killed_is_collected_as_failed(rig, monkeypatch):
    """A runner killed before recording its end (SIGKILL, a reboot) is no longer waited on: the
    next command collects its run as failed, with what its directory keeps, and prunes it."""
    hosts, _, tmp_path = rig
    run_id = _orphaned(rig, monkeypatch, ended=False)
    directory = tmp_path / "alpha" / "runs" / run_id
    os.kill(json.loads((directory / "runner.json").read_text())["pid"], signal.SIGKILL)
    answered = []
    deadline = time.monotonic() + 30
    while not engine.collect(hosts, answered.append, rest=True)[0]:  # until the killed runner's lock is released
        assert time.monotonic() < deadline
        time.sleep(0.2)
    record = json.loads((store.results() / run_id / "rig.json").read_text())
    assert record["failed"] == {"phase": "runner", "detail": "its runner on alpha ended without recording its end"}
    assert (store.results() / run_id / "plan.json").exists(), "what the host kept is fetched"
    assert [launched["run"] for launched in answered] == [run_id]
    assert not directory.exists() and state.Launched.gone() == []


def test_a_collector_that_died_after_the_removal_leaves_a_good_run_good(rig, monkeypatch):
    """The run ended and was fetched, then its host directory removed, and the collector died
    before removing its record: the next collector records it from the fetched end, not as gone."""
    hosts, _, tmp_path = rig
    run_id = _orphaned(rig, monkeypatch, ended=True)
    launched = next(value for value in state.Launched.gone() if value["run"] == run_id)
    directory = tmp_path / "alpha" / "runs" / run_id
    engine.sync.fetch(connect(hosts.host("alpha")), str(directory), store.results() / run_id)
    HostRun(connect(hosts.host("alpha")), run_id, launched["agent"]).fetched()
    assert not directory.exists()
    answered = []
    assert engine.collect(hosts, answered.append, rest=True)[0] == [launched]
    record = json.loads((store.results() / run_id / "rig.json").read_text())
    assert record["failed"] is None and record["codes"] == {"script": 0}


def test_a_stop_while_staging_with_the_host_away_ends_the_caller_at_once(rig, monkeypatch):
    """Staging retries a lost host; stopped meanwhile, the caller stops retrying: nothing of the
    run started, so nothing waits for the host's return."""
    hosts, source, _ = rig
    monkeypatch.setattr(engine, "BACKOFF_FIRST", 30.0)
    attempts = []

    def away(*arguments, **options):
        attempts.append(time.monotonic())
        # mrig stop, half a second into the 30 s wait for the host
        threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM)).start()
        raise engine.HostUnavailable("alpha: unreachable (the link is down)")

    monkeypatch.setattr(engine.sync, "push", away)
    started = time.monotonic()
    with pytest.raises(engine.RunFailed) as failed:
        engine.execute(request(source, jobs=[("job", engine.Step("job", (PY, "-c", "")))], host="alpha"),
                       engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert "before it started" in failed.value.detail and time.monotonic() - started < 10
    assert len(attempts) == 1


def test_only_a_lost_connection_is_retried(monkeypatch):
    """An unreachable host is tried again; a command that failed on a reachable one fails the
    run at once."""
    monkeypatch.setattr(engine, "BACKOFF_FIRST", 0.01)
    noted = []
    run = type("Run", (), {"note": lambda self, text: noted.append(text)})()
    attempts = iter([engine.HostUnavailable("alpha: unreachable (dropped)"), "done"])

    def flaky():
        outcome = next(attempts)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert engine._retrying(run, "mirror", flaky) == "done" and len(noted) == 1

    def failing():
        raise CommandFailed("alpha: rsync exited 23")

    with pytest.raises(CommandFailed):
        engine._retrying(run, "mirror", failing)
    assert len(noted) == 1


def test_a_failed_answer_is_reported_and_the_run_stays_owed(rig, monkeypatch, capsys):
    """An answer that raises is shown on stderr with the run, its host and the error; the
    command goes on, the run stays on the host and here, and a later command answers it once."""
    hosts, _, tmp_path = rig
    run_id = _orphaned(rig, monkeypatch, ended=True)

    def failing(launched):
        raise SystemExit("the answered command could not plan")

    assert engine.collect(hosts, failing, rest=True)[0] == []
    err = capsys.readouterr().err
    assert run_id in err and "alpha" in err and "SystemExit: the answered command could not plan" in err
    assert [launched["run"] for launched in state.Launched.gone()] == [run_id]
    assert (tmp_path / "alpha" / "runs" / run_id).exists()
    assert cli._collect(sys.stderr, rest=True) == 1 and cli._collect(sys.stderr, rest=True) == 0
    results_dir = store.results() / run_id
    assert (results_dir / engine.CODE).read_text() == "0"
    assert capsys.readouterr().err.count(f"mrig: {run_id} ended without its caller") == 1
    assert not (tmp_path / "alpha" / "runs" / run_id).exists() and state.Launched.gone() == []


def _holding(remote, run: str, kind: str, released: bool = False) -> HostRun:
    """Another run's runner holding `kind` until stopped, or, `released`, holding it and then
    releasing it."""
    plan = {"info": {"who": "other", "purpose": "p", "estimate": 3600},
            "ops": [{"op": "hold", "kind": kind}, *([{"op": "release", "kind": kind}] * released),
                    {"op": "run", "argv": [PY, "-c", "import time; time.sleep(3600)"], "cwd": remote.root, "env": {}}]}
    with HostRun.staging(remote, run) as start:
        host = start(plan)
    for event in host.follow(0):
        if event["event"] == "held":
            return host
    raise AssertionError("the other run never held its device")


def test_a_step_is_skipped_where_its_path_exists(rig):
    """A failing build whose `unless` path exists never runs, and the run goes on."""
    hosts, source, tmp_path = rig
    (tmp_path / "exported").write_text("x")
    build = engine.Step("build", (PY, "-c", "import sys; sys.exit(3)"), unless=str(tmp_path / "exported"))
    outcome = engine.execute(request(source, builds=[build], jobs=[("job", engine.Step("job", (PY, "-c", "")))]),
                             engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert outcome.codes == {"job": 0}
    assert "exists" in (outcome.results / "logs" / "build-source.log").read_text()


def test_headers_are_exported_once_for_each_catalog_a_host_runs(rig):
    """A run acquiring headers builds the catalog tool and exports only where the host has no
    export of its catalog, which its models, lock and pinned artifacts key; one acquiring weights always
    builds the tool."""
    hosts, source, _ = rig
    for path in engine.CATALOG_INPUTS:
        (source.directory / path).parent.mkdir(parents=True, exist_ok=True)
        (source.directory / path).write_text("one catalog")

    def ops(acquires):
        work = dataclasses.replace(request(source).work("alpha"), acquires=acquires)
        candidate = engine.Candidate(connect(hosts.host("alpha")), {}, work,
                                     {engine.build_key(work, side): 0 for side in work.sides}, 0.0, (60.0,))
        return [op for op in engine.plan_for(request(source), candidate, "planned").ops if op["op"] == "run"]

    key = engine.catalog_key(source)
    exported = f"{hosts.host('alpha').root}/catalog-headers/{key}/headers/index.json"
    (build,) = [op for op in ops(engine.Headers(None)) if op["argv"][:2] == ["cargo", "build"]]
    (export,) = [op for op in ops(engine.Headers(None)) if "headers" in op["argv"][1:2]]
    assert build["unless"] == export["unless"] == exported
    (weights,) = [op for op in ops(engine.Weights("m:v")) if op["argv"][:2] == ["cargo", "build"]]
    assert "unless" not in weights
    for path in engine.CATALOG_INPUTS:  # each file keys it: a model added without moving the lock exports its own headers
        (source.directory / path).write_text("another catalog")
        assert engine.catalog_key(source) != key
        (source.directory / path).write_text("one catalog")
        assert engine.catalog_key(source) == key


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


def test_a_run_whose_checkouts_cannot_grow_fails_before_it_builds(rig):
    hosts, source, tmp_path = rig
    built = tmp_path / "built"
    build = engine.Step("build", (PY, "-c", "import sys; open(sys.argv[1], 'w').write('x')", str(built)))
    run = request(source, builds=[build], jobs=[("job", engine.Step("job", (PY, "-c", "")))])
    work = run.work("alpha")
    for alias in ("alpha", "beta"):  # a build no volume holds
        (tmp_path / alias / "rig" / "builds").mkdir(parents=True)
        (tmp_path / alias / "rig" / "builds" / f"{engine.build_key(work, work.sides[0])}.json").write_text(
            json.dumps({"size": 10**18}))
    with pytest.raises(engine.RunFailed) as failed:
        engine.execute(run, engine.new_run_id("test"), hosts=hosts, announce=lambda line: None)
    assert failed.value.phase == "install-source"
    assert "GB short" in failed.value.detail and failed.value.outcome.host in failed.value.detail
    assert not built.exists()


def test_a_run_writes_its_report_when_asked(rig):
    """Asked while its job runs, once the caller has seen the job's output, the run reports it;
    the job ends only once that report is written."""
    hosts, source, tmp_path = rig
    go = tmp_path / "go"
    job = engine.Step("job", tuple(until_exists(go, before="print('working', flush=True)\n")))
    run_id = engine.new_run_id("test")
    report = store.results() / run_id / engine.REPORT

    def ask():
        # Asked once the job runs (its log is the caller's: its SIGUSR1 handler is in place),
        # and again until a report holds the job's output, which the caller sees on its own
        # time; the job waits, so every report is written while it runs.
        while not (store.results() / run_id / "logs" / "job-job.log").exists():
            time.sleep(0.05)
        while not (report.exists() and "'working'" in report.read_text()):
            os.kill(os.getpid(), signal.SIGUSR1)
            time.sleep(0.1)
        go.touch()

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
    assert (tmp_path / "local" / "checkouts" / "slot-0" / "src" / "data.txt").exists()
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


def test_a_runs_estimate_is_of_its_held_time_its_queued_time_shown_beside_it(tmp_path, monkeypatch, capsys):
    """A step queued 22 minutes behind another run, after 6 of its own, must not read "0s left
    of 25m00s": a run is judged by its held time alone, by the host's one account of
    its waits, which its caller mirrors; its queued time is shown beside it, never counted."""
    now = time.time()
    queued, held = state.account({"started": now - 1680, "waited": 1300.0, "waiting_since": now - 20}, now)
    assert (queued, held) == pytest.approx((1320.0, 360.0))
    entry = {"estimate": 1500.0, "remaining": 1140.0, "over": 0.0, "queued": 1320.0}
    assert engine.against_estimate(entry) == "queued 22m00s, 19m00s left of its 25m00s estimate"
    assert engine.against_estimate({**entry, "queued": 0.0}) == "19m00s left of its 25m00s estimate"
    # Queued well past its estimate, its work within it: followed to its answer, never overrun.
    directory, child = _fake_run(tmp_path, 2)
    live = {"started": time.time() - 1680, "estimate": 600.0, "waited": 1300.0, "waiting_since": None}
    monkeypatch.setattr(state, "run_state", lambda run: live)
    assert results.follow(directory.name, directory, child.pid, None) == 0
    child.wait()
    assert "the answer" in capsys.readouterr().out
    # Its work past its estimate: overrun, whatever it queued.
    live.update(started=time.time() - 2000, waited=1300.0)  # 700 s held of 600
    (tmp_path / "again").mkdir()
    directory, child = _fake_run(tmp_path / "again", 30)
    assert results.follow(directory.name, directory, child.pid, None) == results.OVERRUN
    child.kill()


def test_a_run_past_its_estimate_returns_its_report_and_what_to_do(tmp_path, monkeypatch, capsys):
    live = {"started": time.time(), "estimate": 600.0}
    monkeypatch.setattr(state, "run_state", lambda run: live)
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


def test_a_run_whose_process_ends_without_an_answer_returns_at_once_with_its_output(tmp_path, capsys):
    # A run refused before it registers (its plan exits with why) has no estimate to end at:
    # its caller learns of its end from its process, which is the caller's own child.
    directory = tmp_path / "20261023T000000Z-fake-000002"
    directory.mkdir()
    with open(directory / engine.TRANSCRIPT, "w") as transcript:
        child = subprocess.Popen([PY, "-c", "import sys; sys.exit('memory: refused')"], stdout=transcript,
                                 stderr=subprocess.STDOUT)
    ended = []
    follower = threading.Thread(target=lambda: ended.append(results.follow(directory.name, directory, child.pid, None)),
                                daemon=True)
    follower.start()
    follower.join(10)
    assert ended == [1]
    output = capsys.readouterr().out
    assert "memory: refused" in output and "ended without an answer" in output


def test_waiting_records_its_estimate_and_justification(rig, capsys):
    directory = store.results() / "20261023T000000Z-fake-000001"
    directory.mkdir(parents=True)
    (directory / engine.TRANSCRIPT).write_text("the answer\n")
    (directory / engine.CODE).write_text("2")
    code = cli.main(["wait", "fake-000001", "--estimate", "10m", "--justification", "a cold build of 40 crates"])
    assert code == 2 and "the answer" in capsys.readouterr().out
    assert engine.waits(directory)[0]["justification"] == "a cold build of 40 crates"
    assert engine.waits(directory)[0]["estimate_seconds"] == 600


def _follower(ignores_stop: bool) -> subprocess.Popen:
    """A stand-in run caller: one that takes SIGTERM, or one hung past it; returned once it has
    set its handling up (its "ready" line)."""
    script = ("import signal, sys, time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignores_stop else "")
              + "print('ready', flush=True)\ntime.sleep(60)\n")
    follower = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert follower.stdout.readline() == "ready\n"
    return follower


def test_a_stopped_caller_that_ends_is_left_to_fetch_its_run(rig, monkeypatch):
    follower = _follower(ignores_stop=False)
    record = state.RunState("20261105T000000Z-step-aaaaaa", {"host": "alpha"})
    record.value["pid"] = follower.pid
    record.write()
    assert cli._stop(argparse.Namespace(run="aaaaaa")) == 0
    follower.wait(5)


def test_a_stopped_caller_hung_past_its_stop_is_ended_and_its_runner_stopped(rig, monkeypatch):
    follower = _follower(ignores_stop=True)
    record = state.RunState("20261105T000000Z-step-bbbbbb", {"host": "alpha"})
    record.value["pid"] = follower.pid
    record.write()
    state.Launched("20261105T000000Z-step-bbbbbb", "alpha", ["agent"], Path("results"), ()).write()
    monkeypatch.setattr(cli, "STOP_SECONDS", 1.0)
    stopped = []
    monkeypatch.setattr(cli, "connect", lambda host: host)
    monkeypatch.setattr(cli, "HostRun", lambda remote, run, agent: types.SimpleNamespace(stop=lambda: stopped.append(run) or {"stopped": True}))
    monkeypatch.setattr(cli.Hosts, "load", lambda: types.SimpleNamespace(host=lambda name: name))
    assert cli._stop(argparse.Namespace(run="bbbbbb")) == 0
    assert not state.alive(follower.pid)  # `alive` reaps it, so its own wait reads no signal
    assert stopped == ["20261105T000000Z-step-bbbbbb"]
