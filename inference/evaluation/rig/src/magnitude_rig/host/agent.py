"""The rig's host agent: one detached runner per run, carrying out the run's plan and taking and
releasing typed holds on the host for it.

Deployed with the rest of the host package (`connection.connect`) and run with the host's
system Python, so it is a single standard-library file (Python 3.9+) importing nothing from
this package: it also runs alone. The package is named by its content, so callers of other
versions never replace one another's, and a runner records the agent it runs (`runner.json`,
[`invocation`]), which its follow and stop use.

    agent.py start   --root R --run ID   (prints {"staging"}; then the plan on stdin; prints
                                          {"agent", "pid"}; or, once started, {"started"})
    agent.py follow  --root R --run ID --after SEQ
    agent.py stop    --root R --run ID
    agent.py fetched --root R --run ID
    agent.py status  --root R [--source KEY ...]
    agent.py install --from S --to D

A hold is what a phase of a run needs from the host. Kinds, and which conflict:

    build         compiling             conflicts with timed
    device        running on the GPU    conflicts with device, timed
    timed         measuring             conflicts with build, device, timed
    models        the shared model store, while acquiring into it
    install       installing a source into a checkout (and evicting): conflicts with timed
    checkout:NAME one checkout (its source, build target and caches), the run's alone: never
                  requested, only claimed (`checkout`), so it never waits; eviction never
                  removes a checkout with a live hold

Kinds conflict only as listed (`models` with itself); holds of one session never conflict with
each other. Requests are served in arrival order: a request is granted when no live hold
conflicts with it and no live earlier request does. A session claims its checkouts first, then
takes its holds in the order install (released before any other), build, models, device or
timed, and requests device or timed only while holding no build or models hold, so no two
sessions ever wait on each other.

Every request and hold is an entry `<root>/rig/entries/<sequence>-<pid>-<n>.json`, which its
session keeps open under an exclusive OS lock for as long as it lives: an entry whose lock can
be taken belongs to a dead session and is removed by whoever looks next, so nothing outlives
its session. Entries are created, changed and judged only under `<root>/rig/guard.lock`. A
device or timed hold also takes `<root>/lock`, the device lock other tools on the host take.

A run's life on the host does not depend on its caller. Everything the caller provides (each
side's source, the request's files) is in the run's directory before `start`, and the runner's
only waits are on host holds. `start` holds the run's directory for its caller while it stages
it (`stage.lock`, taken before it prints `staging`), so nothing evicts a stage before its run
starts. The caller then sends the plan, or closes the connection without one when it stopped
or died, and the directory is a gone run's. Given the plan, `start` writes it to the run's
directory and starts the runner in a session of its own, detached from the connection that
started it, and returns once the runner owns the run, so one lock or the other holds the
directory throughout. A later `start` of the run (its caller retrying a lost connection)
supersedes an earlier one still waiting for its plan, which a silent link loss would keep
waiting for hours; one starting the runner it waits for. The runner carries out
the plan's operations in order, each event numbered in `events.jsonl` and each phase's output
in `logs/<phase>.log`:

    {"op": "phase", "phase": P}
    {"op": "checkout", "side": S, "source": KEY, "build": B, "fallback": N,
     "commit": C, "seconds": T}               claim a checkout for side S (below), whose
                                              build is of the steps B names (their targets
                                              and profiles) from commit C, N the bytes it
                                              needs where this host has measured none and T
                                              the seconds its caller priced it at; later
                                              operations name its directory `{checkout:S}`
    {"op": "hold", "kind": K}                 events: waiting (blocking: [entry, ...]), held
    {"op": "release", "kind": K}
    {"op": "evict", "side": S, "host": H}     make room for the builds (below): waits, its
                                              install hold released, while live runs' checkouts
                                              will cover it (events: waiting, held); fails the
                                              run when even they cannot
    {"op": "run", "argv": [...], "cwd": C, "env": {...}, "input": S,
     "check": B, "job": J, "capture": NAME, "stdout_to": PATH, "unless": PATH}
                                              events: output, exited. `check` fails the run
                                              on a nonzero exit; `job` records the exit code
                                              as that job's; `capture` keeps the last stdout
                                              line as `{capture:NAME}` in later operations;
                                              `stdout_to` writes the stdout to PATH; `unless`
                                              skips the step (a note) where PATH exists
    {"op": "built", "side": S}                record the checkout's bytes after a build, and
                                              the build's seconds by its warmth

Every job captures its own hang. Its processes are told a directory, `logs/stacks/` of the run
(`STACKS`); a process that reports on SIGUSR1 (the engine's diagnostics, which every harness and
the watcher's tests install) creates `stacks-<pid>.txt` there once its handler is in place, so
the runner signals only processes that will not die of it, and only the job's own (its process
group). The runner asks each such process once for its report (`Watch`), which the process
appends to its file without stopping: when the run's held time passes its estimate, or when a
test binary's results stop for `STALL_FACTOR` times the longest gap between its recent results
(at least `STALL_FLOOR`). Each ask is an event (`stacks`), which the caller shows and names in
the run's report; with reason to ask and no process to ask, it says that once instead.

A step ends with its command's exit: its process group is killed then, and its output waited for
only `OUTPUT_GRACE_SECONDS` more (a process that left the group may hold it open; that is said).
A logged line keeps its first `LINE_HEAD` bytes and its last `LINE_TAIL`, what lies between
counted: a test's assertion printed whole, gigabytes on one line, would otherwise hold the caller,
which streams every line, minutes behind a run long ended. A step whose output is read
(`capture`, `stdout_to`) is kept whole.

The run ends after its last operation, at the first failure, or on `stop`: the runner kills
its command's whole process group, withdraws its entries, writes `ended.json` (each job's
exit code, each phase's seconds, the failure if any) and the `ended` event, and exits.
`follow` prints the events after a sequence number until `ended` (output only as it is written
from then on: each phase's log keeps what came before), so a caller that loses its connection
follows again from the last event it saw; `fetched` removes the directory of an ended run once
the caller has fetched it. The runner withdraws every entry before it records its end, so an
ended run holds nothing of the host while its caller fetches it.

Each checkout lives in `<root>/checkouts/<NAME>/` and belongs to no source: a run claims the
one no live run holds that was last installed from its source (`source`), else the one
most recently claimed, else a new one, and holds it alone for its whole life, so runs never build,
install or read in each other's checkout and each builds in parallel with its own cargo lock.
The run's install rewrites only the files whose bytes differ, each with the time it is written,
so cargo, whose freshness is by file time against its last build in that same directory,
rebuilds exactly what differs from whatever source the checkout last held, and the build's
provenance proves each workspace unit against the bytes now there (`provenance.py`). The
directory never changes, so neither do the paths cargo keys artifacts and build scripts by.

`used` is touched whenever a checkout is claimed, and `built` records its bytes after every
build (`size`) and, per build B (the build steps' targets and profiles), in
`<root>/rig/builds/<B>.json`, what a build of B measured on this host: `size`, what the latest
build of B into a cold checkout (its target built from nothing) added, the build's whole size;
and `growth`, the most a build of B into a checkout that had built B before (its `builds`) added.
A build target only grows, so what bounds a build is what it adds (`need`): a checkout that has
built B before may add B's `growth`, any other B's `size`; where this host has measured neither,
the claim's `fallback`, the largest size of B (else of any build) measured on any host the caller
placed among. A claimed checkout's entry carries its need until its build ends, and every
building checkout may still add its need less what it has grown since its last recorded build,
so several runs building at once on one host each count. A claimed checkout grown past
`CHECKOUT_LIMIT` times the largest build measured here (artifacts of dependencies and features
no longer built) has its target removed, and builds cold, so no checkout holds more than that;
the number of checkouts is the most runs ever held at once, and eviction (below) removes the
least recently used whenever a build needs the space.

A claim knows how warm its checkout's target is for the run's source (`WARMTH`), and its entry
carries that (`build`) and the seconds its caller priced the build at (`seconds`) until the
side's build ends. `built` then records, with B's measurements, the build's seconds, the commit
it built (`to`, also the checkout's `commit` from then), the commit the checkout last built
before it (`from`, None from a cold target) and whether the checkout had built B before
(`built`): its `builds`, the latest `BUILDS_KEPT`. Whether a build was incremental (its checkout
had built B at an ancestor of its commit) needs the repository, so the caller judges it from
those, as it prices a new run's build from them (`engine.build_seconds`), and `status` reports
what it needs: each side's checkout's `commit` and whether it built B. From the live entries
alone, `status` reports the build seconds each live run still owes the host's CPU (`building`):
builds share it, so a run placed beside them builds that much slower.

A run's time so far is two quantities, one definition (`account`): `queued`, every wait of its
session for a hold (`WAITED`), and `held`, the rest, its work. Its estimate is of its work: it
is past its estimate only when its held time is, however long it queued. The runner announces
each change to its waits (`queue` events), so its caller judges it by the same account
(`state.account`). `status` prices each live run by the time it is expected to take still
(`remaining`): its estimate less its held time, so a run queued for a hold keeps what its work
still needs; once past its estimate, as long again as it has taken. A new request waits behind every run that holds or is queued for a hold that
conflicts with it, each once (`waits`).

Each runner owns its run's directory, `<root>/runs/<ID>/`, keeping `session.lock` in it under
an exclusive OS lock for as long as it lives. A directory with `ended.json` is kept until its
caller fetches it; any other directory whose locks (its runner's and its caller's `stage.lock`)
can be taken and that no live entry names (an agent of another version locks none) belongs to
a run that is gone and only holds space, as does one whose caller staged into it and never
started it.

`evict` removes, least recently used first, the checkouts no live run holds and the directories
of gone runs, never an ended one not yet fetched, until the volume has those needs free, and
reports it (`needed_bytes`, the paths removed relative to the root). Short of it, the run waits
while what the live runs it does not wait ahead of hold would cover its own need: their
checkouts, which come free (evictable, their need no longer due) as those runs end. It waits
without its install hold (no timed hold queues behind it) in a `room` request, whose place in
arrival order is its place among the runs waiting for room: a run counts neither the need nor
the release of those waiting after it, which wait for it in turn, so no two wait on each other.
A run that even every such checkout could not cover is refused before the build starts rather
than filled during it. `status` reports the same accounting for a new run (`room`, and each
side's `need` as measured here), from which its caller estimates its wait for room
(`engine.room_wait`).
"""

from __future__ import annotations

import argparse
import errno
import filecmp
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

POLL_SECONDS = 0.5
# The environment variable naming the directory a job's processes report their stacks to (the
# engine's `diagnostics::STACKS`).
STACKS = "MAGNITUDE_STACKS"
# A test binary's results stopped this many times the longest gap between its recent results
# (`STALL_RECENT` of them), and at least `STALL_FLOOR` seconds (libtest's own "running for over
# 60 seconds" mark), is a hang to report.
STALL_FACTOR, STALL_RECENT, STALL_FLOOR = 10, 20, 60.0
TEST_RESULT = re.compile(r"^test .+ \.\.\. (ok|FAILED|ignored)")
REPORTS = re.compile(r"^stacks-(\d+)\.txt$")
TERMINATE_GRACE_SECONDS = 10.0
# A logged line's bytes kept: its head, and its last `LINE_TAIL`. A line can be gigabytes (one
# assertion's); streamed whole, its caller shows the job running long after it has ended. A step
# whose output is read (`capture`, `stdout_to`) is kept whole.
LINE_HEAD, LINE_TAIL = 1 << 20, 1 << 10
# How long a command's output may stay open once its process group is killed: past it, a process
# that left the group (a session of its own) holds it, and it is no longer waited for.
OUTPUT_GRACE_SECONDS = 10.0
_WINDOWS = os.name == "nt"
# This agent's module in the deployed host package.
MODULE = "magnitude_rig.host.agent"
# A timed measurement shares its host with nothing that loads it: another run's build, device
# work, source installation (with checkout eviction) or model acquisition.
CONFLICTS = {
    ("build", "timed"),
    ("device", "device"),
    ("device", "timed"),
    ("timed", "timed"),
    ("install", "timed"),
    ("models", "models"),
    ("models", "timed"),
}
DEVICE_KINDS = ("device", "timed")
ENTRY_LOCK_OFFSET = 1 << 30
RUN_LOCK = "session.lock"  # in a run's directory, locked by its runner while it lives
STAGE_LOCK = "stage.lock"  # in a run's directory, locked by `start` while its caller stages it
STAGE_STATE = "stage.json"  # the `start` holding STAGE_LOCK: its pid, and whether it waits for a plan
WAITED = "waited.json"  # in a run's directory: how long its session has waited (`Session._account`)
RUN_LOCKS = (RUN_LOCK, STAGE_LOCK, STAGE_STATE, WAITED)  # a run directory's files the host's processes alone use
# In a run's directory: the plan, the runner's pid and agent, its events, its end and its own output.
PLAN, RUNNER, EVENTS, ENDED = "plan.json", "runner.json", "events.jsonl", "ended.json"
RUNNER_LOG = "runner.log"
# The entries' meaning. Runs of different protocols never share a host's holds: a run enters
# its first request only when no run of another protocol is live, waiting without an entry
# until then, and then judges only its own protocol's entries. Agents of other versions take
# every entry of another protocol to conflict with every request, so a run of another protocol
# that comes later waits on this one whole. No run waits on another protocol's while it holds
# anything (nor while one waits on it), so waits across protocols form no cycle: no run holding
# a checkout waits for the device behind a run of another protocol that, holding its own
# checkout, waits for that one.
PROTOCOL = 4
# In a checkout: the key of the source last installed into it.
SOURCE = "source"
# A claimed checkout past this many times the largest build measured is cleared (its target).
CHECKOUT_LIMIT = 2
# In a checkout: the builds (`engine.build_key`) it has built since its target was last empty.
BUILT = "builds"
# How warm a claimed checkout's target is for the run's source, so how long its build takes:
# last installed from that source, from another source (rebuilds what differs), or built from
# nothing (a new checkout, or one whose target was cleared).
WARMTH = (SAME, OTHER, COLD) = (0, 1, 2)
# A build's seconds while no build is measured on the host: what a run that has claimed no
# checkout yet owes, and what an entry of another version, which records no price, does.
BUILD_PRIOR_SECONDS = 1200
# The latest builds of each build kept with its measurements.
BUILDS_KEPT = 8
# In a checkout: the commit it last built.
COMMIT = "commit"


def conflicts(a: str, b: str) -> bool:
    if a.startswith("checkout:") or b.startswith("checkout:"):
        return False  # claimed, never requested: the claim takes only a checkout no one holds
    return (a, b) in CONFLICTS or (b, a) in CONFLICTS


class FileLock:
    """A non-blocking OS lock on a file; closing it (or exiting) releases it.

    Windows locks a byte range and refuses reads of it, so an entry, whose content is read
    while it is locked, locks a byte far past its content (`offset`); the device lock keeps
    byte 0, where other tools on the host lock it."""

    def __init__(self, path: str, mode: str = "a+b", offset: int = 0):
        self.path = path
        self.offset = offset
        self.file = open(path, mode)  # not inherited by children (PEP 446)

    def try_acquire(self, exclusive: bool = True) -> bool:
        """Whether the lock is now held; False only when another holder has it. Any other
        failure raises."""
        if _WINDOWS:
            import msvcrt

            self.file.seek(self.offset)
            try:
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno != errno.EACCES:  # what a byte locked by another reports
                    raise
                return False
        else:
            import fcntl

            flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            try:
                fcntl.flock(self.file.fileno(), flag | fcntl.LOCK_NB)
            except BlockingIOError:  # EWOULDBLOCK: another holder has it
                return False
        return True

    def close(self) -> None:
        self.file.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Rig:
    """The host's rig directory: entries and the guard serializing every change to them."""

    def __init__(self, root: str):
        self.root = root
        self.directory = os.path.join(root, "rig")
        self.entries = os.path.join(self.directory, "entries")
        os.makedirs(self.entries, exist_ok=True)

    def guarded(self, action):
        guard = FileLock(os.path.join(self.directory, "guard.lock"))
        try:
            while not guard.try_acquire():
                time.sleep(0.01)
            return action()
        finally:
            guard.close()

    def live(self) -> list:
        """Live entries in arrival order, removing those whose session is gone. Call guarded."""
        found = []
        for name in sorted(os.listdir(self.entries)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.entries, name)
            try:
                probe = FileLock(path, "r+b", ENTRY_LOCK_OFFSET)
            except FileNotFoundError:
                continue  # withdrawn since listed
            try:
                if probe.try_acquire(exclusive=False):
                    probe.close()
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    continue
                probe.file.seek(0)
                entry = json.loads(probe.file.read().decode())
            except ValueError:
                continue
            finally:
                probe.close()
            entry["entry"] = name
            found.append(entry)
        return found

    def checkout(self, key: str) -> str:
        return os.path.join(self.root, "checkouts", key)

    def run(self, run: str) -> str:
        return os.path.join(self.root, "runs", run)


def account(started: float, waited: dict, now: float) -> tuple[float, float]:
    """A run's seconds since it `started`, as (queued, held): every wait of its session for a
    hold (`waited`: its `WAITED`, the waits ended and when the current one began; empty for a
    run of another version, which records none) and the rest, its work, which its estimate is
    judged against. The one definition of both, for the host's status and the caller's."""
    since = waited.get("since")
    queued = waited.get("waited", 0.0) + (0.0 if since is None else now - since)
    return queued, now - started - queued


def remaining(estimate: float, done: float) -> float:
    """Seconds a run `done` seconds into its work is expected to take still: what its estimate
    leaves; once past it, the estimate says nothing of the end, so as long again as it has taken
    (more than its overrun so far, and growing with it)."""
    return estimate - done if done < estimate else done


class Session:
    def __init__(self, rig: Rig, info: dict):
        self.rig = rig
        self.info = info
        self.entries = {}  # kind -> (path, lock)
        self.held = set()
        self.builds = {}  # checkout kind -> its `Owed` build, until that build ends
        self.device = None
        self._count = 0
        self._waited, self._since = 0.0, None  # seconds of waits ended; when the current began
        # Told (waited, waiting) at each change to what the session waits for: its runner's events.
        self.on_account = None

        def own():
            directory = rig.run(info["run"])
            os.makedirs(directory, exist_ok=True)
            lock = FileLock(os.path.join(directory, RUN_LOCK))
            if not lock.try_acquire():
                raise OSError("run %s already has a live session" % info["run"])
            return lock

        self.run_lock = rig.guarded(own)  # released when the runner's process ends

    def built(self, key: str, seconds: float) -> dict:
        """Record the build in checkout `key` that took `seconds`: its bytes, what its build
        added (into a cold checkout, its build's `size`; into one that had built it before, at
        most its `growth`), and the build among its build's `builds`; its entry owes no build
        from now."""
        directory = self.rig.checkout(key)
        after = checkout_bytes(directory)  # outside the guard: a large target takes seconds
        kind = "checkout:" + key

        def record():
            pending = self.builds.pop(kind)
            added = after - _recorded(os.path.join(directory, "size"))
            _record(os.path.join(directory, "size"), after)
            measured = _measured(self.rig.directory, pending.build)
            built = _built(directory)
            if pending.warmth == COLD:
                measured["size"] = added
            elif pending.build in built:
                measured["growth"] = max(added, measured.get("growth", 0))
            measured["builds"] = [*measured.get("builds", []), {
                "seconds": round(seconds), "from": None if pending.warmth == COLD else _read(os.path.join(directory, COMMIT)),
                "to": pending.commit, "built": pending.build in built,
            }][-BUILDS_KEPT:]
            with open(os.path.join(directory, COMMIT), "w") as stream:
                stream.write(pending.commit)
            os.makedirs(os.path.join(self.rig.directory, MEASURED), exist_ok=True)
            _write_json(_measured_path(self.rig.directory, pending.build), measured)
            if pending.build not in built:
                _write_json(os.path.join(directory, BUILT), sorted(built | {pending.build}))
            self._write(kind, "held")
            return {"bytes": after, "added_bytes": added}

        return self.rig.guarded(record)

    def _write(self, kind: str, state: str) -> None:
        path, lock = self.entries[kind]
        entry = dict(self.info, kind=kind, state=state, changed=_now())
        if kind in self.builds:
            pending = self.builds[kind]
            entry["build"], entry["need"], entry["seconds"] = pending.warmth, pending.need, pending.seconds
        lock.file.seek(0)
        lock.file.truncate()
        lock.file.write(json.dumps(entry).encode())
        lock.file.flush()

    def _account(self, blocked: bool = False) -> None:
        """Record in the run's directory (`WAITED`) how long the session has waited, for an
        entry it does not hold yet or, `blocked`, to enter its first: the seconds of waits ended
        (`waited`) and when the current one began (`since`, None while it waits for nothing), so
        `status` counts only its work as its progress. Call guarded, after every change to what
        it waits for."""
        waiting = blocked or any(kind not in self.held for kind in self.entries)
        if waiting == (self._since is not None):
            return
        now = time.time()
        if waiting:
            self._since = now
        else:
            self._waited, self._since = self._waited + now - self._since, None
        _write_json(os.path.join(self.rig.run(self.info["run"]), WAITED), {"waited": self._waited, "since": self._since})
        if self.on_account is not None:
            self.on_account(self._waited, self._since is not None)

    def _others(self, live: list) -> list:
        """Live runs of another protocol, which keep this session's first entry out."""
        others = [e for e in live if e.get("protocol") != PROTOCOL]
        return others if not self.entries else []

    def _enter(self, live: list, kind: str, state: str) -> None:
        """A new entry for `kind`, after every live one. Call guarded."""
        sequence = 1 + max((int(e["entry"].split("-")[0]) for e in live), default=0)
        self._count += 1
        name = "%012d-%d-%d.json" % (sequence, os.getpid(), self._count)
        path = os.path.join(self.rig.entries, name)
        lock = FileLock(path, "w+b", ENTRY_LOCK_OFFSET)
        if not lock.try_acquire():  # a new file under the guard: nothing else can hold it
            raise OSError("cannot lock the new entry %s" % path)
        self.entries[kind] = (path, lock)
        self._write(kind, state)

    def request(self, kind: str) -> list:
        """Enter a request for `kind`; nothing, or, for the session's first request while a
        run of another protocol is live, those runs' entries, entering none (see
        `blocking`)."""

        def enter():
            live = self.rig.live()
            others = self._others(live)
            if not others:
                self._enter(live, kind, "waiting")
            self._account(blocked=bool(others))
            return others

        return self.rig.guarded(enter)

    def claim(self, source: str, build: str, fallback: int, commit: str, seconds: float):
        """Hold a checkout for a run of `source` that builds `build` from `commit` (see the module
        documentation; `fallback`, its need where this host has measured none; `seconds`, what
        its caller priced the build at): its name and
        nothing, or, for the session's first entry while a run of another protocol is live,
        None and those runs' entries, claiming none."""
        base = os.path.join(self.rig.root, "checkouts")

        def take():
            live = self.rig.live()
            others = self._others(live)
            if others:
                self._account(blocked=True)
                return None, others
            name = choose(base, source, held_checkouts(live))
            if name is None:
                names = set(_directories(base))
                name = next("slot-%d" % n for n in range(len(names) + 1) if "slot-%d" % n not in names)
                warmth = COLD
            else:
                warmth = warmth_of(base, name, source)
            directory = os.path.join(base, name)
            os.makedirs(directory, exist_ok=True)
            kind = "checkout:" + name
            self.builds[kind] = Owed(warmth, build, fallback, need(self.rig.directory, directory, build, fallback),
                                     commit, seconds)
            self._enter(live, kind, "held")
            self.held.add(kind)
            self._account()
            with open(os.path.join(directory, "used"), "w") as stream:
                stream.write(_now())
            with open(os.path.join(directory, SOURCE), "w") as stream:
                stream.write(source)
            return name, []

        return self.rig.guarded(take)

    def cap(self, name: str):
        """Clear the claimed checkout `name`'s target when it holds more than `CHECKOUT_LIMIT`
        times the largest build measured on this host: the bytes it held, or None."""
        directory = self.rig.checkout(name)
        held = _recorded(os.path.join(directory, "size"))
        if held <= CHECKOUT_LIMIT * largest(self.rig.directory):
            return None
        shutil.rmtree(os.path.join(directory, "target"), ignore_errors=True)
        _clear(os.path.join(directory, BUILT))
        _record(os.path.join(directory, "size"), checkout_bytes(directory))
        kind = "checkout:" + name

        def cold():
            pending = self.builds[kind]
            self.builds[kind] = Owed(COLD, pending.build, pending.fallback,
                                     need(self.rig.directory, directory, pending.build, pending.fallback),
                                     pending.commit, pending.seconds)
            self._write(kind, "held")

        self.rig.guarded(cold)
        return held

    def blocking(self, kind: str) -> list:
        """Entries that keep `kind` from being held now; empty once it is held."""

        def judge():
            mine = os.path.basename(self.entries[kind][0])
            ahead = True
            blockers = []
            for entry in self.rig.live():
                if entry["entry"] == mine:
                    ahead = False
                    continue
                if entry["run"] == self.info["run"]:
                    continue
                if entry.get("protocol") != PROTOCOL:
                    continue  # a run of another protocol that came after this one waits on it
                if not conflicts(entry["kind"], kind):
                    continue
                if entry["state"] == "held" or ahead:
                    blockers.append(entry)
            if blockers:
                return blockers
            if kind in DEVICE_KINDS and self.device is None:
                device = FileLock(os.path.join(self.rig.root, "lock"))
                if not device.try_acquire():
                    device.close()
                    return [{"kind": "device", "who": "a tool holding %s" % device.path,
                             "run": None, "state": "held"}]
                self.device = device
            self._write(kind, "held")
            self.held.add(kind)
            self._account()
            return []

        return self.rig.guarded(judge)

    def release(self, kind: str) -> None:
        def withdraw():
            entry = self.entries.pop(kind, None)
            if entry is None:
                return
            path, lock = entry
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            lock.close()
            self.held.discard(kind)
            self._account()
            if self.device is not None and not self.held.intersection(DEVICE_KINDS):
                self.device.close()
                self.device = None

        self.rig.guarded(withdraw)

    def release_all(self) -> None:
        for kind in list(self.entries):
            self.release(kind)

    def evict(self, keep: str) -> dict:
        base = os.path.join(self.rig.root, "checkouts")
        # Each building checkout's bytes, measured outside the guard as `built` measures them
        # (a large target takes seconds, and every hold and status waits on the guard); under
        # it, only which checkouts are held is read again, and one held meanwhile is measured.
        sizes = {}

        def act():
            live = self.rig.live()
            held = held_checkouts(live)
            if (held | {keep}) - set(sizes):
                return None
            before = shutil.disk_usage(self.rig.root).free
            running = {e["run"] for e in live}
            keys = os.listdir(base) if os.path.isdir(base) else []
            fallback = largest(self.rig.directory)
            entries = {e["kind"].split(":", 1)[1]: e for e in live if e["kind"].startswith("checkout:")}

            def growth(key):  # what checkout `key` may still add
                bound = owed(entries[key], fallback)
                return bound - min(bound, max(0, sizes[key] - _recorded(os.path.join(base, key, "size"))))

            # This run's checkouts, and the other runs', but those of runs waiting for room after it.
            later = _waiting_after(live, self.entries.get("room"))
            checkouts = {key: entry["run"] for key, entry in entries.items()}
            mine = {key for key, run in checkouts.items() if run == self.info["run"]}
            others = {key: run for key, run in checkouts.items() if key not in mine and run not in later}
            needed = sum(growth(key) for key in mine | set(others))
            runs = os.path.join(self.rig.root, "runs")
            candidates = [
                (_mtime(os.path.join(base, key, "used")), "checkouts", key)
                for key in keys if key not in held and key != keep
            ] + [
                (_mtime(os.path.join(runs, run)), "runs", run)
                for run in (os.listdir(runs) if os.path.isdir(runs) else [])
                if run not in running and os.path.isdir(os.path.join(runs, run))
            ]
            removed = []
            for _, kind, name in sorted(candidates):
                if shutil.disk_usage(self.rig.root).free >= needed:
                    break
                directory = os.path.join(self.rig.root, kind, name)
                if kind == "runs" and (_owned(directory) or _owned(directory, STAGE_LOCK)
                                       or os.path.exists(os.path.join(directory, ENDED))):
                    continue  # a live run's, one its caller is staging, or an ended one not fetched
                shutil.rmtree(directory, ignore_errors=True)
                removed.append("%s/%s" % (kind, name))
            return {
                "removed": removed,
                "free_bytes": shutil.disk_usage(self.rig.root).free,
                "freed_bytes": shutil.disk_usage(self.rig.root).free - before,
                "needed_bytes": needed,
                "own_bytes": sum(growth(key) for key in mine),
                "releasable_bytes": sum(sizes[key] for key in others),
                "releasing": sorted(set(others.values())),
            }

        while True:
            for key in (self.rig.guarded(lambda: held_checkouts(self.rig.live())) | {keep}) - set(sizes):
                sizes[key] = checkout_bytes(os.path.join(base, key))
            evicted = self.rig.guarded(act)
            if evicted is not None:
                return evicted


def _waiting_after(live: list, room) -> set:
    """The runs waiting for room after the `room` request entry (its path; None: after none,
    as one not waiting is after every run that is)."""
    if room is None:
        return set()
    mine = os.path.basename(room[0])
    return {e["run"] for e in live if e["kind"] == "room" and e["entry"] > mine}


def held_checkouts(live: list) -> set:
    return {e["kind"].split(":", 1)[1] for e in live if e["kind"].startswith("checkout:")}


def _directories(base: str) -> list:
    names = os.listdir(base) if os.path.isdir(base) else []
    return [name for name in names if os.path.isdir(os.path.join(base, name))]


def _waited(rig: Rig, run: str) -> dict:
    """Run `run`'s `WAITED`; empty until it first waits."""
    try:
        with open(os.path.join(rig.run(run), WAITED)) as stream:
            return json.load(stream)
    except FileNotFoundError:
        return {}


def _read(path: str):
    try:
        with open(path) as stream:
            return stream.read()
    except FileNotFoundError:
        return None


def choose(base: str, source: str, held: set):
    """The checkout, of those not in `held`, a run of `source` takes: the one last installed
    from it, else the one most recently claimed (most likely nearest the trunk), then used;
    None when every one is held."""
    free = [name for name in _directories(base) if name not in held]
    if not free:
        return None

    def rank(name):
        recorded = os.path.join(base, name, SOURCE)  # written at each claim
        return _read(recorded) == source, _mtime(recorded), _mtime(os.path.join(base, name, "used"))

    return max(free, key=rank)


def warmth_of(base: str, name: str, source: str) -> int:
    """How warm existing checkout `name` is for a run of `source` (`WARMTH`)."""
    return SAME if _read(os.path.join(base, name, SOURCE)) == source else OTHER


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


def _owned(run: str, lock: str = RUN_LOCK) -> bool:
    """Whether a live process holds this run directory's `lock`: its runner's (`RUN_LOCK`), or
    `start`'s while its caller stages it (`STAGE_LOCK`). Call guarded."""
    try:
        probe = FileLock(os.path.join(run, lock), "r+b")
    except FileNotFoundError:
        return False
    try:
        return not probe.try_acquire()
    finally:
        probe.close()


class Owed:
    """A claimed checkout's build, until it ends: its warmth, its build (`engine.build_key`), the
    claim's fallback, the bytes it may add (`need`), the commit it builds and the seconds its
    caller priced it at."""

    def __init__(self, warmth: int, build: str, fallback: int, need: int, commit: str, seconds: float):
        self.warmth, self.build, self.fallback, self.need = warmth, build, fallback, need
        self.commit, self.seconds = commit, seconds


MEASURED = "builds"  # in `<root>/rig`: each build's measurements, `<build>.json`


def _measured_path(directory: str, build: str) -> str:
    return os.path.join(directory, MEASURED, build + ".json")


def _measured(directory: str, build: str) -> dict:
    """What builds of `build` measured on the host whose rig directory is `directory`: its
    `size` and its `growth`, each once measured (see the module documentation); none while no
    build of it was. Written whole (`_write_json`), so a record that does not read is a fault."""
    try:
        with open(_measured_path(directory, build)) as stream:
            return json.load(stream)
    except FileNotFoundError:
        return {}


def slowest(directory: str) -> float:
    """The slowest build measured on the host, `BUILD_PRIOR_SECONDS` while none is."""
    found = os.path.join(directory, MEASURED)
    names = os.listdir(found) if os.path.isdir(found) else []
    return max((build["seconds"] for name in names if name.endswith(".json")
                for build in _measured(directory, name[:-len(".json")]).get("builds", [])),
               default=BUILD_PRIOR_SECONDS)


def largest(directory: str) -> int:
    """The largest build `size` measured on the host, 0 when none is."""
    found = os.path.join(directory, MEASURED)
    names = os.listdir(found) if os.path.isdir(found) else []
    return max((_measured(directory, name[:-len(".json")]).get("size", 0)
                for name in names if name.endswith(".json")), default=0)


def _built(checkout: str) -> set:
    """The builds checkout `checkout` has built since its target was last empty."""
    try:
        with open(os.path.join(checkout, BUILT)) as stream:
            return set(json.load(stream))
    except FileNotFoundError:
        return set()


def need(directory: str, checkout: str, build: str, fallback):
    """The bytes a build of `build` may add to `checkout` (None for none there yet, a cold
    one) on the host whose rig directory is `directory`: its `growth` where the checkout has
    built it before and it was measured, else its `size`; `fallback` where neither was."""
    measured = _measured(directory, build)
    if checkout is not None and build in _built(checkout) and "growth" in measured:
        return measured["growth"]
    return measured.get("size", fallback)


def owed(entry: dict, fallback: int) -> int:
    """The bytes a live checkout entry's build may still add: its need until its build ends;
    for an entry of another version, which records none, `fallback` while it builds."""
    if "build" not in entry:
        return 0
    return entry.get("need", fallback)


def _recorded(path: str) -> int:
    """A count `built` recorded (a checkout's `size`), 0 when none is; each record is written
    whole (`_record`), so one that does not read is a fault, never unrecorded."""
    try:
        with open(path) as stream:
            return int(stream.read().strip())
    except FileNotFoundError:
        return 0


def _write_json(path: str, value: dict) -> None:
    staged = "%s.%d.tmp" % (path, os.getpid())
    with open(staged, "w") as stream:
        json.dump(value, stream)
    os.replace(staged, path)


def _record(path: str, value: int) -> None:
    staged = "%s.%d.tmp" % (path, os.getpid())
    with open(staged, "w") as stream:
        stream.write(str(value))
    os.replace(staged, path)


def checkout_bytes(directory: str) -> int:
    """The bytes a checkout's files hold now (what `built` records as its `size`)."""
    total = 0
    for parent, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.lstat(os.path.join(parent, name)).st_size
            except FileNotFoundError:
                pass  # removed since listed
    return total


NOTE = "rig: "  # how the rig's own lines start in a phase's log


class Events:
    """A run's numbered events (`events.jsonl`) and each phase's output (`logs/<phase>.log`),
    appended as they happen, so a caller reads them whenever it is connected."""

    def __init__(self, directory: str):
        self._lock = threading.Lock()
        self._path = os.path.join(directory, EVENTS)
        self._logs = os.path.join(directory, "logs")
        os.makedirs(self._logs, exist_ok=True)
        self._sequence = 0
        self.phase = "start"

    def emit(self, event: str, **fields) -> None:
        with self._lock:
            self._sequence += 1
            line = json.dumps(dict({"seq": self._sequence, "event": event, "at": time.time()}, **fields))
            with open(self._path, "a") as stream:
                stream.write(line + "\n")

    def output(self, stream: str, text: str) -> None:
        with self._lock:
            with open(os.path.join(self._logs, self.phase + ".log"), "a") as log:
                log.write(text if stream == "stdout" else "[stderr] " + text)
        self.emit("output", phase=self.phase, stream=stream, text=text)

    def note(self, message: str) -> None:
        self.output("stdout", NOTE + message[:400] + "\n")


def kill_group(process: subprocess.Popen, grace: float) -> None:
    """Terminate the child's whole process group, escalating to a kill after `grace`."""
    if _WINDOWS:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Watch:
    """One job's watch for its hang (see the module documentation): the processes of its group
    that report on SIGUSR1 (`stacks-<pid>.txt` in `directory`), each asked once, when `held()`
    passes `estimate` or its test results stall."""

    def __init__(self, directory: str, group: int, held, estimate: float, emit):
        self.directory, self.group, self.held, self.estimate, self.emit = directory, group, held, estimate, emit
        self.asked = set()
        self.unanswered = False  # said that it had reason to ask but no process to ask
        self.last = None  # when the last test result was printed
        self.gaps = []  # the recent gaps between test results, seconds

    def saw(self, text: str) -> None:
        if not TEST_RESULT.match(text):
            return
        now = time.monotonic()
        if self.last is not None:
            self.gaps = (self.gaps + [now - self.last])[-STALL_RECENT:]
        self.last = now

    def reason(self):
        """Why the job's processes are to report now, or None."""
        if self.held() >= self.estimate:
            return "its held time passed its %ds estimate" % self.estimate
        if self.last is None:
            return None
        stalled = time.monotonic() - self.last
        threshold = max(STALL_FLOOR, STALL_FACTOR * max(self.gaps, default=0.0))
        if stalled >= threshold:
            return "no test result for %ds (%d times the longest recent gap, at least %ds)" % (
                stalled, STALL_FACTOR, STALL_FLOOR)
        return None

    def check(self) -> None:
        if _WINDOWS:
            return  # no SIGUSR1
        why = self.reason()
        if why is None:
            return
        for name in sorted(os.listdir(self.directory)):
            found = REPORTS.match(name)
            pid = int(found.group(1)) if found else None
            if pid is None or pid in self.asked:
                continue
            try:
                if os.getpgid(pid) != self.group:
                    continue  # not the job's (a pid reused since)
                os.kill(pid, signal.SIGUSR1)
            except OSError:
                continue  # ended
            self.asked.add(pid)
            self.emit(pid, os.path.join("logs", "stacks", name), why)
        if not self.asked and not self.unanswered:
            self.unanswered = True
            self.emit(None, None, why)  # nothing of the job reports its stacks: said once


class Child:
    """One command in its own process group, its output passed on line by line and its
    standard output kept."""

    def __init__(self, request: dict, on_output, *, whole: bool):
        env = dict(os.environ)
        env.update(request.get("env", {}))
        data = request.get("input")
        options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if _WINDOWS
            else {"start_new_session": True}
        )
        self.process = subprocess.Popen(
            request["argv"],
            cwd=request.get("cwd"),
            env=env,
            stdin=subprocess.DEVNULL if data is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **options,
        )
        self._on_output = on_output
        self._whole = whole  # its output is read: kept whole, never cut (`LINE_HEAD`)
        self.stdout = []
        self.code = None
        self._pumps = [
            threading.Thread(target=self._pump, args=(self.process.stdout, "stdout"), daemon=True),
            threading.Thread(target=self._pump, args=(self.process.stderr, "stderr"), daemon=True),
        ]
        for pump in self._pumps:
            pump.start()
        if data is not None:
            threading.Thread(target=self._feed, args=(data,), daemon=True).start()
        self.finished = threading.Event()
        threading.Thread(target=self._wait, daemon=True).start()

    def _feed(self, data: str) -> None:
        try:
            self.process.stdin.write(data.encode())
            self.process.stdin.close()
        except OSError:
            pass

    def _pump(self, stream, name: str) -> None:
        for line in iter(self._readline(stream), b""):
            text = line.decode("utf-8", "replace")
            if name == "stdout" and self._whole:
                self.stdout.append(text)
            self._on_output(name, text)

    def _readline(self, stream):
        """The stream's next line: whole for a step whose output is read; else at most its
        `LINE_HEAD` bytes and its last `LINE_TAIL`, what lies between counted, never held."""
        if self._whole:
            return stream.readline

        def bounded() -> bytes:
            head = stream.readline(LINE_HEAD)
            if len(head) < LINE_HEAD or head.endswith(b"\n"):
                return head
            elided, tail = 0, b""
            while True:
                piece = stream.readline(LINE_HEAD)
                tail = (tail + piece)[-LINE_TAIL:]
                elided += len(piece)
                if not piece or piece.endswith(b"\n"):
                    break
            elided -= len(tail)
            return head + b" [... %d bytes of this line not kept by the rig ...] " % elided + tail

        return bounded

    def _wait(self) -> None:
        code = self.process.wait()
        if not _WINDOWS:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + OUTPUT_GRACE_SECONDS
        for pump in self._pumps:
            pump.join(max(0.0, deadline - time.monotonic()))
        if any(pump.is_alive() for pump in self._pumps):
            self._on_output("stderr", "agent: the command exited %d, but a process that left its group still "
                                      "holds its output open; its exit is the step's, not waited on further\n" % code)
        self.code = code
        self.finished.set()

    def kill(self) -> None:
        kill_group(self.process, TERMINATE_GRACE_SECONDS)
        self.finished.wait(TERMINATE_GRACE_SECONDS)


def user_tool_path() -> None:
    """Put the user's tool directories (uv, cargo) first on PATH for every child."""
    home = os.path.expanduser("~")
    tools = [os.path.join(home, ".local", "bin"), os.path.join(home, ".cargo", "bin")]
    os.environ["PATH"] = os.pathsep.join(tools + [os.environ.get("PATH", "")])


def _public(entry: dict) -> dict:
    return {k: entry[k] for k in ("kind", "who", "run", "purpose", "estimate", "started", "state") if k in entry}


class Failed(Exception):
    """The run ends here: a step failed, or it was stopped."""


class Runner:
    """Carries out one run's plan to its end; see the module documentation."""

    def __init__(self, rig: Rig, run: str):
        self.rig = rig
        self.directory = rig.run(run)
        with open(os.path.join(self.directory, PLAN)) as stream:
            self.plan = json.load(stream)
        self.events = Events(self.directory)
        self.stopping = threading.Event()
        info = dict(self.plan["info"], run=run, started=time.time(), protocol=PROTOCOL,
                    pid=os.getpid(), hostname=socket.gethostname())
        self.session = Session(rig, info)
        self.session.on_account = lambda waited, waiting: self.events.emit("queue", waited=waited, waiting=waiting)
        self.child = None
        self.codes = {}
        self.phases = []
        self.bound = {}  # `{NAME}` in later operations: `capture:C` and `checkout:SIDE`
        self.checkouts = {}  # side -> its checkout's name
        self._phase_started = time.monotonic()

    def enter(self, phase: str) -> None:
        now = time.monotonic()
        if self.events.phase != "start":
            self.phases.append({"phase": self.events.phase, "seconds": round(now - self._phase_started, 3)})
        self.events.phase, self._phase_started = phase, now
        self.events.emit("phase", phase=phase)

    def _fill(self, text: str) -> str:
        for name, value in self.bound.items():
            text = text.replace("{%s}" % name, value)
        return text

    def claim(self, side: str, source: str, build: str, fallback: int, commit: str, seconds: float) -> None:
        reported = None
        while True:
            self._check_stop()
            name, others = self.session.claim(source, build, fallback, commit, seconds)
            if name is not None:
                break
            if [e.get("run") for e in others] != reported:
                reported = [e.get("run") for e in others]
                self.events.note("waiting for a checkout behind %s" % ", ".join(
                    "%s %s" % (e["kind"], e.get("run")) for e in others))
                self.events.emit("waiting", kind="checkout", blocking=[_public(e) for e in others])
            time.sleep(POLL_SECONDS)
        self.checkouts[side] = name
        self.bound["checkout:" + side] = self.rig.checkout(name)
        self.events.emit("held", kind="checkout:" + name)
        self.events.note("checkout %s for %s" % (name, side))
        cleared = self.session.cap(name)
        if cleared is not None:
            self.events.note("cleared checkout %s's target: it held %.1f GB, past %d times the largest "
                             "build measured" % (name, cleared / 1e9, CHECKOUT_LIMIT))

    def _check_stop(self) -> None:
        if self.stopping.is_set():
            raise Failed("stopped by mrig stop")

    def hold(self, kind: str) -> None:
        reported = None
        entered = False
        while True:
            self._check_stop()
            if not entered:
                others = self.session.request(kind)
                entered = not others
            blockers = [_public(e) for e in (self.session.blocking(kind) if entered else others)]
            if not blockers:
                self.events.emit("held", kind=kind)
                return
            if [b.get("run") for b in blockers] != reported:
                reported = [b.get("run") for b in blockers]
                shown = ", ".join("%s %s" % (b["kind"], b.get("run")) for b in blockers)
                self.events.note("waiting for %s behind %s" % (kind, shown))
                self.events.emit("waiting", kind=kind, blocking=blockers)
            time.sleep(POLL_SECONDS)

    def make_room(self, side: str, host: str) -> None:
        """Evict until the building checkouts' needs fit; short of it, wait for the live runs'
        checkouts that would cover it, without the install hold, then evict again; fail when
        even they could not (see the module documentation)."""
        reported = None
        while True:
            evicted = self.session.evict(self.checkouts[side])
            if evicted["removed"]:
                self.events.note("evicted least recently used %s" % evicted["removed"])
                self.events.emit("evicted", removed=evicted["removed"], freed_bytes=evicted["freed_bytes"])
            needed, free = evicted["needed_bytes"], evicted["free_bytes"]
            if free >= needed:
                self.session.release("room")
                return
            short = (needed - free) / 1e9
            if free + evicted["releasable_bytes"] < evicted["own_bytes"]:
                raise Failed(
                    "%s: its building checkouts may grow by %.1f GB but only %.1f GB is free with "
                    "every unheld checkout and ended run evicted (%.1f GB short), and the live runs' "
                    "checkouts would not cover its own %.1f GB as they end"
                    % (host, needed / 1e9, free / 1e9, short, evicted["own_bytes"] / 1e9))
            if "room" not in self.session.entries:
                self.session.request("room")
            if evicted["releasing"] != reported:
                reported = evicted["releasing"]
                self.events.note("waiting for room: %.1f GB short until %s end" % (short, ", ".join(reported)))
                self.events.emit("waiting", kind="room", blocking=[
                    {"kind": "disk", "run": run, "state": "held"} for run in reported])
            self.session.release("install")
            runs = lambda: self.rig.guarded(lambda: {e["run"] for e in self.rig.live()})
            live = runs()
            while live <= runs():  # until one of them ends: what frees room
                self._check_stop()
                time.sleep(POLL_SECONDS)
            self.hold("install")

    def execute(self, op: dict) -> None:
        argv = [self._fill(argument) for argument in op["argv"]]
        unless = self._fill(op["unless"]) if "unless" in op else None
        if unless is not None and os.path.exists(unless):
            self.events.note("skipped `%s`: %s exists" % (" ".join(argv)[:160], unless))
            return
        request = dict(op, argv=argv, cwd=self._fill(op["cwd"]),
                       env={k: self._fill(v) for k, v in op.get("env", {}).items()})
        self.events.note("$ " + " ".join(argv))
        watch = None
        if "job" in op:
            stacks = os.path.join(self.directory, "logs", "stacks")
            os.makedirs(stacks, exist_ok=True)
            request["env"] = dict(request["env"], **{STACKS: stacks})
        output = self.events.output
        if "job" in op:
            output = lambda stream, text: (watch is not None and watch.saw(text), self.events.output(stream, text))
        try:
            self.child = Child(request, output, whole="capture" in op or "stdout_to" in op)
        except OSError as error:
            self.events.output("stderr", "agent: %s\n" % error)
            code, stdout = 127, ""
        else:
            if "job" in op:
                watch = Watch(stacks, self.child.process.pid, self._held, self.session.info["estimate"], self._asked)
            while not self.child.finished.wait(POLL_SECONDS):
                if self.stopping.is_set():
                    self.child.kill()
                if watch is not None:
                    watch.check()
            code, stdout = self.child.code, "".join(self.child.stdout)
            self._check_stop()
        self.events.emit("exited", code=code)
        if "job" in op:
            self.codes[op["job"]] = code
        elif op.get("check", True) and code != 0:
            raise Failed("`%s` exited %d" % (" ".join(argv)[:160], code))
        if "capture" in op:
            lines = stdout.strip().splitlines()
            self.bound["capture:" + op["capture"]] = lines[-1] if lines else ""
        if "stdout_to" in op:
            with open(op["stdout_to"], "w") as stream:
                stream.write(stdout)

    def _held(self) -> float:
        """The run's held time now (`account`)."""
        session = self.session
        return account(session.info["started"], {"waited": session._waited, "since": session._since}, time.time())[1]

    def _asked(self, pid, path, why: str) -> None:
        if pid is None:
            self.events.note("%s, but no live process of the job reports its stacks (none registered, or none left)"
                             % why)
        else:
            self.events.note("asked process %d for its stacks: %s; they go to %s" % (pid, why, path))
        self.events.emit("stacks", pid=pid, path=path, reason=why)

    def operate(self, op: dict) -> None:
        kind = op["op"]
        if kind == "phase":
            self.enter(op["phase"])
        elif kind == "checkout":
            self.claim(op["side"], op["source"], op["build"], op["fallback"], op["commit"], op["seconds"])
        elif kind == "hold":
            self.hold(op["kind"])
        elif kind == "release":
            self.session.release(op["kind"])
        elif kind == "evict":
            self.make_room(op["side"], op["host"])
        elif kind == "run":
            self.execute(op)
        elif kind == "built":
            built = self.session.built(self.checkouts[op["side"]], time.monotonic() - self._phase_started)
            self.events.note("checkout holds %.1f GB, %.1f GB added by this build"
                             % (built["bytes"] / 1e9, built["added_bytes"] / 1e9))
        else:
            raise Failed("unknown operation %s" % op)

    def carry_out(self) -> dict:
        failed = None
        try:
            for op in self.plan["ops"]:
                self._check_stop()
                self.operate(op)
        except Failed as failure:
            failed = {"phase": self.events.phase, "detail": str(failure)}
        except Exception as error:  # the run's end is recorded whatever ends it
            failed = {"phase": self.events.phase, "detail": "agent: %r" % error}
        if self.child is not None and not self.child.finished.is_set():
            self.child.kill()
        self.session.release_all()
        self.phases.append({"phase": self.events.phase,
                            "seconds": round(time.monotonic() - self._phase_started, 3)})
        ended = {"codes": self.codes, "phases": self.phases, "failed": failed}
        staged = os.path.join(self.directory, ".%s.%d" % (ENDED, os.getpid()))
        with open(staged, "w") as stream:
            json.dump(ended, stream)
        os.replace(staged, os.path.join(self.directory, ENDED))
        self.events.emit("ended", **ended)
        return ended


def invocation() -> list:
    """What runs this agent after the Python executable: the deployed host package's agent
    module (`PACKAGE magnitude_rig.host.agent`, the package a zip), or this file where it runs
    alone."""
    package = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return [package, MODULE] if os.path.isfile(package) else [os.path.abspath(__file__)]


def runner(arguments) -> int:
    """The detached runner `start` launches: owns the run until its plan ends."""
    user_tool_path()
    current = Runner(Rig(arguments.root), arguments.run)
    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), lambda signum, frame: current.stopping.set())
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # no connection's end ends the run
    staged = os.path.join(current.directory, ".%s.%d" % (RUNNER, os.getpid()))
    with open(staged, "w") as stream:
        json.dump({"pid": os.getpid(), "agent": invocation()}, stream)
    os.replace(staged, os.path.join(current.directory, RUNNER))
    current.carry_out()
    return 0


def start(arguments) -> int:
    """Hold the run's directory while its caller stages it (`STAGE_LOCK`, held until this
    process ends), and say so; then write the plan on stdin to it and start its runner
    detached, printing the runner's pid and agent once it owns the run, or on stderr why it
    does not. Stdin closed without a plan withdraws: nothing starts. Once a runner has owned
    the run (`RUNNER`), every later `start` of it prints that runner (`started`) and stages
    nothing, so a caller whose reply was lost follows the run it started."""
    rig = Rig(arguments.root)
    directory = rig.run(arguments.run)

    def hold():
        os.makedirs(directory, exist_ok=True)
        lock = FileLock(os.path.join(directory, STAGE_LOCK))
        if lock.try_acquire():
            _write_json(state, {"pid": os.getpid(), "waiting": True})
            return lock
        lock.close()
        # An earlier attempt holds it, which only its holder writes `state` for. One still
        # waiting for its plan is one its caller gave up, which this attempt retries: a silent
        # link loss would keep it waiting until the connection's keepalive ends it, hours
        # later. It is superseded. One starting the runner is waited for.
        with open(state) as stream:
            earlier = json.load(stream)
        if earlier["waiting"]:
            try:
                os.kill(earlier["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass  # ending already
        return None

    state = os.path.join(directory, STAGE_STATE)
    staging = rig.guarded(hold)  # kept open, so held, until this process ends
    while staging is None:  # an earlier attempt's, until it ends
        time.sleep(POLL_SECONDS)
        staging = rig.guarded(hold)
    started = os.path.join(directory, RUNNER)
    if os.path.exists(started):  # an earlier attempt started its runner and its reply was lost
        with open(started) as stream:
            print(json.dumps({"started": json.load(stream)}), flush=True)
        return 0
    print(json.dumps({"staging": arguments.run}), flush=True)
    plan = sys.stdin.read()
    if not plan:
        return 0
    # From here no later attempt supersedes this one: it starts the runner, or ends.
    rig.guarded(lambda: _write_json(state, {"pid": os.getpid(), "waiting": False}))
    with open(os.path.join(directory, PLAN), "w") as stream:
        stream.write(plan)
    options = (
        {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
        if _WINDOWS
        else {"start_new_session": True}
    )
    with open(os.path.join(directory, RUNNER_LOG), "ab") as log:
        process = subprocess.Popen(
            [sys.executable, *invocation(), "runner", "--root", arguments.root, "--run", arguments.run],
            cwd=arguments.root, stdin=subprocess.DEVNULL, stdout=log, stderr=log, **options,
        )
    path = os.path.join(directory, RUNNER)
    while not os.path.exists(path):
        if process.poll() is not None:
            with open(os.path.join(directory, RUNNER_LOG), errors="replace") as stream:
                print("the runner exited %d: %s" % (process.returncode, stream.read()[-2000:]), file=sys.stderr)
            return 1
        time.sleep(0.05)
    with open(path) as stream:
        print(stream.read())
    return 0


def follow(arguments) -> int:
    """Print the run's events after `--after`, as they are written, through `ended`; a runner
    gone without ending prints `vanished`. Output is printed only as it is written from now on:
    what came before is in its phase's log, which the caller fetches, so a caller following again
    after a lost connection reads at once how the run stands, over however slow a link."""
    directory = Rig(arguments.root).run(arguments.run)
    path = os.path.join(directory, EVENTS)
    position = 0
    live = False  # past the events written before this follow began
    while True:
        gone = not _owned(directory)  # read before the events, so none written before it is missed
        if os.path.exists(path):
            with open(path, "rb") as stream:
                stream.seek(position)
                while True:
                    line = _event_line(stream)
                    if line is None:
                        break
                    position = stream.tell()
                    event = json.loads(line)
                    if event["seq"] > arguments.after and (live or event["event"] != "output"):
                        sys.stdout.write(line)
                    if event["event"] == "ended":
                        sys.stdout.flush()
                        return 0
            sys.stdout.flush()
        live = True
        if gone:
            print(json.dumps({"seq": 0, "event": "vanished"}), flush=True)
            return 1
        time.sleep(POLL_SECONDS)


# The longest event line a follow reads whole. A runner of this version writes none longer (a
# logged line is bounded, `LINE_HEAD`); one of an earlier version may have written gigabytes on
# one line, which a follow holding it whole sat on, at 4 GB, for as long as it was followed.
EVENT_BYTES = 1 << 20
SEQ = re.compile(rb'^\{"seq": (\d+), "event": "output".*?"phase": "([^"]*)"')


def _event_line(stream):
    """The next whole event line from `stream` (binary), as text; None until one is written
    whole. A line longer than `EVENT_BYTES` (an output line of an earlier runner) is read through
    in pieces, never held, and stands as an output event saying how much was not forwarded."""
    start = stream.tell()
    head = stream.readline(EVENT_BYTES)
    if head.endswith(b"\n"):
        return head.decode("utf-8", "replace")
    if len(head) < EVENT_BYTES:
        stream.seek(start)
        return None  # still being written
    skipped = len(head)
    while True:
        piece = stream.readline(EVENT_BYTES)
        skipped += len(piece)
        if piece.endswith(b"\n"):
            break
        if len(piece) < EVENT_BYTES:
            stream.seek(start)
            return None  # still being written
    found = SEQ.match(head)
    if found is None:
        raise ValueError("an event line of %d bytes that is no output event" % skipped)
    return json.dumps({"seq": int(found.group(1)), "event": "output", "phase": found.group(2).decode(),
                       "stream": "stderr", "text": "agent: an output line of %d bytes not forwarded\n" % skipped}) + "\n"


def state(arguments) -> int:
    """Print how the run stands: `running` while its runner lives, `ended` once it recorded its
    end, `vanished` when its runner is gone without one (killed, or lost to a reboot), `absent`
    when its directory is gone too (pruned)."""
    directory = Rig(arguments.root).run(arguments.run)
    gone = not _owned(directory)  # read before the end, so an end written before it is seen
    if os.path.exists(os.path.join(directory, ENDED)):
        standing = "ended"
    elif not gone:
        standing = "running"
    elif os.path.isdir(directory):
        standing = "vanished"
    else:
        standing = "absent"
    print(json.dumps({"state": standing}))
    return 0


def stop(arguments) -> int:
    """Stop the run's runner: it ends its command, releases its holds and records the stop.
    Signalled only under the guard while it runs (it holds the run's lock and has recorded no
    end), so its pid is the runner's, never one the OS has since given another process. Prints
    whether it was stopped, and why not."""
    rig = Rig(arguments.root)
    directory = rig.run(arguments.run)

    def deliver():
        if not _owned(directory) or os.path.exists(os.path.join(directory, ENDED)):
            return "run %s is not running" % arguments.run
        with open(os.path.join(directory, RUNNER)) as stream:
            pid = json.load(stream)["pid"]
        if _WINDOWS:  # no signal to handle there: its process tree ends, and its locks with it
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, signal.SIGTERM)
        return None

    reason = rig.guarded(deliver)
    print(json.dumps({"stopped": True} if reason is None else {"stopped": False, "reason": reason}))
    return 0


def fetched(arguments) -> int:
    """Remove an ended run's directory once its caller has fetched it."""
    rig = Rig(arguments.root)
    directory = rig.run(arguments.run)

    def remove():
        if _owned(directory):
            return "run %s is still running" % arguments.run
        shutil.rmtree(directory, ignore_errors=True)
        return None

    refusal = rig.guarded(remove)
    print(json.dumps({"removed": refusal is None, "reason": refusal}))
    return 0 if refusal is None else 1


def status(arguments) -> int:
    if len(arguments.build) != len(arguments.source):
        raise SystemExit("status: one --build for each --source")
    print(json.dumps(report(arguments.root, list(zip(arguments.source, arguments.build)))))
    return 0


def report(root: str, sides: list) -> dict:
    """The host's live entries, each with the time its run is expected to take still and how far
    it is past its estimate (`remaining`, `over`), the wait of a new request of each kind behind
    the runs holding or queued for a conflicting hold (each once, by its `remaining`: `waits`),
    the build seconds each live run still owes (`building`); and, for a new run whose sides are
    `sides` (each a source and its build), the checkout each would claim now, with its warmth
    for that source (None: none is free, a cold build), the commit it last built and whether it
    has built that build (`built`), and the bytes its build may add as measured here (`need`,
    None when unmeasured), each build's measurements here (`measured`), the largest build of any
    (`largest`), the slowest (`slowest`), and the room its builds would find (`room`)."""
    rig = Rig(root)
    base = os.path.join(root, "checkouts")

    def read():
        live = rig.live()
        # Every live run: those with entries, and those staged or claiming, which hold none yet.
        directories = [rig.run(name) for name in _directories(os.path.join(root, "runs"))]
        live_runs = {e.get("run") for e in live} | {
            os.path.basename(d) for d in directories
            if not os.path.exists(os.path.join(d, ENDED)) and (_owned(d) or _owned(d, STAGE_LOCK))
        }
        taken, checkouts, needs = held_checkouts(live), [], []
        for source, build in sides:
            name = choose(base, source, taken)
            taken.add(name)  # None among the names changes no later choice
            checkouts.append(None if name is None else {
                "name": name, "source": _read(os.path.join(base, name, SOURCE)),
                "size": _recorded(os.path.join(base, name, "size")), "warmth": warmth_of(base, name, source),
                "commit": _read(os.path.join(base, name, COMMIT)), "built": build in _built(os.path.join(base, name)),
            })
            needs.append(need(rig.directory, None if name is None else os.path.join(base, name), build, None))
        # Only this protocol's runs record their waits.
        waited = {run: _waited(rig, run) for run in {e["run"] for e in live if e.get("protocol") == PROTOCOL}}
        return live, live_runs, checkouts, needs, waited

    entries, live_runs, checkouts, needs, waited = rig.guarded(read)
    now = time.time()
    runs = {}  # run -> (remaining, over, queued, held), from any of its entries: each carries its run's info
    for e in entries:
        if e["run"] not in runs:
            queued, held = account(e["started"], waited.get(e["run"], {}), now)
            runs[e["run"]] = (remaining(e["estimate"], held), max(0.0, held - e["estimate"]), queued, held)
    seconds = slowest(rig.directory)
    waits = {}
    for kind in ("build", "device", "timed"):
        ahead = {e["run"] for e in entries if e.get("protocol") != PROTOCOL or conflicts(e["kind"], kind)}
        waits[kind] = sum(runs[run][0] for run in ahead)
    legacy = None
    device = FileLock(os.path.join(root, "lock"))
    if not any(e["kind"] in DEVICE_KINDS and e["state"] == "held" for e in entries):
        if not device.try_acquire():
            try:
                with open(os.path.join(root, "holder.json")) as stream:
                    legacy = json.load(stream)
            except (FileNotFoundError, ValueError):
                legacy = {}
    device.close()
    found = {
        "entries": [dict(_public(e), **dict(zip(("remaining", "over", "queued", "held"), runs[e["run"]])))
                    for e in entries],
        "waits": waits,
        "building": [owed for owed in (owing(entries, run, seconds, now) for run in live_runs) if owed > 0],
        "room": room(root, entries, {run: left for run, (left, *_) in runs.items()}),
        "device_lock_held_elsewhere": legacy,
        "free_bytes": shutil.disk_usage(root).free,
        "checkouts": checkouts,
        "needs": needs,
        "measured": {build: _measured(rig.directory, build) for _, build in sides},
        "largest": largest(rig.directory),
        "slowest": seconds,
    }
    return found


def room(root: str, live: list, left: dict) -> dict:
    """The room a new run's builds would find (`evict`), by the time each live run is expected
    to take still (`left`, by run): the bytes free and in unheld checkouts (`available`), what
    the live checkouts' builds still owe (`owed`), and, the first ending first, each
    live checkout's run's end with the bytes that frees and the debt it ends (`ending`: each is
    evictable and owes nothing then). From the recorded sizes and needs, measuring nothing."""
    base = os.path.join(root, "checkouts")
    fallback = largest(os.path.join(root, "rig"))
    held = {e["kind"].split(":", 1)[1]: e for e in live if e["kind"].startswith("checkout:")}
    size = lambda name: _recorded(os.path.join(base, name, "size"))
    return {
        "available": shutil.disk_usage(root).free + sum(size(name) for name in _directories(base) if name not in held),
        "owed": sum(owed(entry, fallback) for entry in held.values()),
        "ending": [[ends, size(name), debt] for ends, name, debt in
                   sorted((left[entry["run"]], name, owed(entry, fallback)) for name, entry in held.items())],
    }


def owing(live: list, run: str, unpriced: float, now: float) -> float:
    """The build seconds live run `run` still owes the host: what its caller priced each build
    of a checkout it holds whose build is owed at (`unpriced` for an entry of another version,
    and once while it has no entry: it is still staged, or claiming its first checkout), less
    how long it has built the side it is building."""
    entries = [e for e in live if e.get("run") == run]
    checkouts = [e for e in entries if e["kind"].startswith("checkout:")]
    owed = float(sum(e.get("seconds", unpriced) for e in checkouts if "build" in e) if entries else unpriced)
    building = [e for e in entries if e["kind"] == "build" and e["state"] == "held"]
    if owed == 0 or not building:
        return owed
    since = max(datetime.fromisoformat(e["changed"]).timestamp()
                for e in building + [c for c in checkouts if "build" not in c])
    return max(0.0, owed - (now - since))


def _clear(path: str) -> None:
    """Remove whatever is at `path`: a real directory with its contents, or a file or link."""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def install(arguments) -> int:
    """Make --to equal --from in content: the staged tree is the manifest.

    A file is written only when its bytes differ from the one there (to a temporary file in its
    directory, then replaced, with the staged mode), and a link only when its target differs;
    a file with the same bytes is left as it is, its mode corrected in place. Its time is then
    when its content last changed here, never the staged copy's: tools that decide by file time
    (cargo) rebuild exactly what changed, in either direction, since a file whose content went
    back to older bytes still reads as changed. Files, links and directories --from does not
    have are removed."""
    source, target = arguments.source, arguments.target
    present = set()
    written = removed = 0
    for current, directories, files in os.walk(source, onerror=_raise):
        relative = os.path.relpath(current, source)
        into = os.path.normpath(os.path.join(target, relative))
        if not (os.path.isdir(into) and not os.path.islink(into)):
            _clear(into)
            os.makedirs(into)
        links = [d for d in directories if os.path.islink(os.path.join(current, d))]
        directories[:] = [d for d in directories if d not in links]
        present.update(os.path.normpath(os.path.join(relative, d)) for d in directories)
        for name in files + links:
            present.add(os.path.normpath(os.path.join(relative, name)))
            origin, destination = os.path.join(current, name), os.path.join(into, name)
            staged = os.path.join(into, ".%s.%d.install" % (name, os.getpid()))
            if os.path.islink(origin):
                pointer = os.readlink(origin)
                if os.path.islink(destination) and os.readlink(destination) == pointer:
                    continue
                os.symlink(pointer, staged)
            else:
                mode = os.stat(origin).st_mode & 0o7777
                if os.path.isfile(destination) and not os.path.islink(destination) \
                        and filecmp.cmp(origin, destination, shallow=False):
                    if os.stat(destination).st_mode & 0o7777 != mode:
                        os.chmod(destination, mode)
                    continue
                shutil.copyfile(origin, staged)
                os.chmod(staged, mode)
            if os.path.isdir(destination) and not os.path.islink(destination):
                shutil.rmtree(destination)
            os.replace(staged, destination)
            written += 1
    for current, directories, files in os.walk(target, topdown=False):
        relative = os.path.relpath(current, target)
        for name in files + directories:
            if os.path.normpath(os.path.join(relative, name)) not in present:
                _clear(os.path.join(current, name))
                removed += 1
    print(json.dumps({"written": written, "removed": removed}))
    return 0


def _raise(error: OSError) -> None:
    raise error


def main() -> int:
    parser = argparse.ArgumentParser(prog="magnitude-rig-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("start", start), ("runner", runner), ("follow", follow), ("state", state),
                          ("stop", stop), ("fetched", fetched)):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--run", required=True)
        command.set_defaults(handler=handler)
        if name == "follow":
            command.add_argument("--after", type=int, default=0)
    command = commands.add_parser("status")
    command.add_argument("--root", required=True)
    command.add_argument("--source", action="append", default=[])
    command.add_argument("--build", action="append", default=[], help="each --source's build (`engine.build_key`)")
    command.set_defaults(handler=status)
    command = commands.add_parser("install")
    command.add_argument("--from", dest="source", required=True)
    command.add_argument("--to", dest="target", required=True)
    command.set_defaults(handler=install)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
