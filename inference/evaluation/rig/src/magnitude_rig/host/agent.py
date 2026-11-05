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
    checkout:NAME one checkout (its source, build target and caches), while the run installs
                  and builds there (and runs its jobs there, for a side whose jobs read it):
                  the run's alone, or shared with the runs of its tree and
                  build (below); never requested, only claimed (`checkout`), so it never
                  waits; eviction never removes a checkout with a live hold
    output:KEY    a build output (below), while the run's later steps read it: shared by every
                  live run of its tree and build; claimed with the checkout, never requested

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
    {"op": "checkout", "side": S, "source": KEY, "tree": TREE, "build": B, "output": O,
     "fallback": N, "commit": C, "seconds": T} claim a checkout for side S (below), whose
                                              tree is TREE (`engine.tree_key`), whose build
                                              is of the steps B names (their targets and
                                              profiles) from commit C, N the bytes it needs
                                              where this host has measured none and T the
                                              seconds its caller priced it at, and hold its
                                              build output O; later operations name their
                                              directories `{checkout:S}` and `{output:S}`
    {"op": "hold", "kind": K}                 events: waiting (blocking: [entry, ...]), held
    {"op": "release", "kind": K}
    {"op": "evict", "side": S, "host": H}     make room for the builds (below): waits, its
                                              install hold released, while live runs' checkouts
                                              will cover it (events: waiting, held); fails the
                                              run when even they cannot
    {"op": "run", "argv": [...], "cwd": C, "env": {...}, "input": S,
     "check": B, "job": J, "capture": NAME, "stdout_to": PATH, "unless": PATH,
     "install": S, "tests": PATH}
                                              events: output, exited. `check` fails the run
                                              on a nonzero exit; `job` records the exit code
                                              as that job's; `capture` keeps the last stdout
                                              line as `{capture:NAME}` in later operations;
                                              `stdout_to` writes the stdout to PATH; `unless`
                                              skips the step (a note) where PATH exists;
                                              `install` names the side whose checkout the
                                              step installs, skipped (a note) when the side
                                              shares a checkout its tree is installed in;
                                              `tests` writes the test executables the
                                              build's cargo steps produced to PATH first
                                              (`[[executable, manifest], ...]`)
    {"op": "built", "side": S, "release": R,  record the units the build's cargo steps
     "provenance": P}                         reference, trim the checkout's target to what
                                              its kept builds reference, record its bytes
                                              and the build's seconds by its warmth;
                                              complete the build's output (the executables
                                              build provenance record P kept) and, with R,
                                              release the checkout (without it, a job of
                                              the side reads it: held until the run ends);
                                              then keep the rig's checkouts within its
                                              share (below); a side sharing its checkout
                                              does none of it but the output and release

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

Each checkout lives in `<root>/checkouts/<NAME>/` and belongs to no source. A run builds a
tree (`tree`: its source, commit and changes, `engine.tree_key`) with a build (B, below). It
shares the checkout live runs hold for that same tree and build once they have installed it:
its install is skipped (the bytes are those already there), and its build waits on cargo's
lock for theirs and finds every unit fresh, so a tree is built once however many runs of it a
host holds. No run shares a checkout with itself (its sides each claim their own) or with a
run of another tree or build. Otherwise it claims the checkout no live run holds that is
warmest for it: one that has built B and holds its tree, then one that has built B (its
dependencies, profiles and features are there, so only the workspace units that differ
rebuild), then one that holds its tree, then one last installed from its source (`source`),
then the most recently claimed, else a new one; and holds it until its build ends (its run
ends, for a side a job of which reads the checkout), so no other tree is installed or built
there while it does. The run's install rewrites only the files whose
bytes differ, each with the time it is written, so cargo, whose freshness is by file time
against its last build in that same directory, rebuilds exactly what differs from whatever
source the checkout last held, and the build's provenance proves each workspace unit against
the bytes now there (`provenance.py`); once installed, the checkout records the tree (`tree`).
The directory never changes, so neither do the paths cargo keys artifacts and build scripts
by.

Unless a job of its side reads the checkout (one that runs cargo there, e.g. a test suite whose
documentation tests compile from the source when they run), nothing reads a checkout once its
build has ended: the run's later steps (acquiring its model, its jobs) read the build's output,
`<root>/outputs/<tree>-<build>/`, which no later install or build changes, so the checkout is
released as the build ends and another tree can install and build there while those steps
queue and run. Once the side's install has written its tree, the
output gets that tree as `src`, its files hard-linked (an install replaces a file whole, so a
link keeps the bytes it was made of); the build's environment (`venv`) is synced there by the
build itself, without editable installs, so it names no checkout. When the build ends, the
output gets every executable of the target's profile and example directories and every test
executable the build's cargo steps produced, linked at their paths under its `target`, and the
list of those test executables with their packages' directories in its `src` (`TESTS`), and is
complete (`OUTPUT_DONE`): a run of the same tree and build that finds it complete reuses it.
Its `src` and `target` are each made aside and moved in under the guard, so the first run to
finish one gives it and an output is never completed from parts; one no live run holds that
is not complete was left by a run that died making it, and is discarded before it is held.
Every executable an output holds is verified at the build's end (`provenance.py`: the job's,
each of the profile and example directories, and each test executable the build produced),
and a verified one read nothing of its source tree at compile time, so an output runs as its
checkout would have. Every live run of the output's tree
and build holds it; the last to end removes it, and the disk owner removes any no live run
holds (one whose run died).

A target keeps what its builds reference and nothing else. The runner carries out each cargo
build step with cargo's JSON messages, showing each diagnostic as the step's own format would
and collecting every unit the step references, fresh or built (its artifacts and its build
scripts' output directories). After the build, `built` keeps that set as B's in the checkout (`references`, the latest `BUILDS_KEPT` builds), and removes from the target's
`deps`, `build` and `examples` directories every hashed artifact no kept build references; a
unit removed that a later build needs is one cargo rebuilds (its outputs are missing), so
trimming never makes a build wrong. A checkout therefore holds the units of the builds it
keeps, each once, and keeps check, test and release builds side by side. The units a checkout
held before it first recorded a build's are kept as the oldest build's (`UNRECORDED`) until
`BUILDS_KEPT` builds have been recorded, so recording begins without removing what earlier
builds left warm, and what none of those builds references then goes. Incremental directories
are never trimmed: rustc names each for a hash of the unit's metadata that no cargo message
reports, so nothing the build records says which a kept build still uses, and a fresh unit
writes nothing there, so its time says nothing either; rustc removes each one's superseded
sessions itself, and eviction bounds the rest with the checkout. Only the build a checkout's
holder carries out trims it or records anything of it (below): a run sharing the checkout
built nothing of its own.

`used` is touched whenever a checkout is claimed, and `built` records its bytes after every
build (`size`) and, per build B (the build steps' targets and profiles), in
`<root>/rig/builds/<B>.json`, what a build of B measured on this host: `size`, what the latest
build of B into a cold checkout (its target built from nothing) added, the build's whole size;
and `growth`, the most a build of B into a checkout that keeps B's references added; each
measured before the build's trim, since the build needs room for all it writes. A run sharing
a checkout records none of these, nor its seconds, and neither trims it nor reclaims: it waited
on its holders' build, which records and trims it, and its entry owes no build, so it never
counts among the host's builds, for room, eviction or the CPU `status` reports. What
bounds a build is what it adds (`need`): a checkout that keeps B may add B's `growth`, any other
B's `size`; where this host has measured neither, the claim's `fallback`, the largest size of B
(else of any build) measured on any host the caller placed among; a run sharing a checkout adds
nothing. A claimed checkout's entry carries its need until its build ends, and every building
checkout may still add its need less what it has grown since its last recorded build, so
several runs building at once on one host each count. Eviction (below) removes the least
recently used checkout whenever a build needs the space, and, after every build, until its
checkouts hold no more than its `SHARE` of the volume (or every one left is held): the rig
budgets its own bytes, so what the volume's other users hold never costs it a warm checkout
until a build is actually short of room, and it removes nothing it did not create.

A claim knows how warm its checkout's target is for the run's source (`WARMTH`), and its entry
carries that (`build`) and the seconds its caller priced the build at (`seconds`) until the
side's build ends. `built` then records, with B's measurements, the build's seconds, the commit
it built (`to`, also the checkout's `commit` from then), the commit the checkout last built
before it (`from`, None from a cold target) and whether the checkout kept B's references
before it (`built`): its `builds`, the latest `BUILDS_KEPT`. Whether a build was incremental
(its checkout had built B at an ancestor of its commit) needs the repository, so the caller
judges it from those, as it prices a new run's build from them (`engine.build_seconds`), and
`status` reports what it needs: each side's checkout's `commit` and whether it keeps B, or that
the side would share a live run's checkout (`shared`) and the seconds that run's build still
owes (`owed`), which is all the side waits for. From the live entries
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
of gone runs, never an ended one not yet fetched, until the volume has those needs free (and
`reclaim`, after a build, unheld checkouts alone until the rig's are within its `SHARE`,
never waiting), and
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
PROTOCOL = 7
# The kinds a session claims rather than requests: a checkout, while it builds there (or its jobs
# read it), and a build output, while its jobs read it.
CLAIMED = ("checkout:", "output:")
# In `<root>/outputs/<tree>-<build>/`: a build's output (`Runner.materialize`), complete once
# `OUTPUT_DONE` is there, kept while a live run holds it.
OUTPUTS, OUTPUT_DONE, TESTS = "outputs", "complete", "tests.json"
# In a checkout: the key of the source last installed into it, and the tree it holds once installed.
SOURCE, TREE = "source", "tree"
# In a checkout: for each build (`engine.build_key`) it keeps, the units its latest build
# referenced and when (`Runner._units`); the target holds those units and nothing else. Its
# format's version is in its name: a checkout whose references another format wrote has
# recorded none of this one's, and its target's units are kept as `UNRECORDED`.
REFERENCES = "references-2.json"
# In `<root>/rig`: what `Session.reclaim` has moved aside to remove outside the guard.
TRASH = "trash"
# The rig's own share of its volume: after every build its checkouts hold at most this fraction
# of the volume's bytes (`Session.reclaim`), whatever the volume's other users hold.
SHARE = 0.4
# A cargo artifact named for its unit: `<name>-<16 hex digits>`, then an extension or nothing.
HASHED = re.compile(r"-([0-9a-f]{16})(?:\.|$)")
# The target directories of a profile whose entries are units' artifacts (`trim`).
UNIT_DIRECTORIES = ("deps", "build", "examples")
# In `REFERENCES`: the units a checkout held before it first recorded a build's, kept as the
# oldest build's until `BUILDS_KEPT` builds of any kind have been recorded since (`left`, the
# count still to come).
UNRECORDED = "unrecorded"
# The cargo commands whose units a build keeps (`Runner._units`).
CARGO_BUILDS = ("build", "check", "test", "bench", "rustc")
# How warm a claimed checkout's target is for the run's source, so how long its build takes:
# last installed from that source, from another source (rebuilds what differs), or built from
# nothing (a new checkout).
WARMTH = (SAME, OTHER, COLD) = (0, 1, 2)
# A build's seconds while no build is measured on the host: what a run that has claimed no
# checkout yet owes, and what an entry of another version, which records no price, does.
BUILD_PRIOR_SECONDS = 1200
# The latest builds of each build kept with its measurements.
BUILDS_KEPT = 8
# In a checkout: the commit it last built.
COMMIT = "commit"


def conflicts(a: str, b: str) -> bool:
    if a.startswith(CLAIMED) or b.startswith(CLAIMED):
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
                # Written whole under the guard: an entry that does not parse is a defect.
                entry = json.loads(probe.file.read().decode())
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
        self.claims = {}  # checkout kind -> the tree and build it holds it for, and whether installed
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

    def built(self, key: str, seconds: float, units) -> dict:
        """Record the build in checkout `key` that took `seconds` and whose cargo steps reference
        `units` (None for a build that ran none): those as its build's references, from when it
        began; the target trimmed to what the kept builds reference (`trim`); its bytes after the
        trim, and what its build added before it (what the build needed room for: into a cold
        checkout, its build's `size`; into one that kept it before, at most its `growth`), with
        the build among its build's `builds`. Only a claim that holds its checkout's build
        records (`Runner` never calls this for a shared one). Its entry owes no build from now."""
        directory = self.rig.checkout(key)
        kind = "checkout:" + key
        build = self.builds[kind].build
        kept = _references(directory)
        before = build in kept
        peak = checkout_bytes(directory)  # outside the guard: a large target takes seconds
        trimmed = {"bytes": 0, "entries": 0}
        if units is not None:
            began = time.time() - seconds
            if not os.path.exists(os.path.join(directory, REFERENCES)):  # what its builds before recorded none of
                # Kept as the oldest build until `BUILDS_KEPT` builds have been recorded (this
                # one the first).
                kept[UNRECORDED] = {"at": 0, "left": BUILDS_KEPT,
                                    "units": sorted(present_units(os.path.join(directory, "target")) - units)}
            if UNRECORDED in kept:
                kept[UNRECORDED]["left"] -= 1
                if kept[UNRECORDED]["left"] == 0:
                    del kept[UNRECORDED]
            kept[build] = {"at": began, "units": sorted(units)}
            kept = dict(sorted(kept.items(), key=lambda item: item[1]["at"])[-BUILDS_KEPT:])
            _write_json(os.path.join(directory, REFERENCES), kept)
            trimmed = trim(os.path.join(directory, "target"), kept)  # outside the guard, as above
        after = peak - trimmed["bytes"]  # `trim` counts what it removes as `checkout_bytes` does

        def record():
            pending = self.builds.pop(kind)
            added = peak - _recorded(os.path.join(directory, "size"))
            _record(os.path.join(directory, "size"), after)
            measured = _measured(self.rig.directory, pending.build)
            if pending.warmth == COLD:
                measured["size"] = added
            elif before:
                measured["growth"] = max(added, measured.get("growth", 0))
            measured["builds"] = [*measured.get("builds", []), {
                "seconds": round(seconds),
                "from": None if pending.warmth == COLD else _read(os.path.join(directory, COMMIT)),
                "to": pending.commit, "built": before,
            }][-BUILDS_KEPT:]
            os.makedirs(os.path.join(self.rig.directory, MEASURED), exist_ok=True)
            _write_json(_measured_path(self.rig.directory, pending.build), measured)
            with open(os.path.join(directory, COMMIT), "w") as stream:
                stream.write(pending.commit)
            self._write(kind, "held")
            return {"bytes": after, "added_bytes": added, "trimmed": trimmed}

        return self.rig.guarded(record)

    def shared_built(self, key: str) -> None:
        """Mark the end of the build of shared checkout `key`, which its holder records (`built`):
        its entry's change is when this run's build ended."""
        self.rig.guarded(lambda: self._write("checkout:" + key, "held"))

    def _write(self, kind: str, state: str) -> None:
        path, lock = self.entries[kind]
        entry = dict(self.info, kind=kind, state=state, changed=_now(), **self.claims.get(kind, {}))
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

    def claim(self, source: str, tree: str, build: str, fallback: int, commit: str, seconds: float):
        """Hold a checkout for a run of `source` whose tree is `tree` and that builds `build`
        from `commit` (see the module documentation; `fallback`, its need where this host has
        measured none; `seconds`, what its caller priced the build at): its name, whether it
        shares it with a live run of that tree and build, and nothing; or, for the session's
        first entry while a run of another protocol is live, None, False and those runs'
        entries, claiming none."""
        base = os.path.join(self.rig.root, "checkouts")

        def take():
            live = self.rig.live()
            others = self._others(live)
            if others:
                self._account(blocked=True)
                return None, False, others
            name = shareable(live, {kind.split(":", 1)[1] for kind in self.held if kind.startswith("checkout:")},
                             tree, build)
            shared = name is not None
            if not shared:
                name = choose(base, source, tree, build, held_checkouts(live))
                if name is None:
                    names = set(_directories(base))
                    name = next("slot-%d" % n for n in range(len(names) + 1) if "slot-%d" % n not in names)
                    warmth = COLD
                else:
                    warmth = warmth_of(base, name, source)
            directory = os.path.join(base, name)
            os.makedirs(directory, exist_ok=True)
            kind = "checkout:" + name
            self.claims[kind] = {"tree": tree, "key": build, "installed": shared}
            if not shared:  # a sharer owes no build: its holders' records and trims the checkout
                self.builds[kind] = Owed(warmth, build, fallback, need(self.rig.directory, directory, build, fallback),
                                         commit, seconds)
            self._enter(live, kind, "held")
            self.held.add(kind)
            self._account()
            with open(os.path.join(directory, "used"), "w") as stream:
                stream.write(_now())
            if not shared:  # its source is about to change: it holds no tree until installed
                with open(os.path.join(directory, SOURCE), "w") as stream:
                    stream.write(source)
                _clear(os.path.join(directory, TREE))
            return name, shared, []

        return self.rig.guarded(take)

    def hold_output(self, key: str) -> None:
        """Hold the build output `key` (another live run of its tree and build may hold it too).
        One no live run holds that is not complete, left by a run that died making it, is
        discarded first: a run completes an output it finds only when a live run is making it."""

        kind = "output:" + key
        if kind in self.entries:  # its other side builds the same tree the same way
            return

        def take():
            live, output = self.rig.live(), os.path.join(self.rig.root, OUTPUTS, key)
            if (os.path.isdir(output) and kind not in {e["kind"] for e in live}
                    and not os.path.exists(os.path.join(output, OUTPUT_DONE))):
                _discard_output(self.rig.root, os.path.join(self.rig.directory, TRASH), key)
            self._enter(live, kind, "held")
            self.held.add(kind)
            os.makedirs(output, exist_ok=True)

        self.rig.guarded(take)

    def end_outputs(self) -> list:
        """Release every output the session holds and remove each no other live run holds: moved
        aside under the guard, removed outside it. The outputs removed."""
        trash = os.path.join(self.rig.directory, TRASH)
        removed = []
        for kind in [kind for kind in self.entries if kind.startswith("output:")]:
            self.release(kind)
            key = kind.split(":", 1)[1]

            def unheld():
                if kind in {e["kind"] for e in self.rig.live()}:
                    return False
                _discard_output(self.rig.root, trash, key)
                return True

            if self.rig.guarded(unheld):
                removed.append(key)
        for name in _directories(trash):
            _remove_tree(os.path.join(trash, name))
        return removed

    def installed(self, name: str) -> None:
        """Record that checkout `name` holds the tree it was claimed for, which runs of that tree
        and build may share from now."""
        kind = "checkout:" + name

        def mark():
            self.claims[kind]["installed"] = True
            with open(os.path.join(self.rig.checkout(name), TREE), "w") as stream:
                stream.write(self.claims[kind]["tree"])
            self._write(kind, "held")

        self.rig.guarded(mark)

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
            fallback = largest(self.rig.directory)
            entries = checkout_entries(live)

            def growth(key):  # what checkout `key` may still add: the most any of its holders owes
                bound = max(owed(entry, fallback) for entry in entries[key])
                return bound - min(bound, max(0, sizes[key] - _recorded(os.path.join(base, key, "size"))))

            # This run's checkouts, and the other runs', but those only runs waiting for room after
            # it hold. A checkout comes free once every run holding it has ended, so one a run
            # waiting after it also holds never counts as coming free: that run waits for it.
            later = _waiting_after(live, self.entries.get("room"))
            holders = {key: {entry["run"] for entry in held} for key, held in entries.items()}
            mine = {key for key, runs in holders.items() if self.info["run"] in runs}
            own = {key for key in mine if "checkout:" + key in self.builds}  # those it builds, not shares
            building = {key for key, runs in holders.items() if key not in mine and not runs <= later}
            others = {key: runs for key, runs in holders.items() if key in building and not runs & later}
            needed = sum(growth(key) for key in mine | building)
            removed = _remove_least_recent(self.rig.root, live, keep, needed)
            return {
                "removed": removed,
                "free_bytes": shutil.disk_usage(self.rig.root).free,
                "freed_bytes": shutil.disk_usage(self.rig.root).free - before,
                "needed_bytes": needed,
                "own_bytes": sum(growth(key) for key in own),
                "releasable_bytes": sum(sizes[key] for key in others),
                "releasing": sorted(set().union(*others.values())),
            }

        while True:
            for key in (self.rig.guarded(lambda: held_checkouts(self.rig.live())) | {keep}) - set(sizes):
                sizes[key] = checkout_bytes(os.path.join(base, key))
            evicted = self.rig.guarded(act)
            if evicted is not None:
                return evicted

    def reclaim(self) -> dict:
        """After a build: remove the checkouts no live run holds, least recently used first,
        until the rig's checkouts hold no more than its `SHARE` of the volume, never waiting (what
        is left may all be held): the checkouts removed and the bytes they held. What the
        volume's other users hold never makes the rig remove a warm checkout here; only a build
        short of room does (`evict`). Each is moved aside under the guard and removed outside it,
        so a large one keeps no other run's holds waiting."""
        root = self.rig.root
        base, trash = os.path.join(root, "checkouts"), os.path.join(self.rig.directory, TRASH)
        budget = SHARE * shutil.disk_usage(root).total
        sizes = {name: _held_bytes(os.path.join(base, name)) for name in _directories(base)}
        removed, freed = [], 0

        def outputs():  # no run reads them (their runs died): removed whatever the rig's share
            found = _unheld_outputs(root, self.rig.live())
            for key in found:
                _discard_output(root, trash, key)
            return found

        removed += ["%s/%s" % (OUTPUTS, key) for key in self.rig.guarded(outputs)]
        while True:
            for name in _directories(trash):  # moved aside, by this reclaim or one that did not finish
                _remove_tree(os.path.join(trash, name))
            if sum(sizes.values()) <= budget:
                break
            moved = self.rig.guarded(lambda: _move_least_recent(root, self.rig.live(), trash))
            if moved is None:
                break
            name, aside = moved
            # One made since `sizes` was taken is measured where it was moved.
            freed += sizes.pop(name) if name in sizes else checkout_bytes(aside)
            removed.append("checkouts/" + name)
        return {"removed": removed, "freed_bytes": freed}


def _discard_output(root: str, trash: str, key: str) -> None:
    """Move build output `key` into `trash`, removed from there outside the guard. Call guarded."""
    os.makedirs(trash, exist_ok=True)
    os.rename(os.path.join(root, OUTPUTS, key),
              os.path.join(trash, "output-%s-%d-%d" % (key, os.getpid(), time.time_ns())))


def _link_tree(origin: str, destination: str) -> None:
    """`destination` as `origin`'s tree: its directories made, its files linked, its links kept."""
    for current, directories, files in os.walk(origin):
        into = os.path.join(destination, os.path.relpath(current, origin))
        os.makedirs(into, exist_ok=True)
        for name in files + [d for d in directories if os.path.islink(os.path.join(current, d))]:
            path = os.path.join(current, name)
            if os.path.islink(path):
                os.symlink(os.readlink(path), os.path.join(into, name))
            else:
                os.link(path, os.path.join(into, name))
        directories[:] = [d for d in directories if not os.path.islink(os.path.join(current, d))]


def _held_bytes(checkout: str) -> int:
    """The bytes checkout `checkout` holds: as its last build recorded them (`size`, a recorded 0
    among them), else, for one no build recorded, measured now (never recorded here: a run
    building in it measures what its build adds against its own record)."""
    size = os.path.join(checkout, "size")
    return _recorded(size) if os.path.exists(size) else checkout_bytes(checkout)


def _remove_tree(path: str) -> None:
    """Remove the directory at `path` and what it holds. An entry gone meanwhile (another
    remover's) is no failure; any other failure raises, so nothing counts as removed that is not."""

    def gone(function, at, error):
        error = error if isinstance(error, BaseException) else error[1]
        if not isinstance(error, FileNotFoundError):
            raise error

    shutil.rmtree(path, **({"onexc": gone} if sys.version_info >= (3, 12) else {"onerror": gone}))


def _least_recent(root: str, live: list, keep) -> list:
    """The checkouts no live run holds (but `keep`) and the directories of gone runs, never an
    ended one not yet fetched, least recently used first: each as `(kind, name)`. Call guarded."""
    base, runs = os.path.join(root, "checkouts"), os.path.join(root, "runs")
    held, running = held_checkouts(live), {e["run"] for e in live}
    candidates = [
        (_mtime(os.path.join(base, key, "used")), "checkouts", key)
        for key in _directories(base) if key not in held and key != keep
    ] + [
        (_mtime(os.path.join(runs, run)), "runs", run)
        for run in _directories(runs) if run not in running
    ]
    found = []
    for _, kind, name in sorted(candidates):
        directory = os.path.join(root, kind, name)
        if kind == "runs" and (_owned(directory) or _owned(directory, STAGE_LOCK)
                               or os.path.exists(os.path.join(directory, ENDED))):
            continue  # a live run's, one its caller is staging, or an ended one not fetched
        found.append((kind, name))
    return found


def _unheld_outputs(root: str, live: list) -> list:
    """The build outputs no live run holds: left by a run that died without removing its own
    (`reclaim` removes them after every build)."""
    held = {e["kind"].split(":", 1)[1] for e in live if e["kind"].startswith("output:")}
    return [key for key in _directories(os.path.join(root, OUTPUTS)) if key not in held]


def _remove_least_recent(root: str, live: list, keep, wanted: float) -> list:
    """Remove the least recently used of `_least_recent` until the volume has `wanted` bytes
    free: the paths removed, relative to the root. Call guarded."""
    removed = []
    for kind, name in _least_recent(root, live, keep):
        if shutil.disk_usage(root).free >= wanted:
            break
        _remove_tree(os.path.join(root, kind, name))
        removed.append("%s/%s" % (kind, name))
    return removed


def _move_least_recent(root: str, live: list, trash: str):
    """Move the least recently used checkout no live run holds into `trash` (no run takes it
    from there): its name and where it was moved, or None when there is none. Call guarded."""
    for kind, name in _least_recent(root, live, None):
        if kind == "checkouts":
            os.makedirs(trash, exist_ok=True)
            aside = os.path.join(trash, "%s-%d-%d" % (name, os.getpid(), time.time_ns()))
            os.rename(os.path.join(root, kind, name), aside)
            return name, aside
    return None


def _waiting_after(live: list, room) -> set:
    """The runs waiting for room after the `room` request entry (its path; None: after none,
    as one not waiting is after every run that is)."""
    if room is None:
        return set()
    mine = os.path.basename(room[0])
    return {e["run"] for e in live if e["kind"] == "room" and e["entry"] > mine}


def held_checkouts(live: list) -> set:
    return {e["kind"].split(":", 1)[1] for e in live if e["kind"].startswith("checkout:")}


def checkout_entries(live: list) -> dict:
    """Each held checkout's entries, one for each run holding it (several share one tree)."""
    found = {}
    for entry in live:
        if entry["kind"].startswith("checkout:"):
            found.setdefault(entry["kind"].split(":", 1)[1], []).append(entry)
    return found


def shareable(live: list, own: set, tree: str, build: str):
    """The checkout live runs hold for tree `tree` and build `build` and have installed it in,
    which a run of the same tree and build shares, but one in `own` (those the run's earlier
    sides hold) or one a run of another protocol holds; None when there is none."""
    for name, entries in sorted(checkout_entries(live).items()):
        if name not in own and all(entry.get("protocol") == PROTOCOL and entry.get("tree") == tree
                                   and entry.get("key") == build and entry.get("installed") for entry in entries):
            return name
    return None


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


def choose(base: str, source: str, tree: str, build: str, held: set):
    """The checkout, of those not in `held`, a run of `source` whose tree is `tree` and that
    builds `build` takes: the warmest for it (see the module documentation), among equals the
    most recently claimed, then used; None when every one is held."""
    free = [name for name in _directories(base) if name not in held]
    if not free:
        return None

    def rank(name):
        directory = os.path.join(base, name)
        kept, same = build in _references(directory), _read(os.path.join(directory, TREE)) == tree
        recorded = os.path.join(directory, SOURCE)  # written at each claim
        return (kept and same, kept, same, _read(recorded) == source, _mtime(recorded),
                _mtime(os.path.join(directory, "used")))

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
    claim's fallback, the bytes it may add (`need`), the commit it builds, the seconds its
    caller priced it at. A claim that shares its checkout owes none: its holders' build is
    the checkout's."""

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


def _references(checkout: str) -> dict:
    """The builds checkout `checkout` keeps, each with the units its latest build referenced
    and when (`REFERENCES`); written whole, so a record that does not read is a fault."""
    try:
        with open(os.path.join(checkout, REFERENCES)) as stream:
            return json.load(stream)
    except FileNotFoundError:
        return {}


def units_of(target: str, paths) -> set:
    """The units of `target` the artifact paths `paths` name: `<profile>/<dir>/<hash>` for each
    under a profile's unit directory (`UNIT_DIRECTORIES`) whose entry there is hashed; paths
    elsewhere (a profile's own outputs, another target) name none."""
    units = set()
    for path in paths:
        parts = os.path.relpath(path, target).split(os.sep)
        if parts[0] == os.pardir:
            continue
        for index, part in enumerate(parts[:-1]):
            if part in UNIT_DIRECTORIES:
                found = HASHED.search(parts[index + 1])
                if found:
                    units.add("/".join(parts[:index + 1] + [found.group(1)]))
                break
    return units


def _unit_directories(target: str) -> list:
    """Each profile's unit directories (`<profile>/<dir>`, and under a target triple
    `<triple>/<profile>/<dir>`), relative to `target`."""
    found = []
    for first in _directories(target):
        for second in _directories(os.path.join(target, first)):
            if second in UNIT_DIRECTORIES:
                found.append(first + "/" + second)
                continue
            found += ["%s/%s/%s" % (first, second, third)
                      for third in _directories(os.path.join(target, first, second)) if third in UNIT_DIRECTORIES]
    return found


def present_units(target: str) -> set:
    """Every unit with an artifact in `target`'s unit directories."""
    return {"%s/%s" % (relative, found.group(1)) for relative in _unit_directories(target)
            for name in os.listdir(os.path.join(target, *relative.split("/"))) if (found := HASHED.search(name))}


def trim(target: str, kept: dict) -> dict:
    """Remove from `target` every hashed entry of a unit directory that no build of `kept`
    (`_references`) references, never its incremental directories (see the module
    documentation): the bytes and the entries removed."""
    units = {unit for build in kept.values() for unit in build["units"]}
    freed = entries = 0
    for relative in _unit_directories(target):
        directory = os.path.join(target, *relative.split("/"))
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            found = HASHED.search(name)
            if found is None or "%s/%s" % (relative, found.group(1)) in units:
                continue
            freed += checkout_bytes(path) if os.path.isdir(path) and not os.path.islink(path) else os.lstat(path).st_size
            entries += 1
            _clear(path)
    return {"bytes": freed, "entries": entries}


def need(directory: str, checkout: str, build: str, fallback):
    """The bytes a build of `build` may add to `checkout` (None for none there yet, a cold
    one) on the host whose rig directory is `directory`: its `growth` where the checkout keeps
    it and it was measured, else its `size`; `fallback` where neither was."""
    measured = _measured(directory, build)
    if checkout is not None and build in _references(checkout) and "growth" in measured:
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
            except ProcessLookupError:
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
        except BrokenPipeError:
            pass  # the child ended before reading its input; its exit says why

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
        self.shared = set()  # the sides sharing a checkout their tree is installed in
        # What the current build's cargo steps reference (None before its first), and whether one
        # left it unknown (`_units`).
        self.units, self.unknown = None, False
        self.tests = []  # the test executables the current build's cargo steps produced: (executable, manifest)
        self.outputs = {}  # side -> its build output's directory
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

    def claim(self, side: str, source: str, tree: str, build: str, fallback: int, commit: str,
              seconds: float) -> None:
        reported = None
        while True:
            self._check_stop()
            name, shared, others = self.session.claim(source, tree, build, fallback, commit, seconds)
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
        if shared:
            self.shared.add(side)
            self.events.note("checkout %s for %s, shared with the live runs of this tree and build" % (name, side))
        else:
            self.events.note("checkout %s for %s" % (name, side))

    def _move_in(self, staged: str, place: str, *rest) -> bool:
        """Move `staged` to `place` under the guard, with `rest` (path, value) written after as
        JSON, unless `place` is there: made by another live run of the same tree and build. Whether
        it moved; one that did not is removed."""

        def move():
            if os.path.exists(place):
                return False
            os.rename(staged, place)
            for path, value in rest:
                _write_json(path, value)
            return True

        moved = self.rig.guarded(move)
        if not moved:
            shutil.rmtree(staged)
        return moved

    def snapshot(self, side: str) -> None:
        """Give the side's output the tree just installed in its checkout, once: its source
        hard-linked file by file (an install replaces a file whole, so a link keeps the bytes it
        was made of). Built aside and moved in, so a run of the same tree that made it first wins."""
        output, checkout = self.outputs[side], self.rig.checkout(self.checkouts[side])
        if os.path.isdir(os.path.join(output, "src")):
            return
        staged = os.path.join(output, ".src.%d" % os.getpid())
        _link_tree(os.path.join(checkout, "src"), staged)
        self._move_in(staged, os.path.join(output, "src"))

    def materialize(self, side: str, proven: str) -> int:
        """Complete the side's output from its build, once: every executable of the target's
        profile and example directories that its build provenance (`proven`, its record) kept,
        and every test executable the build produced, linked at
        its path under the output's `target`, and the list of the test executables with their
        packages' directories in the output's source (`TESTS`). The executables linked."""
        output, checkout = self.outputs[side], self.rig.checkout(self.checkouts[side])
        tests, self.tests = self.tests, []
        if os.path.exists(os.path.join(output, OUTPUT_DONE)):
            return 0
        target, into = os.path.realpath(os.path.join(checkout, "target")), os.path.join(output, "target")
        staged = os.path.join(output, ".target.%d" % os.getpid())
        os.makedirs(staged)
        tests = [(os.path.realpath(path), manifest) for path, manifest in tests]
        with open(proven) as stream:
            executables = set(json.load(stream)["kept"]) | {path for path, _ in tests}
        for path in executables:
            relative = os.path.relpath(path, target)
            os.makedirs(os.path.dirname(os.path.join(staged, relative)), exist_ok=True)
            os.link(path, os.path.join(staged, relative))
        source = os.path.join(checkout, "src")
        listed = [{"executable": os.path.join(into, os.path.relpath(path, target)),
                   "directory": os.path.join(output, "src", os.path.relpath(os.path.dirname(manifest), source))}
                  for path, manifest in tests]
        moved = self._move_in(staged, into, (os.path.join(output, TESTS), listed),
                              (os.path.join(output, OUTPUT_DONE), _now()))
        return len(executables) if moved else 0

    def _units(self, request: dict):
        """For a cargo build step whose units the build keeps (`CARGO_BUILDS`, with the run's
        target): the step with cargo's JSON messages in place of its own format, and the output
        it is carried out with, which shows what that format would have (each diagnostic as
        cargo renders it, its own lines as they are) and adds the paths of each unit's artifacts
        and each build script's output directory to `paths`; None for any other step. A cargo
        build step in a format of its own leaves the build's units unknown (`self.unknown`)."""
        argv = request["argv"]
        if argv[:1] != ["cargo"] or len(argv) < 2 or argv[1] not in CARGO_BUILDS or "CARGO_TARGET_DIR" not in request["env"]:
            return None
        formats = [argument for argument in argv if argument.startswith("--message-format")]
        if formats not in ([], ["--message-format=short"]):
            self.unknown = True
            return None
        messages = "--message-format=json-diagnostic-short" if formats else "--message-format=json-render-diagnostics"
        argv = [argument for argument in argv if not argument.startswith("--message-format")]
        paths = []

        def output(stream: str, text: str) -> None:
            message = None
            if stream == "stdout" and text.startswith("{"):
                try:
                    message = json.loads(text)
                except ValueError:
                    pass
            if message is None:
                self.events.output(stream, text)
            elif message.get("reason") == "compiler-message":
                self.events.output("stderr", message["message"]["rendered"])
            elif message.get("reason") == "compiler-artifact":
                paths.extend(message["filenames"])
                if message.get("executable") and message["profile"]["test"]:
                    self.tests.append((message["executable"], message["manifest_path"]))
            elif message.get("reason") == "build-script-executed":
                paths.append(message["out_dir"])

        return dict(request, argv=argv[:2] + [messages] + argv[2:]), output, paths

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
        even they could not (see the module documentation). A side sharing its checkout adds
        nothing, so it needs no room: its holders' build made it."""
        if side in self.shared:
            return
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
        if "install" in op and op["install"] in self.shared:
            self.events.note("skipped `%s`: checkout %s holds this tree" % (" ".join(argv)[:160], self.checkouts[op["install"]]))
            return
        if "tests" in op:  # the build's test executables, for its provenance step
            _write_json(self._fill(op["tests"]), [list(test) for test in self.tests])
        self.events.note("$ " + " ".join(argv))
        watch = None
        if "job" in op:
            stacks = os.path.join(self.directory, "logs", "stacks")
            os.makedirs(stacks, exist_ok=True)
            request["env"] = dict(request["env"], **{STACKS: stacks})
        output, paths = self.events.output, None
        if "job" in op:
            output = lambda stream, text: (watch is not None and watch.saw(text), self.events.output(stream, text))
        elif not {"install", "capture", "stdout_to"} & set(op) and (found := self._units(request)) is not None:
            request, output, paths = found
        try:
            self.child = Child(request, output, whole="capture" in op or "stdout_to" in op)
        except (FileNotFoundError, PermissionError) as error:
            # The command cannot run, as a shell reports it: 127 not found, 126 not executable.
            self.events.output("stderr", "agent: %s\n" % error)
            code, stdout = (127 if isinstance(error, FileNotFoundError) else 126), ""
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
        if "install" in op:  # its output has the tree before any run can share the checkout
            self.snapshot(op["install"])
            self.session.installed(self.checkouts[op["install"]])
        if paths is not None:
            self.units = (self.units or set()) | units_of(request["env"]["CARGO_TARGET_DIR"], paths)
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
            self.claim(op["side"], op["source"], op["tree"], op["build"], op["fallback"], op["commit"], op["seconds"])
            self.session.hold_output(op["output"])
            self.outputs[op["side"]] = os.path.join(self.rig.root, OUTPUTS, op["output"])
            self.bound["output:" + op["side"]] = self.outputs[op["side"]]
        elif kind == "hold":
            self.hold(op["kind"])
        elif kind == "release":
            self.session.release(op["kind"])
        elif kind == "evict":
            self.make_room(op["side"], op["host"])
        elif kind == "run":
            self.execute(op)
        elif kind == "built":
            seconds = time.monotonic() - self._phase_started
            name = self.checkouts[op["side"]]
            units = None if self.unknown else self.units
            self.units, self.unknown = None, False
            shared = op["side"] in self.shared
            if shared:
                self.session.shared_built(name)
            else:
                built = self.session.built(name, seconds, units)
            linked = self.materialize(op["side"], op["provenance"])
            if op["release"]:
                self.session.release("checkout:" + name)
                self.events.note("released checkout %s: this run's later steps read its build output %s (%d executables)"
                                 % (name, self.outputs[op["side"]], linked))
            else:
                self.events.note("kept checkout %s: a job of %s reads it" % (name, op["side"]))
            if shared:  # it built nothing of its own, so it adds nothing to reclaim
                self.events.note("checkout %s is shared: its holders' build records and trims it" % name)
            else:
                self._reclaim(built)
        else:
            raise Failed("unknown operation %s" % op)

    def _reclaim(self, built: dict) -> None:
        """Report what the build added and trimmed (`Session.built`), then keep the rig's
        checkouts within its share of the volume."""
        trimmed = built["trimmed"]
        self.events.note("checkout holds %.1f GB, %.1f GB added by this build; trimmed %.1f GB its kept builds "
                         "no longer reference: %d unit artifacts"
                         % (built["bytes"] / 1e9, built["added_bytes"] / 1e9, trimmed["bytes"] / 1e9,
                            trimmed["entries"]))
        reclaimed = self.session.reclaim()
        if reclaimed["removed"]:
            self.events.note("reclaimed %.1f GB to keep the rig's checkouts within its %d%% share of the volume: "
                             "removed least recently used %s" % (reclaimed["freed_bytes"] / 1e9, SHARE * 100,
                                                                 reclaimed["removed"]))
            self.events.emit("evicted", removed=reclaimed["removed"], freed_bytes=reclaimed["freed_bytes"])

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
        self.session.end_outputs()
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
    if not len(arguments.build) == len(arguments.tree) == len(arguments.source):
        raise SystemExit("status: one --tree and one --build for each --source")
    print(json.dumps(report(arguments.root, list(zip(arguments.source, arguments.tree, arguments.build)))))
    return 0


def report(root: str, sides: list) -> dict:
    """The host's live entries, each with the time its run is expected to take still and how far
    it is past its estimate (`remaining`, `over`), the wait of a new request of each kind behind
    the runs holding or queued for a conflicting hold (each once, by its `remaining`: `waits`),
    the build seconds each live run still owes (`building`); and, for a new run whose sides are
    `sides` (each a source, its tree and its build), the checkout each would claim now: one it
    would share (`shared`, with the seconds the build of its holders still owes, `owed`), or one
    with its warmth for that source (None: none is free, a cold build), the commit it last built
    and whether it keeps that build (`built`); and the bytes its build may add as measured here (`need`,
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
        taken, own, checkouts, needs = held_checkouts(live), set(), [], []
        for source, tree, build in sides:
            shared = shareable(live, own, tree, build)
            name = shared if shared is not None else choose(base, source, tree, build, taken)
            taken.add(name)  # None among the names changes no later choice
            own.add(name)  # as a claim, a later side never shares an earlier one's
            checkouts.append(None if name is None else {
                "name": name, "source": _read(os.path.join(base, name, SOURCE)),
                "size": _recorded(os.path.join(base, name, "size")),
                "warmth": SAME if shared is not None else warmth_of(base, name, source),
                "commit": _read(os.path.join(base, name, COMMIT)),
                "built": build in _references(os.path.join(base, name)), "shared": shared is not None,
            })
            needs.append(0 if shared is not None else
                         need(rig.directory, None if name is None else os.path.join(base, name), build, None))
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
    holders = checkout_entries(entries)
    for found in checkouts:
        if found is not None and found["shared"]:
            found["owed"] = max(shared_owing(entries, entry, seconds, now) for entry in holders[found["name"]])
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
            except FileNotFoundError:
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
        "measured": {build: _measured(rig.directory, build) for _, _, build in sides},
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
    held = checkout_entries(live)
    size = lambda name: _recorded(os.path.join(base, name, "size"))
    debt = lambda name: max(owed(entry, fallback) for entry in held[name])
    ends = lambda name: max(left[entry["run"]] for entry in held[name])  # once every holder has ended
    return {
        "available": shutil.disk_usage(root).free + sum(size(name) for name in _directories(base) if name not in held),
        "owed": sum(debt(name) for name in held),
        "ending": [[end, size(name), owes] for end, name, owes in sorted((ends(name), name, debt(name)) for name in held)],
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


def shared_owing(live: list, entry: dict, unpriced: float, now: float) -> float:
    """The seconds a run sharing checkout entry `entry`'s checkout waits on its holder's build
    of it: none once that build has ended; else what its holder still owes the host
    (`owing`), but never more than that one build's price, since its holder's other builds are
    not its."""
    if "build" not in entry:
        return 0.0
    return min(entry["seconds"], owing(live, entry["run"], unpriced, now))


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
    command.add_argument("--tree", action="append", default=[], help="each --source's tree (`engine.tree_key`)")
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
