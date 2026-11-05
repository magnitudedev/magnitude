"""Running one request on the rig: placement, then a plan the host carries out on its own.

A request names its backend, the hosts it may run on and its work on any one of them: its
sides (each a source with its build steps), the device hold its jobs need (`device`, `timed`
or none) and the jobs themselves, in the order they run. The work is planned for each
candidate host from that host's facts (an instrument may refuse a host whose fact it needs is
not recorded), and placement takes the candidate where its jobs would start earliest: after its
own build (priced by the host's measured builds of it, `build_seconds`), slowed by every build it
would share the host's CPU with, once the host has disk room for its builds (by their measured
sizes, `room_wait`), and after the holds its jobs wait behind (`place`). A request may derive its
estimate from that start on the host placed (`Request.estimate`). On the host it picks, the caller first stages each side's source
into the run's own `<root>/runs/<run id>/stage-<label>/src` and the request's files into its `inputs` (`sync.py`),
taking no host hold: only the agent's hold on the run's directory, which keeps eviction from it
until the runner owns it, and which a stop before the start withdraws. The run's runner
(`host/agent.py`) then depends on nothing off the host: its only waits are on host holds. It:

1. claims a checkout for each side: the one live runs of the side's tree and build hold, which
   it shares (its install skipped, its build fresh once theirs ends), else the warmest no live
   run holds, its alone until its build ends (its jobs end, for a side a job of which reads the
   checkout: `Step.checkout`; `host/agent.py`), and the side's build output; evicts least
   recently used checkouts and gone runs until every checkout building on the host can add the
   most a build has added there (the run fails before it builds when they cannot), and installs
   the side's stage into the checkout's `src`;
2. under a build hold: installs the recording rustc wrapper, runs the build steps with the
   checkout's target, writes the build provenance of every cargo-built executable the
   side's jobs name and every test executable its build produced, and withdraws every other one of the target it cannot tie to the side's
   source (a job never runs an executable its sources did not produce), trims the checkout's
   target to the units its kept builds reference, records what the build added to the checkout,
   completes the side's build output (its tree, environment and executables) and releases the
   checkout unless a job of the side reads it, and removes least recently used unheld
   checkouts until the rig's are within its share of the volume. From then on the steps of a
   side that released its checkout read its output (`{src}`, `{target}`, `{venv}`), and
   `{checkout}`, which another tree may be building in, is no longer theirs to name;
3. under the models hold: acquires the request's model into the host's model store, or, for a
   run that needs only its headers, exports the catalog's headers (each component's exact
   header, sparse at the file's full size, so it opens as the model with zero weights);
4. under the device hold (after the thermal gate when timed): runs each job, recorded with the
   host's facts and thermal trace, into `<root>/runs/<run id>/<job>/`.

The caller turns the request into that plan, every token filled, starts the runner with it and
follows the runner's events: it mirrors each phase's output into `results/<run id>/logs/<phase>.log` and keeps the run's phase and what it
waits behind. A lost connection loses nothing: the runner goes on, and the caller follows again
from the last event it saw, after a backoff, for as long as it takes (output from then on: what
it missed comes with the logs it fetches). The runner releases every hold as it ends. The caller
then fetches what the run's answer reads (`sync.fetch_answer`) into the repository's results
store (`store.py`: the main checkout's `inference/evaluation/results/<run id>/`), records it and
is answered. Only then, the run no longer in flight, does it fetch the rest (`sync.fetch`), which
no answer waits for, and the runner's agent removes the directory from the host (`complete`).
Each step is kept where the next command finds it (the host's end, the results' record and
exit code, the launched record), so a run whose caller died at any step is taken up from there:
answered by the next `mrig` command and fetched whole by a `mrig fetch` it starts (`collect`).

Holds are taken in that order and build and models are released before the device hold is
requested, so runs never wait on each other in a cycle (`host/agent.py`).

Every phase's time is kept in `rig.json` beside the results, with every `mrig wait` on the run
and its justification. On SIGUSR1 the caller writes what it knows now (its phases, what it
waits behind, its last output, the device's state) to `report.txt` in its results: what its
caller shows when the run passes its estimate (`results.follow`).

Tokens `{src}`, `{target}` (the side's: in its checkout, or in its build output once it releases
the checkout), `{venv}` (the side's environment, in its build output), `{checkout}` (the side's,
which its runner names as it claims it, until it is released), `{output}` (the side's build
output), `{root}`, `{models}`, `{python}`, `{artifacts}`, `{catalog}` (where the run acquires headers: the header export of
the acquiring side's catalog, `catalog_key`), `{engine_cache}` (the
host's engine cache, laid out as the service's cache root: `{engine_cache}/kernels` is the
kernel cache every checkout shares, valid by its content-addressed keys), `{runs}` (the run's
directory, holding every job's), `{inputs}` (the request's files), `{run_dir}` and
`{build_file}` (the job's) are replaced with host paths in build steps, job steps and their
environments.
"""

import dataclasses
import getpass
import json
import math
import hashlib
import os
import shlex
import shutil
import signal
import socket
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import sources, state, store, sync
from .connection import (
    AGENT, HostError, HostRun, HostUnavailable, Remote, _ssh, connect, host_status, run_state,
)
from .host.agent import ENDED, NOTE, SAME
from .host.gate import GateSettings
from .host.provenance import WRAPPER
from .hosts import Hosts
from .sources import Source

# In each run's results (`store.results()`): what the run's process printed, its exit code once it has one, its
# latest report, and every wait on it.
TRANSCRIPT, CODE, REPORT, WAITS = "mrig.out", "mrig.code", "report.txt", "waits.jsonl"
HOST_PACKAGE = "evaluation/rig/src"
CATALOG_TOOL = "{target}/release/magnitude-service-catalog-tool"
COMPILER = "RUSTC_WORKSPACE_WRAPPER"  # cargo runs rustc for workspace crates through it
GATE = GateSettings()
MAKE = "import os, sys; [os.makedirs(p, exist_ok=True) for p in sys.argv[1:]]"
WRITE = "import os, sys; p = sys.argv[1]; os.makedirs(os.path.dirname(p), exist_ok=True); open(p, 'w').write(sys.stdin.read())"
# Reconnecting after a lost connection: the first wait, doubling to the longest. A link that
# drops for a minute is followed again within the minute it returns, and an unreachable host
# costs one ssh attempt a minute.
BACKOFF_FIRST, BACKOFF_LONGEST = 5.0, 60.0


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    # A job that reads its model from storage, not from pages an earlier load cached: its
    # `{artifacts}` name the model's files as `host/fresh.py` gives them.
    fresh: bool = False
    # A host path whose existence makes the step needless: the host skips it there.
    unless: str | None = None
    # A job that reads its side's checkout (cargo building or running from its source and
    # target): the side keeps the checkout until its jobs end, rather than releasing it as its
    # build ends.
    checkout: bool = False


CATALOG_BUILD = Step("catalog tool", ("cargo", "build", "--release", "-p", "magnitude-service-catalog-tool"))
# A header export's index, written last: the export is whole once it exists.
EXPORTED = "{catalog}/headers/index.json"
# What a header export is of (`magnitude_artifacts::catalog_headers`): the catalog's lock and the
# pinned artifacts beside it, the source files `{catalog}` is keyed by.
CATALOG_INPUTS = ("catalog/models.lock.json", "engine/artifacts/src/catalog_headers.rs")


def catalog_key(source: Source) -> str:
    """The key of `source`'s header export: a digest of the files it is exported from, so each
    source reads an export of its own catalog, and a host keeps it for every run of that catalog."""
    digest = hashlib.sha256()
    for path in CATALOG_INPUTS:
        digest.update(path.encode() + b"\0" + (source.directory / path).read_bytes() + b"\0")
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class Side:
    label: str
    source: Source
    builds: tuple[Step, ...]


@dataclass(frozen=True)
class Job:
    side: str  # the label of the side whose build it runs
    name: str  # its run directory under the run's
    step: Step


@dataclass(frozen=True)
class Weights:
    """A catalog model's files, acquired (downloaded only where missing) as `{artifacts}`, and,
    with `headers`, the catalog's header export as `{catalog}`: what a service opens its catalog
    from."""

    model: str
    headers: bool = False


@dataclass(frozen=True)
class Headers:
    """The catalog's exact headers, exported to `{catalog}` (its header bundle and `headers/`),
    and, for a model, its acquisition object over them as `{artifacts}`: the model opens with
    zero weights, nothing of them downloaded."""

    model: str | None  # None: the catalog's headers alone


@dataclass(frozen=True)
class Work:
    """What a request runs on the host it is placed on."""

    sides: tuple[Side, ...]
    jobs: tuple[Job, ...] = ()
    device: str | None = None  # "device" or "timed" for the jobs; None runs them under build
    acquires: Weights | Headers | None = None  # what the jobs read of the catalog
    files: tuple[Path, ...] = ()  # local files the jobs read, as `{inputs}/<name>`


class Refused(Exception):
    """Why a request's work cannot be planned for a host: a fact of it the plan reads is not
    recorded."""


@dataclass(frozen=True)
class Request:
    instrument: str
    purpose: str
    # Seconds, the caller's; or derived once placed: from the host's alias and the seconds until
    # the run's jobs would start there (`_start`: its queues and its build).
    estimate: float | Callable[[str, float], float]
    backend: str
    # Its work on a host (by alias), planned from that host's facts once it is a candidate;
    # raises Refused for a host it cannot be planned for.
    work: Callable[[str], Work]
    host: str | None = None  # a host or pool; None places by backend among every ready host


@dataclass
class Outcome:
    run_id: str
    host: str
    results: Path
    codes: dict[str, int]  # job name -> exit code
    phases: list[dict] = field(default_factory=list)
    estimate: float | None = None  # the run's seconds as placed, when recorded


class RunFailed(Exception):
    def __init__(self, phase: str, detail: str, outcome: Outcome):
        super().__init__(f"{phase}: {detail}")
        self.phase, self.detail, self.outcome = phase, detail, outcome


def new_run_id(instrument: str) -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{instrument}-{os.urandom(3).hex()}"


def who(tree: Path) -> str:
    """The holder as other sessions see it: the coding-agent session, the worktree its run was
    staged from (`Source.tree`), the user."""
    session = state.agent_session()
    names = [f"session:{session}"] if session else []
    return " ".join([*names, tree.name, f"{getpass.getuser()}@{socket.gethostname()}"])


def duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


# Placement ---------------------------------------------------------------------------------


def build_steps(work: Work, side: Side) -> tuple[Step, ...]:
    """The steps `side`'s build runs: its own, and the catalog tool for the side that acquires,
    where it acquires anything but headers the host has exported already (the tool's release
    build is minutes of a run whose tree changed the executor)."""
    if side is not work.sides[0] or work.acquires is None:
        return side.builds
    headers = isinstance(work.acquires, Headers)
    return (*side.builds, dataclasses.replace(CATALOG_BUILD, unless=EXPORTED) if headers else CATALOG_BUILD)


def tree_key(source: sources.Source) -> str:
    """What `source` installs into a checkout: its key, commit and changes (`dirty_digest`), which
    together fix every byte staged, so runs of one tree and build share a checkout and build it
    once (`host/agent.py`)."""
    identity = source.identity
    return hashlib.sha256(json.dumps([source.key, identity.commit, identity.dirty_digest]).encode()).hexdigest()[:16]


def build_key(work: Work, side: Side) -> str:
    """What `side`'s build is, as the hosts measure builds' sizes: its steps' commands, which
    name their targets and profiles (tokens unfilled, so the key is every host's)."""
    steps = [list(step.argv) for step in build_steps(work, side)]
    return hashlib.sha256(json.dumps(steps).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Candidate:
    remote: Remote
    status: dict
    work: Work  # the request's work planned for this host
    # Each build's bytes where a host has measured none (`fallbacks`), by `build_key`.
    fallbacks: dict[str, int]
    room_wait: float | None  # seconds until its builds have room (`room_wait`); None: never
    builds: tuple[float, ...]  # each side's build's seconds here (`build_seconds`)


def room_wait(room: dict, need: int) -> float | None:
    """Seconds until builds needing `need` bytes have room on a host whose agent reports `room`
    (`host/agent.py`, `room`): at once when what is available covers every owed build and them;
    else once enough live runs have ended, the first ending first, for their checkouts to; None
    when even all of them would not."""
    available, owed = room["available"], room["owed"]
    if available >= owed + need:
        return 0.0
    for ends, freed, debt in room["ending"]:
        available, owed = available + freed, owed - debt
        if available >= owed + need:
            return ends
    return None


def platform(remote: Remote) -> tuple[str, str]:
    """What a build's size and time depend on beside its steps: the host's OS and architecture
    (one `cargo check` builds other targets on macOS than on Linux)."""
    return remote.system, remote.machine


def fallbacks(statuses: list[tuple[Work, dict]]) -> dict[str, int]:
    """Each build's bytes for a host that has measured none, from hosts of its platform
    (`platform`): the largest size of that build any of them measured, else the largest of any
    build, else nothing measured, 0."""
    keys = {build_key(work, side) for work, _ in statuses for side in work.sides}
    anything = max((status["largest"] for _, status in statuses), default=0)
    measured = lambda key: [status["measured"][key]["size"] for _, status in statuses
                            if "size" in status["measured"].get(key, {})]
    return {key: max(measured(key), default=anything) for key in keys}


def platform_fallbacks(probed: list[tuple[Remote, dict, Work]]) -> dict[tuple[str, str], dict[str, int]]:
    """`fallbacks` for each platform among the probed hosts (each its remote, status and work),
    from its own hosts alone: another platform builds other targets, so its sizes say nothing."""
    platforms = {platform(remote) for remote, _, _ in probed}
    return {found: fallbacks([(work, status) for remote, status, work in probed if platform(remote) == found])
            for found in platforms}


def _incremental(built: bool, start: str | None, commit: str) -> bool:
    """Whether a build of `commit` in a checkout that last built `start` (None: from a cold
    target) is incremental: the checkout has built that build (`built`) at an ancestor."""
    return built and start is not None and sources.descends(commit, start)


def build_seconds(status: dict, index: int, key: str, commit: str, others: list[dict]) -> float:
    """The seconds side `index`'s build (`key`, of `commit`) should take on the host whose agent
    reports `status`: what the build of the live runs whose checkout it would share still owes
    (it finds every unit fresh after it); incremental when the checkout it would claim has built
    it at an ancestor of `commit`, else cold; priced by the latest build of that class of it on this host, else the
    slowest of that class on the other candidates of its platform (`others`, their statuses); an incremental one
    unmeasured anywhere as a cold one; else the slowest build of any on this host (the agent's
    prior while none is measured)."""
    checkout = status["checkouts"][index]
    if checkout is not None and checkout["shared"]:
        return checkout["owed"]
    incremental = checkout is not None and _incremental(checkout["built"], checkout["commit"], commit)

    def measured(found: dict, wanted: bool) -> list[float]:
        return [build["seconds"] for build in found["measured"].get(key, {}).get("builds", [])
                if _incremental(build["built"], build["from"], build["to"]) == wanted]

    for wanted in (True, False) if incremental else (False,):
        if here := measured(status, wanted):
            return here[-1]
        if elsewhere := [seconds for found in others for seconds in measured(found, wanted)]:
            return max(elsewhere)
    return status["slowest"]


def _candidates(request: Request, names: tuple) -> tuple[list[Candidate], list[str]]:
    """The hosts of `names` (their configurations) the request's work can be planned for, each
    with its status and the wait for room its builds' needs give it, a need this host has not
    measured taken from the others of its platform (`fallbacks`); and why each other one that runs the backend
    was skipped."""

    def probe(config):
        try:
            remote = connect(config)
        except HostError as error:
            return None, str(error)
        if request.backend not in remote.backends:
            return None, None
        if not remote.ready:
            return None, f"{config.alias}: not set up (`mrig setup {config.alias}`)"
        try:
            work = request.work(config.alias)
        except Refused as refused:
            return None, f"{config.alias}: {refused}"
        try:
            sides = [(side.source.key, tree_key(side.source), build_key(work, side)) for side in work.sides]
            return (remote, host_status(remote, sides), work), None
        except HostError as error:
            return None, f"{config.alias}: {error}"

    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(probe, names))
    probed = [c for c, _ in found if c is not None]
    skipped = [reason for _, reason in found if reason]
    alike = lambda remote: [(found, work) for other, found, work in probed if platform(other) == platform(remote)]
    known = platform_fallbacks(probed)


    def candidate(remote: Remote, status: dict, work: Work) -> Candidate:
        own = known[platform(remote)]
        need = sum(own[build_key(work, side)] if measured is None else measured
                   for side, measured in zip(work.sides, status["needs"], strict=True))
        others = [found for found, _ in alike(remote) if found is not status]
        builds = tuple(build_seconds(status, index, build_key(work, side), side.source.identity.commit, others)
                       for index, side in enumerate(work.sides))
        return Candidate(remote, status, work, own, room_wait(status["room"], need), builds)

    return [candidate(*each) for each in probed], skipped


def _start(candidate: Candidate) -> float:
    """Seconds until the run's jobs would start on this host. Its build starts once the host
    has room for it (`Candidate.room_wait`: never, when even the live runs' ending would not
    make it, which ranks it after every host that can fit it) and
    the timed holds ahead of it end, and shares the CPU with every build the host's live runs
    still owe: sharing it equally, each of those delays it by as much of its own as it builds
    alongside it. Its jobs then start once that build ends and the runs their kind waits behind
    (holding or queued for a conflicting hold) have ended, each by the time it is expected to
    take still (`host/agent.py`: past its estimate, as long again as it has taken)."""
    status, kind = candidate.status, candidate.work.device or "build"
    if candidate.room_wait is None or kind != "build" and status["device_lock_held_elsewhere"] is not None:
        return float("inf")
    own = sum(candidate.builds)
    begins = max(candidate.room_wait, status["waits"]["build"])
    built = begins + own + sum(min(owed, own) for owed in status["building"])
    return max(built, status["waits"][kind])


def place(request: Request, hosts: Hosts, run_id: str) -> Candidate:
    """The candidate where the run's jobs would start earliest (`_start`); among equals, the
    one run `run_id` ranks first by a hash of the two, so runs placed at the same moment, before
    any sees another on a host, still spread across equal hosts."""
    named = hosts.resolve(request.host) if request.host else tuple(hosts.host(a) for a in hosts.known())
    candidates, skipped = _candidates(request, named)
    if not candidates:
        where = request.host or "any host"
        detail = "; ".join(skipped) if skipped else f"none runs {request.backend}"
        raise HostError(f"no {request.backend} host among {where}: {detail}")
    return min(
        candidates,
        key=lambda c: (_start(c), hashlib.sha256(f"{run_id} {c.remote.alias}".encode()).digest()),
    )


def placed_estimate(request: Request, candidate: Candidate) -> float:
    """The request's estimate on the candidate placed: the caller's, or derived from the
    seconds until its jobs would start there (`_start`); a host that never would (its room
    never made, its device held outside mrig: the run fails or waits for what no estimate
    knows) counts its builds alone."""
    if not callable(request.estimate):
        return request.estimate
    start = _start(candidate)
    return request.estimate(candidate.remote.alias, start if math.isfinite(start) else sum(candidate.builds))


def _idle(candidate: Candidate) -> bool:
    """Whether nothing is live on the candidate's host (no hold or request, no build owed, no
    device lock held outside mrig) and it has room for the run's build."""
    status = candidate.status
    return (not status["entries"] and not status["building"] and status["device_lock_held_elsewhere"] is None
            and candidate.room_wait == 0)


def idle_elsewhere(request: Request, hosts: Hosts, candidate: Candidate) -> list[str]:
    """When `--host` placed the run on a host that is not idle, the hosts outside what it names
    that could run the request and are idle (`_idle`): the run still goes where it is pinned."""
    if request.host is None or _idle(candidate):
        return []
    pinned = {config.alias for config in hosts.resolve(request.host)}
    others = tuple(hosts.host(alias) for alias in hosts.known() if alias not in pinned)
    return [found.remote.alias for found in _candidates(request, others)[0] if _idle(found)]


def against_estimate(entry: dict) -> str:
    """How a reported entry's run stands against its estimate, which is of its held time, its
    queued time beside it (`host.agent.account`)."""
    estimate = duration(entry["estimate"])
    queued = f"queued {duration(entry['queued'])}, " if entry["queued"] >= 1 else ""
    if entry["over"] > 0:
        return f"{queued}{duration(entry['over'])} past its {estimate} estimate"
    return f"{queued}{duration(entry['remaining'])} left of its {estimate} estimate"


def _holders(status: dict) -> str:
    """The runs ahead on the host: each hold, and each request queued for one, with how its run
    stands against its estimate."""
    entries = [e for e in status["entries"] if not e["kind"].startswith("checkout:")]
    shown = []
    for entry in entries:
        queued = "" if entry["state"] == "held" else "queued "
        purpose = f" \"{entry['purpose']}\"" if entry.get("purpose") else ""
        shown.append(f"{queued}{entry['kind']} {entry.get('run')}{purpose}, {against_estimate(entry)}")
    if status["device_lock_held_elsewhere"] is not None:
        legacy = status["device_lock_held_elsewhere"]
        shown.append(f"device held outside mrig by {legacy.get('who', 'another tool')}")
    return "holders: " + ("; ".join(shown) if shown else "none")


def _room(status: dict) -> str:
    """The host's room as placement counted it: free, and the rig's own to reclaim (the unheld
    checkouts its builds would evict, `host/agent.py`)."""
    reclaimable = status["room"]["available"] - status["free_bytes"]
    return f"room {status['free_bytes'] / 1e9:.1f} GB free + {reclaimable / 1e9:.1f} GB of the rig's own to reclaim"


def launch_line(request: Request, candidate: Candidate, idle: list[str]) -> str:
    """The run's start line; `idle`, the idle hosts its `--host` passed over (`idle_elsewhere`)."""
    status, found = candidate.status, candidate.status["checkouts"][0]
    if found is None:
        target = "no free checkout (cold build)"
    elif found["shared"]:
        target = f"checkout {found['name']}, shared with the live runs of this tree and build (builds once)"
    elif found["warmth"] == SAME:
        target = f"checkout {found['name']}, last built from this source"
    else:
        built = found["source"] or "a source it did not record"
        target = f"checkout {found['name']}, last built from {built} (rebuilds what differs)"
    staged = ", ".join(f"{side.label} {side.source.described()} from {side.source.tree}" for side in candidate.work.sides)
    return (
        f"mrig {request.instrument} · {request.backend} → {candidate.remote.alias} · staging {staged} · "
        f"estimate {duration(request.estimate)} · {_holders(status)} · target: {target} · {_room(status)}"
        + (f" · pinned by --host {request.host} to a busy host while {', '.join(idle)} "
           f"{'is' if len(idle) == 1 else 'are'} idle" if idle else "")
    )


# The plan ----------------------------------------------------------------------------------


class Plan:
    """The run's operations on its host, every token filled (`host/agent.py`)."""

    def __init__(self, request: Request, work: Work, remote: Remote, run_id: str, fallbacks: dict[str, int],
                 builds: tuple[float, ...]):
        self.request, self.work, self.remote, self.run_id = request, work, remote, run_id
        self.fallbacks = fallbacks  # `Candidate.fallbacks`
        self.builds = builds  # `Candidate.builds`
        # The header export the run acquires (`{catalog}`), of a run that acquires headers.
        exports = isinstance(work.acquires, Headers) or (isinstance(work.acquires, Weights) and work.acquires.headers)
        self.catalog = catalog_key(work.sides[0].source) if exports else None
        self.ops: list[dict] = []
        self.released: set[str] = set()  # the sides whose checkout is released: their later steps read its output

    @staticmethod
    def checkout(side: Side) -> str:
        """The side's checkout, which the runner names once it claims it."""
        return "{checkout:%s}" % side.label

    @staticmethod
    def output(side: Side) -> str:
        """The side's build output, which the runner names once it claims the checkout: the tree
        and what its build produced, which no later build changes (`host/agent.py`)."""
        return "{output:%s}" % side.label

    def base(self, side: Side) -> str:
        """Where the side's steps find its tree: its checkout until it is released, then its output."""
        return self.output(side) if side.label in self.released else self.checkout(side)

    def keeps(self, side: Side) -> bool:
        """Whether the side keeps its checkout until its jobs end: a job of it reads the checkout."""
        return any(job.step.checkout for job in self.work.jobs if job.side == side.label)

    def stage(self, side: Side) -> str:
        """Where the caller stages the side's source (as `<stage>/src`) before the run starts."""
        return self.remote.path("runs", self.run_id, f"{sync.STAGE}{side.label}")

    def tokens(self, side: Side, job: Job | None) -> dict[str, str]:
        base = self.base(side)
        run_root = self.remote.path("runs", self.run_id)
        tokens = {
            "src": f"{base}/src",
            "target": f"{base}/target",
            "venv": f"{self.output(side)}/venv",  # its build syncs it there, from the checkout's source
            "output": self.output(side),
            "root": self.remote.root,
            "models": self.remote.models,
            "python": self.remote.python,
            "artifacts": f"{run_root}/artifacts.json",
            "engine_cache": self.remote.path("cache", "engine"),
            "runs": run_root,
            "inputs": f"{run_root}/{sync.RUN_INPUTS}",
        }
        if side.label not in self.released:
            tokens["checkout"] = base
        if self.catalog is not None:
            tokens["catalog"] = self.remote.path("catalog-headers", self.catalog)
        if job is not None:
            tokens["run_dir"] = f"{run_root}/{job.name}"
            tokens["build_file"] = f"{run_root}/{job.name}/build.json"
        return tokens

    def environment(self, step: Step, tokens: dict[str, str]) -> dict[str, str]:
        """The step's environment: cargo's target and the recording wrapper only while the side
        holds its checkout (an output is never built in)."""
        cargo = {"CARGO_TARGET_DIR": tokens["target"], COMPILER: f"{tokens['checkout']}/{WRAPPER}"} \
            if "checkout" in tokens else {}
        return {
            **cargo,
            "UV_CACHE_DIR": f"{self.remote.root}/cache/uv",
            "UV_PROJECT_ENVIRONMENT": tokens["venv"],
            "PIP_CACHE_DIR": f"{self.remote.root}/cache/pip",
            "HF_HUB_CACHE": f"{self.remote.root}/cache/huggingface/hub",
            **self.remote.env,
            **{name: _fill(value, tokens) for name, value in step.env},
        }

    def host_module(self, side: Side, *arguments: str) -> tuple[list[str], dict[str, str]]:
        return host_module(self.remote, self.base(side), *arguments)

    def add(self, op: str, **fields) -> None:
        self.ops.append({"op": op, **fields})

    def phase(self, phase: str) -> None:
        self.add("phase", phase=phase)

    def step(self, side: Side, argv: list[str], env: dict[str, str], *, housekeeping: bool = False,
             **fields) -> None:
        """A step run from the side's source root, or, for host housekeeping (which may precede
        that root), from the host's root."""
        cwd = self.remote.root if housekeeping else f"{self.base(side)}/src"
        self.add("run", argv=argv, cwd=cwd, env=env, **fields)

    def to_json(self) -> dict:
        request = self.request
        return {
            "info": {"who": who(self.work.sides[0].source.tree), "purpose": request.purpose,
                     "estimate": request.estimate},
            "ops": self.ops,
        }


def host_module(remote: Remote, checkout: str, *arguments: str) -> tuple[list[str], dict[str, str]]:
    env = {"PYTHONPATH": f"{checkout}/src/{HOST_PACKAGE}", "PYTHONDONTWRITEBYTECODE": "1"}
    return [remote.python, "-m", "magnitude_rig.host", *arguments], env


def _fill(text: str, tokens: dict[str, str]) -> str:
    for token, value in tokens.items():
        text = text.replace("{" + token + "}", value)
    return text


def _substitute(argv: tuple[str, ...], tokens: dict[str, str]) -> list[str]:
    return [_fill(argument, tokens) for argument in argv]


def _refuse_checkout(step: Step) -> None:
    """Refuse to plan a job that names the checkout of a side that releases it as its build ends:
    another tree may be building there when it runs (`Step.checkout` keeps it)."""
    if any("{checkout}" in text for text in (*step.argv, *(value for _, value in step.env))):
        raise ValueError(f"job {step.name!r} names {{checkout}}, which its side releases as its build "
                         "ends: mark it Step(checkout=True)")


def _texts(plan: Plan, side: Side) -> list[str]:
    """The side's job texts with host paths, which name the executables to prove."""
    texts = []
    for job in plan.work.jobs:
        if job.side == side.label:
            tokens = plan.tokens(side, job)
            texts += _substitute(job.step.argv, tokens)
            texts += [_fill(value, tokens) for _, value in job.step.env]
    return texts


def _prepare(plan: Plan, side: Side) -> None:
    """The side's checkout, eviction and the install of its staged source. The checkout is held
    until the side's build ends (its jobs end, for a side that keeps it), the install hold only
    for the install."""
    remote = plan.remote
    build = build_key(plan.work, side)
    plan.add("checkout", side=side.label, source=side.source.key, tree=tree_key(side.source), build=build,
             output=f"{tree_key(side.source)}-{build}", fallback=plan.fallbacks[build],
             commit=side.source.identity.commit, seconds=plan.builds[plan.work.sides.index(side)])
    plan.phase(f"queue-install-{side.label}")
    plan.add("hold", kind="install")
    plan.phase(f"install-{side.label}")
    plan.add("evict", side=side.label, host=remote.alias)
    plan.step(side, remote.host(AGENT, "install", "--from", f"{plan.stage(side)}/src",
                                "--to", f"{plan.checkout(side)}/src"), {}, housekeeping=True, install=side.label)
    plan.add("release", kind="install")


def _build(plan: Plan, side: Side) -> None:
    plan.phase(f"build-{side.label}")
    argv, env = plan.host_module(side, "compiler", "--root", plan.checkout(side))
    plan.step(side, argv, env)
    for step in build_steps(plan.work, side):
        tokens = plan.tokens(side, None)
        unless = {} if step.unless is None else {"unless": _fill(step.unless, tokens)}
        plan.step(side, _substitute(step.argv, tokens), plan.environment(step, tokens), **unless)
    tokens = plan.tokens(side, None)
    output = plan.remote.path("runs", plan.run_id, f"provenance-{side.label}.json")
    tests = plan.remote.path("runs", plan.run_id, f"tests-{side.label}.json")  # the runner writes it
    argv, env = plan.host_module(side, "provenance", "--output", output, "--source", tokens["src"],
                                 "--target", tokens["target"], "--tests", tests, "--", *_texts(plan, side))
    plan.step(side, argv, env, tests=tests)
    keeps = plan.keeps(side)
    plan.add("built", side=side.label, release=not keeps, provenance=output)
    if not keeps:
        plan.released.add(side.label)


def _acquire(plan: Plan, side: Side) -> None:
    plan.phase("queue-models")
    plan.add("hold", kind="models")
    plan.phase("models")
    tokens = plan.tokens(side, None)
    env = plan.environment(Step("models", ()), tokens)
    tool = _fill(CATALOG_TOOL, tokens)
    acquires, models = plan.work.acquires, plan.remote.models
    if isinstance(acquires, Headers):
        plan.step(side, [tool, "headers", "--output", tokens["catalog"], "--model-store", models], env,
                  unless=_fill(EXPORTED, tokens))
        if acquires.model is not None:
            argv, module_env = plan.host_module(
                side, "header-artifacts", "--export", f"{tokens['catalog']}/headers", "--model", acquires.model,
                "--output", tokens["artifacts"],
            )
            plan.step(side, argv, module_env)
    else:
        if acquires.headers:
            plan.step(side, [tool, "headers", "--output", tokens["catalog"], "--model-store", models], env,
                      unless=_fill(EXPORTED, tokens))
        plan.step(side, [tool, "acquire", "--model", acquires.model, "--model-store", models], env,
                  stdout_to=tokens["artifacts"])
    plan.add("release", kind="models")


def _jobs(plan: Plan) -> None:
    work = plan.work
    sides = {side.label: side for side in work.sides}
    if work.device is not None:
        plan.phase(f"queue-{work.device}")
        plan.add("hold", kind=work.device)
    elif work.jobs:  # host work that needs no device still never runs during a measurement
        plan.phase("queue-build")
        plan.add("hold", kind="build")
    if work.device == "timed":
        plan.phase("gate")
        side = work.sides[0]
        settings = [
            argument
            for name, value in vars(GATE).items()
            for argument in (f"--{name.replace('_', '-')}", str(value))
        ]
        argv, env = plan.host_module(side, "gate", "--log", plan.remote.path("runs", plan.run_id, "thermal-gate.jsonl"), *settings)
        plan.step(side, argv, env)
    for job in work.jobs:
        side = sides[job.side]
        if side.label in plan.released:
            _refuse_checkout(job.step)
        plan.phase(f"job-{job.name}")
        tokens = plan.tokens(side, job)
        plan.step(side, [plan.remote.python, "-c", WRITE, tokens["build_file"]], {}, housekeeping=True,
                  input=json.dumps(side.source.identity.to_json(), indent=2))
        if job.step.fresh:
            fresh = f"{tokens['run_dir']}/artifacts.json"
            argv, _ = plan.host_module(
                side, "fresh", "--artifacts", tokens["artifacts"], "--output", fresh,
                "--clones", f"{tokens['runs']}/{sync.RUN_CLONES}/{job.name}", "--",
                *_substitute(job.step.argv, {**tokens, "artifacts": fresh}),
            )
        else:
            argv = _substitute(job.step.argv, tokens)
        # The job inherits `record`'s environment, the host package on PYTHONPATH among it, so a
        # job running a host module names none of its own.
        record, record_env = plan.host_module(
            side, "record", "--run-dir", tokens["run_dir"], "--alias", plan.remote.alias, "--", *argv,
        )
        plan.step(side, record, {**plan.environment(job.step, tokens), **record_env}, job=job.name)


def plan_for(request: Request, candidate: Candidate, run_id: str) -> Plan:
    """Every operation the run's host carries out, in order."""
    work = candidate.work
    plan = Plan(request, work, candidate.remote, run_id, candidate.fallbacks, candidate.builds)
    for side in work.sides:
        _prepare(plan, side)
    plan.phase("queue-build")
    plan.add("hold", kind="build")
    for side in work.sides:
        _build(plan, side)
    plan.add("release", kind="build")
    if work.acquires is not None:
        _acquire(plan, work.sides[0])
    _jobs(plan)
    return plan


# Following ---------------------------------------------------------------------------------


class Logs:
    """Each phase's step output in `logs/<phase>.log` as the caller sees the runner write it (the
    fetch brings the host's whole copy), the caller's own notes in `logs/caller.log`, and the
    last line any step printed."""

    CALLER = "caller"

    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.last: tuple[float, str] | None = None
        self._lock = threading.Lock()

    def write(self, phase: str, stream: str, text: str) -> None:
        with self._lock, open(self.directory / f"{phase}.log", "a") as log:
            log.write(text if stream == "stdout" else f"[stderr] {text}")
        if text.strip() and not text.startswith(NOTE):
            self.last = (time.monotonic(), text.strip())

    def note(self, message: str) -> None:
        self.write(self.CALLER, "stdout", f"{NOTE}{message[:400]}\n")


class Run:
    """One run as its caller follows it: the runner's events mirrored into the results."""

    def __init__(self, request: Request, candidate: Candidate, run_id: str, results: Path,
                 announce: Callable[[str], None]):
        self.request = request
        self.announce = announce  # where the run's own lines go: its caller's output
        self.remote = candidate.remote
        self.checkouts = candidate.status["checkouts"]  # what each side would claim, at placement
        self.run_id = run_id
        self.results = results
        self.logs = Logs(results / "logs")
        self.started = time.monotonic()
        self.phase = "connect"
        self.phase_started = self.started
        self.phases: list[dict] = []  # as seen here, for the report
        self.state = state.RunState(
            run_id,
            {
                "instrument": request.instrument,
                "purpose": request.purpose,
                "estimate": request.estimate,
                "host": self.remote.alias,
                "results": str(results),
                "started": time.time(),
                "phase": self.phase,
            },
        )
        self.host: HostRun | None = None
        self.package: str | None = None  # the first side's stage, with the host package, once staged
        self.plan_sent = False  # whether a start of the run may have reached the host with its plan
        self.sequence = 0  # the last event followed
        self.stacks: list[str] = []  # each process asked for its stacks: where they go, and why

    def enter(self, phase: str) -> None:
        now = time.monotonic()
        self.phases.append({"phase": self.phase, "seconds": round(now - self.phase_started, 3)})
        self.phase, self.phase_started = phase, now
        self.state.update(phase=phase, phase_started=time.time(), waiting_behind=None)

    def note(self, message: str) -> None:
        self.logs.note(f"{self.phase}: {message}")

    def receive(self, event: dict) -> None:
        """Mirror one runner event here."""
        self.sequence = event["seq"]
        kind = event["event"]
        if kind == "phase":
            self.enter(event["phase"])
        elif kind == "output":
            self.logs.write(event["phase"], event["stream"], event["text"])
        elif kind == "waiting":
            self.state.update(waiting_behind=", ".join(f"{b['kind']} {b.get('run')}" for b in event["blocking"]))
        elif kind == "evicted":  # the host reclaimed its own checkouts and gone runs for this run
            self.announce(f"mrig: {self.remote.alias}: evicted {', '.join(event['removed'])} "
                          f"({event['freed_bytes'] / 1e9:.1f} GB) to make room for this run's build")
        elif kind == "stacks":  # a job's process asked for its stacks: a hang captured, or none to ask
            if event["pid"] is None:
                where = f"none: no live process of the job reports its stacks ({event['reason']})"
                self.announce(f"mrig: {self.remote.alias}: {event['reason']}, but no live process of the job "
                              "reports its stacks (none registered, or none left)")
            else:
                where = f"{self.results / event['path']} ({event['reason']})"
                self.announce(f"mrig: {self.remote.alias}: process {event['pid']} reports its stacks to {where}")
            self.stacks.append(where)
        elif kind == "held":
            self.state.update(waiting_behind=None)
        elif kind == "queue":  # the host's account of its waits changed (`state.account`)
            self.state.update(waited=event["waited"], waiting_since=time.time() if event["waiting"] else None)

    def stage(self, plan: Plan, stop: threading.Event) -> None:
        """Put everything the run reads from here into its directory on the host, before it
        starts: each side's source in its stage (sending only what the checkout it would claim
        at placement lacks) and the request's files in its inputs, sent from their copy in its
        results, which the record hashes (they are never fetched back); no more once `stop` is
        set."""
        work, sides = plan.work, plan.work.sides
        inputs = plan.tokens(sides[0], None)["inputs"]
        made = [f"{plan.stage(side)}/src" for side in sides] + ([inputs] if work.files else [])
        _retrying(self, "stage", lambda: _ssh(self.remote, self.remote.command([self.remote.python, "-c", MAKE, *made])[-1]),
                  stop)
        for side, found in zip(sides, self.checkouts):
            if stop.is_set():
                return
            started = time.monotonic()
            alike = None if found is None else self.remote.path("checkouts", found["name"], "src")
            _retrying(self, "stage", lambda side=side, alike=alike: sync.push(
                self.remote, list(side.source.paths), side.source.directory, f"{plan.stage(side)}/src",
                alike=alike), stop)
            self.note(f"staged {len(side.source.paths)} files of {side.label} in {time.monotonic() - started:.1f} s")
        if work.files:
            local = self.results / sync.RUN_INPUTS
            local.mkdir(exist_ok=True)
            for path in work.files:
                shutil.copyfile(path, local / path.name)
            _retrying(self, "stage", lambda: sync.push(self.remote, [path.name for path in work.files], local, inputs),
                      stop)
        self.package = plan.stage(sides[0])

    # the report

    def write_report(self) -> None:
        """Write what the run knows now to `report.txt`, atomically: each request (a thread of
        its own per SIGUSR1) stages its own file, so reports asked at once each land whole."""
        path = self.results / REPORT
        staged = path.with_name(f".{REPORT}.{os.getpid()}.{threading.get_ident()}")
        staged.write_text(self.report() + "\n")
        staged.replace(path)

    def report(self) -> str:
        now = time.monotonic()
        phases = [*self.phases, {"phase": self.phase, "seconds": now - self.phase_started}]
        queued, held = state.account(self.state.value, time.time())
        lines = [
            (f"mrig: {self.request.instrument} {self.run_id} ({self.remote.alias}) at "
             f"{duration(held)} held (queued {duration(queued)} more), estimate {duration(self.request.estimate)}"),
            "  phases:       " + " · ".join(f"{p['phase']} {duration(p['seconds'])}" for p in phases),
        ]
        waiting = self.state.value.get("waiting_behind")
        if waiting:
            lines.append(f"  waiting:      behind {waiting}")
        if self.logs.last is not None:
            at, text = self.logs.last
            lines.append(f"  last output:  {duration(now - at)} ago, {text[:160]!r}")
        else:
            lines.append("  last output:  none yet")
        lines.append(f"  device:       {self.device_facts()}")
        lines += [f"  stacks:       {where}" for where in self.stacks]
        lines.append(f"  log:          {self.logs.directory / (self.phase + '.log')}")
        lines.append(f"  stop:         mrig stop {self.run_id}")
        return "\n".join(lines)

    def device_facts(self) -> str:
        if self.package is None:
            return "not read yet: the host's probe arrives with the run's source, still staging"
        argv, env = host_module(self.remote, self.package, "probe")
        line = shlex.join(["env", *(f"{k}={v}" for k, v in env.items()), *argv])
        try:
            reading = json.loads(_ssh(self.remote, line, timeout=30))
        except (HostError, ValueError) as error:
            return f"unavailable ({error})"
        utilization = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(reading["utilization_pct"].items()))
        return utilization or f"no utilization reading ({reading['errors']})"


class Stopped(Exception):
    """The caller was stopped while the host was away, before anything of the run started."""


def _retrying(run: Run, what: str, action: Callable[[], object],
              abandon: threading.Event | None = None) -> object:
    """`action` until it reaches the host, after a lost connection backing off as it lasts; a
    failed command fails the run (`CommandFailed`). With `abandon`, for an action nothing
    started depends on, it ends once `abandon` is set (`Stopped`) rather than waiting for the
    host."""
    backoff = BACKOFF_FIRST
    while True:
        try:
            return action()
        except HostUnavailable as error:
            if abandon is not None and abandon.is_set():
                raise Stopped(f"{what}: {error}") from error
            run.note(f"{what} failed ({error}); again in {backoff:.0f} s")
            if abandon is not None and abandon.wait(backoff):
                raise Stopped(f"{what}: {error}") from error
            if abandon is None:
                time.sleep(backoff)
            backoff = min(2 * backoff, BACKOFF_LONGEST)


def _stop_when(run: Run, stop: threading.Event, followed: threading.Event) -> None:
    """Deliver the caller's stop to the runner, once asked, however long the host is away;
    nothing once the run has been followed to its end."""
    stop.wait()
    if not followed.is_set():
        run.state.update(stopping=time.time())  # `mrig stop` waits for this
        _retrying(run, "stop", run.host.stop)


def _follow(run: Run) -> dict:
    """Follow the runner to its end, through lost connections; its `ended` event."""
    backoff = BACKOFF_FIRST
    while True:
        try:
            for event in run.host.follow(run.sequence):
                backoff = BACKOFF_FIRST
                if event["event"] == "vanished":
                    return _vanished(run.phase, run.remote.alias)
                run.receive(event)
                if event["event"] == "ended":
                    return event
        except HostUnavailable as error:
            run.note(f"connection lost ({error}); following again in {backoff:.0f} s")
            time.sleep(backoff)
            backoff = min(2 * backoff, BACKOFF_LONGEST)


def _announce(line: str) -> None:
    print(line, flush=True)


def execute(request: Request, run_id: str, *, hosts: Hosts | None = None,
            announce: Callable[[str], None] = _announce, command: tuple[str, ...] = ()) -> Outcome:
    """Run `request` as `run_id` to its end, fetch what its answer reads and return where its
    results are; raises RunFailed when a phase before the jobs fails (the jobs' own exit codes
    are the outcome's). The run stays in flight until its caller, answered, `complete`s it.
    SIGUSR1 writes the run's report meanwhile. `command` is the mrig command line that asked for
    the run, kept until it is fetched so a later command answers it (`collect`)."""
    results = store.results() / run_id
    current: list[Run] = []

    def reported(signum, frame):
        threading.Thread(target=_write_report, args=(current, request, results), daemon=True).start()

    previous = signal.signal(signal.SIGUSR1, reported)
    try:
        return _execute(request, run_id, results, hosts or Hosts.load(), announce, current, command)
    finally:
        signal.signal(signal.SIGUSR1, previous)


def _write_report(current: list[Run], request: Request, results: Path) -> None:
    if current:
        current[0].write_report()
        return
    results.mkdir(parents=True, exist_ok=True)
    path = results / REPORT
    staged = path.with_name(f".{REPORT}.{os.getpid()}.{threading.get_ident()}")
    staged.write_text(f"mrig: {request.instrument} is still choosing its host\n")
    staged.replace(path)


def _launch(run: Run, plan: Plan, stop: threading.Event) -> HostRun | None:
    """Stage the run under the host's hold on its directory, then start it; None when stopped
    first, which withdraws the hold, so nothing starts. A run an earlier attempt started (its
    reply lost) is its runner's: followed, never staged again. Stopped while the host is away,
    before its plan was ever sent, it raises `Stopped`: nothing of it can have started."""
    if stop.is_set() and not run.plan_sent:
        raise Stopped("start: stopped before its plan was sent")
    with HostRun.staging(run.remote, run.run_id) as start:
        if isinstance(start, HostRun):
            return start
        run.stage(plan, stop)
        if stop.is_set():
            return None
        run.plan_sent = True  # from here a runner may own the run, whatever the reply
        return start(plan.to_json())


def _execute(request: Request, run_id: str, results: Path, hosts: Hosts,
             announce: Callable[[str], None], current: list[Run], command: tuple[str, ...]) -> Outcome:
    candidate = place(request, hosts, run_id)
    request = dataclasses.replace(request, estimate=placed_estimate(request, candidate))
    announce(launch_line(request, candidate, idle_elsewhere(request, hosts, candidate)))
    run = Run(request, candidate, run_id, results, announce)
    current.append(run)
    stop = threading.Event()
    # Stopping the caller stops the run: the runner ends its command and releases its holds.
    previous = signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    try:
        plan = plan_for(request, candidate, run_id)
        run.enter("stage")
        # A connection lost while staging loses the host's hold on the directory: stage again
        # under a new one; stopped meanwhile, before its plan was sent, nothing starts.
        try:
            run.host = _retrying(run, "start", lambda: _launch(run, plan, stop))
        except Stopped:
            run.host = None
        # The caller's phases: every one entered here, the last closing at the runner's first.
        caller = len(run.phases) + 1
        if run.host is None:
            ended = {"codes": {}, "phases": [], "failed": {
                "phase": run.phase, "detail": "stopped by mrig stop before it started"}}
        else:
            state.Launched(run_id, run.remote.alias, run.host.agent, results, command).write()
            followed = threading.Event()
            threading.Thread(target=_stop_when, args=(run, stop, followed), daemon=True).start()
            ended = _follow(run)
            followed.set()  # the runner has ended, its holds released: no stop goes to its pid after it
            run.enter("fetch")
            _retrying(run, "fetch", lambda: sync.fetch_answer(run.remote, run.remote.path("runs", run_id), results))
    finally:
        signal.signal(signal.SIGTERM, previous)
    outcome = Outcome(run_id, run.remote.alias, results, ended["codes"], estimate=request.estimate)
    failed = ended["failed"]
    last = {"phase": run.phase, "seconds": round(time.monotonic() - run.phase_started, 3)}  # fetch, or stage
    outcome.phases = [*run.phases[:caller], *ended["phases"], last]
    record((request, candidate.work), run_id, run.remote.alias, outcome, failed)
    if failed is not None:
        raise RunFailed(failed["phase"], failed["detail"], outcome)
    return outcome


def waits(results: Path) -> list[dict]:
    """Every `mrig wait` on the run: when, the new estimate and why it was needed."""
    path = results / WAITS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def record(placed: tuple[Request, Work] | None, run_id: str, host: str, outcome: Outcome,
           failed: dict | None) -> None:
    """The run's `rig.json`: what it was (the request and its work on its host, when its caller
    records it), where, each phase's time and how it ended."""
    entry = {
        "run": run_id,
        "seconds": round(sum(p["seconds"] for p in outcome.phases), 3),
        "phases": outcome.phases,
        "host": host,
        "codes": outcome.codes,
        "failed": failed,
        "waits": waits(outcome.results),
        "inputs": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "bytes": path.stat().st_size}
                   for path in sorted((outcome.results / "inputs").glob("*")) if path.is_file()},
    }
    if placed is not None:
        request, work = placed
        entry.update({
            "instrument": request.instrument,
            "purpose": request.purpose,
            "estimate_seconds": request.estimate,
            "backend": request.backend,
            "sources": {
                side.label: {"name": side.source.name, "key": side.source.key, "branch": side.source.branch,
                             "tree": str(side.source.tree), "described": side.source.described(),
                             **side.source.identity.to_json()}
                for side in work.sides
            },
            "acquires": None if work.acquires is None else {
                "kind": type(work.acquires).__name__.lower(), "model": work.acquires.model,
            },
        })
    outcome.results.mkdir(parents=True, exist_ok=True)
    (outcome.results / "rig.json").write_text(json.dumps(entry, indent=2))


def complete(run_id: str, hosts: Hosts | None = None) -> None:
    """Close run `run_id`, which its caller has answered: it is no longer in flight, and the rest
    of its directory is fetched and removed from the host (`_fetch_rest`), unless a collector
    claimed it first and does that itself. A fetch that fails is reported on stderr and left for
    a later command."""
    with state.Launched.claim(run_id) as claimed:
        state.RunState.remove(run_id)
        if claimed:
            _owed(state.Launched.read(run_id), lambda launched: _fetch_rest(hosts or Hosts.load(), launched))


def collect(hosts: Hosts, answer: Callable[[dict], None], *, rest: bool) -> tuple[list[dict], list[dict]]:
    """Take up every run this machine launched whose caller is gone, from where it stopped: one
    unanswered whose runner no longer runs is fetched (what its answer reads), recorded from its
    own record of its end (or, its runner gone without one, as failed) and `answer`ed from its
    launched record (`_answer`); then, with `rest`, its rest is fetched and the host prunes it
    (`_fetch_rest`). The launched records of the runs it answered, and those of the answered runs
    whose rest it left to fetch. Each run is taken up by the one command holding its claim
    (`state.Launched.claim`); one whose step fails is reported on stderr and stays, on the host
    and here, for a later command, never failing this one."""
    answered, unfetched = [], []
    for launched in state.Launched.gone():
        with state.Launched.claim(launched["run"]) as claimed:
            if not claimed:
                continue  # another command is taking it up, or has
            results = Path(launched["results"])
            if not (results / CODE).exists():
                if not _owed(launched, lambda launched: _answer(hosts, launched, answer)):
                    continue  # still running, or left for a later command
                answered.append(launched)
            if rest:
                _owed(launched, lambda launched: _fetch_rest(hosts, launched))
            else:
                unfetched.append(launched)
    return answered, unfetched


def _owed(launched: dict, step: Callable[[dict], bool | None]) -> bool:
    """`step` of a launched run; whether it was taken. One its host fails (`HostError`) or whose
    answered command exits (`SystemExit`) is reported on stderr and left for a later command;
    anything else is a fault of the rig and raised."""
    try:
        return step(launched) is not False
    except (HostError, SystemExit) as error:
        print(f"mrig: taking up {launched['run']} from {launched['host']} failed, "
              f"left for a later command: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        return False


def _vanished(phase: str, host: str) -> dict:
    """The end recorded for a run whose runner is gone without recording its own."""
    return {"codes": {}, "phases": [], "failed": {
        "phase": phase, "detail": f"its runner on {host} ended without recording its end"}}


def _answer(hosts: Hosts, launched: dict, answer: Callable[[dict], None]) -> bool:
    """Answer one claimed launched run unless its runner still runs; whether it was answered.
    Recorded first, unless its caller recorded it before it died. A runner gone without
    recording its end (killed, or lost to a reboot) is recorded as failed, with whatever of its
    directory the host still keeps."""
    results = Path(launched["results"])
    if not (results / "rig.json").exists():
        remote = connect(hosts.host(launched["host"]))
        standing = run_state(remote, launched["run"])
        if standing == "running":
            return False  # a later command looks again
        if standing == "absent":
            # One an earlier collector fetched whole, with its end, and had removed before it died.
            results.mkdir(parents=True, exist_ok=True)
        else:
            sync.fetch_answer(remote, remote.path("runs", launched["run"]), results)
        ended = (json.loads((results / ENDED).read_text()) if (results / ENDED).exists()
                 else _vanished("runner", launched["host"]))
        outcome = Outcome(launched["run"], launched["host"], results, ended["codes"], ended["phases"])
        record(None, launched["run"], launched["host"], outcome, ended["failed"])
    answer(launched)
    return True


def _fetch_rest(hosts: Hosts, launched: dict) -> None:
    """Fetch the rest of one answered run's directory, have the host prune it and remove its
    launched record."""
    remote = connect(hosts.host(launched["host"]))
    if run_state(remote, launched["run"]) != "absent":
        sync.fetch(remote, remote.path("runs", launched["run"]), Path(launched["results"]))
        HostRun(remote, launched["run"], launched["agent"]).fetched()
    state.Launched.remove(launched["run"])
