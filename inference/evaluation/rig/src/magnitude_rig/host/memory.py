#!/usr/bin/env python3
"""`mrig memory`'s measurement: what a load's worker holds outside its charge, on a dedicated
GPU or in unified memory, and the costs fitted to those loads. Standard library only, so it
also runs alone on a machine outside the rig.

The costs it measures are assessment's, in
`inference/engine/executor/src/planning/assessment.rs`.

Which measurement a device takes is whose memory it is (`--memory`), as the engine's
topology states it, never its backend: a discrete GPU's own (`Own`), or the host's (`Host`:
Apple silicon, the host CPU, and an integrated GPU on any backend, GB10's CUDA among them).

Dedicated memory (`DeviceProcessCosts`): for each model, loads
it with the standalone engine in an isolated kernel cache, samples the
GPU's memory while it loads, and splits what the device lost into the terms
assessment prices:

- context: what the device lost when the engine opened it (the `device
  opened` moment) before any allocation;
- programs: what it lost by `programs prepared` beyond the context and what
  the engine's ledger counts allocated there (`device_allocated_bytes`), per
  native kernel of the load's kernel inventory (`magnitude-assess --breakdown`,
  `kernels`);
- residual: what it lost at load end beyond context, programs and allocated.

The GPU must be otherwise idle: device-wide memory is sampled (`mrig memory`
holds the device for it). Run it from any directory; it writes only under
--scratch.

Unified memory (`HostProcessCosts`): the device's memory is the
host's, so a load's own memory is its worker's. `magnitude-harness memory`
loads each model's target plain in a worker process, as the service loads,
and prints its charge; the load's own memory is the worker's footprint at
load end (`loaded`) less what its plan holds then (device and upload
storage, and the exact host terms), and its charge is the runtime,
vocabulary and program terms. The projector and drafter are not loaded:
vision is a term of its own, and a drafter's graphs are planned. Under
`--weight-cap` the load is planned under the cap, its routed experts
streamed, and the same comparison says whether the plan charges everything a
streamed load holds; the storage the load held is shown beside its plan's,
term by term: the plan's storage terms beside Seismic's charge at load end in
each (the engine's `MemoryChargeReconciliation`), with what no term holds.
With `--residency` the load is at a template's context plus the tokens it decodes after it
(lane 0 of the template file, which the harness renders), and also reads how many pages of
its expert stacks the file cache holds once loaded and once decoded (it must hold none);
with `--pressure-store STORE` too, another process takes what that store gives past the
guarantee the load's ledger keeps free there over the middle half of that decode, and the
owner's resizes and releases are reported in order with the decode's tokens against the same
decode on a resident load: none in the steady decode before the press, each matched to the
reading and term that caused it, and each shrink returning memory to the OS.

  python3 memory.py --bin DIR --backend cuda|vulkan|metal|cpu --memory dedicated|unified \\
      --scratch DIR [--weight-cap BYTES] [--context TOKENS] \\
      [--residency TEMPLATE TOKENS [--pressure-store host|device]] \\
      [--artifacts ACQUISITION.json] [TARGET.gguf[,PROJECTOR.gguf[,DRAFT.gguf]] ...]

DIR holds magnitude-engine and magnitude-assess (an unpacked engine
installation or a release build) for dedicated memory, magnitude-harness
(a release build of the evaluation harness) for unified memory. `--artifacts`
names a catalog acquisition object (`catalog-tool acquire`) whose target,
projector and drafter are measured. Results: --scratch/device-costs.json or
host-costs.json and a table on stdout; from a machine outside the rig, send
both back for calibration.

Fitting the unified-memory costs (`HostProcessCosts::of`) to loads:

  python3 memory.py --fit HOST-COSTS.json ...

over every load in the files (`mrig memory` leaves each run's in
`inference/evaluation/results/<run>/memory/`): per backend, the runtime,
per-vocabulary-entry and per-kernel costs are the least-overestimating ones
under which no load's own memory exceeds its charge (a linear program in
three costs, solved at its vertices). It prints them and each load's
overcharge under the charge it was measured with and under the fit.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

WINDOWS = platform.system() == "Windows"
BACKENDS = ("cuda", "vulkan", "metal", "cpu")
# By the device's memory, as the engine's topology states it (`mrig device`): its own, sampled
# device-wide, or the host's, read as the worker's footprint.
RESULTS = {"dedicated": "device-costs.json", "unified": "host-costs.json"}
PRESSURE = "pressure.json"  # `--pressure-store`'s resizes, releases, pressure and tokens


def exe(directory, name):
    return str(Path(directory) / (name + (".exe" if WINDOWS else "")))


class Sampler:
    """Device-wide used bytes, sampled until stopped, as (time, bytes)."""

    def __init__(self):
        self.samples, self.running = [], True
        self.read = self.reader()

    def reader(self):
        if shutil.which("nvidia-smi"):
            def nvidia():
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, check=True).stdout.split()
                return int(out[0]) * 2**20
            return nvidia
        cards = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used")) if not WINDOWS else []
        if cards:
            return lambda: int(cards[0].read_text())
        if WINDOWS:
            counter = r"\GPU Adapter Memory(*)\Dedicated Usage"
            def adapter():
                out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                      f"((Get-Counter '{counter}').CounterSamples | Measure-Object -Property CookedValue -Maximum).Maximum"],
                                     capture_output=True, text=True, check=True).stdout.strip()
                return int(float(out))
            return adapter
        sys.exit("no GPU memory reading: need nvidia-smi, amdgpu sysfs, or Windows GPU counters")

    def start(self):
        def loop():
            while self.running:
                try:
                    self.samples.append((time.time(), self.read()))
                except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
                    print(f"sample failed: {error}", file=sys.stderr)  # a missed sample is not fatal
                time.sleep(0.1)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.thread.join()

    def at(self, moment_time):
        """The first sample at or after a moment (the device as the moment left it)."""
        later = [value for when, value in self.samples if when >= moment_time]
        return later[0] if later else self.samples[-1][1]


def engine_args(paths, port, cache):
    parts = paths.split(",")
    args = ["--model", parts[0], "--port", str(port), "--cache-dir", cache, "--served-model", "m"]
    args += ["--projector", parts[1]] if len(parts) > 1 and parts[1] else ["--no-projector"]
    if len(parts) > 2 and parts[2]:
        args += ["--draft", parts[2]]
    return args


def run_engine(bindir, backend, paths, cache, log, port):
    """Start the engine, wait until it serves, send one chat, stop it. Returns
    the timestamped stderr lines."""
    proc = subprocess.Popen([exe(bindir, "magnitude-engine"), "--device", backend] + engine_args(paths, port, cache),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    lines = []

    def drain():
        for line in proc.stderr:
            lines.append((time.time(), line.rstrip("\n")))
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 3600
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            urllib.request.urlopen(url + "/health", timeout=1)
            break
        except OSError:  # not serving yet
            time.sleep(1)
    if proc.poll() is None:
        body = json.dumps({"model": "m", "max_tokens": 64,
                           "messages": [{"role": "user", "content": "Write three sentences about memory."}]}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body,
                                                          {"content-type": "application/json"}), timeout=600).read()
        except OSError as error:
            print(f"chat failed: {error}", file=sys.stderr)
        time.sleep(1)
        lines.append((time.time(), "SCRIPT: after chat"))
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    reader.join(5)
    Path(log).write_text("\n".join(f"{when:.3f} {line}" for when, line in lines))
    return lines


def moments(lines):
    """The load's memory moments, the last of each name (the Prepare worker's
    come first), with their times."""
    found = {}
    for when, line in lines:
        if (moment := _moment(line)) is not None:
            found[moment["moment"]] = (when, moment)
    return found


EVENT = "magnitude-harness: event "
MOMENT = "load memory"


def _moment(line):
    """The memory moment a line reports, if it does: the harness's JSON event or the engine
    executable's log line of `load memory`."""
    if line.startswith(EVENT):
        event = json.loads(line[len(EVENT):])
        return json.loads(event["record"]) if event["message"] == MOMENT else None
    _, found, record = line.partition(f"{MOMENT} record=")
    return json.loads(record) if found else None
SHRINK_STEPS = ("queued", "planned")  # a shrink's steps on the submitting thread; nothing moves
YIELD = "memory released: the cache the ledger took"
SHORTAGE = "a real memory shortage"
READ = "memory read"
# What a resize record names its cause by: the store it resized in, the reading whose derived target
# it followed (`Ledger::generation` there, which that reading's record carries) and the term of
# that reading's record whose move since `before_generation` set the target: the reading the store's
# previous resize followed, which the engine judges the move against. A store's first resize
# follows no earlier one: its term is `load`, whatever reading it names as before.
CAUSE = ("store", "reading", "term")
BEFORE = "before_generation"
LOAD = "load"
# What a shortage's response took beyond the cache: new work waiting, sequences preempted, the
# model unloaded (spec §4.5, steps 2-4), each only once the bank sits at its floor.
COSTLIER = ("admission_waits", "preempt", "unload")


def releases(lines):
    """The owner's memory events, each with its time in Unix seconds, in order: the bank's resizes
    (`memory released`, a shrink, and `memory grown`, `resize` naming which) with their cause
    (`CAUSE`) and, for a shrink, the bytes it returned to the OS (`returned`), the cache the ledger
    took released at a boundary (`yield`) and each real shortage (`shortage`, its cause, and its
    response's `COSTLIER` steps): the scheduler's events the worker prints, one JSON object each, an
    unread value absent."""
    found = []
    for line in lines:
        if not line.startswith(EVENT):
            continue
        event = json.loads(line[len(EVENT):])
        if event["message"] in ("memory released", "memory grown"):
            resize = "Shrink" if event["message"] == "memory released" else "Growth"
            found.append({"at": event["at"], "resize": resize, "ms": event["micros"] / 1e3,
                          "bytes": event.get("released" if resize == "Shrink" else "grown"),
                          **{key: event.get(key) for key in ("bank_slots", "held_before", "held_after", "holding",
                                                              "from", "to", *CAUSE, BEFORE)}})
            if resize == "Shrink":
                # A shrink's steps on the submitting thread (ms), and what it returned to the OS;
                # a growth takes none of them.
                found[-1].update({key: event[f"{key}_micros"] / 1e3 for key in SHRINK_STEPS})
                found[-1]["returned"] = event.get("returned")
            # When a resize's other steps happened.
            found[-1].update({key: event["at"] - event[f"{key}_before"] / 1e6
                              for key in ("decided", "prepared", "installed", "released")})
        elif event["message"] == YIELD:
            found.append({"at": event["at"], "yield": event["micros"] / 1e3,
                          "held_before": event["held_before"], "held_after": event["held_after"]})
        elif event["message"] == SHORTAGE:
            found.append({"at": event["at"], "shortage": event["cause"], "store": event["store"],
                          **{key: event.get(key) for key in COSTLIER}})
    return found


def readings(lines):
    """The owner's memory readings (`memory read`, the ledger's estimate of one store at a reading),
    by store, each store's in the order they were read: the records as printed."""
    found = {}
    for line in lines:
        if line.startswith(EVENT):
            event = json.loads(line[len(EVENT):])
            if event["message"] == READ:
                found.setdefault(event["store"], []).append(event)
    return found


def untraced(resizes, read):
    """Each resize matched to its cause: the reading its record names (`reading`, among `read`, the
    store's readings) and the term there that moved from the reading its record names as before
    (`before_generation`, the reading the store's previous resize followed). A resize follows a
    target the ledger derives afresh at each reading, and a target moves only where a term of the
    estimate does: one with no reading, no term, or a term that did not move is untraced, and
    steady decode moves no term. A store's first resize is the load's own sizing (`load`), and
    only that one. The failures, one a resize."""
    failures = []
    resized = set()
    for event in resizes:
        what = f"{event['resize']} of {event.get('holding')} {event.get('from')}→{event.get('to')} at {event['at']}"
        if any(event.get(key) is None for key in CAUSE):
            failures.append(f"{what} is untraced: its record names no " + ", ".join(
                key for key in CAUSE if event.get(key) is None))
            continue
        first = event["store"] not in resized
        resized.add(event["store"])
        store = {record.get("generation"): record for record in read.get(event["store"], [])}
        term, before = event["term"], event.get(BEFORE)
        if first:
            if term != LOAD:
                failures.append(f"{what} is the first resize of {event['store']} but names {term}, not {LOAD}")
            continue
        if term == LOAD or before is None:
            failures.append(f"{what} names {term} with before reading {before}, "
                            f"though an earlier resize of {event['store']} came before it")
            continue
        missing = [generation for generation in (event["reading"], before) if generation not in store]
        if missing:
            failures.append(f"{what} follows reading {event['reading']} from {before} of {event['store']}, "
                            f"and no record holds " + ", ".join(map(str, missing)))
        elif term not in store[event["reading"]] or term not in store[before]:
            failures.append(f"{what} names {term}, which the readings of {event['store']} do not carry")
        elif store[event["reading"]][term] == store[before][term]:
            failures.append(f"{what} names {term}, which held at {store[before][term]} "
                            f"from reading {before} to {event['reading']}")
    return failures


def steady_resizes(pressure, resizes):
    """The resizes decided while the decode ran steady: from its first token to the press's start,
    with the occupier's approach already held (it leaves the load the guarantee its ledger keeps
    free, so no term of the estimate moves). Steady decode makes none; each one there is a
    failure."""
    # The outputs' times; the admission's (token 0) is where a press from the prompt starts.
    outputs = [at for count, at in pressure.get("steps") or [] if count > 0]
    started = pressure.get("started", {}).get("at")
    if not outputs or started is None:
        return ["no steady decode: the decode or the press never began"]
    first = outputs[0]
    steady = [event for event in resizes if first <= event["decided"] < started]
    print(f"steady decode: {started - first:.2f} s before the press, {len(steady)} resizes in it")
    return [f"{len(steady)} resizes in steady decode, before the press: "
            + ", ".join(f"{event['resize']} of {event.get('holding')} at {event['decided']}" for event in steady)
            ] if steady else []


STEP_READS = ("misses", "missed_bytes", "read_micros", "parked_micros")


def served_steps(lines):
    """Each streamed step's reads as it completed (`expert step completed`): its time in Unix
    seconds, its misses, the bytes they read and how long its reads ran and were parked."""
    found = []
    for line in lines:
        if line.startswith(EVENT):
            event = json.loads(line[len(EVENT):])
            if event["message"] == "expert step completed":
                found.append({"at": event["at"], **{key: event[key] for key in STEP_READS}})
    return found


def report_stalls(pressure, released, start, served):
    """What each resize in the pressure's window costs the decode, apart from steps that are slow on
    their own. The window is the decode's span under the held pressure (the harness's `window`
    tokens); a resize outside it is listed, not judged. A resize costs the decode what it holds
    the submitting thread (its decision and plan: `ms`); its release, from install to released,
    slowed device work beside it by about a tenth on a GB10, so a release spanning many steps
    explains little of any one slow step among them and none is credited. Each step's excess
    is its wait less the step there, the median of the window's other waits (`step_at` says why
    that is enough); where resizes follow one another every step, the other steps carry them
    too. A resize lands within one step when its thread time is within the step.
    A step slower than the step there by more than a step, past the thread time of the resize
    decided in it, is a slow step
    of its own, reported as one with what the streamed steps that completed in it read (`served`,
    the worker's `expert step completed` events). Returns the gate's failures: a window that tested nothing, and
    each resize in it not within one step."""
    resizes = [event for event in released if "resize" in event]
    steps = pressure.get("steps", [])
    window = pressure.get("window") or {}
    at_token = dict((count, at) for count, at in steps)
    opened, closed = at_token.get(window.get("from")), at_token.get(window.get("to"))
    if opened is None or closed is None:
        print("the pressure's window did not close: NOTHING TESTED")
        return ["the pressure's window did not close: nothing tested"]
    # Each output's wait for a token since the one before.
    waits = [((at - before_at) / (count - before_count), before_at, at)
             for (before_count, before_at), (count, at) in zip(steps, steps[1:]) if count > before_count]
    inside = [event for event in resizes if opened <= event["decided"] <= closed]
    print(f"window: tokens {window['from']}–{window['to']} ({opened - start:.2f}–{closed - start:.2f} s) under the "
          f"held pressure; {len(inside)} of {len(resizes)} resizes in it")
    if not inside:
        print("NO RESIZE IN THE WINDOW: the cell tested no resize under pressure")
        return ["no resize in the pressure's window: nothing tested"]
    # Only the window's waits: before it the decode is unpressed, after it the memory comes back.
    waits = [wait for wait in waits if opened <= wait[1] and wait[2] <= closed]
    if len(waits) < 2:
        print("the window holds fewer than two steps: NOTHING TESTED")
        return ["the pressure's window holds fewer than two steps: nothing tested"]

    def step_at(index):
        """The step there: the median of the window's other waits. A verdict's tolerance is a whole
        step (a resize within one step, a step slow by more than one), so the step there need only
        be known within a step, and the step's drift across a window stays inside that: 21.9–33.6 ms
        on Metal, 23–51 ms on a GB10 as its bank fell to 46 slots, each within a step of its
        window's median. The median keeps a few slow steps out of it, and needs no count of
        neighbours."""
        return statistics.median(wait for other, (wait, _, _) in enumerate(waits) if other != index)

    ms = lambda seconds: f"{seconds * 1e3:.1f}"
    print("| s | resize | slots | thread ms | release ms | step ms | excess at decision "
          "| worst excess during release | within one step |")
    print("|---|---|---|---|---|---|---|---|---|")
    excess = lambda index: waits[index][0] - step_at(index)
    overlap = lambda index, since, until: max(0.0, min(waits[index][2], until) - max(waits[index][1], since))
    # What resizes cost each step: the thread time of those decided in it.
    cost = [0.0] * len(waits)
    for event in resizes:
        for index, wait in enumerate(waits):
            if wait[1] <= event["decided"] < wait[2]:
                cost[index] += event["ms"] / 1e3
    within = 0
    for event in inside:
        thread = event["ms"] / 1e3
        decided = next((index for index, wait in enumerate(waits) if wait[1] <= event["decided"] < wait[2]), None)
        during = [index for index in range(len(waits)) if overlap(index, event["installed"], event["released"]) > 0]
        anchor = decided if decided is not None else (during[0] if during else None)
        step = step_at(anchor) if anchor is not None else None
        at_decision = excess(decided) if decided is not None else None
        release = max((excess(index) for index in during), default=None)
        fits = step is not None and thread <= step
        within += fits
        print(f"| {event['decided'] - start:.2f} | {event['resize']} | {event['bank_slots']} | {event['ms']:.1f} "
              f"| {ms(event['released'] - event['installed'])} | {ms(step) if step else '-'} "
              f"| {ms(at_decision) if at_decision is not None else '-'} | {ms(release) if release is not None else '-'} "
              f"| {'yes' if fits else 'NO'} |")
    print(f"resizes in the window within one step, by thread time: {within} of {len(inside)}")
    slow = [index for index in range(len(waits)) if excess(index) - cost[index] > step_at(index)]
    for index in slow:
        wait, since, until = waits[index]
        # The streamed steps that completed while the decode waited: what they read.
        read = [step for step in served if since < step["at"] <= until]
        cause = (f"its steps missed {sum(step['misses'] for step in read)} experts, "
                 f"{sum(step['missed_bytes'] for step in read) / 1e6:.1f} MB read on demand, "
                 f"{sum(step['read_micros'] for step in read) / 1e3:.1f} ms reading "
                 f"({sum(step['parked_micros'] for step in read) / 1e3:.1f} ms parked)"
                 if read else "no streamed step completed in it")
        print(f"slow step at {since - start:.2f} s: {ms(wait)} ms against a {ms(step_at(index))} ms step, "
              f"{ms(excess(index) - cost[index])} ms beyond what resizes cost it ({ms(cost[index])} ms); {cause}")
    print(f"slow steps of their own in the window: {len(slow)}")
    return ([f"{len(inside) - within} of {len(inside)} resizes in the window not within one step"]
            if within < len(inside) else [])


def memory_samples(lines, name):
    """Each of the owner's memory readings of the store named `name` (the ledger's estimate): its
    time, what the load holds there (committed and cache), the ceiling its budget leaves that beside
    the guarantee the ledger keeps free (`Estimate::guarantee`), and the bank's derived targets, in
    slots, of its expert and host holdings there."""
    return [{"at": event["at"], "charge": event["committed"] + event["cache"],
             "ceiling": event["budget"] - event["guarantee"],
             # The machine's wired ceiling and wire count (`wired_limit`, `wire_count`; zero where
             # no wired ceiling bounds the store): none off Metal.
             **({"wired": (event["wired_limit"], event["wire_count"])} if event["wired_limit"] else {}),
             "targets": {"Experts": event["bank_target"], "Host": event["host_target"]}}
            for event in readings(lines).get(name, [])]


def within_ceiling(samples, growths, held_from, freeing):
    """The load under the ceilings it answers to. The machine's: the wire count of any reading
    (`wired`, the OS's bytes wired machine-wide, ours included) over the wired limit the runtime
    reads (`min(iogpu.wired_limit_mb, recommendedMaxWorkingSetSize)`) is a failure, held pressure
    or not. The occupier's: a reading over what it leaves is the host's move, which the load must
    answer before the pressure ends, and never one of its own growth steps. The failures, and how
    many readings under the held pressure were over the occupier's."""
    wired = [sample for sample in samples if sample.get("wired") and sample["wired"][1] > sample["wired"][0]]
    failures = ([f"the machine wires {worst['wired'][1]} bytes over its {worst['wired'][0]} wired ceiling at "
                 f"{len(wired)} of {len(samples)} readings, first at {wired[0]['at']}"
                 for worst in [max(wired, key=lambda sample: sample["wired"][1] - sample["wired"][0])]]
                if wired else [])
    held = [sample for sample in samples if held_from <= sample["at"] <= freeing]
    if not held:
        return [*failures, "unexercised pressure: no reading of the load's charge under the held pressure"], 0
    over = lambda sample: sample["charge"] > sample["ceiling"]
    if over(held[-1]):
        failures.append(f"the load charges {held[-1]['charge']} over the {held[-1]['ceiling']} the occupier leaves "
                        "as the pressure ends")
    for before, after in zip(held, held[1:]):
        if over(after) and not over(before) and any(before["at"] < event["installed"] <= after["at"] for event in growths):
            failures.append(f"a growth step took the load over the ceiling at {after['at']}")
    return failures, sum(map(over, held))


# The ledger's counts of readings whose accounting failed, cumulative per store: an error in a
# store with an exact OS count of what is out of RAM (macOS), where unobserved bytes are the
# ledger's own charge (`accounting_violations`); on a store under a wired ceiling (Metal) the
# readings left unobserved or short of the process's wired bytes (`unobserved_readings`,
# `unwired_readings`), none in a correct run.
COUNTERS = {"accounting_violations": False, "unobserved_readings": True, "unwired_readings": True}


def uncounted(records):
    """A counter of the ledger's accounting that is nonzero at a store's last reading has counted
    during the run (`COUNTERS`, cumulative from its first reading); the one-per-store failures name
    it. `records`: each store's readings, as `readings` gives them; a counter that applies only
    under a wired ceiling (`COUNTERS`, true) is judged on stores that read one."""
    failures = []
    for store, store_records in records.items():
        wired = any(record["wired_limit"] for record in store_records)
        for name, needs_ceiling in COUNTERS.items():
            first, last = store_records[0][name], store_records[-1][name]
            if last and (wired or not needs_ceiling):
                failures.append(f"{store}: {name} counted {last} by reading {store_records[-1].get('generation')} "
                                f"({first} at its first of {len(store_records)})")
    return failures


def others_after_shrinks(resizes, records, held_from, freeing):
    """A shrink frees the process's memory and moves no other program's, so the reading's `others`
    (what the store uses beyond its supply and the process's own bytes in use) must not rise across
    one by more than the bytes the shrink freed, the most a store's reclaim lag can show as others':
    the ledger counts the bytes freed at once, the device shows them later. Judged from the reading
    the shrink followed (its `reading`, the last before it was decided, where the occupier's press
    was already shown) to the readings until its release and the first after it, and only for a
    shrink wholly inside the held pressure (`held_from` to `freeing`), where no occupier is still
    taking memory. The log carries no reclaim lag figure of the store's, so the freed bytes are
    its bound. The failures."""
    failures = []
    for event in resizes:
        if event["resize"] != "Shrink" or not (held_from <= event["decided"] and event["released"] <= freeing):
            continue
        if not event.get("bytes") or event.get("store") is None or event.get("reading") is None:
            continue  # nothing freed, or untraced (`untraced` fails it)
        store = records.get(event["store"], [])
        base = next((record for record in store if record.get("generation") == event["reading"]), None)
        later = [record for record in store if event["decided"] < record["at"] <= event["released"]]
        later += [record for record in store if record["at"] > event["released"]][:1]
        what = f"the shrink of {event['holding']} {event['from']}→{event['to']} at {event['decided']}"
        if base is None or not later:
            failures.append(f"{what}: others unverified, no reading " + ("it followed" if base is None else "after it"))
            continue
        rise = max(record["others"] for record in later) - base["others"]
        if rise > event["bytes"]:
            failures.append(f"{what}: others rose {rise} bytes over reading {event['reading']}, "
                            f"more than the {event['bytes']} it freed")
    return failures


def costlier(events, samples, selected, name):
    """A real shortage of the pressed store (named `name`) is answered in order (spec §4.5): the
    cache yields first, and new work waits, sequences are preempted or the model unloads
    (`COSTLIER`) only once the bank sits at its floor. So a costlier response is judged by the
    reading that answered it (the store's first at or after it, `samples`): each of the `selected`
    holdings' derived targets there at its floor. A shortage of another store is that store's,
    and one whose record lacks its response leaves the order unverified. The failures."""
    failures = []
    for event in events:
        if "shortage" not in event or event["store"] != name:
            continue
        if any(event.get(key) is None for key in COSTLIER):
            failures.append(f"release ordering unverified: the shortage at {event['at']} records no "
                            + ", ".join(key for key in COSTLIER if event.get(key) is None))
            continue
        taken = [key for key in COSTLIER if event[key]]
        if not taken:
            continue
        reading = min((sample for sample in samples if sample["at"] >= event["at"]),
                      key=lambda sample: sample["at"], default=None)
        if reading is None:
            failures.append(f"release ordering unverified: no reading of {name} answered the shortage at {event['at']}")
            continue
        # A holding with no derived target (a table) holds its floor always.
        above = [f"{holding} at {reading['targets'][holding]} slots over its floor of {value['floor']}"
                 for holding, value in selected.items()
                 if reading["targets"].get(holding, value["floor"]) > value["floor"]]
        if above:
            failures.append(f"costlier response ({', '.join(taken)}) at {event['at']} before the pressed store's "
                            "holdings reached their floor: " + "; ".join(above))
    return failures


def pressure_transitions(charge, events, samples):
    """A physical cell exercises one holding's shrink under the ceiling the occupier leaves
    (`within_ceiling`), answers the press costlier than the cache only at its floor (`costlier`),
    then regrows every shrunk holding to all it held, or to the bank's derived target once the press
    is back where that is less."""
    pressure = charge["pressure"]
    required = ("started", "approached", "occupied", "freeing", "released", "freed", "window")
    missing = [name for name in required if name not in pressure]
    if missing:
        return ["unexercised pressure window: missing " + ", ".join(missing)]
    window = pressure["window"]
    if not (isinstance(window.get("from"), int) and isinstance(window.get("to"), int)
            and window["from"] < window["to"]):
        return ["unexercised pressure: incomplete held window"]
    observed = [pressure["approached"].get("before"), pressure["occupied"].get("observation"),
                pressure["released"].get("observation")]
    if any(not value for value in observed):
        return ["pressure lacks before/held/released physical-store observations"]
    store = observed[0]["store"]
    if any(value["store"] != store for value in observed):
        return ["pressure observations name different physical stores"]
    bank = charge.get("bank")
    if not bank:
        return ["unexercised pressure: load has no elastic bank"]
    holdings = bank["holdings"]
    selected = {name: value for name, value in holdings.items() if value["stores"] == [store]}
    if not selected:
        return ["unexercised pressure: no holding exclusively occupies the selected store"]
    # The child records physical release; the parent's `freed` is when it joined the child.
    start = max(pressure["started"]["at"], pressure["approached"]["at"])
    freed = pressure["released"]["at"]
    resizes = [event for event in events if event.get("holding") in selected]
    if any(not all(isinstance(event.get(key), (int, float))
                   for key in ("decided", "installed", "released")) for event in resizes):
        return ["physical transitions unverified: missing resize lifecycle timestamps"]
    shrunk = {event["holding"] for event in resizes if event["resize"] == "Shrink"
              and start <= event["released"] <= freed and event["to"] < event["from"] and event.get("bytes", 0) > 0}
    failures = []
    if not shrunk:
        failures.append("unexercised pressure: selected store's holdings never shrank")
    growths = [event for event in resizes if event["resize"] == "Growth"]
    held, over = within_ceiling(samples, growths, pressure["occupied"]["at"], pressure["freeing"]["at"])
    failures += held
    print(f"readings over the ceiling under the held pressure, each answered by a release: {over}")
    failures += costlier(events, samples, selected, observed[0]["name"])
    current = {name: value["initial"] for name, value in selected.items()}
    # Replay backing changes at their lifecycle boundary, not delayed log delivery.
    for event in sorted(resizes, key=lambda event: event["released"] if event["resize"] == "Shrink"
                        else event["installed"]):
        current[event["holding"]] = event["to"]
    # All it held, or less where the bank's derived target once the press is back (the ledger's
    # room, up to where the holding's misses stop falling) says so: the target while the decode
    # follows it, from the press's release to the occupier's exit (`freed`), which comes after the
    # decode, when no step is left to grow in.
    exited = pressure["freed"]["at"]
    back = [sample for sample in samples if freed <= sample["at"] < exited]
    targets = back[-1]["targets"] if back else {}
    wanted = {name: min(selected[name]["initial"], targets.get(name, selected[name]["initial"])) for name in shrunk}
    failures += [f"{name} did not regrow after the pressure: {current[name]} of the {wanted[name]} its target holds"
                 for name in shrunk if current[name] < wanted[name]]
    return failures


def report_pressure(charge, lines, resident_tokens, scratch):
    """The forced-pressure gate: the pressed store's holdings shrink, the load keeps under what the
    occupier leaves, and regrows once the memory is free (`pressure_transitions`); steady decode
    makes no resize (`steady_resizes`) and every resize traces to its cause (`untraced`); every
    shrink returns memory to the OS and needs none (what the process holds never rises across
    one); and the decode's tokens equal the resident decode's. Returns the gate's failures, empty
    when it passes, with the window's (`report_stalls`)."""
    events = releases(lines)
    pressure = charge["pressure"]
    tokens = charge["decoded_tokens"]
    samples = memory_samples(lines, pressure.get("approached", {}).get("before", {}).get("name"))
    failures = pressure_transitions(charge, events, samples)
    resizes = [event for event in events if "resize" in event]
    failures += steady_resizes(pressure, resizes)
    records = readings(lines)
    failures += untraced(resizes, records)
    failures += uncounted(records)
    held = (pressure.get("occupied", {}).get("at"), pressure.get("freeing", {}).get("at"))
    if None not in held:  # a pressure without them fails in `pressure_transitions`
        failures += others_after_shrinks(resizes, records, *held)
    # A shrink frees memory to others: its empty blocks return to the OS (`returned`), and what the
    # process holds never rises across it. One that returns nothing freed no one anything.
    bank = [event for event in resizes if event["resize"] == "Shrink"]
    kept = [event for event in bank if not event.get("returned")]
    if kept:
        failures.append(f"{len(kept)} of {len(bank)} bank shrinks returned nothing to the OS")
    grew = [event for event in bank if event["held_after"] > event["held_before"]]
    if grew:
        failures.append(f"{len(grew)} of {len(bank)} bank shrinks grew what the process holds")
    if tokens is None or resident_tokens is None:
        failures.append("no exact-output comparison: pressured or reference decode did not complete")
    elif tokens != resident_tokens:
        failures.append("pressured decode differs from unpressured reference")
    if not failures:
        failures += report_stalls(pressure, resizes, pressure["started"]["at"], served_steps(lines))
    report = {"pressure": pressure, "bank": charge.get("bank"), "events": events, "samples": samples, "failures": failures,
              "tokens": tokens, "reference_tokens": resident_tokens}
    (scratch / PRESSURE).write_text(json.dumps(report, indent=1))
    print(f"physical pressure: {len(events)} owner events; tokens "
          f"{'identical' if tokens is not None and tokens == resident_tokens else 'unverified/different'}")
    for moment, value in (("before", pressure.get("approached", {}).get("before")),
                          ("held", pressure.get("occupied", {}).get("observation")),
                          ("released", pressure.get("released", {}).get("observation"))):
        if value:
            print(f"{moment}: store={value['store']}, headroom={value['headroom_bytes']}")
    for event in events:
        if event.get("holding"):
            print(f"{event['resize']}: {event['holding']} {event['from']}→{event['to']}, "
                  f"backing bytes={event['bytes']}, returned={event.get('returned', '-')}, "
                  f"cause {event.get('term')} at reading {event.get('reading')}")
    for failure in failures:
        print(f"FAILED: {failure}")
    print(f"results in {scratch / PRESSURE}")
    return failures


def parser() -> argparse.ArgumentParser:
    """The measurement's options; `refusal` judges them together."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", nargs="+", metavar="HOST-COSTS.json", help="fit the costs to these loads")
    parser.add_argument("--bin")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--memory", choices=tuple(RESULTS),
                        help="whose memory the device's is: its own (dedicated) or the host's (unified), as "
                             "the engine's topology states it (`memory`: Own or Host)")
    parser.add_argument("--scratch")
    parser.add_argument("--port", type=int, default=18977)
    parser.add_argument("--artifacts", help="a catalog acquisition object (JSON) to measure")
    parser.add_argument("--weight-cap", type=int, metavar="BYTES",
                        help="load under this weight cap; dedicated memory requires --pressure-store")
    parser.add_argument("--kernels", metavar="DIR",
                        help="the kernel cache every load prepares from, as a served load's (its executor's "
                             "costs price the bank's growth)")
    parser.add_argument("--context", type=int, metavar="TOKENS",
                        help="load at this context; dedicated memory requires --pressure-store")
    parser.add_argument("--residency", nargs=2, metavar=("TEMPLATE", "TOKENS"),
                        help="under --weight-cap: load at the context of the template file TEMPLATE plus TOKENS, "
                             "decode TOKENS tokens after its lane 0, and read the pages of the expert stacks the "
                             "file cache holds once loaded and once decoded")
    parser.add_argument("--pressure-store", choices=("host", "device"),
                        help="under --residency: another process takes what this physical store gives past the "
                             "guarantee the load's ledger keeps free there, from a quarter of the decode to three "
                             "quarters of it; report the owner's resizes and releases (the bank's slots, each "
                             "step's time, what the process held on each side) and the decode's tokens against a "
                             "resident load's (the forced-pressure gate)")
    parser.add_argument("--pressure-prompt", action="store_true",
                        help="under --pressure-store: the pressure starts once the request is admitted, so its "
                             "prompt's groups run under it")
    parser.add_argument("models", nargs="*")
    return parser


def refusal(options) -> str | None:
    """Why these options (`parser`'s) do not name one measurement, or None: the one judge of
    them, which `mrig memory` also applies to the command it plans."""
    if options.pressure_store and not options.residency:
        return "--pressure-store applies while a streamed load decodes: it takes --residency"
    if options.pressure_prompt and not options.pressure_store:
        return "--pressure-prompt says when the pressure starts: it takes --pressure-store"
    if options.fit:
        return None
    if not (options.bin and options.backend and options.memory and options.scratch):
        return "a measurement takes --bin, --backend, --memory and --scratch"
    if (options.weight_cap or options.context) and options.memory == "dedicated" and not options.pressure_store:
        return "--weight-cap and --context measure a unified-memory load"
    if options.residency and not options.weight_cap:
        return "--residency reads a streamed load's experts: it takes --weight-cap"
    if options.residency and options.context:
        return "--residency loads at its template's context plus the tokens it decodes: it takes no --context"
    if not (options.models or options.artifacts):
        return "give models or --artifacts"
    return None


def main():
    command = parser()
    options = command.parse_args()
    if (refused := refusal(options)) is not None:
        command.error(refused)
    if options.fit:
        fit_loads([load for path in options.fit for load in json.loads(Path(path).read_text())])
        return
    models = [(None, paths, [paths.split(",")[0]]) for paths in options.models]
    if options.artifacts:
        acquired = json.loads(Path(options.artifacts).read_text())
        target_files = acquired.get("target_files") or [acquired["target"]]
        models.append((options.artifacts, acquired["target"], target_files))
    scratch = Path(options.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    (device_costs if options.memory == "dedicated" and not options.pressure_store else host_costs)(options, models, scratch)


def host_costs(options, models, scratch):
    """Each model's target loaded plain: its own memory at load end against its charge. A load, a
    or a resident decode that failed fails the measurement once every model is reported:
    what it would have tested is untested. So does every check that failed: expert pages in the
    file cache, and the forced-pressure gate (`report_pressure`)."""
    results, failed, failures = [], [], []
    for index, (_artifacts, _paths, target_files) in enumerate(models):
        target = target_files[0]
        name = Path(target).stem
        cap = ("--weight-cap", str(options.weight_cap)) if options.weight_cap else ()
        context = ("--context", str(options.context)) if options.context else ()
        log = scratch / f"{index}-{name}.log"
        decode = ()
        if options.residency:
            template, tokens = options.residency
            # The harness renders the template's lane 0 and loads at its context plus the tokens.
            decode = ("--template", template, "--decode-tokens", tokens)
        # The pressured decode is held to the resident one's tokens, so it approves no CPU site: a
        # site's admitted arithmetic differs from the device's within the precision system, and a
        # shrink moves misses to it. The site's arithmetic is judged by its own instrument (kl).
        unsited = scratch / "unsited.json"
        if options.pressure_store:
            unsited.write_text(json.dumps({"cpu_sites": False}))
        pressure = (("--pressure-store", options.pressure_store,
                     *(("--pressure-prompt",) if options.pressure_prompt else ()), "--numerical-profile", str(unsited))
                    if options.pressure_store else ())
        kernels = ("--cache-dir", options.kernels) if options.kernels else ()
        load = subprocess.run([exe(options.bin, "magnitude-harness"), "memory", "--device", options.backend,
                               *cap, *context, *decode, *pressure, *kernels, *target_files],
                              capture_output=True, text=True, check=False)  # a failure is reported below
        log.write_text(load.stderr)
        found = moments((None, line) for line in load.stderr.splitlines())
        if load.returncode != 0 or "loaded" not in found:
            failure = next((line for line in reversed(load.stderr.splitlines())
                            if line.startswith("magnitude-harness: ") and "event {" not in line), "")
            print(f"{name}: the load failed (exit {load.returncode}): {failure}; see {log}", file=sys.stderr)
            failed.append(name)
            continue
        charge = json.loads(load.stdout.splitlines()[-1])
        loaded = found["loaded"][1]
        if options.memory == "unified":
            footprint = loaded["footprint_bytes"]
            storage = loaded["device_allocated_bytes"]
            # What the load holds at load end of what its plan prices: its storage (every byte the ledger
            # counts the device allocating in unified memory, device and upload, block slack and driver
            # objects among them), and its exact host terms.
            own = footprint - storage - charge["plan_host_bytes"]
            charged = charge["charged_runtime_bytes"] + charge["charged_vocabulary_bytes"] + charge["charged_program_bytes"]
            results.append({**charge, "host": platform.node(), "os": platform.system(),
                            "loaded_footprint_bytes": footprint,
                            # Storage as the load held it, and the rest of the footprint beside it.
                            "loaded_storage_bytes": storage,
                            "loaded_malloc_bytes": loaded["malloc_live_bytes"],
                            # Seismic's charge at load end by the plan term that charges it.
                            "loaded_storage_terms": loaded["charge"],
                            "own_bytes": own, "charged_bytes": charged, "overcharge_bytes": charged - own})
            (scratch / RESULTS["unified"]).write_text(json.dumps(results, indent=1))
        if options.pressure_store and charge["decoded_tokens"] is None:
            # The decode ended under pressure (the model unloaded): its resizes and releases are reported.
            failures += [f"{name}: {failure}" for failure in report_pressure(charge, load.stderr.splitlines(), None, scratch)]
        elif options.pressure_store:
            # The same decode resident, after the streamed load, so its cached reads never serve the
            # streamed one: from the prompt the pressured decode was sized to, as many tokens as it
            # decoded, the tokens it must equal.
            same = ("--template", decode[1], "--decode-tokens", str(len(charge["decoded_tokens"])),
                    "--decode-prompt-tokens", str(charge["decode_prompt_tokens"]))
            resident = subprocess.run([exe(options.bin, "magnitude-harness"), "memory", "--device", options.backend,
                                       *same, *kernels, *target_files],
                                      capture_output=True, text=True, check=False)  # a failure is reported below
            (scratch / f"{index}-{name}-resident.log").write_text(resident.stderr)
            if resident.returncode != 0:
                print(f"{name}: the resident decode failed (exit {resident.returncode})", file=sys.stderr)
                failed.append(name)
                continue
            resident_tokens = json.loads(resident.stdout.splitlines()[-1])["decoded_tokens"]
            failures += [f"{name}: {failure}"
                         for failure in report_pressure(charge, load.stderr.splitlines(), resident_tokens, scratch)]
    if options.memory == "dedicated":
        (scratch / RESULTS["dedicated"]).write_text(json.dumps({"pressure_report": PRESSURE, "failures": failures, "untested": failed}, indent=1))
        if failed or failures:
            sys.exit("; ".join([*failed, *failures]))
        return
    mb = lambda value: f"{value / 1e6:9.1f}"
    print("| model | weight cap | vocabulary | kernels | footprint at load | storage held / planned "
          "| rest / planned host terms | own | charged | overcharge |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        rest = r["loaded_footprint_bytes"] - r["loaded_storage_bytes"]
        print(f"| {Path(r['model']).name} | {r['weight_cap'] or '-'} | {r['vocabulary']} | {r['kernels']} "
              f"| {mb(r['loaded_footprint_bytes'])} | {mb(r['loaded_storage_bytes'])} / {mb(r['plan_storage_bytes'])} "
              f"| {mb(rest)} / {mb(r['plan_host_bytes'])} | {mb(r['own_bytes'])} | {mb(r['charged_bytes'])} "
              f"| {mb(r['overcharge_bytes'])} |")
    if results:
        print("\nMB. own: the footprint at load end less the storage it holds and the plan's host terms; "
              "charged: its host process costs; a negative overcharge is memory the plan does not charge")
    for r in results:
        planned, held = r["plan_storage_terms"], r["loaded_storage_terms"]
        terms = [*planned, *(term for term in held if term not in planned and term != "allocated")]
        print(f"\n{Path(r['model']).name}, weight cap {r['weight_cap'] or '-'}, context {r['context']}: "
              "storage by term at load end (the recorded graphs and the device process's bytes are charged "
              "to the entries whose allocations made them, so they read in the total alone)")
        print("| term | planned MB | charged MB | charged − planned (bytes) |")
        print("|---|---|---|---|")
        for term in terms:
            plan, charge = planned.get(term), held.get(term)
            shown = lambda recorded: "unrecorded" if recorded is None else mb(recorded)
            difference = "-" if plan is None or charge is None else charge - plan
            print(f"| {term} | {shown(plan)} | {shown(charge)} | {difference} |")
        print(f"| total charged | {mb(r['plan_storage_bytes'])} | {mb(held['allocated'])} "
              f"| {held['allocated'] - r['plan_storage_bytes']} |")
        residency = r["expert_residency"]
        if residency:
            print(f"expert stacks' pages in the file cache, of {residency['pages']}: "
                  f"{residency['loaded']} once loaded, {residency['decoded']} after decoding "
                  f"{residency['decode_tokens']} tokens (it must hold none)")
            if residency["loaded"] or residency["decoded"]:
                failures.append(f"{Path(r['model']).name}: the file cache holds expert pages "
                                f"({residency['loaded']} loaded, {residency['decoded']} decoded)")
    if results:
        print(f"\nresults in {scratch / RESULTS['unified']}; fit across loads with --fit")
    for failure in failures:
        print(f"FAILED: {failure}")
    if failed or failures:
        sys.exit("; ".join([*([f"untested: {', '.join(failed)}"] if failed else []),
                            *([f"{len(failures)} checks failed"] if failures else [])]))


def solve(rows):
    """Three linear equations (rows of coefficients and right side), solved exactly; None
    when singular."""
    rows = [[Fraction(value) for value in row] for row in rows]
    for column in range(3):
        pivot = next((r for r in range(column, 3) if rows[r][column] != 0), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(3):
            if r != column and rows[r][column] != 0:
                factor = rows[r][column] / rows[column][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[column])]
    return [rows[r][3] / rows[r][r] for r in range(3)]


def fit(loads):
    """The costs (runtime, per vocabulary entry, per kernel) that cover every load's own memory
    with the least summed overcharge: the best feasible vertex of the linear program."""
    counts = lambda load: (1, load["vocabulary"], load["kernels"])
    # Constraints a·x >= b: every load's charge covers its own memory, every cost is >= 0.
    constraints = [(counts(load), load["own_bytes"]) for load in loads]
    constraints += [((1, 0, 0), 0), ((0, 1, 0), 0), ((0, 0, 1), 0)]
    best = None
    for chosen in itertools.combinations(constraints, 3):
        costs = solve([[*a, b] for a, b in chosen])
        if costs is None or any(sum(c * x for c, x in zip(a, costs)) < b for a, b in constraints):
            continue
        over = sum(sum(c * x for c, x in zip(counts(load), costs)) - load["own_bytes"] for load in loads)
        if best is None or over < best[0]:
            best = (over, costs)
    return best[1]


def fit_loads(loads):
    for backend in sorted({load["backend"] for load in loads}):
        measured = [load for load in loads if load["backend"] == backend]
        runtime, per_token, per_kernel = fit(measured)
        print(json.dumps({"backend": backend, "loads": len(measured), "runtime": round(runtime),
                          "per_token": round(per_token), "per_kernel": round(per_kernel)}))
        for load in measured:
            fitted = runtime + per_token * load["vocabulary"] + per_kernel * load["kernels"]
            print(json.dumps({"model": Path(load["model"]).name, "host": load["host"], "own_bytes": load["own_bytes"],
                              "vocabulary": load["vocabulary"], "kernels": load["kernels"],
                              "overcharge_measured_bytes": load["overcharge_bytes"],
                              "overcharge_fitted_bytes": round(fitted - load["own_bytes"])}))


def device_costs(options, models, scratch):
    """Each model's load on a dedicated GPU: what the device lost, split by term."""
    cache = str(scratch / "kernel-cache")
    results = []
    for index, (artifacts_path, paths, target_files) in enumerate(models):
        name = Path(target_files[0]).stem
        model_args = ("--artifacts", artifacts_path) if artifacts_path else (paths,)
        assess = subprocess.run([exe(options.bin, "magnitude-assess"), "--device", options.backend,
                                 "--kernels", cache, "--breakdown", *model_args],
                                capture_output=True, text=True, check=False)  # a failure is reported below
        breakdown = [json.loads(l.split(": memory ", 1)[1]) for l in assess.stderr.splitlines() if ": memory {" in l]
        verdict = json.loads(assess.stdout.splitlines()[0]) if assess.stdout.strip() else {}
        if not breakdown:
            print(f"{name}: assessment failed:\n{assess.stderr}", file=sys.stderr)
            continue
        breakdown = breakdown[-1]
        # The first run prepares the kernel cache (its search runs in a child
        # process); the measured run then only loads.
        sampler = Sampler()
        idle = sampler.read()
        run_engine(options.bin, options.backend, paths, cache, scratch / f"{index}-{name}-prepare.log", options.port)
        # A driver returns an exited process's memory lazily: measure from
        # the idle device, not from what the preparing run left behind.
        settled = time.time() + 180
        while sampler.read() > idle + (64 << 20) and time.time() < settled:
            time.sleep(1)
        baseline = sampler.read()
        if baseline > idle + (64 << 20):
            print(f"{name}: the device did not return to idle ({baseline - idle} bytes above it)", file=sys.stderr)
        sampler.start()
        lines = run_engine(options.bin, options.backend, paths, cache, scratch / f"{index}-{name}.log", options.port)
        sampler.stop()
        found = moments(lines)
        if "loaded" not in found:
            print(f"{name}: the engine did not load; see {scratch}/{index}-{name}.log", file=sys.stderr)
            continue
        def lost(moment, sampler=sampler, found=found, baseline=baseline):
            return sampler.at(found[moment][0]) - baseline

        def allocated(moment, found=found):
            return found[moment][1]["device_allocated_bytes"]

        context = lost("device opened")
        programs = lost("programs prepared") - context - allocated("programs prepared")
        loaded = lost("loaded")
        residual = loaded - context - programs - allocated("loaded")
        peak = max(value for _, value in sampler.samples) - baseline
        kernels = breakdown["kernels"]
        planned = breakdown["serving_bytes"] - breakdown["state_bytes"] + breakdown["startup_state_bytes"]
        result = {
            "model": paths, "backend": options.backend, "host": platform.node(), "os": platform.system(),
            "kernels": kernels, "idle_used_bytes": idle, "baseline_used_bytes": baseline,
            "context_bytes": context, "program_bytes": programs,
            "program_bytes_per_kernel": programs // max(kernels, 1),
            "loaded_lost_bytes": loaded, "peak_lost_bytes": peak,
            "allocated_bytes": allocated("loaded"),
            "residual_bytes": residual, "planned_at_load_bytes": planned,
            "required_bytes": [domain["requiredBytes"] for domain in verdict.get("memory", [])],
            "breakdown": breakdown,
        }
        results.append(result)
        (scratch / RESULTS["dedicated"]).write_text(json.dumps(results, indent=1))
    gb = lambda value: f"{value / 1e9:7.3f}" if value is not None else "      -"
    print("| model | kernels | context | programs | per kernel | allocated | residual | lost at load | peak | planned at load |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {Path(r['model'].split(',')[0]).name} | {r['kernels']} | {gb(r['context_bytes'])} | {gb(r['program_bytes'])} "
              f"| {r['program_bytes_per_kernel'] / 1e6:.1f} MB | {gb(r['allocated_bytes'])} | {gb(r['residual_bytes'])} "
              f"| {gb(r['loaded_lost_bytes'])} | {gb(r['peak_lost_bytes'])} | {gb(r['planned_at_load_bytes'])} |")
    if results:
        print(f"\nper-kernel cost bound: {max(r['program_bytes_per_kernel'] for r in results) / 1e6:.1f} MB; "
              f"context: {max(r['context_bytes'] for r in results) / 1e6:.0f} MB; results in {scratch / RESULTS['dedicated']}")


if __name__ == "__main__":
    main()
