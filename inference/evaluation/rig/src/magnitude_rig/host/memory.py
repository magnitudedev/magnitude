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
  the engine holds there (`device_held_bytes`), per native kernel of the
  load's kernel inventory (`magnitude-assess --breakdown`, `kernels`);
- residual: what it lost at load end beyond context, programs and held.

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
With `--residency` the load also reads how many pages of its expert stacks
the file cache holds once loaded and once it has decoded a fixture (it must hold none);
with `--pressure BYTES` too, another process presses the load BYTES below its
planning line over the middle half of that decode, and the owner's releases are reported in order
with the decode's tokens against the same decode on a resident load.
With `--recommit GB[,GB...]` nothing loads: it times an in-place recommit of
a reserved tensor (an elastic expert bank's resize) at each commitment.

  python3 memory.py --bin DIR --backend cuda|vulkan|metal|cpu --memory dedicated|unified \\
      --scratch DIR [--weight-cap BYTES] [--context TOKENS] \\
      [--residency HARNESS FIXTURES TOKENS [--pressure BYTES]] \\
      [--artifacts ACQUISITION.json] [TARGET.gguf[,PROJECTOR.gguf[,DRAFT.gguf]] ...]
  python3 memory.py --bin DIR --backend BACKEND --memory unified --scratch DIR --recommit GB[,GB...]

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
RECOMMIT = "recommit.json"  # `--recommit`'s results
PRESSURE = "pressure.json"  # `--pressure`'s releases, pressure and tokens
# The plan's storage term that bounds what the backend holds beside its allocations (its blocks' free
# space, its recorded graphs), where every other term is exact.
BOUND = "beyond_allocations"


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
SHRINK_STEPS = ("queued", "planned", "moves")  # a shrink's steps on the submitting thread


def releases(lines):
    """The owner's releases (`memory released`), the bank's growth steps (`memory grown`, rung
    `Growth`) and memory conditions (`memory condition changed`), each with its time in Unix seconds,
    in order: the scheduler's events the worker prints, one JSON object each, an unread value
    absent."""
    found = []
    for line in lines:
        if not line.startswith(EVENT):
            continue
        event = json.loads(line[len(EVENT):])
        if event["message"] in ("memory released", "memory grown"):
            rung = event["rung"] if event["message"] == "memory released" else "Growth"
            found.append({"at": event["at"], "rung": rung, "ms": event["micros"] / 1e3,
                          "bytes": event.get("grown" if rung == "Growth" else "released"),
                          **{key: event.get(key) for key in ("bank_slots", "charged_before", "charged_after",
                                                              "footprint_before", "footprint_after", "holding", "from", "to", "cause",
                                                              "reclaim_device_deficit_bytes", "reclaim_host_deficit_bytes")}})
            if rung == "Bank":
                # A shrink's steps on the submitting thread (ms); a growth takes none of them.
                found[-1].update({key: event[f"{key}_micros"] / 1e3 for key in SHRINK_STEPS})
            if rung in ("Bank", "Growth"):
                # When a resize's other steps happened.
                found[-1].update({key: event["at"] - event[f"{key}_before"] / 1e6
                                  for key in ("decided", "prepared", "installed", "released")})
        elif event["message"] == "the expert bank's storage refused a resize":
            # This event's `at` is a holding size, not the event timestamp.
            # Retain the source reason; a partial refusal makes backing proof
            # incomplete even when logical slots later reach their floor.
            found.append({"refusal": event})
        elif event["message"] == "memory condition changed":
            found.append({"at": event["at"], "condition": event["to"]})
        elif event["message"] == "expert bank growth decision":
            # What a growth with room enough reaches: where the load's latest miss record says
            # the holding's misses stop falling, at most its ceiling.
            found.append({"at": event["at"], "decision": event["holding"], "justified": event["justified"]})
    return found


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


def report_stalls(pressure, released, start, release_step_ratio, served):
    """What each resize in the pressure's window costs the decode, apart from steps that are slow on
    their own. The window is the decode's span under the held pressure (the harness's `window`
    tokens); a resize outside it is listed, not judged. A resize costs the decode what it holds
    the submitting thread (its decision, plan and moves: `ms`) and what its device work beside the
    decode slows the steps it overlaps (its release, from install to released), at most
    `release_step_ratio` less one of the time it overlaps each: the executor's costs measured that
    ratio (device work beside a release over it alone), and none read, a release explains nothing.
    On a GB10 a release slowed device work beside it by about a tenth, so a release spanning many
    steps explains little of any one slow step among them. Each step's excess
    is its wait less the step there, the median of the window's other waits (`step_at` says why
    that is enough); where resizes follow one another every step, the other steps carry them
    too. A resize lands within one step when its
    thread time and the worst excess of the steps its release overlaps are each within the step.
    A step slower than the step there by more than a step, which no resize's thread or release
    touches, or slower than the thread time of the resize decided in it explains, is a slow step
    of its own, reported as one with what the streamed steps that completed in it read (`served`,
    the worker's `expert step completed` events). The instrument's own pause, giving its memory back from inside the
    decode's output, is no step's. Returns the gate's failures: a window that tested nothing, and
    each resize in it not within one step."""
    resizes = [event for event in released if event["rung"] in ("Bank", "Growth")]
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
    if "freeing" in pressure:
        pause = (pressure["freeing"]["at"], pressure["freed"]["at"])
        waits = [wait for wait in waits if wait[2] <= pause[0] or wait[1] >= pause[1]]
    inside = [event for event in resizes if opened <= event["decided"] <= closed]
    print(f"window: tokens {window['from']}–{window['to']} ({opened - start:.2f}–{closed - start:.2f} s) under the "
          f"held pressure; {len(inside)} of {len(resizes)} resizes in it")
    if not inside:
        print("NO RESIZE IN THE WINDOW: the cell tested no resize under pressure")
        return ["no resize in the pressure's window: nothing tested"]
    # Only the window's waits: before it the decode is unpressed, after it the instrument drains the
    # outputs it queued while giving its memory back.
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
    print("| s | resize | slots | thread ms | moves ms | release ms | step ms | excess at decision "
          "| worst excess during release | within one step |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    excess = lambda index: waits[index][0] - step_at(index)
    overlap = lambda index, since, until: max(0.0, min(waits[index][2], until) - max(waits[index][1], since))
    # What resizes can cost each step: the thread time of those decided in it, and what each release
    # running during it slows it by, at most the executor's measured ratio less one of the time they
    # overlap (`release_step_ratio`, the move point beside a release over it alone). Unread, a
    # release explains nothing.
    slowing = max(0.0, release_step_ratio - 1.0) if release_step_ratio is not None else 0.0
    if release_step_ratio is None:
        print("a release's cost to a step beside it is unread (the executor's costs): it explains no step")
    slowed = lambda index, event: slowing * overlap(index, event["installed"], event["released"])
    cost = [0.0] * len(waits)
    for event in resizes:
        for index, wait in enumerate(waits):
            if wait[1] <= event["decided"] < wait[2]:
                cost[index] += event["ms"] / 1e3
            cost[index] += slowed(index, event)
    within = 0
    for event in inside:
        thread = event["ms"] / 1e3
        decided = next((index for index, wait in enumerate(waits) if wait[1] <= event["decided"] < wait[2]), None)
        during = [index for index in range(len(waits)) if overlap(index, event["installed"], event["released"]) > 0]
        anchor = decided if decided is not None else (during[0] if during else None)
        step = step_at(anchor) if anchor is not None else None
        at_decision = excess(decided) if decided is not None else None
        # Each step its release overlaps is slower by at most what the release slows it by.
        release = max((min(excess(index), slowed(index, event))
                       for index in during), default=None)
        fits = step is not None and thread <= step and (release is None or release <= step)
        within += fits
        moves = f"{event['moves']:.1f}" if event["rung"] == "Bank" else "-"
        print(f"| {event['decided'] - start:.2f} | {event['rung']} | {event['bank_slots']} | {event['ms']:.1f} "
              f"| {moves} | {ms(event['released'] - event['installed'])} | {ms(step) if step else '-'} "
              f"| {ms(at_decision) if at_decision is not None else '-'} | {ms(release) if release is not None else '-'} "
              f"| {'yes' if fits else 'NO'} |")
    print(f"resizes in the window within one step, by thread time and release: {within} of {len(inside)}")
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
    """Each of the owner's memory readings of the store named `name`: its time, what the load charges
    it, and the ceiling everything else on the host leaves that charge above the planning line."""
    events = (json.loads(line[len(EVENT):]) for line in lines if line.startswith(EVENT))
    return [{"at": event["at"], "charge": event["charged"],
             "ceiling": event["charged"] + event["headroom"] - event["planning"]}
            for event in events if event["message"] == "memory read" and event["store"] == name]


def within_ceiling(samples, growths, held_from, freeing):
    """The load under what the occupier leaves: a reading over the ceiling is the host's move, which
    the load must answer before the pressure ends, and never one of its own growth steps. The failures,
    and how many readings under the held pressure were over."""
    held = [sample for sample in samples if held_from <= sample["at"] <= freeing]
    if not held:
        return ["unexercised pressure: no reading of the load's charge under the held pressure"], 0
    over = lambda sample: sample["charge"] > sample["ceiling"]
    failures = []
    if over(held[-1]):
        failures.append(f"the load charges {held[-1]['charge']} over the {held[-1]['ceiling']} the occupier leaves "
                        "as the pressure ends")
    for before, after in zip(held, held[1:]):
        if over(after) and not over(before) and any(before["at"] < event["installed"] <= after["at"] for event in growths):
            failures.append(f"a growth step took the load over the ceiling at {after['at']}")
    return failures, sum(map(over, held))


def pressure_transitions(charge, events, samples):
    """A physical cell exercises one holding's shrink under the ceiling the occupier leaves
    (`within_ceiling`), then regrows every shrunk holding to all it held."""
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
    # A growth the store refuses leaves the holding as it was, and memory freeing piecemeal
    # refuses one; a refused shrink is what leaves its backing unknown.
    refused = [event["refusal"] for event in events
               if event.get("refusal", {}).get("holding") in selected and event["refusal"].get("cause") != "Growth"]
    if refused:
        return [f"physical backing unverified after resize refusal: {event}" for event in refused]
    # The child records physical release; parent `freed` includes shutdown/join
    # and remains the stall instrument's exclusion boundary only.
    start = max(pressure["started"]["at"], pressure["approached"]["at"])
    freed = pressure["released"]["at"]
    resizes = [event for event in events if event.get("holding") in selected]
    if any(not all(isinstance(event.get(key), (int, float))
                   for key in ("decided", "installed", "released")) for event in resizes):
        return ["physical transitions unverified: missing resize lifecycle timestamps"]
    shrunk = {event["holding"] for event in events if event.get("rung") == "Bank"
              and event.get("holding") in selected and start <= event["released"] <= freed
              and event["to"] < event["from"] and event.get("bytes", 0) > 0}
    failures = []
    if not shrunk:
        failures.append("unexercised pressure: selected store's holdings never shrank")
    growths = [event for event in resizes if event.get("rung") == "Growth"]
    held, over = within_ceiling(samples, growths, pressure["occupied"]["at"], pressure["freeing"]["at"])
    failures += held
    print(f"readings over the ceiling under the held pressure, each answered by a release: {over}")
    current = {name: value["initial"] for name, value in selected.items()}
    # Replay backing changes at their lifecycle boundary, not delayed log delivery.
    ordered = sorted((event for event in events if "refusal" not in event), key=lambda event: event["released"] if event.get("rung") == "Bank"
                     else event["installed"] if event.get("rung") == "Growth" else event["at"])
    for event in ordered:
        if event.get("holding") in selected:
            current[event["holding"]] = event["to"]
        if event.get("rung") in ("Prefix", "Optional", "Preemption") and start <= event["at"] <= freed:
            deficits = {"Device": event.get("reclaim_device_deficit_bytes"),
                        "Host": event.get("reclaim_host_deficit_bytes")}
            if event.get("cause") != "Reclaim" or any(
                    not isinstance(deficits.get(value["storage_kind"]), int) for value in selected.values()):
                failures.append("release ordering unverified: costlier release has live-claim or missing reclaim attribution")
            elif any(deficits[value["storage_kind"]] > 0 and current[name] > value["physical_floor"]
                     for name, value in selected.items()):
                failures.append("costlier resource released before pressured-store holdings exhausted reclaimable backing")
    # All it held, or less where the latest growth decision's miss record says more saves nothing.
    justified = {event["decision"]: event["justified"] for event in events if "decision" in event}
    wanted = {name: min(selected[name]["initial"], justified.get(name, selected[name]["initial"])) for name in shrunk}
    failures += [f"{name} did not regrow after the pressure: {current[name]} of the {wanted[name]} its record justifies"
                 for name in shrunk if current[name] < wanted[name]]
    return failures


def report_pressure(charge, lines, resident_tokens, scratch):
    """The forced-pressure gate: the pressed store's holdings shrink before any costlier release, the
    load keeps under what the occupier leaves, and regrows once the memory is free
    (`pressure_transitions`), no bank step needs new memory (its charge and footprint fall), and the
    decode's tokens equal the resident decode's. Returns the gate's failures, empty when it passes,
    with the window's (`report_stalls`)."""
    events = releases(lines)
    pressure = charge["pressure"]
    tokens = charge["decoded_tokens"]
    samples = memory_samples(lines, pressure.get("approached", {}).get("before", {}).get("name"))
    failures = pressure_transitions(charge, events, samples)
    # A shrink needs no new memory: charge and footprint fall at every step. The charge read is
    # the device's, so a discrete bank's host tier leaves it where it was.
    bank = [event for event in events if event.get("rung") == "Bank"]
    grew = [event for event in bank
            if (event["charged_after"] > event["charged_before"] if event["holding"] == "Host"
                else event["charged_after"] >= event["charged_before"])
            or (event["footprint_before"] is not None and event["footprint_after"] > event["footprint_before"])]
    if grew:
        failures.append(f"{len(grew)} of {len(bank)} bank steps grew their charge or footprint")
    if tokens is None or resident_tokens is None:
        failures.append("no exact-output comparison: pressured or reference decode did not complete")
    elif tokens != resident_tokens:
        failures.append("pressured decode differs from unpressured reference")
    if not failures:
        failures += report_stalls(pressure, [event for event in events if "rung" in event],
                                  pressure["started"]["at"], charge["release_step_ratio"], served_steps(lines))
    report = {"pressure": pressure, "bank": charge.get("bank"), "events": events, "samples": samples, "failures": failures,
              "tokens": tokens, "reference_tokens": resident_tokens}
    (scratch / PRESSURE).write_text(json.dumps(report, indent=1))
    print(f"physical pressure: {len(events)} owner events; tokens "
          f"{'identical' if tokens is not None and tokens == resident_tokens else 'unverified/different'}")
    for moment, value in (("before", pressure.get("approached", {}).get("before")),
                          ("held", pressure.get("occupied", {}).get("observation")),
                          ("released", pressure.get("released", {}).get("observation"))):
        if value:
            print(f"{moment}: store={value['store']}, headroom={value['headroom_bytes']}, "
                  f"claimable={value['ceiling_bytes']}")
    for event in events:
        if event.get("holding"):
            print(f"{event['rung']}: {event['holding']} {event['from']}→{event['to']}, "
                  f"backing bytes={event['bytes']}")
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
                        help="load under this weight cap; dedicated memory requires --pressure")
    parser.add_argument("--kernels", metavar="DIR",
                        help="the kernel cache every load prepares from, as a served load's (its executor's "
                             "costs price the bank's growth)")
    parser.add_argument("--context", type=int, metavar="TOKENS",
                        help="load at this context; dedicated memory requires --pressure")
    parser.add_argument("--residency", nargs=3, metavar=("HARNESS", "FIXTURES", "TOKENS"),
                        help="under --weight-cap: read the expert pages and drive the requested decode; "
                             "stacks the file cache holds once loaded and once the load has decoded "
                             "TOKENS tokens of the first of FIXTURES (rendered by the harness "
                             "executable HARNESS)")
    parser.add_argument("--recommit", metavar="GB[,GB...]",
                        help="unified memory, loading nothing: what an in-place recommit of a reserved tensor (an "
                             "elastic expert bank's resize) costs at each commitment")
    parser.add_argument("--pressure", type=int, metavar="BYTES",
                        help="under --residency: another process presses the load BYTES below its planning line "
                             "from a quarter of the decode to three quarters of it; report the owner's releases (the bank's slots, each "
                             "release's time, charge and footprint) and the decode's tokens against a resident "
                             "load's (the forced-pressure gate)")
    parser.add_argument("--pressure-store", choices=("host", "device"))
    parser.add_argument("--pressure-prompt", action="store_true",
                        help="under --pressure: the pressure starts once the request is admitted, so its "
                             "prompt's groups run under it")
    parser.add_argument("models", nargs="*")
    return parser


def refusal(options) -> str | None:
    """Why these options (`parser`'s) do not name one measurement, or None: the one judge of
    them, which `mrig memory` also applies to the command it plans."""
    if options.pressure is not None and options.pressure <= 0:
        return "--pressure requires a positive byte deficit"
    if bool(options.pressure) != bool(options.pressure_store):
        return "--pressure and --pressure-store are required together"
    if options.pressure and not options.residency:
        return "--pressure applies while a streamed load decodes: it takes --residency"
    if options.pressure_prompt and not options.pressure:
        return "--pressure-prompt says when --pressure starts: it takes --pressure"
    if options.fit:
        return None
    if not (options.bin and options.backend and options.memory and options.scratch):
        return "a measurement takes --bin, --backend, --memory and --scratch"
    if options.recommit:
        if options.memory == "dedicated" or options.models or options.artifacts or options.weight_cap \
                or options.context or options.residency:
            return "--recommit takes a unified-memory device and loads nothing"
        return None
    if (options.weight_cap or options.context) and options.memory == "dedicated" and not options.pressure:
        return "--weight-cap and --context measure a unified-memory load"
    if options.residency and not (options.weight_cap and options.context and options.artifacts):
        return "--residency reads a streamed load's experts: it takes --weight-cap, --context and --artifacts"
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
    if options.recommit:
        scratch = Path(options.scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        recommit_costs(options, scratch)
        return
    models = list(options.models)
    if options.artifacts:
        acquired = json.loads(Path(options.artifacts).read_text())
        models.append(",".join([acquired["target"], (acquired.get("projector") or {}).get("path", ""),
                                (acquired.get("drafter") or {}).get("path", "")]).rstrip(","))
    scratch = Path(options.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    (device_costs if options.memory == "dedicated" and not options.pressure else host_costs)(options, models, scratch)


def recommit_costs(options, scratch):
    """What an in-place recommit costs at each commitment (`magnitude-harness recommit`): the
    median and range of a one-row shrink and regrow, and the median per committed GB."""
    run = subprocess.run([exe(options.bin, "magnitude-harness"), "recommit", "--device", options.backend,
                          "--gigabytes", options.recommit],
                         capture_output=True, text=True, check=False)  # a failure is reported below
    (scratch / "recommit.log").write_text(run.stderr)
    if run.returncode != 0:
        print(f"recommit timing failed (exit {run.returncode}); see {scratch / 'recommit.log'}", file=sys.stderr)
        return
    results = [json.loads(line) for line in run.stdout.splitlines()]
    (scratch / RECOMMIT).write_text(json.dumps(results, indent=1))
    span = lambda times: f"{statistics.median(times):7.1f} ({min(times):.1f}–{max(times):.1f})"
    print("| committed GB | reserve ms | shrink one row ms | regrow one row ms | resize ms per GB |")
    print("|---|---|---|---|---|")
    for r in results:
        gigabytes = r["committed_bytes"] / 1e9
        per = statistics.median(r["shrink_ms"] + r["grow_ms"]) / gigabytes
        print(f"| {gigabytes:.2f} | {r['reserve_ms']:.1f} | {span(r['shrink_ms'])} | {span(r['grow_ms'])} "
              f"| {per:.1f} |")
    print(f"\n{options.backend}, rows of {results[0]['row_bytes'] / 1024:g} KiB (the device's granule); median (min–max) of "
          f"{len(results[0]['shrink_ms'])} resizes each way")


def host_costs(options, models, scratch):
    """Each model's target loaded plain: its own memory at load end against its charge. A load, a
    rendering or a resident decode that failed fails the measurement once every model is reported:
    what it would have tested is untested. So does every check that failed: storage beyond its
    bound, expert pages in the file cache, and the forced-pressure gate (`report_pressure`)."""
    results, failed, failures = [], [], []
    for index, paths in enumerate(models):
        target = paths.split(",")[0]
        name = Path(target).stem
        cap = ("--weight-cap", str(options.weight_cap)) if options.weight_cap else ()
        context = ("--context", str(options.context)) if options.context else ()
        log = scratch / f"{index}-{name}.log"
        decode = ()
        if options.residency:
            harness, fixtures, tokens = options.residency
            # The prompt leaves room in the context for the tokens decoded after it.
            render = subprocess.run([harness, "render", "--artifacts", options.artifacts, "--fixtures", fixtures,
                                     "--context", str(options.context - int(tokens)),
                                     "--output", str(scratch / f"{index}-render")],
                                    capture_output=True, text=True, check=False)  # a failure is reported below
            if render.returncode != 0:
                log.write_text(render.stderr)
                print(f"{name}: rendering the fixtures failed (exit {render.returncode}); see {log}", file=sys.stderr)
                failed.append(name)
                continue
            decode = ("--decode", str(scratch / f"{index}-render" / "rendered.jsonl"), "--decode-tokens", tokens)
        pressure = (("--pressure", str(options.pressure), "--pressure-store", options.pressure_store, *(("--pressure-prompt",) if options.pressure_prompt else ()))
                    if options.pressure else ())
        kernels = ("--cache-dir", options.kernels) if options.kernels else ()
        load = subprocess.run([exe(options.bin, "magnitude-harness"), "memory", "--device", options.backend,
                               *cap, *context, *decode, *pressure, *kernels, target],
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
            storage = loaded["device_held_bytes"]
            # What the load holds at load end of what its plan prices: its storage (every charge the device
            # makes in unified memory, device and upload)
            # as held, which is the plan's exactly but for the bytes beyond allocations the plan bounds,
            # and its exact host terms.
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
        if options.pressure and charge["decoded_tokens"] is None:
            # The decode ended under pressure (the ladder past its last rung): its releases are reported.
            failures += [f"{name}: {failure}" for failure in report_pressure(charge, load.stderr.splitlines(), None, scratch)]
        elif options.pressure:
            # The same decode resident, after the streamed load, so its cached reads never serve the
            # streamed one: from the prompt the pressured decode was sized to, as many tokens as it
            # decoded, the tokens it must equal.
            same = ("--decode", decode[1], "--decode-tokens", str(len(charge["decoded_tokens"])),
                    "--decode-prompt-tokens", str(charge["decode_prompt_tokens"]))
            resident = subprocess.run([exe(options.bin, "magnitude-harness"), "memory", "--device", options.backend,
                                       *context, *same, *kernels, target],
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
        terms = [*planned, *(term for term in held if term not in planned and term != "charged")]
        print(f"\n{Path(r['model']).name}, weight cap {r['weight_cap'] or '-'}, context {r['context']}: "
              f"storage by term at load end ({BOUND}: what the backend holds beside its allocations, which "
              "the plan bounds over the load's life)")
        print("| term | planned MB | charged MB | charged − planned (bytes) |")
        print("|---|---|---|---|")
        for term in terms:
            plan, charge = planned.get(term), held.get(term)
            shown = lambda recorded: "unrecorded" if recorded is None else mb(recorded)
            difference = "-" if plan is None or charge is None else charge - plan
            if term == BOUND and difference != "-":
                difference = f"{difference} ({'within' if charge <= plan else 'EXCEEDS'} its bound)"
                if charge > plan:
                    failures.append(f"{Path(r['model']).name}: storage beyond allocations exceeds its bound by "
                                    f"{charge - plan} bytes")
            print(f"| {term} | {shown(plan)} | {shown(charge)} | {difference} |")
        # Every allocation's term is planned exactly; the bound is compared on its own row.
        allocations = r["plan_storage_bytes"] - planned[BOUND]
        print(f"| total charged | {mb(allocations)} | {mb(held['charged'])} "
              f"| {held['charged'] - allocations} |")
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
    for index, paths in enumerate(models):
        name = Path(paths.split(",")[0]).stem
        assess = subprocess.run([exe(options.bin, "magnitude-assess"), "--device", options.backend,
                                 "--kernels", cache, "--breakdown", paths],
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

        def held(moment, found=found):
            return found[moment][1].get("device_held_bytes") or found[moment][1]["device_charged_bytes"]

        context = lost("device opened")
        programs = lost("programs prepared") - context - held("programs prepared")
        loaded = lost("loaded")
        residual = loaded - context - programs - held("loaded")
        peak = max(value for _, value in sampler.samples) - baseline
        kernels = breakdown["kernels"]
        planned = breakdown["serving_bytes"] - breakdown["state_bytes"] + breakdown["loaded_state_bytes"]
        result = {
            "model": paths, "backend": options.backend, "host": platform.node(), "os": platform.system(),
            "kernels": kernels, "idle_used_bytes": idle, "baseline_used_bytes": baseline,
            "context_bytes": context, "program_bytes": programs,
            "program_bytes_per_kernel": programs // max(kernels, 1),
            "loaded_lost_bytes": loaded, "peak_lost_bytes": peak,
            "held_bytes": held("loaded"), "charged_bytes": found["loaded"][1]["device_charged_bytes"],
            "unallocated_bytes": found["loaded"][1].get("device_unallocated_bytes"),
            "residual_bytes": residual, "planned_at_load_bytes": planned,
            "required_bytes": [domain["requiredBytes"] for domain in verdict.get("memory", [])],
            "breakdown": breakdown,
        }
        results.append(result)
        (scratch / RESULTS["dedicated"]).write_text(json.dumps(results, indent=1))
    gb = lambda value: f"{value / 1e9:7.3f}" if value is not None else "      -"
    print("| model | kernels | context | programs | per kernel | held | residual | lost at load | peak | planned at load |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {Path(r['model'].split(',')[0]).name} | {r['kernels']} | {gb(r['context_bytes'])} | {gb(r['program_bytes'])} "
              f"| {r['program_bytes_per_kernel'] / 1e6:.1f} MB | {gb(r['held_bytes'])} | {gb(r['residual_bytes'])} "
              f"| {gb(r['loaded_lost_bytes'])} | {gb(r['peak_lost_bytes'])} | {gb(r['planned_at_load_bytes'])} |")
    if results:
        print(f"\nper-kernel cost bound: {max(r['program_bytes_per_kernel'] for r in results) / 1e6:.1f} MB; "
              f"context: {max(r['context_bytes'] for r in results) / 1e6:.0f} MB; results in {scratch / RESULTS['dedicated']}")


if __name__ == "__main__":
    main()
