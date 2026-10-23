#!/usr/bin/env python3
"""`mrig memory`'s measurement: what a load's worker holds of a dedicated GPU outside its
charge. Standard library only, so it also runs alone on a machine outside the rig.

For each model, loads it with the standalone engine in an isolated kernel
cache, samples the GPU's memory while it loads, and splits what the device
lost into the terms assessment prices (`DeviceProcessCosts` in
`inference/engine/executor/src/planning/assessment.rs`):

- context: what the device lost when the engine opened it (the `device
  opened` moment) before any allocation;
- programs: what it lost by `programs prepared` beyond the context and what
  the engine holds there (`device_held_bytes`), per native kernel of the
  load's kernel inventory (`magnitude-assess --breakdown`, `kernels`);
- residual: what it lost at load end beyond context, programs and held.

The GPU must be otherwise idle: device-wide memory is sampled (`mrig memory`
holds the device for it). Run it from any directory; it writes only under
--scratch.

  python3 memory.py --bin DIR --backend cuda|vulkan --scratch DIR \\
      [--artifacts ACQUISITION.json] [TARGET.gguf[,PROJECTOR.gguf[,DRAFT.gguf]] ...]

DIR holds magnitude-engine and magnitude-assess (an unpacked engine
installation or a release build). `--artifacts` names a catalog acquisition
object (`catalog-tool acquire`) whose target, projector and drafter are
measured. Results: --scratch/device-costs.json and a table on stdout; from a
machine outside the rig, send both back for calibration.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

WINDOWS = platform.system() == "Windows"


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
        if "magnitude-engine: memory {" in line:
            moment = json.loads(line.split("magnitude-engine: memory ", 1)[1])
            found[moment["moment"]] = (when, moment)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", required=True)
    parser.add_argument("--backend", required=True, choices=["cuda", "vulkan"])
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--port", type=int, default=18977)
    parser.add_argument("--artifacts", help="a catalog acquisition object (JSON) to measure")
    parser.add_argument("models", nargs="*")
    options = parser.parse_args()
    models = list(options.models)
    if options.artifacts:
        acquired = json.loads(Path(options.artifacts).read_text())
        models.append(",".join([acquired["target"], (acquired.get("projector") or {}).get("path", ""),
                                (acquired.get("drafter") or {}).get("path", "")]).rstrip(","))
    if not models:
        parser.error("give models or --artifacts")
    scratch = Path(options.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
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
        planned = breakdown["serving_bytes"] - breakdown["state_bytes"] + breakdown["startup_state_bytes"]
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
        (scratch / "device-costs.json").write_text(json.dumps(results, indent=1))
    gb = lambda value: f"{value / 1e9:7.3f}" if value is not None else "      -"
    print("| model | kernels | context | programs | per kernel | held | residual | lost at load | peak | planned at load |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {Path(r['model'].split(',')[0]).name} | {r['kernels']} | {gb(r['context_bytes'])} | {gb(r['program_bytes'])} "
              f"| {r['program_bytes_per_kernel'] / 1e6:.1f} MB | {gb(r['held_bytes'])} | {gb(r['residual_bytes'])} "
              f"| {gb(r['loaded_lost_bytes'])} | {gb(r['peak_lost_bytes'])} | {gb(r['planned_at_load_bytes'])} |")
    if results:
        print(f"\nper-kernel cost bound: {max(r['program_bytes_per_kernel'] for r in results) / 1e6:.1f} MB; "
              f"context: {max(r['context_bytes'] for r in results) / 1e6:.0f} MB; results in {scratch / 'device-costs.json'}")


if __name__ == "__main__":
    main()
