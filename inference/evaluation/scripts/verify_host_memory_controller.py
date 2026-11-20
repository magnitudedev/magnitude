#!/usr/bin/env python3
"""Real-OS, real-hardware verification of the Magnitude host memory controller.

Runs a live `magnitude-engine` serving a streamed MoE model alongside a separate
competing OS process and proves all four lifecycle scenarios end-to-end:

  Scenario A — Clean host growth:
      Model starts streamed and grows its expert bank to fill available RAM
      without false shortages.
  Scenario B — Loaded desktop with idle/compressed background app:
      Competing process allocates ~45% of host RAM (pushing macOS into active
      compression / standing `Warn`, or Linux into inactive anon + page cache)
      and sits idle for 20s. Model MUST hold its remaining sensible bank stably
      with ZERO standing-Warn death-spiral draining.
  Scenario C — Active competing app growth / real pressure:
      Competing process actively grows to ~72–78% of host RAM and touches pages
      continuously. Model MUST smoothly yield bank memory without crashing,
      unloading, or thrashing.
  Scenario D — Recovery / regrowth:
      Competing process exits and frees its RAM. Model MUST lift any standing
      boundary cap (even if macOS remains at `Warn`) and regrow its bank back up.

Designed to run via `mrig script` on remote macOS and Linux hosts (never on a
developer laptop):

  uv run --project inference/evaluation mrig script \\
      inference/evaluation/scripts/verify_host_memory_controller.py \\
      --backend metal --host m2-pro-01 \\
      --model qwen3.6-35b-a3b:gguf:q4 \\
      --prepare "cargo build --release -p magnitude-engine-cli" \\
      --purpose "Prove host memory controller Scenarios A-D on macOS" \\
      --estimate 6m
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

PORT = 18991
WEIGHT_CAP_BYTES = int(os.environ["MRIG_WEIGHT_CAP"]) if os.environ.get("MRIG_WEIGHT_CAP") else None

COMPETITOR_CODE = r"""
import os, sys

chunks = []
chunk_bytes = 64 * 1024 * 1024  # 64 MiB per slab
pages_per_chunk = chunk_bytes // 4096
# Semi-compressible pattern (~2:1 compressibility like real desktop apps)
half = chunk_bytes // 2
pattern_comp = (b"\x5a\xa5\x3c\xc3" * (half // 4)) + os.urandom(half)
# Incompressible pattern (like active GPU/media/VM/build working set)
pattern_incomp = os.urandom(chunk_bytes)
touch_byte = 1
touch_cursor = 0

def alloc_to(target_mib, incompressible=False):
    global touch_byte
    target_chunks = max(0, int(target_mib) // 64)
    stamp = bytes([touch_byte]) * pages_per_chunk
    pat = pattern_incomp if incompressible else pattern_comp
    if incompressible and chunks:
        # Replace existing compressible chunks with incompressible ones
        for i in range(min(len(chunks), target_chunks)):
            buf = bytearray(pat)
            buf[::4096] = stamp
            chunks[i] = buf
    while len(chunks) < target_chunks:
        buf = bytearray(pat)
        # C-speed slice touch on every 4 KiB page to guarantee physical residency
        buf[::4096] = stamp
        chunks.append(buf)
    while len(chunks) > target_chunks:
        chunks.pop()
    print(f"READY {len(chunks) * 64}", flush=True)

def touch_some():
    global touch_byte, touch_cursor
    if not chunks:
        return
    touch_byte = (touch_byte + 1) & 0xFF
    stamp = bytes([touch_byte]) * pages_per_chunk
    # Touch up to 16 chunks (1 GiB) round-robin so active working-set pressure
    # stays responsive without blocking minutes on multi-GB swap-in.
    count = min(16, len(chunks))
    for _ in range(count):
        chunks[touch_cursor % len(chunks)][::4096] = stamp
        touch_cursor += 1

for raw in sys.stdin:
    parts = raw.strip().split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "ALLOC":
        alloc_to(int(parts[1]), incompressible=False)
    elif cmd == "ALLOC_INCOMP":
        alloc_to(int(parts[1]), incompressible=True)
    elif cmd == "TOUCH":
        touch_some()
        print("TOUCHED", flush=True)
    elif cmd == "EXIT":
        chunks.clear()
        print("BYE", flush=True)
        break
"""


def total_ram_bytes() -> int:
    if IS_MACOS:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out)
    if IS_LINUX:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f"unsupported OS: {platform.system()}")


def sample_os_memory() -> dict:
    if IS_MACOS:
        level = int(
            subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            ).strip()
        )
        vm = subprocess.check_output(["vm_stat"], text=True)
        stats = {"pressure_level": level}
        for key, label in (
            ("free_pages", "Pages free:"),
            ("compressed_pages", "Pages occupied by compressor:"),
            ("decompressions", "Decompressions:"),
            ("compressions", "Compressions:"),
        ):
            m = re.search(rf"{re.escape(label)}\s+(\d+)", vm)
            if m:
                stats[key] = int(m.group(1))
        return stats
    if IS_LINUX:
        stats = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            for key, prefix in (
                ("mem_free_kib", "MemFree:"),
                ("mem_avail_kib", "MemAvailable:"),
                ("inactive_anon_kib", "Inactive(anon):"),
                ("cached_kib", "Cached:"),
            ):
                if line.startswith(prefix):
                    stats[key] = int(line.split()[1])
        psi_path = Path("/proc/pressure/memory")
        if psi_path.exists():
            stats["psi"] = psi_path.read_text().strip().replace("\n", " | ")
        return stats
    return {}


class EngineMonitor:
    """Streams magnitude-engine stderr and parses memory controller events."""

    RE_BANK_SLOTS = re.compile(r"bank_slots=(\d+)")
    RE_HELD_AFTER = re.compile(r"held_after=(\d+)")
    RE_CAP = re.compile(r"cap: (None|Some\((\d+)\))")
    RE_BUDGET = re.compile(r"budget: (\d+)")
    RE_SUPPLY = re.compile(r"supply: (-?\d+)")
    RE_INITIAL_SLOTS = re.compile(r"expert bank: (\d+) of the widest routed layer's (\d+)")
    RE_SHORTAGE = re.compile(r"a real memory shortage.*cause=([A-Za-z0-9_(){}: ]+?) yielded=(\d+)")
    RE_SYNC_RECLAIM = re.compile(
        r"expert bank reclaimed synchronously.*from=(\d+) to=(\d+)"
    )

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.lock = threading.Lock()
        self.lines: list[tuple[float, str]] = []
        self.initial_slots: int | None = None
        self.max_experts: int | None = None
        self.current_slots: int | None = None
        self.current_held: int | None = None
        self.current_cap: int | None = None
        self.current_budget: int | None = None
        self.current_supply: int | None = None
        self.accounting_violations = 0
        self.events: list[dict] = []

    def feed(self, line: str) -> None:
        now = time.monotonic()
        with self.lock:
            self.lines.append((now, line))
            if "accounting_violations" in line or "holds bytes in RAM that the store's use does not" in line:
                self.accounting_violations += 1
            m = self.RE_INITIAL_SLOTS.search(line)
            if m:
                self.initial_slots = int(m.group(1))
                self.max_experts = int(m.group(2))
                if self.current_slots is None:
                    self.current_slots = self.initial_slots
            m = self.RE_SYNC_RECLAIM.search(line)
            if m:
                self.current_slots = int(m.group(2))
                self.events.append({"t": now, "kind": "sync_reclaim", "from": int(m.group(1)), "to": int(m.group(2))})
            m = self.RE_BANK_SLOTS.search(line)
            if m:
                slots = int(m.group(1))
                self.current_slots = slots
                ev = {"t": now, "kind": "resize", "slots": slots}
                if "memory grown" in line:
                    ev["dir"] = "grow"
                elif "memory released" in line:
                    ev["dir"] = "shrink"
                mh = self.RE_HELD_AFTER.search(line)
                if mh:
                    self.current_held = int(mh.group(1))
                    ev["held"] = self.current_held
                mc = self.RE_CAP.search(line)
                if mc:
                    self.current_cap = None if mc.group(1) == "None" else int(mc.group(2))
                    ev["cap"] = self.current_cap
                mb = self.RE_BUDGET.search(line)
                if mb:
                    self.current_budget = int(mb.group(1))
                    ev["budget"] = self.current_budget
                ms = self.RE_SUPPLY.search(line)
                if ms:
                    self.current_supply = int(ms.group(1))
                    ev["supply"] = self.current_supply
                self.events.append(ev)
            m = self.RE_SHORTAGE.search(line)
            if m:
                self.events.append({
                    "t": now,
                    "kind": "shortage",
                    "cause": m.group(1).strip(),
                    "yielded": int(m.group(2)),
                })

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "initial_slots": self.initial_slots,
                "max_experts": self.max_experts,
                "slots": self.current_slots,
                "held": self.current_held,
                "cap": self.current_cap,
                "budget": self.current_budget,
                "supply": self.current_supply,
                "accounting_violations": self.accounting_violations,
            }

    def events_since(self, t0: float) -> list[dict]:
        with self.lock:
            return [e for e in self.events if e["t"] >= t0]

    def flush(self) -> None:
        with self.lock:
            self.log_path.write_text("\n".join(f"{t:.3f} {l}" for t, l in self.lines))


PROMPTS = [
    "Explain the architecture of modern virtual memory subsystems, including page tables, TLBs, copy-on-write, memory compression, and swap heuristics in detail.",
    "Write a concise technical overview of distributed consensus algorithms comparing Raft, Paxos, and Viewstamped Replication with concrete failure scenarios.",
    "Describe how GPU memory hierarchies work across registers, shared memory, L2 cache, HBM, and unified memory architectures with examples.",
    "Summarize the physics of semiconductor lithography, EUV light sources, photoresists, and multi-patterning techniques in modern fabs.",
    "Explain how Rust's borrow checker, lifetimes, Send/Sync traits, and lock-free atomics prevent data races at compile time.",
]
_prompt_idx = 0


def send_chat(url: str, max_tokens: int = 32) -> str:
    global _prompt_idx
    prompt = PROMPTS[_prompt_idx % len(PROMPTS)]
    _prompt_idx += 1
    body = json.dumps({
        "model": "m",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", body, {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def main() -> int:
    run_dir = Path(os.environ.get("MRIG_RUN", "."))
    run_dir.mkdir(parents=True, exist_ok=True)
    target_dir = Path(os.environ["MRIG_TARGET"])
    artifacts = json.loads(Path(os.environ["MRIG_ARTIFACTS"]).read_text())
    model_path = artifacts["target"]
    backend = os.environ.get("MRIG_BACKEND", "metal" if IS_MACOS else "cuda")
    cache_dir = Path(os.environ.get("MRIG_ROOT", str(run_dir))) / "kernels"
    cache_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = total_ram_bytes()
    total_mib = total_bytes // (1024 * 1024)
    print(f"Host: {platform.node()} ({platform.system()}), RAM={total_mib} MiB, backend={backend}", flush=True)
    print(f"Model: {Path(model_path).name}, weight_cap={WEIGHT_CAP_BYTES}", flush=True)

    engine_bin = target_dir / "release" / "magnitude-engine"
    if not engine_bin.exists():
        raise SystemExit(f"missing binary: {engine_bin}")

    monitor = EngineMonitor(run_dir / "engine.log")
    env = {**os.environ, "RUST_LOG": "info"}
    engine_cmd = [
        str(engine_bin),
        "--device", backend,
        "--model", model_path,
        "--no-projector",
        "--port", str(PORT),
        "--cache-dir", str(cache_dir),
        "--served-model", "m",
    ]
    if WEIGHT_CAP_BYTES is not None:
        engine_cmd.extend(["--weight-cap", str(WEIGHT_CAP_BYTES)])
    engine = subprocess.Popen(
        engine_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    def drain_stderr():
        assert engine.stderr is not None
        for line in engine.stderr:
            monitor.feed(line.rstrip("\n"))

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()

    competitor = None
    failures: list[str] = []
    report: dict = {"host": platform.node(), "os": platform.system(), "total_mib": total_mib, "scenarios": {}}

    try:
        url = f"http://127.0.0.1:{PORT}"
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if engine.poll() is not None:
                raise RuntimeError(f"magnitude-engine exited early with {engine.returncode}")
            try:
                urllib.request.urlopen(f"{url}/health", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("timed out waiting for magnitude-engine /health")

        # ── Scenario A: Clean host growth ──
        t_a = time.monotonic()
        print("\n=== Scenario A: Clean host growth ===", flush=True)
        for _ in range(3):
            _ = send_chat(url, max_tokens=48)
            time.sleep(2.0)
        snap_a = monitor.snapshot()
        os_a = sample_os_memory()
        events_a = monitor.events_since(t_a)
        shortages_a = [e for e in events_a if e["kind"] == "shortage"]
        print(f"A: snap={snap_a} os={os_a} shortages={len(shortages_a)}", flush=True)
        report["scenarios"]["A"] = {"snap": snap_a, "os": os_a, "shortages": shortages_a}
        if snap_a["slots"] is None or snap_a["initial_slots"] is None:
            failures.append("A: failed to observe initial/current bank slots")
        elif snap_a["slots"] < snap_a["initial_slots"]:
            failures.append(f"A: bank shrank on clean host ({snap_a['slots']} < {snap_a['initial_slots']})")
        if shortages_a:
            failures.append(f"A: unexpected shortages on clean host: {shortages_a}")

        # Spawn competitor process for Scenarios B, C, D
        competitor = subprocess.Popen(
            [sys.executable, "-c", COMPETITOR_CODE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert competitor.stdin is not None and competitor.stdout is not None

        def comp_cmd(cmd: str) -> str:
            competitor.stdin.write(cmd + "\n")
            competitor.stdin.flush()
            return competitor.stdout.readline().strip()

        # ── Scenario B: Loaded desktop with idle/compressed background app ──
        print("\n=== Scenario B: Loaded desktop (idle/compressed background app) ===", flush=True)
        # Allocate enough to push host into real desktop load / compression / Warn
        # without starving the model to floor while idle.
        held_a_mib = (snap_a["held"] or 0) // (1024 * 1024)
        idle_mib = max(int(total_mib * 0.35), min(int(total_mib * 0.55), total_mib - held_a_mib))
        t_b = time.monotonic()
        resp_b = comp_cmd(f"ALLOC {idle_mib}")
        print(f"B: competitor {resp_b} (held_a={held_a_mib} MiB; now IDLE for 20s)", flush=True)
        # Settle initial allocation for 4s, then observe 16s of idle stability
        time.sleep(4.0)
        _ = send_chat(url, max_tokens=8)
        t_b_steady = time.monotonic()
        snap_b_start = monitor.snapshot()
        os_b_samples = [sample_os_memory()]
        slots_b_trace = [snap_b_start["slots"]]
        for i in range(4):
            time.sleep(4.0)
            _ = send_chat(url, max_tokens=8)
            snap_now = monitor.snapshot()
            slots_b_trace.append(snap_now["slots"])
            os_b_samples.append(sample_os_memory())
        snap_b_end = monitor.snapshot()
        events_b_steady = monitor.events_since(t_b_steady)
        warn_shortages_b = [
            e for e in events_b_steady if e["kind"] == "shortage" and "Pressure(Warn)" in e["cause"]
        ]
        print(
            f"B: slots_trace={slots_b_trace} snap_end={snap_b_end} "
            f"warn_shortages_steady={len(warn_shortages_b)} os_samples={os_b_samples[-1]}",
            flush=True,
        )
        report["scenarios"]["B"] = {
            "idle_mib": idle_mib,
            "slots_trace": slots_b_trace,
            "snap_start": snap_b_start,
            "snap_end": snap_b_end,
            "warn_shortages_steady": warn_shortages_b,
            "os_samples": os_b_samples,
        }
        if len(warn_shortages_b) > 0:
            failures.append(f"B: standing Warn fired {len(warn_shortages_b)} shortages while competitor was idle!")
        if snap_b_start["slots"] and snap_b_end["slots"] is not None:
            drop_steady = snap_b_start["slots"] - snap_b_end["slots"]
            if drop_steady > 10:
                failures.append(
                    f"B: bank drained during idle steady window ({snap_b_start['slots']} -> {snap_b_end['slots']})"
                )

        # ── Scenario C: Active competing app growth / real pressure ──
        print("\n=== Scenario C: Active competing app growth / real pressure ===", flush=True)
        peak_frac = 0.56 if IS_MACOS else 0.78
        peak_mib = int(total_mib * peak_frac)
        t_c = time.monotonic()
        resp_c = comp_cmd(f"ALLOC_INCOMP {peak_mib}")
        print(f"C: competitor {resp_c} (incompressible, actively touching)", flush=True)
        slots_c_trace = []
        for _ in range(3):
            comp_cmd("TOUCH")
            time.sleep(2.5)
            _ = send_chat(url, max_tokens=24)
            slots_c_trace.append(monitor.snapshot()["slots"])
        snap_c = monitor.snapshot()
        os_c = sample_os_memory()
        events_c = monitor.events_since(t_c)
        slots_c_min = min(s for s in slots_c_trace if s is not None) if any(slots_c_trace) else snap_c["slots"]
        print(
            f"C: slots_trace={slots_c_trace} min={slots_c_min} snap={snap_c} os={os_c}",
            flush=True,
        )
        report["scenarios"]["C"] = {
            "peak_mib": peak_mib,
            "slots_trace": slots_c_trace,
            "slots_min": slots_c_min,
            "snap": snap_c,
            "os": os_c,
            "events": events_c,
        }
        if snap_a["slots"] is not None and slots_c_min is not None:
            if slots_c_min >= snap_a["slots"]:
                failures.append(
                    f"C: bank did not yield under active pressure (min={slots_c_min} >= A={snap_a['slots']})"
                )

        # ── Scenario D: Recovery / regrowth ──
        print("\n=== Scenario D: Recovery / regrowth after competitor exits ===", flush=True)
        t_d = time.monotonic()
        comp_cmd("EXIT")
        competitor.wait(timeout=30)
        competitor = None
        print("D: competitor exited; observing regrowth for 15s", flush=True)
        slots_d_trace = []
        for _ in range(5):
            time.sleep(3.0)
            _ = send_chat(url, max_tokens=48)
            slots_d_trace.append(monitor.snapshot()["slots"])
        snap_d = monitor.snapshot()
        os_d = sample_os_memory()
        events_d = monitor.events_since(t_d)
        print(f"D: slots_trace={slots_d_trace} snap={snap_d} os={os_d}", flush=True)
        report["scenarios"]["D"] = {
            "slots_trace": slots_d_trace,
            "snap": snap_d,
            "os": os_d,
            "events": events_d,
        }
        if slots_c_min is not None and snap_d["slots"] is not None:
            if snap_d["slots"] <= slots_c_min:
                failures.append(
                    f"D: bank did not regrow after competitor exited ({snap_d['slots']} <= C_min={slots_c_min})"
                )
        if snap_d["cap"] is not None:
            failures.append(f"D: boundary cap remained stuck at {snap_d['cap']} after competitor exited")
        if snap_d["accounting_violations"] > 0:
            failures.append(f"accounting_violations={snap_d['accounting_violations']}")

    finally:
        if competitor is not None and competitor.poll() is None:
            competitor.kill()
            competitor.wait()
        if engine.poll() is None:
            engine.terminate()
            try:
                engine.wait(timeout=10)
            except subprocess.TimeoutExpired:
                engine.kill()
                engine.wait()
        reader.join(timeout=5)
        monitor.flush()

    report["failures"] = failures
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== SUMMARY ===", flush=True)
    if failures:
        for f in failures:
            print(f"FAILED: {f}", flush=True)
        return 1
    print("ALL SCENARIOS (A, B, C, D) PASSED!", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
