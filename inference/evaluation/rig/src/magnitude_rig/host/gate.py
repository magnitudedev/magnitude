"""The thermal gate: measurement starts only when temperatures are flat and the host is idle.

A window passes when, over its whole span:
- every temperature group (mean over that group's sensors) is present in every sample,
  its spread is small and its least-squares trend does not rise;
- CPU and every reported GPU are idle on average;
- macOS thermal pressure is nominal throughout;
- no utilization probe failed.
Missing readings never pass: an unverifiable host fails explicitly after the timeout.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .probes import GROUPS, Probe, Reading
from .trace import sample_line, timed_read


@dataclass(frozen=True)
class GateSettings:
    window_seconds: float = 60.0
    interval_seconds: float = 2.0
    max_spread_c: float = 1.5
    max_rise_c_per_minute: float = 0.5
    max_cpu_pct: float = 15.0
    max_gpu_pct: float = 5.0
    timeout_seconds: float = 900.0


@dataclass(frozen=True)
class Sample:
    seconds: float
    reading: Reading


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reasons: List[str]
    channels: Dict[str, dict]


def _slope_per_minute(points: List[Tuple[float, float]]) -> float:
    count = len(points)
    mean_t = sum(t for t, _ in points) / count
    mean_v = sum(v for _, v in points) / count
    variance = sum((t - mean_t) ** 2 for t, _ in points)
    if variance == 0:
        return 0.0
    return 60.0 * sum((t - mean_t) * (v - mean_v) for t, v in points) / variance


def judge(samples: List[Sample], settings: GateSettings) -> Verdict:
    """Judge the most recent window of samples (oldest first)."""
    if not samples:
        return Verdict(False, ["no samples"], {})
    end = samples[-1].seconds
    covered = [s for s in samples if s.seconds >= end - settings.window_seconds]
    if samples[0].seconds > end - settings.window_seconds or len(covered) < 3:
        return Verdict(False, [f"window not yet covered ({end - samples[0].seconds:.0f} s)"], {})
    reasons: List[str] = []
    channels: Dict[str, dict] = {}
    groups = [g for g in GROUPS if covered[-1].reading.group_values(g)]
    if not groups:
        latest_errors = covered[-1].reading.errors
        reasons.append(f"no temperature sensors (errors: {latest_errors})")
    for group in groups:
        # The group's series is the mean over the sensors read in every sample of the window:
        # a sensor dropping out of one sample must not shift the mean by changing its subset.
        steady = set.intersection(*(set(s.reading.group_sensors(group)) for s in covered))
        if not steady:
            missing = sum(not s.reading.group_values(group) for s in covered)
            reasons.append(f"{group} temperature missing in {missing} samples"
                           if missing else f"{group} has no sensor read in every sample")
            continue
        points = [
            (s.seconds, sum(s.reading.sensors_c[k] for k in steady) / len(steady)) for s in covered
        ]
        spread = max(v for _, v in points) - min(v for _, v in points)
        slope = _slope_per_minute(points)
        channels[f"{group}_mean_c"] = {"latest": points[-1][1], "spread": spread, "slope_per_minute": slope}
        if spread > settings.max_spread_c:
            reasons.append(f"{group} spread {spread:.2f} C > {settings.max_spread_c} C")
        if slope > settings.max_rise_c_per_minute:
            reasons.append(
                f"{group} rising {slope:.2f} C/min > {settings.max_rise_c_per_minute} C/min"
            )
    keys = sorted({k for s in covered for k in s.reading.utilization_pct})
    if "cpu" not in keys:
        reasons.append("cpu utilization unavailable")
    for key in keys:
        values = [s.reading.utilization_pct.get(key) for s in covered]
        present = [v for v in values if v is not None]
        if len(present) != len(values):
            reasons.append(f"{key} utilization missing in {len(values) - len(present)} samples")
            continue
        mean = sum(present) / len(present)
        limit = settings.max_cpu_pct if key == "cpu" else settings.max_gpu_pct
        channels[f"{key}_pct"] = {"mean": mean, "limit": limit}
        if mean > limit:
            reasons.append(f"{key} busy {mean:.1f}% > {limit}%")
    failed = sorted(
        {f"{k}: {v}" for s in covered for k, v in s.reading.errors.items() if k.startswith("utilization")}
    )
    reasons.extend(failed)
    pressures = {s.reading.thermal_pressure for s in covered if s.reading.thermal_pressure}
    if pressures - {"nominal"}:
        reasons.append(f"thermal pressure {sorted(pressures)}")
    return Verdict(not reasons, reasons, channels)


class GateTimeout(Exception):
    def __init__(self, verdict: Verdict, latest: Reading, waited: float):
        super().__init__(f"thermal gate did not pass in {waited:.0f} s: {'; '.join(verdict.reasons)}")
        self.verdict = verdict
        self.latest = latest


def wait(probe: Probe, settings: GateSettings, log: Path) -> Verdict:
    """Sample until the gate passes; raise GateTimeout with the readings otherwise.

    Every gate sample and its verdict is appended to `log` (JSON Lines) as evidence.
    """
    origin = time.monotonic()
    retained = settings.window_seconds + settings.interval_seconds
    samples: List[Sample] = []
    with log.open("a", buffering=1) as stream:
        while True:
            reading, at, started, ended = timed_read(probe, None)
            seconds = (started + ended) / 2 - origin
            samples.append(Sample(seconds, reading))
            samples = [s for s in samples if s.seconds >= seconds - retained]
            verdict = judge(samples, settings)
            line = sample_line(reading, at, seconds, "gate", ended - started)
            line["verdict"] = asdict(verdict)
            stream.write(json.dumps(line, allow_nan=False) + "\n")
            if verdict.passed:
                return verdict
            waited = time.monotonic() - origin
            if waited >= settings.timeout_seconds:
                raise GateTimeout(verdict, reading, waited)
            time.sleep(max(0.0, settings.interval_seconds - (time.monotonic() - started)))
