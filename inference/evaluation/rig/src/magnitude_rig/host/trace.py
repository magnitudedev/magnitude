"""The run's thermal trace (JSON Lines) and its time-weighted summary.

Sampling covers the whole run including setup, warmup and cleanup, runs outside timed
operations without GPU work, and its cost is recorded, never subtracted from timings.
Probe failures are evidence, not run failures; trace persistence failures fail the run.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from .probes import GROUPS, Probe, Reading, open_probe

TRACE = "thermal.jsonl"
SUMMARY = "thermal-summary.json"


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def group_aggregates(reading: Reading) -> Dict[str, Optional[float]]:
    """Mean and max per sensor group; None when the group has no valid reading."""
    values = {}
    for group in GROUPS:
        group_values = reading.group_values(group)
        values[f"{group}_mean"] = sum(group_values) / len(group_values) if group_values else None
        values[f"{group}_max"] = max(group_values) if group_values else None
    return values


def sample_line(reading: Reading, at: str, offset: float, phase: str, duration: float) -> dict:
    return {
        "at": at,
        "offset_seconds": offset,
        "phase": phase,
        "probe_duration_ms": duration * 1000,
        **reading.to_json(),
        "groups_c": group_aggregates(reading),
    }


def timed_read(probe: Optional[Probe], unavailable: Optional[str]):
    """One reading with its wall-clock time and duration; failures become reading errors."""
    started = time.monotonic()
    at = datetime.now(timezone.utc).isoformat()
    if probe is None:
        reading = Reading({}, {}, {}, None, {"probe": unavailable or "no probe"})
    else:
        try:
            reading = probe.read()
        except Exception as error:
            reading = Reading({}, {}, {}, None, {"probe": f"{type(error).__name__}: {error}"})
    return reading, at, started, time.monotonic()


class TemperatureSeries:
    """Integrate adjacent valid samples only; missing readings never become zero."""

    def __init__(self):
        self.count = 0
        self.start = self.end = self.minimum = self.maximum = None
        self.previous_time = None
        self.area = self.covered = 0.0

    def add(self, seconds: float, value: Optional[float]):
        if self.previous_time is None:
            self.start = value
        elif value is not None and self.end is not None:
            elapsed = seconds - self.previous_time
            self.area += elapsed * (self.end + value) / 2
            self.covered += elapsed
        self.end, self.previous_time = value, seconds
        if value is not None:
            self.count += 1
            self.minimum = value if self.minimum is None else min(value, self.minimum)
            self.maximum = value if self.maximum is None else max(value, self.maximum)

    def summary(self) -> dict:
        return {
            "start_c": self.start,
            "end_c": self.end,
            "mean_c": self.area / self.covered if self.covered else None,
            "min_c": self.minimum,
            "max_c": self.maximum,
            "valid_samples": self.count,
            "covered_seconds": self.covered,
        }


class ThermalRecorder:
    """Own a probe and background sampler for one run directory."""

    def __init__(
        self,
        directory: Path,
        *,
        interval_seconds: float = 1.0,
        probe_factory: Callable[[], Probe] = open_probe,
    ):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("temperature interval must be positive and finite")
        self.directory = directory
        self.interval = interval_seconds
        self.factory = probe_factory
        self._stop = threading.Event()
        self._thread = None
        self._probe: Optional[Probe] = None
        self._error = None
        self._write_error = None
        self._series: Dict[str, TemperatureSeries] = {}
        self._errors: Dict[str, int] = {}
        self._samples = 0
        self._hash = hashlib.sha256()
        self.summary: dict = {"status": "not_started", "scope": "whole_run"}

    def __enter__(self):
        self._stream = (self.directory / TRACE).open("x", buffering=1)
        self._origin = time.monotonic()
        try:
            try:
                self._probe = self.factory()
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
            self._sample("start")
            if self._probe is not None:
                self._thread = threading.Thread(target=self._loop, name="thermal-trace")
                self._thread.start()
        except BaseException:
            if self._probe is not None:
                self._probe.close()
            self._stream.close()
            raise
        return self

    def _loop(self):
        deadline = time.monotonic() + self.interval
        try:
            while not self._stop.wait(max(0, deadline - time.monotonic())):
                self._sample("periodic")
                # Skip missed deadlines instead of catching up with bursts of probes.
                deadline = max(deadline + self.interval, time.monotonic() + 0.001)
        except Exception as error:
            self._write_error = error

    def _sample(self, phase):
        reading, at, started, ended = timed_read(self._probe, self._error)
        seconds = (started + ended) / 2 - self._origin
        values: Dict[str, Optional[float]] = dict(reading.sensors_c)
        values.update(group_aggregates(reading))
        for key in self._series.keys() | values.keys():
            if key not in self._series:
                self._series[key] = TemperatureSeries()
                if self._samples:
                    self._series[key].add(0, None)
            self._series[key].add(seconds, values.get(key))
        for key, message in reading.errors.items():
            error_key = f"{key}: {message}"
            self._errors[error_key] = self._errors.get(error_key, 0) + 1
        line = json.dumps(
            sample_line(reading, at, seconds, phase, ended - started), allow_nan=False
        ) + "\n"
        self._stream.write(line)
        self._hash.update(line.encode())
        self._samples += 1

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self._sample("end")
            if self._write_error is not None:
                raise self._write_error
            available = any(s.count for s in self._series.values())
            self.summary = {
                "schema_version": 2,
                "scope": "whole_run",
                "status": "partial"
                if available and self._errors
                else "available"
                if available
                else "unavailable",
                "source": self._probe.source if self._probe else None,
                "sensor_grouping": self._probe.grouping if self._probe else None,
                "interval_seconds": self.interval,
                "duration_seconds": time.monotonic() - self._origin,
                "sample_count": self._samples,
                "mean_method": "trapezoidal time weighting; gaps excluded",
                "trace": TRACE,
                "trace_sha256": self._hash.hexdigest(),
                "channels": {key: series.summary() for key, series in sorted(self._series.items())},
                "errors": self._errors,
            }
            atomic(self.directory / SUMMARY, self.summary)
        finally:
            if self._probe is not None:
                self._probe.close()
            self._stream.close()
