"""Shared benchmark temperature trace and time-weighted summaries."""

import hashlib
import json
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from magnitude_engine.platform.temperature import AppleSMC, sensor_group
from performance.store import atomic


class TemperatureSeries:
    """Integrate adjacent valid samples only; missing readings never become zero."""

    def __init__(self):
        self.count = 0
        self.start = self.end = self.minimum = self.maximum = None
        self.previous_time = None
        self.area = self.covered = 0.0

    def add(self, seconds: float, value: float | None):
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
    """Own a sensor connection and background sampler for one managed benchmark.

    The context includes setup, warmup and cleanup. SMC calls release the GIL and
    do no GPU work. Sampling cost is recorded, never subtracted from timings.
    Probe failures are evidence, not benchmark failures. Trace persistence failures
    fail the recording, as with the benchmark's other evidence files.
    """

    def __init__(self, directory: Path, *, interval_seconds=1.0, probe_factory=None):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("temperature interval must be positive and finite")
        self.directory = directory
        self.interval = interval_seconds
        self.factory = probe_factory or AppleSMC
        self._stop = threading.Event()
        self._thread = None
        self._probe = None
        self._error = None
        self._write_error = None
        self._series: dict[str, TemperatureSeries] = {}
        self._errors: dict[str, int] = {}
        self._samples = 0
        self._hash = hashlib.sha256()
        self.summary: dict = {"status": "not_started", "scope": "whole_run"}

    def __enter__(self):
        self._stream = (self.directory / "thermals.jsonl").open("x", buffering=1)
        self._origin = time.monotonic()
        try:
            try:
                self._probe = self.factory()
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
            self._sample("start")
            if self._probe is not None:
                self._thread = threading.Thread(target=self._loop, name="benchmark-temperatures")
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
        started = time.monotonic()
        at = datetime.now(UTC).isoformat()
        try:
            reading = (
                self._probe.read()
                if self._probe
                else {
                    "sensors_c": {},
                    "errors": {"probe": self._error},
                }
            )
        except Exception as error:
            reading = {"sensors_c": {}, "errors": {"probe": f"{type(error).__name__}: {error}"}}
        ended = time.monotonic()
        seconds = (started + ended) / 2 - self._origin
        sensors = reading["sensors_c"]
        values = dict(sensors)
        for group in ("cpu", "gpu"):
            group_values = [v for key, v in sensors.items() if sensor_group(key) == group]
            values[f"{group}_mean"] = (
                sum(group_values) / len(group_values) if group_values else None
            )
            values[f"{group}_max"] = max(group_values) if group_values else None
        for key in self._series.keys() | values.keys():
            if key not in self._series:
                self._series[key] = TemperatureSeries()
                if self._samples:
                    self._series[key].add(0, None)
            self._series[key].add(seconds, values.get(key))
        for key, message in reading["errors"].items():
            error_key = f"{key}: {message}"
            self._errors[error_key] = self._errors.get(error_key, 0) + 1
        line = (
            json.dumps(
                {
                    "at": at,
                    "offset_seconds": seconds,
                    "phase": phase,
                    "probe_duration_ms": (ended - started) * 1000,
                    **reading,
                    "groups_c": {
                        key: values[key] for key in ("cpu_mean", "cpu_max", "gpu_mean", "gpu_max")
                    },
                },
                allow_nan=False,
            )
            + "\n"
        )
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
                "schema_version": 1,
                "scope": "whole_run",
                "status": "partial"
                if available and self._errors
                else "available"
                if available
                else "unavailable",
                "source": self._probe.source if self._probe else "AppleSMC",
                "sensor_grouping": "SMC prefixes: CPU-associated Tp/Te/Ts; GPU-associated Tg",
                "interval_seconds": self.interval,
                "duration_seconds": time.monotonic() - self._origin,
                "sample_count": self._samples,
                "mean_method": "trapezoidal time weighting; gaps excluded",
                "trace": "thermals.jsonl",
                "trace_sha256": self._hash.hexdigest(),
                "channels": {key: series.summary() for key, series in sorted(self._series.items())},
                "errors": self._errors,
            }
            atomic(self.directory / "thermals.json", self.summary)
        finally:
            if self._probe is not None:
                self._probe.close()
            self._stream.close()
