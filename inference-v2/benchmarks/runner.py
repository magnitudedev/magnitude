"""Common timing and evidence capture; subjects own component-specific operations."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.contracts import Experiment, Observation
from magnitude_engine.composition import build, digest, dumps
from magnitude_engine.host_info import capture_hardware


@dataclass(frozen=True)
class Sample:
    phase: str
    index: int
    elapsed_ns: int
    observation: Observation | None
    rejection: str | None = None


@dataclass
class Run:
    experiment: Experiment
    environment: dict[str, object]
    samples: list[Sample] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    def record(self) -> dict:
        measured = [
            s.elapsed_ns / 1e6
            for s in self.samples
            if s.phase == "measured" and s.rejection is None
        ]
        ordered = sorted(measured)

        def percentile(fraction: float) -> float | None:
            if not ordered:
                return None
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

        return {
            "schema_version": 2,
            "experiment": self.experiment.record(),
            "environment": self.environment,
            "samples": [asdict(sample) for sample in self.samples],
            "rejections": self.rejections,
            "status": "rejected" if self.rejections else "valid_characterization",
            "timing_boundary": "invoke through completion; reset and validation excluded",
            "statistics_ms": {
                "count": len(measured),
                "median": statistics.median(measured) if measured else None,
                "p95": percentile(0.95),
                "p99": percentile(0.99),
                "max": max(measured) if measured else None,
            },
        }

    def write(self, path: Path) -> None:
        encoded = json.dumps(self.record(), indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        # An explicit output must not silently replace earlier measurement evidence.
        with path.open("x") as stream:
            stream.write(encoded)


def environment(root: Path, *, isolated: bool = False) -> dict[str, object]:
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    digest = hashlib.sha256()
    for directory in (root / "src", root / "benchmarks"):
        for path in sorted(
            p for p in directory.rglob("*") if p.is_file() and p.suffix in (".py", ".json")
        ):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    dependencies = {}
    for name in ("mlx", "mlx-lm", "mlx-vlm", "numpy", "transformers"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "time_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": capture_hardware().model_dump(mode="json"),
        "dependencies": dependencies,
        "revision": git("rev-parse", "HEAD"),
        "dirty_files": git("status", "--porcelain", "--", str(root)),
        "implementation_digest": digest.hexdigest(),
        "process": "fresh benchmark child" if isolated else "benchmark supervisor",
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
    }


def measure(
    experiment: Experiment,
    provenance: dict[str, object],
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> Run:
    result = Run(
        experiment,
        {
            **provenance,
            "composition": experiment.subject.describe(),
            "composition_json": dumps(experiment.subject),
            "composition_digest": digest(experiment.subject),
        },
    )
    try:
        with build(experiment.subject) as subject:
            for phase, count in (
                ("warmup", experiment.warmup),
                ("measured", experiment.repetitions),
            ):
                for index in range(count):
                    elapsed = 0
                    try:
                        subject.reset()
                        start = clock()
                        subject.invoke()
                        subject.complete()
                        elapsed = clock() - start
                        if elapsed < 0:
                            raise ValueError("measurement clock moved backwards")
                        observation = subject.observe()
                        json.dumps(asdict(observation), allow_nan=False)
                        result.samples.append(Sample(phase, index, elapsed, observation))
                    except Exception as error:
                        result.samples.append(
                            Sample(phase, index, elapsed, None, f"{type(error).__name__}: {error}")
                        )
                        raise
    except Exception as error:
        result.rejections.append("".join(traceback.format_exception(error)))
    if not result.rejections:
        if len({s.observation.output_digest for s in result.samples if s.observation}) > 1:
            result.rejections.append("deterministic workload produced differing output digests")
    return result
