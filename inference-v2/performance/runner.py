"""Record before setup, time through completion, persist evidence automatically."""

from __future__ import annotations

import fcntl
import importlib.metadata
import inspect
import os
import platform
import statistics
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import psutil

from performance.assembly import Binding
from performance.records import Observation, Profile, digest, encoded
from performance.store import DEFAULT_STORE, Store, atomic
from performance.theory.catalog import revision


def now() -> str:
    return datetime.now(UTC).isoformat()


def local_profile() -> Profile:
    from magnitude_engine.host_info import capture_hardware

    versions = {}
    for name in ("mlx", "mlx-lm", "mlx-vlm", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return Profile(
        capture_hardware().model_dump(mode="json"),
        {"python": platform.python_version(), "packages": versions},
    )


class Run:
    def __init__(
        self,
        binding: Binding,
        *,
        benchmark: str,
        workload: dict,
        profile: Profile,
        boundary: str,
        warmup: int,
        repetitions: int,
        store: Store,
        clock: Callable[[], int],
        bindings: dict,
        contract_version: str,
    ):
        if not benchmark or warmup < 0 or repetitions < 1 or not boundary:
            raise ValueError("invalid benchmark measurement configuration")
        self.store, self.clock = store, clock
        self.record = {
            "schema_version": 2,
            "id": uuid4().hex,
            "status": "running",
            "started_at": now(),
            "process": {
                "hostname": platform.node(),
                "pid": os.getpid(),
                "started_at": psutil.Process().create_time(),
            },
            "benchmark": benchmark,
            "boundary": boundary,
            "contract_version": contract_version,
            "bindings": bindings,
            "node": binding.path,
            "assembly": binding.assembly.graph.record(),
            "profile": profile.record(),
            "workload": workload,
            "warmup": warmup,
            "repetitions": repetitions,
            "sources": binding.assembly.sources,
            "command": sys.argv,
            "formula_revision": revision(),
            "formula_sources": {
                p.name: p.read_text()
                for p in sorted((Path(__file__).parent / "theory").glob("*.py"))
            },
            "measurement_sources": {
                p.name: p.read_text()
                for p in sorted((Path(__file__).parent / "benchmarks").glob("*.py"))
            },
            "runner_source": Path(__file__).read_text(),
            "case_sources": {
                frame.filename: Path(frame.filename).read_text()
                for frame in inspect.stack()[1:]
                if Path(frame.filename).is_file()
                and "site-packages" not in frame.filename
                and "/performance/" not in frame.filename
            },
            "clock": {
                "name": "perf_counter_ns" if clock is time.perf_counter_ns else "injected",
                "monotonic": clock is time.perf_counter_ns,
            },
            "execution_policy": {
                "warmup": "same workload before measured samples",
                "completion": boundary,
            },
            "samples": [],
        }
        encoded(self.record)
        self.directory = store.root / "runs" / self.record["id"]
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "run.json"
        atomic(self.path, self.record)
        self._measured = False

    def measure(
        self,
        operation,
        *,
        complete=None,
        prepare=None,
        validate=None,
        deterministic: bool = False,
        dimension: str | None = "EXEC",
    ) -> None:
        if self._measured:
            raise RuntimeError(
                "one observation boundary per run; start another run for another boundary"
            )
        self._measured = True
        from magnitude_engine.components import implementation
        from performance.theory.catalog import MODELS

        contract = implementation(
            self.record["assembly"]["nodes"][self.record["node"]]["implementation"]
        ).contract
        allowed = MODELS[contract].dimensions
        if dimension is not None and dimension not in allowed:
            raise ValueError(f"unsupported timing dimension {dimension}")
        self.record["timing_dimension"] = dimension
        from performance.assessment import preflight
        from performance.records import Assembly

        self.record["formulation"] = preflight(
            Assembly.read(self.record["assembly"]),
            self.record["workload"],
            Profile(**self.record["profile"]),
            bindings=self.record["bindings"],
        )
        with (self.directory / "samples.jsonl").open("x") as journal:
            for phase, count in (
                ("warmup", self.record["warmup"]),
                ("measured", self.record["repetitions"]),
            ):
                for index in range(count):
                    sample = {"phase": phase, "index": index, "elapsed_ns": 0}
                    try:
                        if prepare is not None:
                            prepare()
                        start = self.clock()
                        output = operation()
                        if complete is not None:
                            complete(output)
                        elapsed = self.clock() - start
                        if elapsed < 0:
                            raise ValueError("measurement clock moved backwards")
                        sample["elapsed_ns"] = elapsed
                        observation = validate(output) if validate else Observation()
                        if not isinstance(observation, Observation):
                            raise TypeError("validation must return an Observation")
                        if not set(observation.metrics) <= set(allowed):
                            raise ValueError(
                                "observation reports dimensions outside its component contract"
                            )
                        sample["observation"] = asdict(observation)
                        if dimension is not None:
                            if dimension in observation.metrics:
                                raise ValueError(
                                    "timing dimension cannot also be supplied by validation"
                                )
                            sample["observation"]["metrics"][dimension] = elapsed / 1e9
                    except BaseException as error:
                        sample["error"] = f"{type(error).__name__}: {error}"
                        raise
                    finally:
                        self.record["samples"].append(sample)
                        journal.write(encoded(sample) + "\n")
                        journal.flush()
            if deterministic:
                outputs = {s["observation"]["output_digest"] for s in self.record["samples"]}
                if len(outputs) != 1 or not next(iter(outputs)):
                    raise ValueError("deterministic observation outputs differ or lack digests")

    @property
    def median_seconds(self) -> float:
        return (
            statistics.median(
                s["elapsed_ns"]
                for s in self.record["samples"]
                if s["phase"] == "measured" and "error" not in s
            )
            / 1e9
        )


@contextmanager
def recording(
    binding: Binding,
    *,
    benchmark: str,
    workload: dict,
    profile: Profile | None = None,
    boundary: str = "component-through-ready",
    contract_version: str = "1",
    bindings: dict | None = None,
    warmup: int = 2,
    repetitions: int = 7,
    output: Path | None = None,
    clock: Callable[[], int] = time.perf_counter_ns,
):
    """The recording context encloses fixture preparation and cleanup, not just timing."""
    run = Run(
        binding,
        benchmark=benchmark,
        workload=workload,
        profile=profile or local_profile(),
        boundary=boundary,
        warmup=warmup,
        repetitions=repetitions,
        store=Store(output if output is not None else DEFAULT_STORE),
        clock=clock,
        contract_version=contract_version,
        bindings=bindings or {},
    )
    lock = Path(tempfile.gettempdir()) / f"magnitude-inference-measurement-{os.getuid()}.lock"
    lease = lock.open("a")
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another inference measurement owns this machine") from error
        yield run
        if not run._measured:
            raise ValueError("recording ended without a measurement")
        if revision() != run.record["formula_revision"]:
            raise ValueError(
                "theory sources changed during the run; restart with one source revision"
            )
        run.record["status"] = "complete"
    except BaseException as error:
        run.record["status"] = (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        )
        run.record["error"] = "".join(traceback.format_exception(error))
        raise
    finally:
        run.record["completed_at"] = now()
        run.record["checksum"] = digest(run.record)
        try:
            run.store.finalize(run.record)
        finally:
            lease.close()
