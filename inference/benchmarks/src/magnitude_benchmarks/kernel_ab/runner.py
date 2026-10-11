"""Alternating baseline/candidate measurement of one host.

A run has up to four phases.

- Tune: the baseline loads each model with an empty kernel cache, as a user's first load would,
  and records the configuration it chose for every entry. The choice is kept per host, device,
  history codec, baseline build and model, so later runs against the same baseline skip it.
- Measure: `rounds` rounds of every model on both builds, in alternating order (baseline first on
  even rounds, candidate first on odd ones) so drift such as heat affects both sides alike. Both
  sides replay the baseline's configurations (a parameter only the candidate declares takes its
  default), so the comparison is of kernel code. One process per side, model and round measures
  every cell.
- Own tuning (`--own-tuning`): the candidate tunes too, and each side is also measured on its
  own choice: what a fresh load of that build runs, tuning luck included.
- Logits: once per build and model, on the baseline's configurations, the logits of a long prompt
  (prefilled in chunks, so later chunks attend to history) followed by single-row decodes,
  compared across builds.

Each build keeps a kernel cache per host and model, so its programs compile once across rounds
and runs.
"""

import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psutil

try:
    import resource
except ImportError:
    resource = None

from ..host.facts import capture_hardware
from ..host.thermals import ThermalRecorder
from . import models as model_set
from .builds import executable

CONTEXTS = (16384,)
# The tuning load only records configurations; a short context keeps it quick.
TUNE_CONTEXT = 256
LOGITS_PREFILL = 2048
LOGITS_DECODES = 16
TOKEN_RANGE = (100, 20_000)
RUN_TIMEOUT_SECONDS = 3600
SIDES = ("baseline", "candidate")
PREFILL_ROWS = 512


def cells(contexts: tuple[int, ...]) -> list[str]:
    """forward_bench arguments measuring every cell in one process: a prefill of 512 rows and a
    decode step after each context length."""
    return [
        "--cells",
        "prefill,decode",
        "--prefill",
        str(PREFILL_ROWS),
        "--prefill-history",
        ",".join(map(str, contexts)),
        "--context",
        ",".join(map(str, contexts)),
    ]


def logits_tokens() -> list[int]:
    rng = random.Random(20261002)
    return [rng.randrange(*TOKEN_RANGE) for _ in range(LOGITS_PREFILL + LOGITS_DECODES)]


def busy_seconds() -> float:
    times = psutil.cpu_times()
    return sum(times) - times.idle - getattr(times, "iowait", 0.0)


def child_seconds() -> float | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def execute(command: list[str], log_path: Path) -> dict:
    """Run one measurement, recording how many cores other processes kept busy meanwhile: a
    host that was not quiet makes the measurement suspect."""
    started = time.monotonic()
    busy, children = busy_seconds(), child_seconds()
    with log_path.open("w") as log:
        try:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_SECONDS
            )
            status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
    seconds = time.monotonic() - started
    own = None if children is None else child_seconds() - children
    foreign = None if own is None else max(0.0, busy_seconds() - busy - own) / seconds
    return {"status": status, "seconds": seconds, "log": log_path.name, "foreign_cores": foreign}


@dataclass
class Run:
    directory: Path
    store: Path
    builds: dict[str, tuple[dict, Path]]
    models: dict[str, Path]
    device: str
    kv_codec: str
    contexts: tuple[int, ...]
    log: object
    plan: dict = field(default_factory=dict)

    def save(self):
        (self.directory / "run.json").write_text(json.dumps(self.plan, indent=2))

    def tool(self, side: str, name: str) -> str:
        return str(self.builds[side][1] / executable(name))

    def build_name(self, side: str) -> str:
        return self.builds[side][1].name

    def common(self, side: str, model: str) -> list[str]:
        """The model, the build's persistent kernel cache, the device and the history codec."""
        cache = self.store / "caches" / self.build_name(side) / model
        return [
            "--model",
            str(self.models[model]),
            "--cache-dir",
            str(cache),
            "--device",
            self.device,
            "--kv-codec",
            self.kv_codec,
        ]

    def pins(self, side: str, model: str) -> Path:
        return self.store / "pins" / self.build_name(side) / f"{model}.json"

    def invoke(self, command: list[str], name: str, **record) -> bool:
        self.log(" ".join(f"{key}={value}" for key, value in record.items()))
        outcome = execute(command, self.directory / f"{name}.log")
        self.plan["results"].append({**record, **outcome})
        self.save()
        if outcome["status"] != "ok":
            self.log(f"  {outcome['status']}; see {outcome['log']}")
        return outcome["status"] == "ok"

    def tune(self, sides: tuple[str, ...]):
        """A fresh tune of every model whose choice this host has not kept for the build."""
        for model in self.models:
            for side in sides:
                pins = self.pins(side, model)
                if pins.is_file():
                    self.plan["results"].append(
                        {
                            "phase": "tune",
                            "side": side,
                            "model": model,
                            "status": "kept",
                            "pins": str(pins),
                        }
                    )
                    continue
                pins.parent.mkdir(parents=True, exist_ok=True)
                name = f"tune-{side}-{model}"
                scratch = self.directory / "caches" / name
                command = [self.tool(side, "forward_bench"), "bench", *self.common(side, model)]
                command[command.index("--cache-dir") + 1] = str(scratch)
                self.invoke(
                    command
                    + ["--output", str(self.directory / f"{name}.json"), *cells((TUNE_CONTEXT,))]
                    + ["--tuning-record", str(pins)],
                    name,
                    phase="tune",
                    side=side,
                    model=model,
                    output=f"{name}.json",
                    pins=str(pins),
                )

    def measure(self, rounds: int, modes: tuple[str, ...]):
        for index in range(rounds):
            for model in self.models:
                for mode in modes:
                    for side in SIDES if index % 2 == 0 else SIDES[::-1]:
                        pins = self.pins("baseline" if mode == "same" else side, model)
                        if not pins.is_file():
                            continue
                        name = f"r{index}-{mode}-{side}-{model}"
                        self.invoke(
                            [self.tool(side, "forward_bench"), "bench", *self.common(side, model)]
                            + ["--output", str(self.directory / f"{name}.json")]
                            + cells(self.contexts)
                            + ["--tuning-replay", str(pins)],
                            name,
                            phase="measure",
                            mode=mode,
                            round=index,
                            side=side,
                            model=model,
                            output=f"{name}.json",
                        )

    def logits(self):
        """Both sides on the baseline's configurations, so any difference is the code's."""
        tokens = ",".join(map(str, logits_tokens()))
        for model in self.models:
            pins = self.pins("baseline", model)
            if not pins.is_file():
                continue
            for side in SIDES:
                name = f"logits-{side}-{model}"
                self.invoke(
                    [self.tool(side, "token_logits"), *self.common(side, model)]
                    + ["--tokens", tokens, "--output", str(self.directory / f"{name}.f32")]
                    + ["--prefill", str(LOGITS_PREFILL), "--tuning-replay", str(pins)],
                    name,
                    phase="logits",
                    side=side,
                    model=model,
                    output=f"{name}.f32",
                )


def run(
    workspace: Path,
    root: Path,
    builds: dict[str, tuple[dict, Path]],
    selected: tuple[model_set.BenchModel, ...],
    own_tuning: bool,
    rounds: int,
    device: str,
    kv_codec: str,
    contexts: tuple[int, ...],
    log,
) -> Path:
    host = platform.node().split(".")[0]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "runs" / f"{stamp}-{host}"
    directory.mkdir(parents=True)
    plan = {
        "started": stamp,
        "host": platform.node(),
        "platform": platform.platform(),
        "hardware": capture_hardware().model_dump(mode="json"),
        "device": device,
        "kv_codec": kv_codec,
        "own_tuning": own_tuning,
        "rounds": rounds,
        "contexts": contexts,
        "builds": {side: built for side, (built, _) in builds.items()},
        "models": [model.name for model in selected],
        "results": [],
    }
    (directory / "run.json").write_text(json.dumps(plan, indent=2))
    store = root / "hosts" / host / f"{device}-{kv_codec}"
    thermals = ThermalRecorder(directory)
    with thermals:
        models = {
            model.name: model_set.fetch(workspace, root / "models", model, log)
            for model in selected
        }
        session = Run(directory, store, builds, models, device, kv_codec, contexts, log, plan)
        session.tune(SIDES if own_tuning else ("baseline",))
        session.measure(rounds, ("same", "own") if own_tuning else ("same",))
        session.logits()
    plan["thermals"] = thermals.summary
    plan["finished"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session.save()
    return directory
