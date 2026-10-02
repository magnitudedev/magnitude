"""Alternating baseline/candidate measurement of one host.

A run has three phases.

- Tune: each build loads each model with an empty kernel cache `tunes` times, as a user's first
  load would, recording the configuration it chose for every entry. The report compares choices
  between builds and between a build's own tunes.
- Measure: `rounds` rounds of every model on both builds, in alternating order (baseline first on
  even rounds, candidate first on odd ones) so drift such as heat affects both sides alike. Every
  round replays fixed configurations twice: the baseline's first tune on both sides, which
  compares kernel code, and each side's own first tune, which is what a fresh load runs.
- Logits: once per build and model, on the same configurations, the logits of a long prompt
  (prefilled in chunks, so later chunks attend to history) followed by single-row decodes,
  compared across builds.
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

PREFILL_ROWS = (512, 4096)
DECODE_CONTEXTS = (256, 4096, 16384)
HISTORIES = (4096, 16384)
LOGITS_PREFILL = 2048
LOGITS_DECODES = 16
TOKEN_RANGE = (100, 20_000)
RUN_TIMEOUT_SECONDS = 3600
SIDES = ("baseline", "candidate")


def invocations(histories: tuple[int, ...]) -> list[tuple[str, list[str]]]:
    """The forward_bench invocations of one model on one build; the first is the tuning one."""
    base = [
        "--cells",
        "prefill,decode",
        "--prefill",
        ",".join(map(str, PREFILL_ROWS)),
        "--context",
        ",".join(map(str, DECODE_CONTEXTS)),
    ]
    with_history = [
        (
            f"history-{history}",
            ["--cells", "prefill", "--prefill", "512", "--prefill-history", str(history)],
        )
        for history in histories
    ]
    return [("base", base), *with_history]


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


def pin_key(pin: dict) -> tuple:
    return pin["entry"], pin["bindings"], json.dumps(pin["statics"], sort_keys=True)


@dataclass
class Run:
    directory: Path
    builds: dict[str, tuple[dict, Path]]
    models: dict[str, Path]
    device: str
    kv_codec: str
    log: object
    plan: dict = field(default_factory=dict)

    def save(self):
        (self.directory / "run.json").write_text(json.dumps(self.plan, indent=2))

    def tool(self, side: str, name: str) -> str:
        return str(self.builds[side][1] / executable(name))

    def common(self, model: str, cache: str) -> list[str]:
        return [
            "--model",
            str(self.models[model]),
            "--cache-dir",
            str(self.directory / "caches" / cache),
            "--device",
            self.device,
            "--kv-codec",
            self.kv_codec,
        ]

    def invoke(self, command: list[str], name: str, **record) -> bool:
        self.log(" ".join(f"{key}={value}" for key, value in record.items()))
        outcome = execute(command, self.directory / f"{name}.log")
        self.plan["results"].append({**record, **outcome})
        self.save()
        if outcome["status"] != "ok":
            self.log(f"  {outcome['status']}; see {outcome['log']}")
        return outcome["status"] == "ok"

    def tune(self, tunes: int, base: list[str]):
        for index in range(tunes):
            for model in self.models:
                for side in SIDES if index % 2 == 0 else SIDES[::-1]:
                    name = f"tune{index}-{side}-{model}"
                    self.invoke(
                        [self.tool(side, "forward_bench"), "bench", *self.common(model, name)]
                        + ["--output", str(self.directory / f"{name}.json"), *base]
                        + ["--tuning-record", str(self.directory / f"{name}-pins.json")],
                        name,
                        phase="tune",
                        round=index,
                        side=side,
                        model=model,
                        invocation="base",
                        output=f"{name}.json",
                        pins=f"{name}-pins.json",
                    )

    def shared_pins(self, model: str) -> Path | None:
        """The baseline's first-tune configurations, as both sides replay them: an entry
        parameter the baseline lacks (a parameter the candidate added) takes the candidate's
        own first-tune value, and one the candidate lacks is dropped."""
        path = self.directory / f"shared-{model}-pins.json"
        sources = [self.directory / f"tune0-{side}-{model}-pins.json" for side in SIDES]
        if not all(source.is_file() for source in sources):
            return None
        baseline, candidate = (json.loads(source.read_text()) for source in sources)
        own = {pin_key(pin): pin for pin in candidate}
        shared = []
        for pin in baseline:
            mine = own.pop(pin_key(pin), None)
            if mine is not None:
                for field_name in ("params", "launch_params"):
                    pin[field_name] = {
                        name: pin[field_name].get(name, value)
                        for name, value in mine[field_name].items()
                    }
            shared.append(pin)
        path.write_text(json.dumps(shared + list(own.values()), indent=2))
        return path

    def measure(self, rounds: int, histories: tuple[int, ...]):
        """Each round measures both sides twice: on the same configurations (the baseline's
        tuning), which compares kernel code, and each on its own first tune, which is what a
        fresh load of that build runs."""
        shared = {model: self.shared_pins(model) for model in self.models}
        for index in range(rounds):
            for model in self.models:
                for mode in ("same", "own"):
                    for side in SIDES if index % 2 == 0 else SIDES[::-1]:
                        own = self.directory / f"tune0-{side}-{model}-pins.json"
                        # The shared set is the baseline's own tune, extended for the candidate.
                        pins = shared[model] if mode == "same" and side == "candidate" else own
                        if pins is None or not pins.is_file():
                            continue
                        for invocation, arguments in invocations(histories):
                            name = f"r{index}-{mode}-{side}-{model}-{invocation}"
                            ok = self.invoke(
                                [self.tool(side, "forward_bench"), "bench"]
                                + self.common(model, f"tune0-{side}-{model}")
                                + ["--output", str(self.directory / f"{name}.json"), *arguments]
                                + ["--tuning-replay", str(pins)],
                                name,
                                phase="measure",
                                mode=mode,
                                round=index,
                                side=side,
                                model=model,
                                invocation=invocation,
                                output=f"{name}.json",
                            )
                            if not ok:
                                break

    def logits(self):
        """Both sides on the same configurations, so any difference is the code's."""
        tokens = ",".join(map(str, logits_tokens()))
        for model in self.models:
            for side in SIDES:
                pins = self.directory / (
                    f"shared-{model}-pins.json"
                    if side == "candidate"
                    else f"tune0-baseline-{model}-pins.json"
                )
                name = f"logits-{side}-{model}"
                replay = ["--tuning-replay", str(pins)] if pins.is_file() else []
                self.invoke(
                    [self.tool(side, "token_logits"), *self.common(model, f"tune0-{side}-{model}")]
                    + ["--tokens", tokens, "--output", str(self.directory / f"{name}.f32")]
                    + ["--prefill", str(LOGITS_PREFILL), *replay],
                    name,
                    phase="logits",
                    side=side,
                    model=model,
                    invocation="logits",
                    output=f"{name}.f32",
                )


def run(
    workspace: Path,
    root: Path,
    builds: dict[str, tuple[dict, Path]],
    selected: tuple[model_set.BenchModel, ...],
    tunes: int,
    rounds: int,
    device: str,
    kv_codec: str,
    histories: tuple[int, ...],
    log,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "runs" / f"{stamp}-{platform.node().split('.')[0]}"
    directory.mkdir(parents=True)
    plan = {
        "started": stamp,
        "host": platform.node(),
        "platform": platform.platform(),
        "hardware": capture_hardware().model_dump(mode="json"),
        "device": device,
        "kv_codec": kv_codec,
        "tunes": tunes,
        "rounds": rounds,
        "histories": histories,
        "builds": {side: built for side, (built, _) in builds.items()},
        "models": [model.name for model in selected],
        "results": [],
    }
    (directory / "run.json").write_text(json.dumps(plan, indent=2))
    thermals = ThermalRecorder(directory)
    with thermals:
        models = {
            model.name: model_set.fetch(workspace, root / "models", model, log)
            for model in selected
        }
        session = Run(directory, builds, models, device, kv_codec, log, plan)
        session.tune(tunes, invocations(histories)[0][1])
        session.measure(rounds, histories)
        session.logits()
    plan["thermals"] = thermals.summary
    plan["finished"] = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session.save()
    return directory
