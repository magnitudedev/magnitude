"""What one comparison measures: the two sides, the inputs and the measurement settings.

The controller writes the plan (`plan.json`) into the comparison's inputs; the host runner
reads it. It holds no host paths, so the same plan runs on any host or locally.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from magnitude_remote.build import BuildIdentity

SIDES = ("base", "candidate")
Side = Literal["base", "candidate"]
WORKTREE = "worktree"
TuningMode = Literal["fresh", "replay-base"]


@dataclass(frozen=True)
class Build:
    """One side: its source (`worktree` or a git ref) and its contract build identity (a ref
    has no uncommitted digest)."""

    source: str
    identity: BuildIdentity

    @property
    def worktree(self) -> bool:
        return self.source == WORKTREE

    @property
    def commit(self) -> str:
        return self.identity.commit

    @classmethod
    def from_json(cls, data: dict) -> "Build":
        return cls(data["source"], BuildIdentity(**data["identity"]))


@dataclass(frozen=True)
class Tuning:
    """`fresh`: each side tunes afresh with the same budget on its first engine run and reuses
    that tuning afterwards. `replay-base`: the base's first run records its tuned configurations
    and every later run of either side replays them, so speed compares kernel code alone."""

    seconds: float
    mode: TuningMode


@dataclass(frozen=True)
class Plan:
    builds: dict[str, Build]
    # The working tree that supplies the measurement tools (harness sources, references,
    # this package) on every host, whichever sides are compared.
    tools: BuildIdentity
    # The prebuilt fixture sets (`magnitude-fixtures build`): the controller's directory of
    # `<family>.jsonl`, and each measured family's set digest (SHA-256 of its bytes), in
    # measurement order. References are keyed by the rendering of exactly these sets.
    fixtures: str
    fixture_sets: dict[str, str]
    contexts: tuple[int, ...]
    concurrency: tuple[int, ...]
    rounds: int
    tuning: Tuning
    quality: bool
    speed: bool
    quality_bar: float
    speed_bar: float
    command: tuple[str, ...]

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(self.fixture_sets)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Plan":
        return cls(
            builds={side: Build.from_json(data["builds"][side]) for side in SIDES},
            tools=BuildIdentity(**data["tools"]),
            fixtures=data["fixtures"],
            fixture_sets=dict(data["fixture_sets"]),
            contexts=tuple(data["contexts"]),
            concurrency=tuple(data["concurrency"]),
            rounds=data["rounds"],
            tuning=Tuning(**data["tuning"]),
            quality=data["quality"],
            speed=data["speed"],
            quality_bar=data["quality_bar"],
            speed_bar=data["speed_bar"],
            command=tuple(data["command"]),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def read(cls, path: Path) -> "Plan":
        return cls.from_json(json.loads(path.read_text()))
