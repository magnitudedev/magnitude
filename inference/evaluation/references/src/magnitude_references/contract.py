"""The evaluation contract formats a reference reads and writes (`specs/26-10-08/evaluation-contracts.md`):
rendered fixtures (harness `render` output), the catalog acquisition object, and logits outputs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from magnitude_references.digest import sha256_file

RENDERED_KEYS = ("fixture_id", "fixture_set_digest", "model", "target_sha256", "template_sha256", "context_target",
                 "tokens", "scored_spans", "generation_prefix_tokens")


class ContractError(ValueError):
    """An input does not conform to the evaluation contracts."""


@dataclass(frozen=True)
class RenderedFixture:
    fixture_id: str
    fixture_set_digest: str
    model: str
    target_sha256: str
    template_sha256: str
    tokens: tuple[int, ...]
    scored_spans: tuple[tuple[int, int], ...]

    def positions(self) -> list[int]:
        """The rows a producer emits, in span order: positions start-1 .. end-2 of each span. The row
        at position p is the output after tokens[0..=p], predicting tokens[p+1], so every scored token
        i is scored by the row at i-1."""
        return [position for start, end in self.scored_spans for position in range(start - 1, end - 1)]

    def targets(self) -> list[int]:
        """The scored tokens, aligned with `positions()`: tokens[p + 1] for each row position p."""
        return [self.tokens[position + 1] for position in self.positions()]


@dataclass(frozen=True)
class RenderedSet:
    path: Path
    digest: str  # SHA-256 of the file bytes
    fixtures: tuple[RenderedFixture, ...]

    @property
    def model(self) -> str:
        return self.fixtures[0].model

    @property
    def fixture_set_digest(self) -> str:
        return self.fixtures[0].fixture_set_digest

    @property
    def template_sha256(self) -> str:
        return self.fixtures[0].template_sha256


def parse_rendered(record: dict, line: int) -> RenderedFixture:
    missing = [key for key in RENDERED_KEYS if key not in record]
    if missing:
        raise ContractError(f"rendered fixture line {line}: missing {missing}")
    tokens = record["tokens"]
    if not tokens or not all(type(token) is int and token >= 0 for token in tokens):
        raise ContractError(f"line {line}: tokens must be a non-empty list of non-negative integers")
    spans = []
    previous_end = 1  # token 0 has no predicting row
    for span in record["scored_spans"]:
        if not (isinstance(span, list) and len(span) == 2 and all(type(v) is int for v in span)):
            raise ContractError(f"line {line}: scored span {span!r} is not [start, end)")
        start, end = span
        if not previous_end <= start < end <= len(tokens):
            raise ContractError(f"line {line}: scored span [{start}, {end}) is empty, starts at token 0, is out of "
                                f"order or overlapping, or lies outside {len(tokens)} tokens")
        spans.append((start, end))
        previous_end = end
    if not spans:
        raise ContractError(f"line {line}: fixture {record['fixture_id']!r} has no scored span")
    return RenderedFixture(record["fixture_id"], record["fixture_set_digest"], record["model"],
                           record["target_sha256"], record["template_sha256"], tuple(tokens), tuple(spans))


def read_rendered(path: Path) -> RenderedSet:
    """A rendered-fixture JSON Lines file; every fixture must share model, target, template and set."""
    path = Path(path)
    fixtures = []
    with path.open("r", encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                fixtures.append(parse_rendered(json.loads(text), line))
    if not fixtures:
        raise ContractError(f"{path}: no rendered fixtures")
    first = fixtures[0]
    for fixture in fixtures:
        for field in ("model", "target_sha256", "template_sha256", "fixture_set_digest"):
            if getattr(fixture, field) != getattr(first, field):
                raise ContractError(f"{path}: fixture {fixture.fixture_id!r} has a different {field}")
    identifiers = [fixture.fixture_id for fixture in fixtures]
    if len(set(identifiers)) != len(identifiers):
        raise ContractError(f"{path}: duplicate fixture ids")
    return RenderedSet(path, sha256_file(path), tuple(fixtures))


@dataclass(frozen=True)
class Artifacts:
    """The catalog acquisition object: the resolved local files of one catalog model."""
    model: str
    target: Path
    target_sha256: str
    record: dict  # the object as read, recorded verbatim in result records


def read_artifacts(path: Path) -> Artifacts:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [key for key in ("model", "target", "target_sha256", "drafter", "projector") if key not in record]
    if missing:
        raise ContractError(f"{path}: catalog acquisition object lacks {missing}")
    return Artifacts(record["model"], Path(record["target"]), record["target_sha256"], record)


def check_match(rendered: RenderedSet, artifacts: Artifacts) -> None:
    """The rendered fixtures were rendered for exactly this catalog model and target file."""
    if rendered.model != artifacts.model:
        raise ContractError(f"rendered for model {rendered.model!r}, artifacts are {artifacts.model!r}")
    if rendered.fixtures[0].target_sha256 != artifacts.target_sha256:
        raise ContractError("rendered target_sha256 differs from the acquired target's")


class LogitsWriter:
    """A logits output: `<name>.f32` (little-endian float32 rows of `vocabulary`) and `<name>.json`."""

    def __init__(self, directory: Path, name: str, producer: str, model: str, rendered_digest: str, vocabulary: int):
        self.directory, self.name = Path(directory), name
        self.header = {"producer": producer, "model": model, "rendered_digest": rendered_digest,
                       "vocabulary": vocabulary}
        self.rows: list[dict] = []
        self.stream = (self.directory / f"{name}.f32").open("wb")

    def write(self, fixture_id: str, positions: list[int], logits: np.ndarray) -> None:
        """Rows for `positions` of a fixture, in emission order; logits [len(positions), vocabulary]."""
        if logits.shape != (len(positions), self.header["vocabulary"]):
            raise ValueError(f"logits {logits.shape} for {len(positions)} rows of {self.header['vocabulary']}")
        np.ascontiguousarray(logits, dtype="<f4").tofile(self.stream)
        self.rows.extend({"fixture_id": fixture_id, "position": position} for position in positions)

    def close(self) -> None:
        self.stream.close()
        (self.directory / f"{self.name}.json").write_text(json.dumps(self.header | {"rows": self.rows}) + "\n")


def read_logits(directory: Path, name: str) -> tuple[dict, np.ndarray]:
    """A logits output's header and its rows as a read-only float32 map [rows, vocabulary]."""
    header = json.loads((Path(directory) / f"{name}.json").read_text())
    values = np.memmap(Path(directory) / f"{name}.f32", dtype="<f4", mode="r")
    return header, values.reshape(len(header["rows"]), header["vocabulary"])
