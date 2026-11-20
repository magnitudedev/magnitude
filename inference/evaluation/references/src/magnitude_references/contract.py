"""The evaluation contract formats a reference reads and writes:
rendered sessions (harness `render` output: each lane of a template, sized and tokenized), the catalog
acquisition object, and logits outputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from magnitude_rig.host.identity import sha256

RENDERED_KEYS = ("template", "lane", "model", "target_sha256", "chat_template_sha256", "request_options", "context",
                 "history", "tokens", "prompt_sha256", "scored_spans", "generation_prefix_tokens")


class ContractError(ValueError):
    """An input does not conform to the evaluation contracts."""


@dataclass(frozen=True)
class RenderedSession:
    """One lane of a template as rendered for one model."""
    template: str
    lane: int
    model: str
    target_sha256: str
    chat_template_sha256: str
    prompt_sha256: str
    tokens: tuple[int, ...]
    scored_spans: tuple[tuple[int, int], ...]

    @property
    def name(self) -> str:
        """`<template>/<lane>`: the session's name in messages, keys and file names."""
        return f"{self.template}/{self.lane}"

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
    """The rendered lanes of one template."""
    path: Path
    digest: str  # SHA-256 of the file bytes
    sessions: tuple[RenderedSession, ...]

    @property
    def model(self) -> str:
        return self.sessions[0].model

    @property
    def template(self) -> str:
        return self.sessions[0].template

    @property
    def chat_template_sha256(self) -> str:
        return self.sessions[0].chat_template_sha256

    @property
    def prompt_sha256(self) -> str:
        """The set's prompt digest: SHA-256 of its lanes' prompt digests, one per line in file order."""
        return hashlib.sha256("\n".join(session.prompt_sha256 for session in self.sessions).encode()).hexdigest()


def parse_rendered(record: dict, line: int) -> RenderedSession:
    missing = [key for key in RENDERED_KEYS if key not in record]
    if missing:
        raise ContractError(f"rendered session line {line}: missing {missing}")
    if type(record["lane"]) is not int or record["lane"] < 0:
        raise ContractError(f"line {line}: lane must be a non-negative integer")
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
        raise ContractError(f"line {line}: session {record['template']}/{record['lane']} has no scored span")
    return RenderedSession(record["template"], record["lane"], record["model"], record["target_sha256"],
                           record["chat_template_sha256"], record["prompt_sha256"], tuple(tokens), tuple(spans))


def read_rendered(path: Path) -> RenderedSet:
    """A rendered-session JSON Lines file: the lanes of one template, sharing model, target and chat template."""
    path = Path(path)
    sessions = []
    with path.open("r", encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                sessions.append(parse_rendered(json.loads(text), line))
    if not sessions:
        raise ContractError(f"{path}: no rendered sessions")
    first = sessions[0]
    for session in sessions:
        for field in ("template", "model", "target_sha256", "chat_template_sha256"):
            if getattr(session, field) != getattr(first, field):
                raise ContractError(f"{path}: session {session.name!r} has a different {field}")
    lanes = [session.lane for session in sessions]
    if len(set(lanes)) != len(lanes):
        raise ContractError(f"{path}: duplicate lanes")
    return RenderedSet(path, sha256(path), tuple(sessions))


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
    """The rendered sessions were rendered for exactly this catalog model and target file."""
    if rendered.model != artifacts.model:
        raise ContractError(f"rendered for model {rendered.model!r}, artifacts are {artifacts.model!r}")
    if rendered.sessions[0].target_sha256 != artifacts.target_sha256:
        raise ContractError("rendered target_sha256 differs from the acquired target's")


LOGITS = "logits"  # the stem of a reference's or harness's logits output
CHUNK_ROWS = 32  # rows of a logits map read at once, bounding the memory a pass over it holds


class LogitsWriter:
    """A logits output: `<name>.f32` (little-endian float32 rows of `vocabulary`) and `<name>.json`."""

    def __init__(self, directory: Path, name: str, producer: str, model: str, rendered_digest: str, vocabulary: int):
        self.directory, self.name = Path(directory), name
        self.header = {"producer": producer, "model": model, "rendered_digest": rendered_digest,
                       "vocabulary": vocabulary}
        self.rows: list[dict] = []
        self.stream = (self.directory / f"{name}.f32").open("wb")

    def write(self, session: RenderedSession, positions: list[int], logits: np.ndarray) -> None:
        """Rows for `positions` of a session, in emission order; logits [len(positions), vocabulary]."""
        if logits.shape != (len(positions), self.header["vocabulary"]):
            raise ValueError(f"logits {logits.shape} for {len(positions)} rows of {self.header['vocabulary']}")
        np.ascontiguousarray(logits, dtype="<f4").tofile(self.stream)
        self.rows.extend({"template": session.template, "lane": session.lane, "position": position}
                         for position in positions)

    def close(self) -> None:
        self.stream.close()
        (self.directory / f"{self.name}.json").write_text(json.dumps(self.header | {"rows": self.rows}) + "\n")


def read_logits(directory: Path, name: str) -> tuple[dict, np.ndarray]:
    """A logits output's header and its rows as a read-only float32 map [rows, vocabulary]."""
    header = json.loads((Path(directory) / f"{name}.json").read_text())
    values = np.memmap(Path(directory) / f"{name}.f32", dtype="<f4", mode="r")
    return header, values.reshape(len(header["rows"]), header["vocabulary"])
