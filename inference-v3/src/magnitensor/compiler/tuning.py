"""Immutable measured costs selected by structural workload identity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class TuningKey:
    region: str
    geometry: tuple[int, ...]
    representations: tuple[str, ...]
    precision: str
    capability: str
    compiler: str


@dataclass(frozen=True, slots=True)
class TuningRecord:
    key: TuningKey
    candidate: str
    latency_seconds: float
    parameters: tuple[tuple[str, int | float | str], ...] = ()

    def __post_init__(self) -> None:
        if self.latency_seconds <= 0:
            raise ValueError("tuning latency must be positive")


class TuningDatabase:
    def __init__(self, records: tuple[TuningRecord, ...] = ()) -> None:
        entries: dict[tuple[TuningKey, str], TuningRecord] = {}
        for record in records:
            identity = (record.key, record.candidate)
            if identity in entries:
                raise ValueError(f"duplicate tuning record for {identity!r}")
            entries[identity] = record
        self._records = MappingProxyType(entries)

    def lookup(self, key: TuningKey, candidate: str) -> TuningRecord | None:
        return self._records.get((key, candidate))

    def candidates(self, key: TuningKey) -> tuple[TuningRecord, ...]:
        return tuple(record for (stored, _), record in self._records.items() if stored == key)

    @classmethod
    def load(cls, path: Path) -> TuningDatabase:
        payload = json.loads(path.read_text())
        records = []
        for item in payload:
            key = TuningKey(**item["key"])
            records.append(
                TuningRecord(
                    key,
                    item["candidate"],
                    item["latency_seconds"],
                    tuple(tuple(pair) for pair in item.get("parameters", ())),
                )
            )
        return cls(tuple(records))

    def save(self, path: Path) -> None:
        payload = []
        for record in sorted(
            self._records.values(), key=lambda item: (repr(item.key), item.candidate)
        ):
            item = asdict(record)
            item["parameters"] = list(record.parameters)
            payload.append(item)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


EMPTY_TUNING = TuningDatabase()
