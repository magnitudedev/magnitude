"""Portable, validated evidence records; importing never executes bundled code."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from .archive import RecordedConfiguration
from .characterization import Characterization
from .evidence import RunEvidence, fingerprint
from .records import Measurement, Record
from .store import ObservationStore


class SeriesLink(Record):
    configuration: str
    occurrence: int = Field(ge=0)
    series: str


class EvidenceBundle(Record):
    version: Literal[1] = 1
    configurations: tuple[RecordedConfiguration, ...]
    measurements: tuple[Measurement, ...]
    characterizations: tuple[Characterization, ...]
    links: tuple[SeriesLink, ...]
    runs: tuple[RunEvidence, ...]
    artifacts: dict[str, str] = Field(default_factory=dict)


def export_bundle(store: ObservationStore, destination: Path) -> None:
    connection = store._connection
    connection.execute("BEGIN")
    try:
        bundle = EvidenceBundle(
            configurations=store.configurations(),
            measurements=tuple(
                Measurement.model_validate_json(row[0])
                for row in connection.execute(
                    "SELECT record FROM formula_measurements ORDER BY sequence"
                )
            ),
            characterizations=tuple(
                Characterization.model_validate_json(row[0])
                for row in connection.execute(
                    "SELECT record FROM device_characterizations ORDER BY sequence"
                )
            ),
            links=tuple(
                SeriesLink(configuration=c, occurrence=o, series=s)
                for c, o, s in connection.execute(
                    "SELECT configuration,occurrence,series FROM formula_configuration_series"
                )
            ),
            runs=store.runs(),
        )
    finally:
        connection.rollback()
    value = bundle.model_dump(mode="json")
    for identity in {digest for m in bundle.measurements for digest in m.artifacts.values()}:
        try:
            value["artifacts"][identity] = base64.b64encode(store.artifact(identity)).decode(
                "ascii"
            )
        except FileNotFoundError:
            pass  # Evidence remains browsable without every optional source attachment.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump({"sha256": fingerprint(value), "evidence": value}, stream, allow_nan=False)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def import_bundle(store: ObservationStore, source: Path) -> int:
    envelope = json.loads(source.read_text())
    if fingerprint(envelope["evidence"]) != envelope["sha256"]:
        raise ValueError("evidence bundle checksum mismatch")
    bundle = EvidenceBundle.model_validate(envelope["evidence"])
    blobs = {
        identity: base64.b64decode(content, validate=True)
        for identity, content in bundle.artifacts.items()
    }
    if any(hashlib.sha256(content).hexdigest() != identity for identity, content in blobs.items()):
        raise ValueError("bundle artifact checksum mismatch")
    # Validate every cross-reference with the ordinary publication contracts before
    # touching the destination. Bundles are complete evidence closures, not scripts.
    with tempfile.TemporaryDirectory() as directory:
        with ObservationStore(Path(directory) / "validated.sqlite") as staged:
            for item in bundle.configurations:
                staged.publish_configuration(item)
            pending = list(bundle.measurements)
            profiles = list(bundle.characterizations)
            while pending or profiles:
                progress = False
                for item in pending[:]:
                    if (
                        item.roofline is None
                        or staged._connection.execute(
                            "SELECT 1 FROM device_characterizations WHERE identity=?",
                            (item.roofline.characterization,),
                        ).fetchone()
                    ):
                        staged.publish(item)
                        pending.remove(item)
                        progress = True
                for item in profiles[:]:
                    if all(
                        staged._connection.execute(
                            "SELECT 1 FROM formula_measurements WHERE identity=?",
                            (rate.measurement,),
                        ).fetchone()
                        for rate in item.rates
                    ):
                        staged.publish_characterization(item)
                        profiles.remove(item)
                        progress = True
                if not progress:
                    raise ValueError("bundle has missing or cyclic characterization evidence")
            configurations = {c.identity: c for c in bundle.configurations}
            series = {s.identity: s for s in staged.series()}
            for link in bundle.links:
                staged.link_series(
                    configurations[link.configuration], link.occurrence, series[link.series]
                )
            for run in bundle.runs:
                staged.publish_run(run)
            tables = {
                "formula_configurations": ("identity", "record"),
                "formula_series": ("identity", "record"),
                "formula_measurements": (
                    "identity",
                    "series",
                    "outcome",
                    "median_seconds",
                    "record",
                ),
                "device_characterizations": ("identity", "cache_key", "record"),
                "formula_configuration_series": ("configuration", "occurrence", "series"),
                "model_runs": ("identity", "model", "created", "record"),
            }
            inserted = 0
            connection = store._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                for table, columns in tables.items():
                    keys = columns[:2] if table == "formula_configuration_series" else columns[:1]
                    for row in staged._connection.execute(
                        f"SELECT {','.join(columns)} FROM {table}"
                    ):
                        where = " AND ".join(f"{key}=?" for key in keys)
                        prior = connection.execute(
                            f"SELECT {','.join(columns)} FROM {table} WHERE {where}",
                            row[: len(keys)],
                        ).fetchone()
                        if prior is not None:
                            if prior != row:
                                raise ValueError(f"immutable evidence conflict in {table}")
                            continue
                        connection.execute(
                            f"INSERT INTO {table}({','.join(columns)}) "
                            f"VALUES({','.join('?' for _ in columns)})",
                            row,
                        )
                        inserted += table == "model_runs"
                for content in blobs.values():
                    store.put_artifact(content)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
    return inserted
