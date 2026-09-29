"""Transactional append-only measurements with a stable formula-series index."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

from .records import History, JobResult, JobStart, Measurement, Outcome, Series, Visibility


def _enable_wal(connection: sqlite3.Connection) -> None:
    # Changing journal mode upgrades a schema/read lock. SQLite may reject a
    # competing upgrade immediately despite busy_timeout, rather than deadlock
    # two new connections. Retry only that known transient transition; once WAL
    # exists, opening a reader/worker does not request another mode change.
    deadline = monotonic() + 5
    while True:
        try:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            if mode != "wal":
                mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if mode != "wal":
                raise RuntimeError(f"formula observation store requires WAL mode, got {mode}")
            return
        except sqlite3.OperationalError as error:
            code = getattr(error, "sqlite_errorcode", 0) & 0xFF
            remaining = deadline - monotonic()
            if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or remaining <= 0:
                raise
            sleep(min(0.01, remaining))


class ObservationStore:
    """A connection belongs to its worker; UI readers open their own connection.

    Each successful publication atomically installs a complete validated record
    and its queryable index. Failure/cancellation never replaces a good record.
    """

    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = path
        self.read_only = read_only
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            path.resolve().as_uri() + "?mode=ro" if read_only else path,
            uri=read_only, timeout=0,
        )
        self._connection.create_function(
            "observed_at", 1, lambda timestamp: datetime.fromisoformat(timestamp).timestamp(),
            deterministic=True,
        )
        if read_only:
            self._connection.execute("PRAGMA query_only=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            return
        # WAL initialization owns its bounded retry deadline.
        try:
            _enable_wal(self._connection)
            self._connection.execute("PRAGMA busy_timeout=5000")
            self._connection.execute("PRAGMA foreign_keys=ON")
            version = self._connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"unsupported formula observation store version {version}")
            with self._connection:
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS formula_series (
                        identity TEXT PRIMARY KEY,
                        record TEXT NOT NULL
                    )
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS formula_measurements (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        identity TEXT NOT NULL UNIQUE,
                        series TEXT NOT NULL REFERENCES formula_series(identity),
                        outcome TEXT NOT NULL,
                        median_seconds REAL,
                        record TEXT NOT NULL
                    )
                """)
                self._connection.execute("""
                    CREATE INDEX IF NOT EXISTS measurement_history
                    ON formula_measurements(series, sequence DESC)
                """)
                self._connection.execute("""
                    CREATE INDEX IF NOT EXISTS measurement_best
                    ON formula_measurements(series, outcome, median_seconds)
                """)
                self._connection.execute("PRAGMA user_version=1")
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS formula_configurations (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        identity TEXT NOT NULL UNIQUE,
                        record TEXT NOT NULL
                    )
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS formula_configuration_series (
                        configuration TEXT NOT NULL REFERENCES formula_configurations(identity),
                        occurrence INTEGER NOT NULL,
                        series TEXT NOT NULL REFERENCES formula_series(identity),
                        PRIMARY KEY(configuration, occurrence)
                    )
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS device_characterizations (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        identity TEXT NOT NULL UNIQUE,
                        cache_key TEXT NOT NULL,
                        record TEXT NOT NULL
                    )
                """)
                self._connection.execute("""
                    CREATE INDEX IF NOT EXISTS characterization_key
                    ON device_characterizations(cache_key, sequence DESC)
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS formula_jobs (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        identity TEXT NOT NULL UNIQUE,
                        measurement TEXT REFERENCES formula_measurements(identity),
                        record TEXT NOT NULL
                    )
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS formula_job_starts (
                        identity TEXT PRIMARY KEY,
                        record TEXT NOT NULL
                    )
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS formula_visibility (
                        job TEXT NOT NULL REFERENCES formula_jobs(identity),
                        client TEXT NOT NULL,
                        record TEXT NOT NULL,
                        PRIMARY KEY(job, client)
                    )
                """)
                self._connection.execute("""
                    CREATE TABLE IF NOT EXISTS model_runs (
                        identity TEXT PRIMARY KEY,
                        model TEXT NOT NULL,
                        created TEXT NOT NULL,
                        record TEXT NOT NULL
                    )
                """)
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS model_history ON model_runs(model,created)")
        except BaseException:
            self._connection.close()
            raise

    def publish_configuration(self, configuration) -> None:
        from .archive import RecordedConfiguration

        payload = configuration.model_dump_json()
        record = RecordedConfiguration.model_validate_json(payload)
        with self._connection:
            existing = self._connection.execute("SELECT record FROM formula_configurations WHERE identity=?",
                                                (record.identity,)).fetchone()
            if existing is not None and existing[0] != payload:
                raise ValueError("recorded configuration identity is immutable")
            self._connection.execute("INSERT OR IGNORE INTO formula_configurations(identity, record) VALUES(?,?)",
                                     (record.identity, payload))

    def configurations(self):
        from .archive import RecordedConfiguration

        return tuple(RecordedConfiguration.model_validate_json(row[0]) for row in self._connection.execute(
            "SELECT record FROM formula_configurations ORDER BY sequence DESC"))

    def link_series(self, configuration, occurrence: int, series: Series) -> None:
        stored = self._connection.execute(
            "SELECT record FROM formula_configurations WHERE identity=?", (configuration.identity,),
        ).fetchone()
        if stored is None or configuration.model_validate_json(stored[0]) != configuration:
            raise ValueError("series requires the published immutable configuration")
        target = next((item for item in configuration.formulas if item.occurrence == occurrence), None)
        if target is None or (target.definition, target.semantics) != (series.formula, series.semantics):
            raise ValueError("series does not describe the recorded occurrence")
        if configuration.device != series.device:
            raise ValueError("series belongs to another physical configuration")
        with self._connection:
            prior = self._connection.execute(
                "SELECT series FROM formula_configuration_series WHERE configuration=? AND occurrence=?",
                (configuration.identity, occurrence)).fetchone()
            if prior is not None and prior[0] != series.identity:
                raise ValueError("prepared configuration input conditions changed")
            self._connection.execute(
                "INSERT OR IGNORE INTO formula_configuration_series(configuration,occurrence,series) VALUES(?,?,?)",
                (configuration.identity, occurrence, series.identity))

    def recorded_history(self, configuration, occurrence: int) -> History | None:
        row = self._connection.execute("""
            SELECT s.record FROM formula_configuration_series c
            JOIN formula_series s ON s.identity=c.series WHERE c.configuration=? AND c.occurrence=?
        """, (configuration.identity, occurrence)).fetchone()
        return self.history(Series.model_validate_json(row[0])) if row else None

    def recorded_overview(self, configuration):
        """One indexed scalar read, not parsing every observation to draw a tree."""
        return {row[0]: (row[1], row[2]) for row in self._connection.execute("""
            SELECT c.occurrence,
                (SELECT median_seconds FROM formula_measurements m
                 WHERE m.series=c.series AND m.outcome='complete' ORDER BY observed_at(json_extract(record, '$.created')) DESC, identity DESC LIMIT 1),
                (SELECT outcome FROM formula_measurements m
                 WHERE m.series=c.series ORDER BY observed_at(json_extract(record, '$.created')) DESC, identity DESC LIMIT 1)
            FROM formula_configuration_series c WHERE c.configuration=?
        """, (configuration.identity,))}

    def recorded_rooflines(self, configuration):
        """Read model summaries without deserializing every kernel/host sample."""
        from .records import Roofline

        rows = self._connection.execute("""
            SELECT c.occurrence,
                (SELECT json_extract(record, '$.roofline') FROM formula_measurements m
                 WHERE m.series=c.series AND m.outcome='complete' ORDER BY observed_at(json_extract(record, '$.created')) DESC, identity DESC LIMIT 1)
            FROM formula_configuration_series c WHERE c.configuration=?
        """, (configuration.identity,))
        return {occurrence: Roofline.model_validate_json(payload)
                for occurrence, payload in rows if payload is not None}

    def publish(self, measurement: Measurement) -> None:
        payload = measurement.model_dump_json()
        # Validate nested runtime dataclasses on ingress as well as on load.
        measurement = Measurement.model_validate_json(payload)
        if measurement.roofline is not None:
            from .characterization import Characterization

            row = self._connection.execute("SELECT record FROM device_characterizations WHERE identity=?",
                                           (measurement.roofline.characterization,)).fetchone()
            if row is None:
                raise ValueError("roofline refers to missing device characterization")
            profile = Characterization.model_validate_json(row[0])
            if (profile.device != measurement.series.device or
                    measurement.implementation is not None and profile.compiler != measurement.implementation.compiler):
                raise ValueError("roofline device/compiler differs from its measurement")
            for limit in measurement.roofline.limits:
                if not any(rate.measurement == limit.measurement and rate.value == limit.rate and
                           rate.dtype == limit.rate_dtype and rate.resource == limit.demand.resource and
                           rate.source == limit.demand.source and
                           rate.unit.dimension == f"{limit.demand.unit.dimension}/time"
                           for rate in profile.rates):
                    raise ValueError("roofline resource reference is not backed by its characterization")
        series = measurement.series
        series_payload = series.model_dump_json()
        with self._connection:
            existing = self._connection.execute(
                "SELECT record FROM formula_series WHERE identity=?", (series.identity,),
            ).fetchone()
            if existing is not None and Series.model_validate_json(existing[0]) != series:
                raise ValueError("formula-series identity collision")
            self._connection.execute(
                "INSERT OR IGNORE INTO formula_series(identity,record) VALUES(?,?)",
                (series.identity, series_payload),
            )
            existing = self._connection.execute(
                "SELECT record FROM formula_measurements WHERE identity=?", (measurement.identity,),
            ).fetchone()
            if existing is not None:
                if Measurement.model_validate_json(existing[0]) != measurement:
                    raise ValueError("measurement identities are immutable")
                return
            self._connection.execute(
                """INSERT INTO formula_measurements(identity,series,outcome,median_seconds,record)
                   VALUES(?,?,?,?,?)""",
                (measurement.identity, series.identity, measurement.outcome.value,
                 measurement.median_seconds, payload),
            )

    def history(self, series: Series, *, limit: int = 32) -> History:
        if limit < 1:
            raise ValueError("history limit must be positive")
        # Keep latest/success/best consistent if another worker publishes while
        # a UI reader assembles this snapshot. WAL readers don't block writers.
        self._connection.execute("BEGIN")
        try:
            rows = self._connection.execute(
                "SELECT record FROM formula_measurements WHERE series=? ORDER BY observed_at(json_extract(record, '$.created')) DESC, identity DESC LIMIT ?",
                (series.identity, limit),
            ).fetchall()
            latest_success = self._connection.execute(
                """SELECT record FROM formula_measurements WHERE series=? AND outcome=?
                   ORDER BY observed_at(json_extract(record, '$.created')) DESC, identity DESC LIMIT 1""",
                (series.identity, Outcome.COMPLETE.value),
            ).fetchone()
            best = self._connection.execute(
                """SELECT record FROM formula_measurements WHERE series=? AND outcome=?
                   ORDER BY median_seconds, observed_at(json_extract(record, '$.created')) DESC, identity DESC LIMIT 1""",
                (series.identity, Outcome.COMPLETE.value),
            ).fetchone()
            observations = tuple(Measurement.model_validate_json(row[0]) for row in rows)
            return History(
                series=series, latest=observations[0] if observations else None,
                latest_success=Measurement.model_validate_json(latest_success[0]) if latest_success else None,
                best=Measurement.model_validate_json(best[0]) if best else None,
                observations=observations,
            )
        finally:
            self._connection.rollback()

    def series(self) -> tuple[Series, ...]:
        return tuple(Series.model_validate_json(row[0]) for row in self._connection.execute(
            "SELECT record FROM formula_series ORDER BY identity",
        ))

    def characterization(self, key):
        from .characterization import Characterization

        row = self._connection.execute(
            "SELECT record FROM device_characterizations WHERE cache_key=? ORDER BY sequence DESC LIMIT 1",
            (key,),
        ).fetchone()
        return Characterization.model_validate_json(row[0]) if row else None

    def publish_characterization(self, characterization) -> None:
        from .characterization import Characterization

        payload = characterization.model_dump_json()
        record = Characterization.model_validate_json(payload)
        with self._connection:
            for rate in record.rates:
                row = self._connection.execute(
                    "SELECT record FROM formula_measurements WHERE identity=?", (rate.measurement,),
                ).fetchone()
                if row is None:
                    raise ValueError("device rate refers to missing measured evidence")
                measurement = Measurement.model_validate_json(row[0])
                metric = next((item for item in measurement.metrics if item.name == rate.metric), None)
                if (measurement.outcome != Outcome.COMPLETE or metric is None or
                        metric.value != rate.value or metric.unit != rate.unit or
                        measurement.series.device != record.device or
                        measurement.series.protocol != record.protocol.measurement or
                        measurement.implementation.compiler != record.compiler):
                    raise ValueError("device rate does not match its checked measurement")
                if rate.source is not None and not any(condition.source == rate.source
                                                       for condition in measurement.series.sources):
                    raise ValueError("source rate provenance differs from its measured source path")
            prior = self._connection.execute(
                "SELECT record FROM device_characterizations WHERE identity=?", (record.identity,),
            ).fetchone()
            if prior is not None:
                if prior[0] != payload:
                    raise ValueError("characterization identities are immutable")
                return
            self._connection.execute(
                "INSERT INTO device_characterizations(identity,cache_key,record) VALUES(?,?,?)",
                (record.identity, record.key, payload),
            )

    def publish_job(self, result: JobResult) -> None:
        payload = result.model_dump_json()
        result = JobResult.model_validate_json(payload)
        with self._connection:
            existing = self._connection.execute(
                "SELECT record FROM formula_jobs WHERE identity=?", (result.identity,),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload:
                    raise ValueError("job results are immutable")
                return
            self._connection.execute(
                "INSERT INTO formula_jobs(identity,measurement,record) VALUES(?,?,?)",
                (result.identity, result.measurement, payload),
            )

    def begin_job(self, start: JobStart) -> None:
        payload = start.model_dump_json()
        start = JobStart.model_validate_json(payload)
        with self._connection:
            self._connection.execute(
                "INSERT INTO formula_job_starts(identity,record) VALUES(?,?)", (start.identity, payload),
            )

    def unfinished_jobs(self) -> tuple[JobStart, ...]:
        """Explicit unfinished records; absence of completion is not success.

        A reader must not call another live worker's request 'crashed' merely
        because it is still unfinished. Worker identity is retained for diagnosis.
        """
        return tuple(JobStart.model_validate_json(row[0]) for row in self._connection.execute(
            """SELECT starts.record FROM formula_job_starts AS starts
               LEFT JOIN formula_jobs AS jobs ON jobs.identity=starts.identity
               WHERE jobs.identity IS NULL ORDER BY starts.rowid""",
        ))

    def publish_visibility(self, visibility: Visibility) -> None:
        payload = visibility.model_dump_json()
        visibility = Visibility.model_validate_json(payload)
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO formula_visibility(job,client,record) VALUES(?,?,?)",
                (visibility.job, visibility.client, payload),
            )

    def visibility(self, job: str) -> tuple[Visibility, ...]:
        return tuple(Visibility.model_validate_json(row[0]) for row in self._connection.execute(
            "SELECT record FROM formula_visibility WHERE job=? ORDER BY client", (job,),
        ))

    def jobs(self, *, limit: int = 32) -> tuple[JobResult, ...]:
        if limit < 1:
            raise ValueError("job history limit must be positive")
        return tuple(JobResult.model_validate_json(row[0]) for row in self._connection.execute(
            "SELECT record FROM formula_jobs ORDER BY sequence DESC LIMIT ?", (limit,),
        ))

    def close(self) -> None:
        self._connection.close()

    def measurement(self, identity: str) -> Measurement:
        row = self._connection.execute(
            "SELECT record FROM formula_measurements WHERE identity=?", (identity,)).fetchone()
        if row is None:
            raise KeyError(identity)
        return Measurement.model_validate_json(row[0])

    def put_artifact(self, content: bytes) -> str:
        if self.read_only:
            raise sqlite3.OperationalError("readonly evidence store")
        identity = hashlib.sha256(content).hexdigest()
        directory = self.path.parent / (self.path.name + ".artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / identity
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("artifact content identity conflict")
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            temporary.replace(path)
        return identity

    def artifact(self, identity: str) -> bytes:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("artifact identity must be a SHA256 digest")
        content = (self.path.parent / (self.path.name + ".artifacts") / identity).read_bytes()
        if hashlib.sha256(content).hexdigest() != identity:
            raise ValueError("artifact checksum mismatch")
        return content

    def publish_run(self, run) -> None:
        from .evidence import RunEvidence

        run = RunEvidence.model_validate_json(run.model_dump_json())
        for identity in run.measurements:
            measurement = self.measurement(identity)
            if run.scope.kind == "formula" and (
                measurement.series.formula != run.scope.formula
                or measurement.series.semantics != run.scope.semantics
                or measurement.series.fixture != run.context.workload.realization
            ):
                raise ValueError("run and formula measurement contract or boundary differ")
            if measurement.series.device != run.context.hardware:
                raise ValueError("run and formula measurement hardware differ")
            if (run.correctness == "passed" and not measurement.checked or
                    run.status == "complete" and measurement.observed_seconds is None):
                raise ValueError("run qualification exceeds its measured evidence")
        if run.configuration is not None and not self._connection.execute(
            "SELECT 1 FROM formula_configurations WHERE identity=?", (run.configuration,)
        ).fetchone():
            raise ValueError("run refers to missing configuration")
        payload = run.model_dump_json()
        with self._connection:
            prior = self._connection.execute(
                "SELECT record FROM model_runs WHERE identity=?", (run.identity,)).fetchone()
            if prior is not None and RunEvidence.model_validate_json(prior[0]) != run:
                raise ValueError("run identities are immutable")
            self._connection.execute(
                "INSERT OR IGNORE INTO model_runs(identity,model,created,record) VALUES(?,?,?,?)",
                (run.identity, run.context.model.identity, run.created.astimezone(UTC).isoformat(), payload))

    def runs(self, model: str | None = None):
        from .evidence import RunEvidence

        # Device-free readers may open historical stores that predate model indexing.
        if not self._connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='model_runs'"
        ).fetchone():
            return ()
        rows = self._connection.execute(
            "SELECT record FROM model_runs" + (" WHERE model=?" if model is not None else "")
            + " ORDER BY created DESC,identity", (model,) if model is not None else ())
        return tuple(RunEvidence.model_validate_json(row[0]) for row in rows)

    def models(self):
        return tuple({run.context.model.identity: run.context.model
                      for run in reversed(self.runs())}.values())

    def __enter__(self) -> ObservationStore:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
