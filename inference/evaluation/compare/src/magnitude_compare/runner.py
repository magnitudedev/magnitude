"""The host side of a comparison: run the schedule, judge it, and write the result record.

Runs on the evaluated host (or locally) as the remote job, after both builds and the model are
in place and the thermal gate passed. Every harness run writes its own result record into the
comparison's run directory; the comparison consumes those records and the logits outputs. A
failed step is recorded and the rest still run, so one failure does not hide other evidence.

Logits outputs of the builds are deleted once judged (they are gigabytes); references stay in
the host's reference cache.
"""

import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from magnitude_references.digest import canonical, sha256_file

from . import quality, rendering, report, speed
from .plan import SIDES, Plan
from .reference import ReferenceFailed, ReferenceRefused, check, obtain
from .steps import TOOLS, EngineRun, HostPaths, QualityCell, schedule

RESULT = "result.json"
REPORT = "report.md"
TUNING_TRACE = "tuning.jsonl"
EXIT = {"pass": 0, "failed": 1, "regressed": 2}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _progress(message: str) -> None:
    print(f"magnitude-compare: {message}", file=sys.stderr, flush=True)


class Runner:
    def __init__(self, plan: Plan, paths: HostPaths):
        self.plan = plan
        self.paths = paths
        self.run_dir = Path(paths.run_dir)
        self.failures: list[dict] = []
        self.tuning: list[dict] = []

    def fail(self, step: str, error: str, log: Path | None = None) -> None:
        _progress(f"{step}: {error}")
        failure = {"step": step, "error": error}
        if log is not None:
            failure["log"] = str(log.relative_to(self.run_dir))
        self.failures.append(failure)

    def engine(self, run: EngineRun) -> dict | None:
        """Run one harness invocation; its result record, or None when it failed."""
        directory = Path(self.paths.run(run.name))
        directory.mkdir(parents=True, exist_ok=True)
        log = directory / "log.txt"
        _progress(f"{run.name}: {shlex.join(run.argv)}")
        with log.open("w") as stream:
            code = subprocess.run(run.argv, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code != 0:
            self.fail(run.name, f"magnitude-harness exited {code}", log)
            return None
        record = json.loads((directory / RESULT).read_text())
        if not run.tunes:
            return record
        self.tuning.append({"run": run.name, "side": run.side, **record["measurements"]["tuning"]})
        with (self.run_dir / TUNING_TRACE).open("a") as trace:
            for line in (directory / record["traces"]["tuning"]).read_text().splitlines():
                trace.write(json.dumps({"run": run.name, **json.loads(line)}) + "\n")
        return record

    def measure_speed(
        self, runs: tuple[EngineRun, ...]
    ) -> tuple[list[speed.Sample], list, list[dict]]:
        """Samples, drafting measurements, and the rounds whose builds timed different tokens."""
        samples, drafting = [], []
        rendered: dict[tuple[str, int], dict[str, Path]] = {}
        for run in runs:
            record = self.engine(run)
            if record is None:
                continue
            measurements = record["measurements"]
            samples.extend(
                speed.Sample(run.side, run.round, run.family, cell)
                for cell in measurements["speed"]
            )
            if "drafting" in measurements:
                drafting.append((run.side, run.family, measurements["drafting"]))
            directory = Path(self.paths.run(run.name))
            rendered.setdefault((run.family, run.round), {})[run.side] = (
                directory / rendering.RENDERED
            )
        differing = []
        for (family, round_index), files in rendered.items():
            if len(files) < len(SIDES):
                continue
            differences = rendering.compare_files(files["base"], files["candidate"])
            if differences:
                differing.append(
                    {"family": family, "round": round_index, "differences": differences}
                )
                self.fail(
                    f"speed/r{round_index}-{family}",
                    "the builds render different tokens (template or tokenizer behaviour changed), "
                    "so their timings are of different input",
                )
        return samples, drafting, differing

    def measure_quality(self, cell: QualityCell, artifacts: Path) -> tuple[list[dict], dict]:
        """The cell's quality measurements and its comparison row. The candidate is measured on
        the base's tokens; its own rendering is compared separately, as a finding."""
        row: dict = {"family": cell.family, "context": cell.context}
        directories = {side: Path(self.paths.run(cell.runs[side].name)) for side in SIDES}
        base = self.engine(cell.runs["base"])
        if base is None:
            return [], {**row, "skipped": "the base logits run failed"}
        rendered = directories["base"] / rendering.RENDERED
        own = self.engine(cell.render)
        if own is not None:
            # A change in template or tokenizer behaviour: reported, not a failure, since the
            # candidate is still measured on the base's exact tokens.
            row["rendering"] = rendering.compare_files(
                rendered, Path(self.paths.run(cell.render.name)) / rendering.RENDERED
            )
        candidate = self.engine(cell.runs["candidate"])
        try:
            return self._judge_quality(
                cell, row, base, candidate is not None, directories, artifacts
            )
        finally:
            for directory in directories.values():
                for logits in directory.glob("*.f32"):
                    logits.unlink()

    def _judge_quality(
        self, cell, row, base, candidate, directories, artifacts
    ) -> tuple[list[dict], dict]:
        rendered = directories["base"] / rendering.RENDERED
        digest = base["rendered_digest"]
        row["rendered_digest"] = digest
        positions: dict[str, quality.Positions] = {}
        reference = None
        try:
            log = Path(self.paths.run(cell.name)) / "reference.log"
            reference = obtain(
                rendered,
                artifacts,
                Path(self.paths.identity(TOOLS)),
                Path(self.paths.references),
                log,
            )
            row["reference"] = reference.to_json()
            check(reference)
        except (ReferenceFailed, ReferenceRefused) as error:
            self.fail(f"{cell.name}/reference", str(error))
            reference = None
        try:
            if reference is not None:
                positions["base-reference"] = quality.compare_outputs(
                    reference.directory, directories["base"]
                )
                if candidate:
                    positions["candidate-reference"] = quality.compare_outputs(
                        reference.directory, directories["candidate"]
                    )
            if candidate:
                positions["candidate-base"] = quality.compare_outputs(
                    directories["base"], directories["candidate"]
                )
        except quality.Incomparable as error:
            self.fail(cell.name, str(error))
        measurements = [
            {
                "family": cell.family,
                "context": cell.context,
                "pair": pair,
                "rendered_digest": digest,
                **quality.summarize(values.kl, values.agree),
                "worst": quality.worst(values),
            }
            for pair, values in positions.items()
        ]
        if "base-reference" in positions and "candidate-reference" in positions:
            row["changes"] = [
                change.to_json()
                for change in quality.judge(
                    positions["base-reference"],
                    positions["candidate-reference"],
                    self.plan.quality_bar,
                )
            ]
        if "candidate-base" in positions:
            builds = quality.summarize(
                positions["candidate-base"].kl, positions["candidate-base"].agree
            )
            row["builds_identical"] = builds["max_kl"] == 0.0 and builds["top1_agreement"] == 1.0
        return measurements, row

    def run(self) -> int:
        started = _now()
        artifacts_path = Path(self.paths.artifacts)
        artifacts = json.loads(artifacts_path.read_text())
        for family, digest in self.plan.fixture_sets.items():
            shipped = sha256_file(Path(self.paths.fixtures(family)))
            if shipped != digest:
                raise ValueError(
                    f"the {family} fixture set on this host is {shipped}, not {digest}"
                )
        plan = schedule(self.plan, self.paths)
        samples, drafting, speed_rendering = self.measure_speed(plan.speed)
        measured_quality, quality_rows = [], []
        for cell in plan.quality:
            measurements, row = self.measure_quality(cell, artifacts_path)
            measured_quality.extend(measurements)
            quality_rows.append(row)
        speed_rows = speed.compare(samples, self.plan.speed_bar) if samples else []
        drafting_rows = speed.compare_drafting(drafting, self.plan.speed_bar) if drafting else []
        for row in [*speed_rows, *drafting_rows]:
            if row["verdict"] == "missing":
                self.fail("speed", f"{_cell_name(row)} was measured on the {row['only']} only")
        regressed = any(
            row["verdict"] == "regressed" for row in [*speed_rows, *drafting_rows]
        ) or any(
            change["verdict"] == "regressed"
            for row in quality_rows
            for change in row.get("changes", [])
        )
        verdict = "failed" if self.failures else "regressed" if regressed else "pass"
        measurements: dict = {}
        if samples:
            measurements["speed"] = [
                {
                    "side": s.side,
                    "round": s.round,
                    "family": s.family,
                    **{
                        k: s.cell[k]
                        for k in ("phase", "context", "concurrency", "tokens_per_second", "samples")
                    },
                }
                for s in samples
            ]
        if drafting:
            measurements["drafting"] = [
                {"side": side, "family": family, **m} for side, family, m in drafting
            ]
        if measured_quality:
            measurements["quality"] = measured_quality
        if self.tuning:
            measurements["tuning"] = self.tuning
        measurements["comparison"] = {
            "verdict": verdict,
            "quality": quality_rows,
            "speed": speed_rows,
            "speed_rendering": speed_rendering,
            "drafting": drafting_rows,
            "failures": self.failures,
            "bars": {"quality": self.plan.quality_bar, "speed": self.plan.speed_bar},
            "tuning_mode": self.plan.tuning.mode,
            "rounds": self.plan.rounds,
        }
        sets = [
            {"family": family, "path": f"{self.plan.fixtures}/{family}.jsonl", "digest": digest}
            for family, digest in self.plan.fixture_sets.items()
        ]
        record = {
            "schema": "evaluation-result/1",
            "tool": "compare",
            "started_at": started,
            "finished_at": _now(),
            "build": self.plan.builds["candidate"].identity.to_json(),
            "sides": {
                side: {
                    "source": build.source,
                    "build": build.identity.to_json(),
                    "harness": self.plan.tools.to_json(),
                }
                for side, build in self.plan.builds.items()
            },
            "model": artifacts["model"],
            "artifacts": artifacts,
            "fixture_set": {
                "path": self.paths.inputs,
                "digest": canonical({entry["family"]: entry["digest"] for entry in sets}),
                "sets": sets,
            },
            "rendered_digest": None,
            "host": {},
            "command": list(self.plan.command),
            "measurements": measurements,
            "traces": {"tuning": TUNING_TRACE} if self.tuning else {},
        }
        (self.run_dir / RESULT).write_text(json.dumps(record, indent=2) + "\n")
        (self.run_dir / REPORT).write_text(report.render(record))
        _progress(f"verdict: {verdict} ({self.run_dir / REPORT})")
        return EXIT[verdict]


def _cell_name(row: dict) -> str:
    if "phase" in row:
        return f"{row['family']} {row['phase']} @ {row['context']} x{row['concurrency']}"
    return f"{row['family']} drafting"
