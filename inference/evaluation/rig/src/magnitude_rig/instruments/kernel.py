"""`kernel`: how fast is this kernel entry, in each configuration, at the shapes this model runs
it at, and how well does the tuner's search choose? The tuner's own survey in the model's
Prepare alone, from its headers alone: the catalog's header export opens as the model with zero
weights, so nothing of it is downloaded or imported. Every admissible configuration of the
named entry (each launch's own choices included, for an entry whose launches are formed
separately) is formed, validated and timed at each of the entry's tuning points for this model
and context, with the tuner's own cache-defeating rotations; the rest of the model is searched
in a token budget. It reports per point the declared default, the configuration the tuner
chose, and the fastest, each the median of its samples with its deviation; then the production
search replayed against the survey: how often it lands within 1/2/5% of the true best at the
evaluations this load allows it, and the per-point gap of one configuration for all points.

The survey publishes into a kernel cache of its own, so the token budget it gives the rest of
the model never reaches the checkout's tuning.
"""

import argparse
import json

from ..engine import Headers, Step
from . import Answer, Instrument, JobResult, Measure, Plan, harness

OTHERS_TUNING_SECONDS = "1"  # the rest of the model is not what this measures


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=15, help="samples per point of every configuration (default 15)")


def plan(arguments, source) -> Plan:
    job = Step("kernel", (
        harness.EXECUTABLE, "prepare",
        "--artifacts", "{artifacts}", "--fixtures", f"{{inputs}}/{arguments.fixture.name}",
        "--output", "{run_dir}", "--build-identity", "{build_file}",
        "--cache-dir", "{checkout}/kernels-survey", "--device", arguments.backend,
        "--context", str(arguments.context),
        "--fresh-tuning", "--tuning-budget", OTHERS_TUNING_SECONDS,
        "--tuning-survey", "{run_dir}/survey", "--tuning-survey-entries", arguments.entry,
        "--tuning-survey-samples", str(arguments.samples),
    ))  # fmt: skip
    return Plan(builds=(harness.BUILD,), jobs=(("kernel", job),), device="timed",
                acquires=Headers(arguments.model), files=(arguments.fixture,))


def _parameters(configuration: dict) -> str:
    """A configuration as text: its entry-wide values, then each launch's own choices, named
    `L<launch ordinal>.<parameter>` (a launch formed separately chooses its own)."""
    values = {**configuration.get("statics", {}), **configuration.get("params", {})}
    for ordinal, launch in enumerate(configuration.get("launches", [])):
        values.update({f"L{ordinal}.{name}": value for name, value in launch.items()})
    return ",".join(f"{name}={value}" for name, value in sorted(values.items())) or "(none)"


def _difference(configuration: str, default: str) -> str:
    """The parameters where `configuration` differs from the default, or `default`."""
    base = dict(item.split("=", 1) for item in default.split(",") if "=" in item)
    changed = [item for item in configuration.split(",") if "=" in item and base.get(item.split("=", 1)[0]) != item.split("=", 1)[1]]
    return ",".join(changed) or "default"


def _times(record: dict) -> dict[str, dict[str, tuple[float, float]]]:
    """Per configuration (as text), per point: (median, deviation) seconds."""
    found: dict[str, dict[str, tuple[float, float]]] = {}
    for configuration in record["configurations"]:
        outcome = configuration["outcome"]
        if "Measured" not in outcome:
            continue
        found[_parameters(configuration["configuration"])] = {
            point["point"]: (point["median_seconds"], point["deviation_seconds"])
            for point in outcome["Measured"]["points"]
        }
    return found


def _shown(at: dict[str, tuple[float, float]], name: str) -> str:
    """A configuration's time at a point, in µs with its deviation, or a dash when unmeasured."""
    if name not in at:
        return f"{'—':>13}"
    median, deviation = at[name]
    return f"{1e6 * median:>7.1f}±{1e6 * deviation:<5.1f}"


def _search(replay: str) -> list[str]:
    """The production objective's rows of the harness's replay report (`replay.md`, one section
    per surveyed entry instance), with each instance's weighted per-point gap."""
    lines = []
    for line in replay.splitlines():
        if line.startswith("### "):
            lines.append(f"  search on {line[4:]}")
        elif line.startswith("| keyed (production)"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            evaluations, within1, within2, within5, median, p95, _, n95 = cells[1:]
            lines.append(f"    {evaluations:>5} evaluations: within 1% {within1}, 2% {within2}, 5% {within5}"
                         f" · median excess {median}, p95 {p95} · n95(2%) {n95}")
        elif line.startswith("Weighted per-point gap"):
            lines.append(f"    one configuration for all points: {line[len('Weighted '):].rstrip()}")
        elif line.startswith("No point has demand"):
            lines.append(f"    {line}")
    return lines


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["kernel"]
    surveys = sorted((result.directory / "survey").glob("*.json"))
    output = result.output()
    if result.code != 0 or not surveys:
        detail = "no survey written: the entry is not tuned for this model" if result.code == 0 else ""
        return Answer((f"harness exited {result.code} {detail}".strip(), *output[-15:]), False)
    lines, measures = [], {}
    for path in surveys:
        survey = json.loads(path.read_text())
        record = survey["result"]
        times = _times(record)
        default = _parameters(record["served"][0])
        chosen = _parameters(record["overall"])
        bindings = survey["bindings"] if len(survey["bindings"]) <= 60 else survey["bindings"][:57] + "..."
        lines.append(f"{survey['entry']} [{bindings}] · {len(times)} configurations measured")
        lines.append(f"  default {default}")
        lines.append(f"  tuner's choice: {_difference(chosen, default)}")
        lines.append(f"  {'point':<20}{'default µs':>13}{'choice µs':>13}{'best µs':>13}  best, as changed from the default")
        for point in record["points"]:
            label = point["label"]
            at = {name: by_point[label] for name, by_point in times.items() if label in by_point}
            if not at:
                continue
            best = min(at, key=lambda name: at[name][0])
            shown = "".join(_shown(at, name) for name in (default, chosen, best))
            lines.append(f"  {label[:19]:<20}{shown}  {_difference(best, default)}")
            if chosen in at:
                median, deviation = at[chosen]
                measures[f"{survey['entry']} {survey['bindings']} {label}"] = Measure(
                    1e6 * median, "µs", 1e6 * deviation, lower_is_better=True
                )
    lines += _search((result.directory / "survey" / "replay.md").read_text())
    return Answer(tuple(lines), True, measures)


INSTRUMENT = Instrument(
    name="kernel",
    kind="speed",
    question="How fast is this kernel entry, per configuration, at this model's shapes, and how well does the search choose?",
    observable="one entry's time per configuration and shape",
    selectors=("backend", "host", "build", "model", "context", "fixture", "entry", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
    comparable=True,
)
