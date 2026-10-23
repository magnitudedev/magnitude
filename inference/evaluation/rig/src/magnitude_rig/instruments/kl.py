"""`kl`: how far is one build's model from the float32 reference, at one fixture family and
context? The harness `logits` on the checkout's kernel cache, then `magnitude-references kl`
against the host's cached reference of the same rendered tokens, built when missing (the
reference is the slow part the first time; it is cached for every later run and build).

`mrig ab kl` runs both builds on the base's tuning (`harness`), measures the candidate on the
base's exact tokens, judges it against the base with a paired interval, and says whether the
two builds' outputs are identical.

`--selection TRACE` asks what a tuning selection costs in precision: the build runs every unit
at its declared default, then the configurations one load of the trace chose (on the same
tokens), and the selection is judged against the defaults as a candidate against a base, beside
what the trace states of each chosen unit's precision (its error in the rounding steps that
govern its admission, squared: a dimensionless contribution relative to resolution noise, not
a predicted KL). Under `ab` with a selection, both builds run the selection.
"""

import argparse
import json
from pathlib import Path

from ..engine import Step, Weights
from . import Answer, Instrument, JobResult, Pairing, Plan, harness

REFERENCES = "{venv}/bin/magnitude-references"
BUILDS = (
    harness.BUILD,
    Step("references", ("uv", "sync", "--frozen", "--project", "{src}/evaluation",
                        "--package", "magnitude-references")),
)


def _trace(text: str) -> Path:
    """A tuning trace, or the run directory holding one."""
    path = Path(text)
    trace = path / "tuning.jsonl" if path.is_dir() else path
    if not trace.is_file():
        raise argparse.ArgumentTypeError(f"{trace} is not a tuning trace")
    return trace


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", type=_trace, metavar="TRACE",
                        help="a tuning trace (or its run directory): judge its chosen configurations against the defaults")
    parser.add_argument("--load", type=int, default=1, help="which load of the selection's trace (default 1)")
    parser.add_argument("--unit", help="select only the units matching this (any part of `entry [bindings] {statics}`)")


def _selecting(arguments) -> tuple[str, ...]:
    return (
        "--tuning-select", f"{{inputs}}/{arguments.selection.name}",
        "--tuning-select-load", str(arguments.load),
        *(("--tuning-select-unit", arguments.unit) if arguments.unit else ()),
    )


def logits(arguments, *, rendered: str | None = None, pinning: tuple[str, ...] = ()) -> Step:
    sizing = ("--rendered", rendered) if rendered else ("--context", str(arguments.context))
    return Step("logits", (
        harness.EXECUTABLE, "logits", "--artifacts", "{artifacts}",
        "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--output", "{run_dir}",
        "--build-identity", "{build_file}", "--cache-dir", "{checkout}/kernels",
        "--device", arguments.backend, *sizing, *pinning,
    ))  # fmt: skip


def render(arguments) -> Step:
    return Step("render", (
        harness.EXECUTABLE, "render", "--artifacts", "{artifacts}",
        "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--output", "{run_dir}",
        "--context", str(arguments.context),
    ))  # fmt: skip


def judge(logits_run: str, *, base: str | None = None, own_rendering: str | None = None) -> Step:
    extra = (("--base", f"{{runs}}/{base}") if base else ()) + (
        ("--own-rendering", f"{{runs}}/{own_rendering}/rendered.jsonl") if own_rendering else ()
    )
    return Step("kl", (
        REFERENCES, "kl", "--logits", f"{{runs}}/{logits_run}", *extra,
        "--artifacts", "{artifacts}", "--build-identity", "{build_file}",
        "--cache", "{root}/references", "--output", "{run_dir}/kl.json",
    ))  # fmt: skip


def plan(arguments, source) -> Plan:
    if arguments.selection is None:
        jobs = (("logits", logits(arguments)), ("kl", judge("logits")))
    else:
        jobs = (
            ("defaults", logits(arguments, pinning=("--tuning-defaults",))),
            ("selected", logits(arguments, rendered="{runs}/defaults/rendered.jsonl", pinning=_selecting(arguments))),
            ("kl", judge("selected", base="defaults")),
        )
    files = (arguments.fixture,) + ((arguments.selection,) if arguments.selection else ())
    return Plan(builds=BUILDS, jobs=jobs, device="device", acquires=Weights(arguments.model), files=files)


def _failed(results: dict[str, JobResult]) -> Answer | None:
    for name, result in results.items():
        if result.code != 0:
            return Answer((f"{name} exited {result.code}", *result.output()[-15:]), False)
    return None


def _line(label: str, summary: dict) -> str:
    return (
        f"{label}: mean KL {summary['mean_kl']:.5f} · p99 {summary['p99_kl']:.4f} · "
        f"max {summary['max_kl']:.3f} · top-1 {100 * summary['top1_agreement']:.2f}% "
        f"({summary['positions']} positions)"
    )


def _changes(summary: dict) -> tuple[list[str], bool]:
    """The paired judgement's lines, and whether any metric regressed."""
    lines, regressed = [], False
    for change in summary["changes"]:
        low, high = change["interval"]
        lines.append(
            f"  {change['metric']:<16} {change['base']:.5f} → {change['candidate']:.5f} "
            f"({100 * change['relative']:+.1f}%, 95% [{low:+.5f}, {high:+.5f}]) {change['verdict']}"
        )
        regressed |= change["verdict"] == "regressed"
    return lines, regressed


def _selection(arguments, results: dict[str, JobResult]) -> Answer:
    summary = json.loads((results["kl"].directory / "kl.json").read_text())
    selection = json.loads((results["selected"].directory / "selection.json").read_text())
    lines = [
        (f"{arguments.model} · {arguments.fixture.stem} @{arguments.context} · "
         f"{len(selection['units'])} selected units of load {arguments.load}"),
        _line("defaults vs reference", summary["base"]),
        _line("selection vs reference", summary["build"]),
        _line("selection vs defaults", summary["between_builds"]),
    ]
    changes, regressed = _changes(summary)
    lines += changes
    predicted, bound = selection["predicted_contribution"], selection["bound_per_unit"]
    lines.append(
        "trace: contribution "
        + ("unmeasured" if predicted is None else f"{predicted:.4f}")
        + " squared error ratio" + ("" if bound is None else f" (α² {bound:.4f} per unit)")
        + f", admission {selection['admission']}"
    )
    for unit in sorted(selection["units"], key=lambda unit: -(unit["contribution"] or 0))[:8]:
        name = f"{unit['unit']['entry']} [{unit['unit']['bindings']}]"
        shown = "—" if unit["contribution"] is None else f"{unit['contribution']:.4f}"
        lines.append(f"  {name[:60]:<61}{shown:>8}")
    return Answer(tuple(lines), not regressed)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    if (failure := _failed(results)) is not None:
        return failure
    if arguments.selection is not None:
        return _selection(arguments, results)
    summary = json.loads((results["kl"].directory / "kl.json").read_text())
    reference = "cached" if summary["reference"]["cached"] else "built this run"
    return Answer((
        f"{arguments.model} · {arguments.fixture.stem} @{arguments.context} · reference {reference}",
        _line("vs reference", summary["build"]),
    ), True)


def pair_jobs(arguments, base: str, candidate: str) -> tuple[tuple[str, str, Step], ...]:
    """(side, run name, step): the base renders and measures; the candidate is measured on the
    base's exact tokens, its own rendering compared, then judged against the base. Both run the
    base's tuning, or both the selection."""
    if arguments.selection is None:
        base_pinning, candidate_pinning = harness.RECORD, harness.REPLAY
    else:
        base_pinning = candidate_pinning = _selecting(arguments)
    return (
        (base, "base-logits", logits(arguments, pinning=base_pinning)),
        (candidate, "candidate-render", render(arguments)),
        (candidate, "candidate-logits",
         logits(arguments, rendered="{runs}/base-logits/rendered.jsonl", pinning=candidate_pinning)),
        (candidate, "kl", judge("candidate-logits", base="base-logits", own_rendering="candidate-render")),
    )


def pair_answer(arguments, results: dict[str, JobResult]) -> Answer:
    if (failure := _failed(results)) is not None:
        return failure
    summary = json.loads((results["kl"].directory / "kl.json").read_text())
    lines = [
        f"{arguments.model} · {arguments.fixture.stem} @{arguments.context}",
        _line("base vs reference", summary["base"]),
        _line("candidate vs reference", summary["build"]),
        "builds identical: " + ("yes (bit-identical outputs)" if summary["builds_identical"] else
                                f"no (mean KL between builds {summary['between_builds']['mean_kl']:.6f})"),
    ]
    changes, regressed = _changes(summary)
    lines += changes
    differences = summary.get("rendering_differences") or []
    if differences:
        lines.append(f"rendering differs in {len(differences)} fixtures: the candidate renders other tokens")
    return Answer(tuple(lines), not regressed)


INSTRUMENT = Instrument(
    name="kl",
    kind="fidelity",
    question="How far is one build from the float32 reference, at one fixture family and context?",
    observable="divergence from the reference",
    selectors=("backend", "host", "build", "model", "context", "fixture", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
    pairing=Pairing(pair_jobs, pair_answer),
)
