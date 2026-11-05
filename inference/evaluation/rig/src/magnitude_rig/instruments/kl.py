"""`kl`: how far is one build's model from the float32 reference, at one fixture family and
context? The harness `logits` on the host's kernel cache, then `magnitude-references kl`
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

`--weight-cap BYTES` (repeatable) asks whether streaming the experts keeps the output: the
uncapped load's logits and each streamed load's (its experts read from storage, `fresh`), on
the uncapped load's tokens, every entry at its declared default. Each streamed load is judged
against the reference as a candidate against the uncapped load (paired, as `ab kl` judges
builds), and whether it is bit-identical is reported beside: a streamed load that computes a
choice's features in another arithmetic (the CPU site) is held to the reference, never to
bit identity.

`--repeat PASSES` under one `--weight-cap` asks whether one load gives one answer: PASSES passes
over the fixtures on one load, the file left in the page cache, each compared with the first.
"""

import argparse
import dataclasses
import json
from pathlib import Path

from ..engine import Step, Weights
from . import Answer, Instrument, JobResult, Pairing, Plan, environment, harness

REFERENCES = "{venv}/bin/magnitude-references"
BUILDS = (
    harness.BUILD,
    environment("references", "--package", "magnitude-references"),
)


def _trace(text: str) -> Path:
    """A tuning trace, or the run directory holding one."""
    path = Path(text)
    trace = path / "tuning.jsonl" if path.is_dir() else path
    if not trace.is_file():
        raise argparse.ArgumentTypeError(f"{trace} is not a tuning trace")
    return trace


# Each pass's logits against the first's, row by row, each printed then removed from the host:
# whether any row differs at all, where the first does, and the top-two margin where a row's top
# token moved.
REPEATS = (
    "import json, os, sys\n"
    "import numpy\n"
    "run, passes = sys.argv[1], int(sys.argv[2])\n"
    "vocabulary = json.load(open(run + '/logits.json'))['vocabulary']\n"
    "rows = lambda path: numpy.memmap(path, dtype='f4', mode='r').reshape(-1, vocabulary)\n"
    "a = rows(run + '/logits.f32')\n"
    "for index in range(1, passes):\n"
    "    path = f'{run}/logits-{index}.f32'\n"
    "    b = rows(path)\n"
    "    differs = numpy.flatnonzero(numpy.any(a != b, axis=1))\n"
    "    moved = numpy.flatnonzero(numpy.argmax(a, axis=1) != numpy.argmax(b, axis=1))\n"
    "    margins = [float(numpy.diff(numpy.sort(a[row])[-2:])[0]) for row in moved]\n"
    "    print(json.dumps({'pass': index, 'rows': len(a), 'differing': int(differs.size),\n"
    "                      'first_differing': int(differs[0]) if differs.size else None,\n"
    "                      'largest': float(numpy.abs(a - b).max()), 'top_moved': moved.tolist(), 'margins': margins}))\n"
    "    del b\n"
    "    os.unlink(path)\n"
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", type=_trace, metavar="TRACE",
                        help="a tuning trace (or its run directory): judge its chosen configurations against the defaults")
    parser.add_argument("--load", type=int, default=1, help="which load of the selection's trace (default 1)")
    parser.add_argument("--unit", help="select only the units matching this (any part of `entry [bindings] {statics}`)")
    parser.add_argument("--weight-cap", type=int, action="append", metavar="BYTES",
                        help="whether streaming under this weight cap changes the logits (repeatable)")
    parser.add_argument("--repeat", type=int, metavar="PASSES",
                        help="under one --weight-cap: whether PASSES passes on one load, one row a step, give the "
                             "same logits, the file left in the page cache: where they part, and the margin there")


def _selecting(arguments) -> tuple[str, ...]:
    return (
        "--tuning-select", f"{{inputs}}/{arguments.selection.name}",
        "--tuning-select-load", str(arguments.load),
        *(("--tuning-select-unit", arguments.unit) if arguments.unit else ()),
    )


def logits(arguments, *, rendered: str | None = None, pinning: tuple[str, ...] = (),
           weight_cap: int | None = None, fresh: bool | None = None) -> Step:
    sizing = ("--rendered", rendered) if rendered else ("--context", str(arguments.context))
    capped = ("--weight-cap", str(weight_cap)) if weight_cap is not None else ()
    return Step("logits", (
        harness.EXECUTABLE, "logits", "--artifacts", "{artifacts}",
        "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--output", "{run_dir}",
        "--build-identity", "{build_file}", "--cache-dir", "{engine_cache}/kernels",
        "--device", arguments.backend, *sizing, *pinning, *capped,
    ), fresh=weight_cap is not None if fresh is None else fresh)  # fmt: skip


# The SHA-256 of each file named after the output path, as a JSON list in order: run on the
# host, where the logits stay (`sync.HOST_ONLY`).
DIGESTS = (
    "import hashlib, json, sys\n"
    "digests = []\n"
    "for path in sys.argv[2:]:\n"
    "    digest = hashlib.sha256()\n"
    "    with open(path, 'rb') as file:\n"
    "        for block in iter(lambda: file.read(1 << 24), b''):\n"
    "            digest.update(block)\n"
    "    digests.append(digest.hexdigest())\n"
    "json.dump(digests, open(sys.argv[1], 'w'))\n"
)


def render(arguments) -> Step:
    return Step("render", (
        harness.EXECUTABLE, "render", "--artifacts", "{artifacts}",
        "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--output", "{run_dir}",
        "--context", str(arguments.context),
    ))  # fmt: skip


def judge(logits_run: str, *, base: str | None = None, own_rendering: str | None = None,
          keep: bool = False) -> Step:
    """The judgement of `logits_run` against the reference (and `base`), which then removes the
    logits it read from the host, unless a later judgement reads them too (`keep`)."""
    extra = (("--base", f"{{runs}}/{base}") if base else ()) + (
        ("--own-rendering", f"{{runs}}/{own_rendering}/rendered.jsonl") if own_rendering else ()
    ) + (("--keep-logits",) if keep else ())
    return Step("kl", (
        REFERENCES, "kl", "--logits", f"{{runs}}/{logits_run}", *extra,
        "--artifacts", "{artifacts}", "--build-identity", "{build_file}",
        "--cache", "{root}/references", "--output", "{run_dir}/kl.json",
    ))  # fmt: skip


def _repeated(arguments) -> tuple[tuple[str, Step], ...]:
    """`--repeat`'s passes on one load under the one cap, a row a step as a service decodes, the
    file left where the page cache holds it, then each pass against the first."""
    (cap,) = arguments.weight_cap
    step = logits(arguments, pinning=("--tuning-defaults",), weight_cap=cap, fresh=False)
    passes = dataclasses.replace(step, argv=(*step.argv, "--batch", "1", "--passes", str(arguments.repeat)))
    return (("passes", passes),
            ("repeats", Step("repeats", ("{venv}/bin/python", "-c", REPEATS, "{runs}/passes", str(arguments.repeat)))))


def plan(arguments, source, host) -> Plan:
    if arguments.repeat:
        return Plan(builds=BUILDS, jobs=_repeated(arguments), device="device", acquires=Weights(arguments.model),
                    files=(arguments.fixture,))
    if arguments.weight_cap:
        if arguments.selection is not None:
            raise SystemExit("kl: --weight-cap compares loads at the defaults, not a selection")
        defaults = ("--tuning-defaults",)
        jobs = (
            ("logits", logits(arguments, pinning=defaults)),
            *((harness.streamed(cap), logits(arguments, rendered="{runs}/logits/rendered.jsonl",
                                     pinning=defaults, weight_cap=cap))
              for cap in arguments.weight_cap),
            ("digests", Step("digests", (
                "{python}", "-c", DIGESTS, "{run_dir}/digests.json", "{runs}/logits/logits.f32",
                *(f"{{runs}}/{harness.streamed(cap)}/logits.f32" for cap in arguments.weight_cap),
            ))),
            # Each judgement reads the uncapped logits: all but the last keep them on the host.
            *((_judged(cap), judge(harness.streamed(cap), base="logits",
                                   keep=index + 1 < len(arguments.weight_cap)))
              for index, cap in enumerate(arguments.weight_cap)),
        )  # fmt: skip
    elif arguments.selection is None:
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


def _judged(cap: int) -> str:
    """The job judging the load streamed under weight cap `cap` against the reference."""
    return f"kl-{cap}"


def _streamed(arguments, results: dict[str, JobResult]) -> Answer:
    uncapped, *capped = json.loads((results["digests"].directory / "digests.json").read_text())
    lines = [f"{arguments.model} · {arguments.fixture.stem} @{arguments.context} · "
             f"uncapped logits {uncapped[:16]}"]
    regressed = False
    for cap, digest in zip(arguments.weight_cap, capped, strict=True):
        summary = json.loads((results[_judged(cap)].directory / "kl.json").read_text())
        identity = "bit-identical" if digest == uncapped else f"not bit-identical ({digest[:16]})"
        lines += [
            f"streamed {cap}: {identity}",
            _line("  uncapped vs reference", summary["base"]),
            _line("  streamed vs reference", summary["build"]),
            _line("  streamed vs uncapped", summary["between_builds"]),
            *(f"  worst row {row['fixture_id']}@{row['position']}: KL {row['kl']:.4f}, "
              + ("top-1 agrees" if row["agree"] else "top-1 differs")
              for row in summary["build"]["worst"][:1]),
        ]
        changes, worse = _changes(summary)
        lines += changes
        regressed |= worse
    return Answer(tuple(lines), not regressed)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    if (failure := _failed(results)) is not None:
        return failure
    if arguments.repeat:
        return Answer(tuple(results["repeats"].output()), True)
    if arguments.weight_cap:
        return _streamed(arguments, results)
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
    if arguments.weight_cap:
        raise SystemExit("ab kl: --weight-cap compares loads of one build; run kl with it on each")
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
