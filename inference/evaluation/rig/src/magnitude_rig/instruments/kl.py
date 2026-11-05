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

`--weight-cap BYTES` (repeatable) asks whether streaming the experts changes only where they
live: the uncapped load's logits and each streamed load's (its experts read from storage,
`fresh`), on the uncapped load's tokens, every entry at its declared default. A streamed load
whose logits equal the uncapped load's, bit for bit, passes. One whose logits differ fails with
where they part: the rows that differ, the first of them, the largest difference, and its KL
against its base (mean, p99 and max over the rows) and top-1 agreement. A non-finite logit in
either load fails it. A float32 reference of the whole model does not say where two loads
diverge, so none is read. `--context` sizes the prompt: a few hundred tokens already miss at
every streamed layer of a bank below its experts.

Each streamed load must also have missed at every streamed layer, or it did not ask the
question; a load that held its experts whole streamed nothing.

A load whose CPU site served misses computed them in the site's own arithmetic, which its
qualification admitted against α, not exact: its logits are not held to equality, only reported
against its base (KL and top-1). So is any load against a base whose own site served misses: an
uncapped load that does not fit its device streams too, so the answer says whether its base held
its experts whole or streamed, and how. `--without-site` judges such loads on a discrete device:
the base, and beside each streamed load the same load, under a numerical profile that approves no
CPU site, so neither requests nor uses a site verdict and every miss on either side is the
device's. That load's logits must equal the base's, its site serving nothing, and the other's site
must have served misses (its stored verdict admitted it). Without it, a sited load's question is
unanswered, which fails.

`--between` with two `--weight-cap`s asks the same for a model no uncapped load fits: the
second streamed load against the first.

`--repeat PASSES` under one `--weight-cap`, without `--between` or `--without-site`, asks whether
one load gives one answer: PASSES passes over the fixtures on one load, the file left in the page
cache, each compared with the first.
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


# A streamed load's comparison with its base on the host: whether the logits are equal bit for bit,
# and where not, the rows that differ, the first and the largest difference, and the candidate's
# KL against its base (mean, p99 and max over the rows) and top-1 agreement; the rows of each load
# holding a non-finite logit, the first of them, which no difference or KL covers; beside each
# load's expert misses. The candidate's logits are then removed, and the base's unless a later
# comparison reads them (`keep`).
COMPARE = (
    "import json, os, sys\n"
    "import numpy\n"
    "base, candidate, output, keep = sys.argv[1:5]\n"
    "header = json.load(open(os.path.join(base, 'logits.json')))\n"
    "rows = lambda run: numpy.memmap(os.path.join(run, 'logits.f32'), dtype='<f4', mode='r').reshape(-1, header['vocabulary'])\n"
    "a, b = rows(base), rows(candidate)\n"
    "differing, unfinished, largest, kl, agree = [], {'base': [], 'candidate': []}, 0.0, [], 0\n"
    "logsoftmax = lambda x: x - (numpy.log(numpy.exp(x - x.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) + x.max(axis=1, keepdims=True))\n"
    "for start in range(0, len(a), 64):\n"
    "    x, y = a[start:start + 64], b[start:start + 64]\n"
    "    differing += (numpy.flatnonzero((x.view('<u4') != y.view('<u4')).any(axis=1)) + start).tolist()\n"
    "    finite = {'base': numpy.isfinite(x).all(axis=1), 'candidate': numpy.isfinite(y).all(axis=1)}\n"
    "    for load, held in finite.items():\n"
    "        unfinished[load] += (numpy.flatnonzero(~held) + start).tolist()\n"
    "    both = finite['base'] & finite['candidate']\n"
    "    if not both.any():\n"
    "        continue\n"
    "    x, y = x[both].astype(numpy.float64), y[both].astype(numpy.float64)\n"
    "    largest = max(largest, float(numpy.abs(x - y).max()))\n"
    "    p, q = logsoftmax(x), logsoftmax(y)\n"
    "    kl += (numpy.exp(p) * (p - q)).sum(axis=1).tolist()\n"
    "    agree += int((x.argmax(axis=1) == y.argmax(axis=1)).sum())\n"
    "kl = numpy.maximum(numpy.array(kl), 0.0)\n"
    "first = min(unfinished['base'] + unfinished['candidate'], default=None)\n"
    "experts = json.load(open(os.path.join(candidate, 'logits.json')))['experts']\n"
    "summary = {'rows': len(a), 'differing': len(differing), 'largest': largest,\n"
    "           'first': header['rows'][differing[0]] if differing else None, 'experts': experts,\n"
    "           'base_experts': header['experts'],\n"
    "           'nonfinite': {'base': len(unfinished['base']), 'candidate': len(unfinished['candidate']),\n"
    "                         'first': None if first is None else header['rows'][first]},\n"
    "           'kl': {'mean': float(kl.mean()), 'p99': float(numpy.percentile(kl, 99)), 'max': float(kl.max())} if len(kl) else None,\n"
    "           'top1': agree / len(kl) if len(kl) else None}\n"
    "json.dump(summary, open(output, 'w'))\n"
    "del a, b\n"
    "os.unlink(os.path.join(candidate, 'logits.f32'))\n"
    "if keep != 'keep':\n"
    "    os.unlink(os.path.join(base, 'logits.f32'))\n"
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", type=_trace, metavar="TRACE",
                        help="a tuning trace (or its run directory): judge its chosen configurations against the defaults")
    parser.add_argument("--load", type=int, default=1, help="which load of the selection's trace (default 1)")
    parser.add_argument("--unit", help="select only the units matching this (any part of `entry [bindings] {statics}`)")
    parser.add_argument("--weight-cap", type=int, action="append", metavar="BYTES",
                        help="whether streaming under this weight cap changes the logits (repeatable)")
    parser.add_argument("--without-site", action="store_true",
                        help="with --weight-cap on a discrete device: each streamed load also under a profile that "
                             "approves no CPU site, every miss on the GPU, which must equal the uncapped load")
    parser.add_argument("--between", action="store_true",
                        help="with two --weight-cap: the second streamed load against the first, for a model "
                             "no uncapped load fits")
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


def render(arguments) -> Step:
    return Step("render", (
        harness.EXECUTABLE, "render", "--artifacts", "{artifacts}",
        "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--output", "{run_dir}",
        "--context", str(arguments.context),
    ))  # fmt: skip


def judge(logits_run: str, *, base: str | None = None, own_rendering: str | None = None) -> Step:
    """The judgement of `logits_run` against the reference (and `base`), which then removes the
    logits it read from the host."""
    extra = (("--base", f"{{runs}}/{base}") if base else ()) + (
        ("--own-rendering", f"{{runs}}/{own_rendering}/rendered.jsonl") if own_rendering else ()
    )
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


# The numerical profile under which a streamed load approves no CPU site (the standard α), which
# the `profile` job writes on the host.
UNSITED = "{runs}/profile/unsited.json"
UNSITED_PROFILE = Step("profile", ("{python}", "-c", "import json, sys; json.dump({'cpu_sites': False}, open(sys.argv[1], 'w'))",
                                   "{run_dir}/unsited.json"))


def _loads(arguments) -> tuple[str | None, list[tuple[str, int, bool]]]:
    """The base's cap (none: the uncapped load) and each streamed load compared with it: its run,
    cap and whether its profile approves a CPU site."""
    base, *capped = arguments.weight_cap if arguments.between else (None, *arguments.weight_cap)
    loads = [(harness.streamed(cap), cap, True) for cap in capped]
    if arguments.without_site:
        loads += [(f"unsited-{cap}", cap, False) for cap in capped]
    return base, loads


def _capped(arguments) -> tuple[tuple[str, Step], ...]:
    """`--weight-cap`'s loads at the defaults, each streamed load on its base's tokens (the
    uncapped load's, or under `--between` the first streamed load's), then each one's comparison
    with its base. Under `--without-site` the base approves no CPU site either, so every miss it
    takes, where it does not fit its device, is the device's."""
    defaults = ("--tuning-defaults",)
    unsited = ("--numerical-profile", UNSITED)
    base, loads = _loads(arguments)
    base_run = "logits" if base is None else harness.streamed(base)
    jobs = [("profile", UNSITED_PROFILE)] if arguments.without_site else []
    jobs.append((base_run, logits(arguments, pinning=(*defaults, *(unsited if arguments.without_site else ())),
                                  weight_cap=base)))
    for run, cap, approved in loads:
        jobs.append((run, logits(arguments, rendered=f"{{runs}}/{base_run}/rendered.jsonl",
                                 pinning=(*defaults, *(() if approved else unsited)), weight_cap=cap)))
    for index, (run, _, _) in enumerate(loads):
        keep = index + 1 < len(loads)  # a later comparison reads the base's logits
        jobs.append((_compared(run), Step("compare", (
            "{venv}/bin/python", "-c", COMPARE, f"{{runs}}/{base_run}", f"{{runs}}/{run}",
            "{run_dir}/compare.json", "keep" if keep else "drop",
        ))))  # fmt: skip
    return tuple(jobs)


def plan(arguments, source, host) -> Plan:
    if arguments.between and len(arguments.weight_cap or ()) != 2:
        raise SystemExit("kl: --between compares the loads under exactly two --weight-cap")
    if arguments.without_site and (arguments.between or not arguments.weight_cap):
        raise SystemExit("kl: --without-site compares streamed loads with the uncapped load")
    if arguments.repeat and (arguments.between or arguments.without_site or len(arguments.weight_cap or ()) != 1):
        raise SystemExit("kl: --repeat runs its passes on one load, under exactly one --weight-cap, "
                         "without --between or --without-site")
    if arguments.repeat:
        return Plan(builds=BUILDS, jobs=_repeated(arguments), device="device", acquires=Weights(arguments.model),
                    files=(arguments.fixture,))
    if arguments.weight_cap:
        if arguments.selection is not None:
            raise SystemExit("kl: --weight-cap compares loads at the defaults, not a selection")
        jobs = _capped(arguments)
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


def _compared(run: str) -> str:
    """The job comparing streamed load `run` with its base."""
    return f"compare-{run}"


def _parted(summary: dict) -> str:
    """Where a streamed load's logits part from its base's, and its KL and top-1 against it."""
    first, kl = summary["first"], summary["kl"]
    return (f"logits differ at {summary['differing']} of {summary['rows']} rows, first "
            f"{first['fixture_id']}@{first['position']}, largest {summary['largest']:.3g}; "
            f"KL against its base mean {kl['mean']:.3g}, p99 {kl['p99']:.3g}, max {kl['max']:.3g}, "
            f"top-1 {100 * summary['top1']:.2f}%")


def _held(experts: dict | None) -> str:
    """How a load held its experts: whole, or streamed with its misses."""
    if experts is None:
        return "held its experts whole"
    return f"streamed, fewest misses at a streamed layer {experts['least_layer_misses']}, {experts['sited']} sited"


# Whose CPU site served misses, by (the load's, its base's), where either did.
INEXACT = {(True, False): "its CPU site", (False, True): "its base's CPU site", (True, True): "its and its base's CPU sites"}


def _streamed(arguments, results: dict[str, JobResult]) -> Answer:
    """Each streamed load against its base, which the first line says held its experts whole or
    streamed. A load holds when it missed at every streamed layer and its logits are equal, all of
    them finite. A load whose CPU site, or whose base's, served misses is not an equality check
    (the site's arithmetic is admitted against α): it is reported, and holds under
    `--without-site`, whose load approving no CPU site judges its row against a base approving none
    either; there the approved load must have had its site serve misses."""
    base, loads = _loads(arguments)
    summaries = [json.loads((results[_compared(run)].directory / "compare.json").read_text()) for run, _, _ in loads]
    base_experts = summaries[0]["base_experts"]
    base_sited = 0 if base_experts is None else base_experts["sited"]
    assert not arguments.without_site or base_sited == 0, (
        f"the base's CPU site served {base_sited} misses under a profile approving none")
    lines = [f"{arguments.model} · {arguments.fixture.stem} @{arguments.context} · against {base or 'uncapped'}, "
             f"which {_held(base_experts)}"]
    holds = True
    for (run, cap, approved), summary in zip(loads, summaries):
        label = f"streamed {cap}" + ("" if approved else " approving no CPU site")
        experts = summary["experts"]
        if experts is None:
            lines.append(f"{label}: held its experts whole, so it streamed nothing")
            holds = False
            continue
        least, sited = experts["least_layer_misses"], experts["sited"]
        assert approved or sited == 0, f"{label}: its CPU site served {sited} misses under a profile approving none"
        nonfinite = summary["nonfinite"]
        if nonfinite["base"] or nonfinite["candidate"]:
            first = nonfinite["first"]
            outcome = (f"non-finite logits at {nonfinite['base']} rows of its base and {nonfinite['candidate']} "
                       f"of its own of {summary['rows']}, first {first['fixture_id']}@{first['position']}")
            held = False
        else:
            compared = f"equal logits ({summary['rows']} rows)" if summary["differing"] == 0 else _parted(summary)
            if sited or base_sited:
                judged = ("the load approving no CPU site judges it" if arguments.without_site
                          else "unjudged: --without-site judges it")
                outcome = (f"{INEXACT[bool(sited), bool(base_sited)]} served misses in arithmetic admitted "
                           f"against α, so this is not an equality check ({judged}): {compared}")
                held = arguments.without_site
            elif arguments.without_site and approved:
                outcome, held = f"{compared}; its CPU site served no miss, so the site went untested", False
            else:
                outcome, held = compared, summary["differing"] == 0
        lines.append(f"{label}: {outcome}; fewest misses at a streamed layer {least}, {sited} sited")
        holds &= held and least > 0
    return Answer(tuple(lines), holds)


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
