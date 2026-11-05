"""`draft`: how well does the catalog drafter propose for this model? The harness `draft` at
anchors spread over each fixture's scored spans: the target prefills to the anchor, the
drafter (primed as the served path primes it) proposes greedily, and the target verifies; the
accepted prefix is the proposals equal to the target's own greedy choices. Reports the tokens
each verification commits, the share of proposals accepted, and how often proposals also match
the fixture's own next tokens. With `--attribute`, each anchor's draft, verification and one
plain decode step are timed on the device, under the timed hold.

`--weight-cap BYTES` (repeatable) asks whether drafting streamed keeps the output: after the
uncapped load, a load streamed under each cap (its experts read from storage, after the product's
measurement job) drafts at the same anchors, and each anchor's proposals and target tokens are
compared with the uncapped load's: where they part is the first anchor that differs.

Greedy drafting is deterministic, so the counts are exact: `mrig ab draft` compares two builds'
drafters without an interval. Served acceptance (the adaptive controller's choices at serving
concurrency) is `bench`'s.
"""

import argparse
import json
from dataclasses import dataclass

from ..engine import Step, Weights
from . import Answer, Instrument, JobResult, Measure, Plan, harness, lacking, unrecorded
from .assess import measurement
from .binaries import ASSESS_BUILD
from .resolution import median_interval

# What a draft answer measures, which every build's record must hold (`lacking`).
MEASURED = ("proposed", "accepted", "anchors", "tokens_per_verify")


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--anchors", type=int, default=8, help="anchors per fixture (default 8)")
    parser.add_argument("--proposals", type=int, help="proposals per draft (the drafter's own width by default)")
    parser.add_argument("--attribute", action="store_true",
                        help="time each anchor's draft, verification and a plain decode step on the device")
    parser.add_argument("--weight-cap", type=int, action="append", metavar="BYTES",
                        help="also draft streamed under this weight cap, against the uncapped load (repeatable)")


def plan(arguments, source, host) -> Plan:
    if arguments.weight_cap and arguments.attribute:
        raise SystemExit("draft: --weight-cap compares outputs; time them with --attribute alone")

    def job(*capped: str, fresh: bool = False) -> Step:
        return Step("draft", (
            harness.EXECUTABLE, "draft", "--artifacts", "{artifacts}",
            "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--output", "{run_dir}",
            "--build-identity", "{build_file}", "--cache-dir", "{engine_cache}/kernels",
            "--device", arguments.backend, "--context", str(arguments.context),
            "--anchors", str(arguments.anchors),
            *(("--proposals", str(arguments.proposals)) if arguments.proposals else ()),
            *(("--attribute",) if arguments.attribute else ()), *capped,
        ), fresh=fresh)  # fmt: skip
    caps = arguments.weight_cap or ()
    jobs = (("draft", job()), *((("measure", measurement(arguments.backend)),) if caps else ()),
            *((harness.streamed(cap), job("--weight-cap", str(cap), fresh=True)) for cap in caps))
    return Plan(builds=(harness.BUILD, *((ASSESS_BUILD,) if caps else ())), jobs=jobs,
                device="timed" if arguments.attribute else "device",
                acquires=Weights(arguments.model), files=(arguments.fixture,))


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["draft"]
    if result.code != 0:
        return Answer((f"harness draft exited {result.code}", *result.output()[-15:]), False)
    record = json.loads((result.directory / "result.json").read_text())
    if missing := lacking(record, *(f"measurements.drafting.{name}" for name in MEASURED)):
        return Answer((f"this build's harness records no {', '.join(missing)}",), False)
    drafting = record["measurements"]["drafting"]
    proposed, accepted, anchors = drafting["proposed"], drafting["accepted"], drafting["anchors"]
    matched = ("" if lacking(drafting, "matched_fixture") else
               f" · matched the fixture {100 * drafting['matched_fixture'] / max(proposed, 1):.1f}%")
    lines = [
        f"{arguments.model} · {arguments.fixture.stem} @{arguments.context} · {anchors} anchors",
        (f"{drafting['tokens_per_verify']:.3f} tokens per verification · accepted {accepted} of {proposed} "
         f"proposals ({100 * accepted / max(proposed, 1):.1f}%){matched}"),
    ]
    if arguments.attribute:
        rows = [json.loads(line) for line in (result.directory / "draft.jsonl").read_text().splitlines()]
        phases = ("draft", "verify", "plain")
        missing = sorted({path for row in rows for path in lacking(row, *(f"attribution.{p}.device_ms" for p in phases))})
        if missing:
            lines.append(unrecorded("attribution", missing))
        for phase in phases if not missing else ():
            median, half = median_interval([row["attribution"][phase]["device_ms"] for row in rows])
            lines.append(f"  {phase:<7} {median:.3f} ms device (±{half:.3f})")
    streamed, reproduced = _streamed(arguments, results, result)
    lines.extend(streamed)
    measures = {
        "tokens_per_verify": Measure(drafting["tokens_per_verify"], "tokens", None, lower_is_better=False),
        "accepted": Measure(accepted, "proposals", None, lower_is_better=False),
    }
    return Answer(tuple(lines), reproduced, measures)


def _rows(result: JobResult) -> list[tuple]:
    """Each anchor's identity and output, in order: what a streamed load must reproduce."""
    rows = [json.loads(line) for line in (result.directory / "draft.jsonl").read_text().splitlines()]
    return [(row["fixture_id"], row["anchor"], row["proposals"], row["target"]) for row in rows]


@dataclass(frozen=True)
class _Reproduced:
    """A streamed load drafted the uncapped load's proposals and target tokens at every one of
    its anchors, of which there were some."""

    anchors: int


@dataclass(frozen=True)
class _Parted:
    """Why a streamed load's drafting is not shown to reproduce the uncapped load's."""

    lines: tuple[str, ...]


def _compare(cap: int, result: JobResult, reference: list[tuple]) -> _Reproduced | _Parted:
    """One streamed load's drafting against the uncapped load's `reference`, anchor by anchor:
    reproduced only where its job ended well and it drafted the same tokens at the same anchors,
    and at least one."""
    if result.code != 0:
        return _Parted((f"streamed {cap}: harness draft exited {result.code}", *result.output()[-15:]))
    rows = _rows(result)
    if [row[:2] for row in rows] != [row[:2] for row in reference]:
        return _Parted((f"streamed {cap}: drafted at other anchors than the uncapped load",))
    if not rows:
        return _Parted((f"streamed {cap}: no anchor drafted, so nothing compared",))
    differing = [(mine, theirs) for mine, theirs in zip(rows, reference) if mine != theirs]
    if not differing:
        return _Reproduced(len(rows))
    (fixture, anchor, proposals, target), (_, _, base_proposals, base_target) = differing[0]
    return _Parted((f"streamed {cap}: {len(differing)} of {len(rows)} anchors differ, first {fixture} @{anchor}: "
                    f"proposals {proposals} vs {base_proposals}, target {target} vs {base_target}",))


def _streamed(arguments, results: dict[str, JobResult], uncapped: JobResult) -> tuple[list[str], bool]:
    """Each streamed load's drafting against the uncapped load's, and whether every one
    reproduced it. The streamed loads are priced by the product's measurement job: where that
    job failed, none is the load it is named, and the answer fails with it."""
    if not arguments.weight_cap:
        return [], True
    measured = results["measure"]
    verdicts = [_compare(cap, results[harness.streamed(cap)], _rows(uncapped)) for cap in arguments.weight_cap]
    lines = [] if measured.code == 0 else [f"measurement job exited {measured.code}", *measured.output()[-15:]]
    for cap, verdict in zip(arguments.weight_cap, verdicts):
        match verdict:
            case _Reproduced(anchors):
                lines.append(f"streamed {cap}: same proposals and target tokens at all {anchors} anchors")
            case _Parted(parted):
                lines.extend(parted)
    return lines, measured.code == 0 and all(isinstance(verdict, _Reproduced) for verdict in verdicts)


INSTRUMENT = Instrument(
    name="draft",
    kind="fidelity",
    question="How well does the catalog drafter propose for this model, at one fixture family and context?",
    observable="drafter proposals the target accepts",
    selectors=("backend", "host", "build", "model", "context", "fixture", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
    comparable=True,
)
