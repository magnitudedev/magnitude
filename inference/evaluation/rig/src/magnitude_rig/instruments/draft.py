"""`draft`: how well does the catalog drafter propose for this model? The harness `draft` at
anchors spread over each fixture's scored spans: the target prefills to the anchor, the
drafter (primed as the served path primes it) proposes greedily, and the target verifies; the
accepted prefix is the proposals equal to the target's own greedy choices. Reports the tokens
each verification commits, the share of proposals accepted, and how often proposals also match
the fixture's own next tokens. With `--attribute`, each anchor's draft, verification and one
plain decode step are timed on the device, under the timed hold.

Greedy drafting is deterministic, so the counts are exact: `mrig ab draft` compares two builds'
drafters without an interval. Served acceptance (the adaptive controller's choices at serving
concurrency) is `bench`'s.
"""

import argparse
import json

from ..engine import Step, Weights
from . import Answer, Instrument, JobResult, Measure, Plan, harness
from .resolution import median_interval


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--anchors", type=int, default=8, help="anchors per fixture (default 8)")
    parser.add_argument("--proposals", type=int, help="proposals per draft (the drafter's own width by default)")
    parser.add_argument("--attribute", action="store_true",
                        help="time each anchor's draft, verification and a plain decode step on the device")


def plan(arguments, source) -> Plan:
    job = Step("draft", (
        harness.EXECUTABLE, "draft", "--artifacts", "{artifacts}",
        "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--output", "{run_dir}",
        "--build-identity", "{build_file}", "--cache-dir", "{checkout}/kernels",
        "--device", arguments.backend, "--context", str(arguments.context),
        "--anchors", str(arguments.anchors),
        *(("--proposals", str(arguments.proposals)) if arguments.proposals else ()),
        *(("--attribute",) if arguments.attribute else ()),
    ))  # fmt: skip
    return Plan(builds=(harness.BUILD,), jobs=(("draft", job),),
                device="timed" if arguments.attribute else "device",
                acquires=Weights(arguments.model), files=(arguments.fixture,))


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["draft"]
    if result.code != 0:
        return Answer((f"harness draft exited {result.code}", *result.output()[-15:]), False)
    record = json.loads((result.directory / "result.json").read_text())
    drafting = record["measurements"]["drafting"]
    proposed, accepted, anchors = drafting["proposed"], drafting["accepted"], drafting["anchors"]
    lines = [
        f"{arguments.model} · {arguments.fixture.stem} @{arguments.context} · {anchors} anchors",
        (f"{drafting['tokens_per_verify']:.3f} tokens per verification · accepted {accepted} of {proposed} "
         f"proposals ({100 * accepted / max(proposed, 1):.1f}%) · matched the fixture "
         f"{100 * drafting['matched_fixture'] / max(proposed, 1):.1f}%"),
    ]
    if arguments.attribute:
        rows = [json.loads(line) for line in (result.directory / "draft.jsonl").read_text().splitlines()]
        for phase in ("draft", "verify", "plain"):
            median, half = median_interval([row["attribution"][phase]["device_ms"] for row in rows])
            lines.append(f"  {phase:<7} {median:.3f} ms device (±{half:.3f})")
    measures = {
        "tokens_per_verify": Measure(drafting["tokens_per_verify"], "tokens", None, lower_is_better=False),
        "accepted": Measure(accepted, "proposals", None, lower_is_better=False),
    }
    return Answer(tuple(lines), True, measures)


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
