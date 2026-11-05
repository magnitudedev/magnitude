"""`served`: does a load work through the real service? It runs the service executable
(`magnitude-inference serve`) on the host from an installation of the checkout's binary and the
catalog, drives it over its HTTP API as a client does, and reports each stage from what the API
answers and what the service logged (`host/served.py`).

The service runs its own measurement job beside the load, under its own device lock, so this is
the one instrument on which nothing has measured before a load. `--weight-cap` and `--context` are
the service's evaluation bounds: without them a load is the product's, at the model's declared
context, if assessment admits it.

Scenarios, each from an empty kernel cache of the run's own:

- `lifecycle`: a load arrives during the job and preempts it, serves under the no-cost policy,
  the job runs beside the idle load, a request preempts it, `Reprice` reaches the live load, a
  new service's load starts priced; and where each answer parts from a resident load's.
- `stop`: a load stopped while the job runs beside it, then a load in the same service.
- `stop-prepare`: a cold load stopped during its Prepare with a second load queued behind the
  stop: the queued load loads and answers, and no two inference workers ever run at once, by the
  service's own records of each worker's start and of its exit once shown reaped.
- `repeat`: two prompts answered in turn on one load, where each parts from its first answer.
- `failure`: a stretch of the model file read as EIO under two answers: those that named it
  fail, the load serves on, and where its answer parts from the one before.

Every complete answer is then judged against the model's own distributions: the harness loads
the model whole on the same device and teacher-forces each answer as the scored reply of its
conversation, rendered as the service rendered it, so every row is the reference fed the served
tokens before it. The service exposes no logits, so the served tokens, not their distributions,
are held to the reference, by a bound on how far the service's arithmetic can move a margin.

The bound is measured in the run. The service computes what the reference computes but in other
batch shapes, with the kernels its own tuning chose, in their accumulation orders: so the harness
computes the same rows a second time that way, one sequence a row at a time as the service
decodes, with the kernels its own tuning search chooses, on the reference's exact tokens, and Δ
is the difference between the two at each row and token. At a row, the service can answer a
token a other than the reference's top b only by flipping that one pair, and the flip moves a's
margin below b by exactly Δa − Δb: a shift the whole row shares moves no margin, so it is no part
of the bound. The tokens a greedy choice is between are, at each row, its top and every token a
whose margin below it is within |Δa| + |Δb|, the most the measured difference could move it; no
other token can be answered by an error of the size measured. The bound B is the largest
|Δa − Δb| among those competing tokens over every row, and each answered token must be its row's
top or less than B below it. Every token whose margin the measured difference moves by its whole
margin competes (|Δa − Δb| ≤ |Δa| + |Δb|), so B holds every parting the measurement shows
possible; and B is at most 2D, D the largest |Δ| among the competing tokens. The bound is not
taken over the whole vocabulary: its tail sits tens of nats below the top, where logits differ by
several nats, and twice that accepts a corruption many nats below the top, which is what the
judgement exists to fail. Past a parting the reference is fed the served tokens, so the answer
past it is judged as any other row. Garbage from any token on fails, its tokens sitting far below
their rows' tops. A B of zero bounds nothing (the second computation moved no competing margin),
and fails as unsound. Where an answer parts from another is reported, never failed.

The judgement is sound only where the harness's tokens of an answer are the service's: an answer
retokenized to another count fails as unjudged, and so does every answer when the resident
load's own answers fail, as a judge that cannot pass the reference's own answers judges nothing.
"""

import argparse
import json
import re
from pathlib import Path

from ..engine import Step, Weights
from ..host import served
from . import Answer, Instrument, JobResult, Plan, harness, kl

SERVICE = "{target}/release/magnitude-inference"
# The service selects its device itself: what makes `--backend` hold where it would prefer another.
HIDDEN = {"vulkan": (("CUDA_VISIBLE_DEVICES", ""),)}
BUILD = Step("service", ("cargo", "build", "--release", "-p", "magnitude-service-server", "--bin", "magnitude-inference"))
# The judgement on the host, as `{run_dir}/judged.json`: the bound B, the largest move of a margin
# below a row's top between the reference's rows and the service-shaped computation's among the
# tokens competing with that top, where B lies, and for each answer its rows, how many tokens are
# their row's top, the worst margin below the top and its row, how many reach B, and how many rows
# hold a non-finite logit in either, which no margin covers. The logits are then removed.
JUDGE = (
    "import json, os, sys\n"
    "import numpy\n"
    "reference, decoded, output = sys.argv[1:4]\n"
    "header = json.load(open(os.path.join(reference, 'logits.json')))\n"
    "assert json.load(open(os.path.join(decoded, 'logits.json')))['rows'] == header['rows'], 'the two computations scored other rows'\n"
    "tokens = {line['fixture_id']: line['tokens'] for line in map(json.loads, open(os.path.join(reference, 'rendered.jsonl')))}\n"
    "rows = lambda run: numpy.memmap(os.path.join(run, 'logits.f32'), dtype='<f4', mode='r').reshape(-1, header['vocabulary'])\n"
    "a, b = rows(reference), rows(decoded)\n"
    "finite = [bool(numpy.isfinite(a[index]).all() and numpy.isfinite(b[index]).all()) for index in range(len(a))]\n"
    "bound, at = 0.0, None\n"
    "for index in range(len(a)):\n"
    "    if finite[index]:\n"
    "        reference_row, decoded_row = (numpy.asarray(side[index], dtype=numpy.float64) for side in (a, b))\n"
    "        delta = decoded_row - reference_row\n"
    "        top = int(reference_row.argmax())\n"
    "        moved = numpy.abs(delta - delta[top])\n"
    "        competing = reference_row[top] - reference_row <= numpy.abs(delta) + abs(delta[top])\n"
    "        token = int(numpy.flatnonzero(competing)[moved[competing].argmax()])\n"
    "        if moved[token] > bound:\n"
    "            bound, at = float(moved[token]), {**header['rows'][index], 'token': token,\n"
    "                                              'reference': float(reference_row[top] - reference_row[token]),\n"
    "                                              'decoded': float(decoded_row[top] - decoded_row[token])}\n"
    "judged = {}\n"
    "for index, row in enumerate(header['rows']):\n"
    "    entry = judged.setdefault(row['fixture_id'], {'rows': 0, 'top': 0, 'worst': 0.0, 'worst_row': None, 'over': 0, 'nonfinite': 0})\n"
    "    entry['rows'] += 1\n"
    "    if not finite[index]:\n"
    "        entry['nonfinite'] += 1\n"
    "        continue\n"
    "    logits = numpy.asarray(a[index], dtype=numpy.float64)\n"
    "    margin = float(logits.max() - logits[tokens[row['fixture_id']][row['position'] + 1]])\n"
    "    entry['top'] += margin == 0.0\n"
    "    entry['over'] += margin >= bound and margin > 0.0\n"
    "    if margin > entry['worst']:\n"
    "        entry['worst'], entry['worst_row'] = margin, entry['rows'] - 1\n"
    "json.dump({'bound': bound, 'at': at, 'answers': judged}, open(output, 'w'))\n"
    "del a, b\n"
    "os.unlink(os.path.join(reference, 'logits.f32'))\n"
    "os.unlink(os.path.join(decoded, 'logits.f32'))\n"
)


def strip(line: str) -> str:
    """`line` without its terminal colours."""
    return re.sub(r"\x1b\[[0-9;]*m", "", line)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", choices=served.SCENARIOS, default="lifecycle", help="what is driven (default lifecycle)")
    parser.add_argument("--context", type=int, metavar="TOKENS", help="serve at this context")
    parser.add_argument("--weight-cap", type=int, metavar="BYTES", help="stream every capped load under this cap")
    parser.add_argument("--plain", action="store_true", help="decode plainly, the model's drafter left out")


def plan(arguments, source, host) -> Plan:
    bounds = (*(("--context", str(arguments.context)) if arguments.context else ()),
              *(("--weight-cap", str(arguments.weight_cap)) if arguments.weight_cap else ()),
              *(("--plain",) if arguments.plain else ()))
    job = Step("served", (
        "{python}", "-m", "magnitude_rig.host.served", "--service", SERVICE, "--bundle", "{catalog}/model-planner-inputs.bundle",
        "--models", "{models}", "--artifacts", "{artifacts}", "--fixtures", f"{{inputs}}/{arguments.fixture.name}", "--scratch", "{root}/served",
        "--run-dir", "{run_dir}", "--model", arguments.model, "--scenario", arguments.scenario, *bounds,
    ), HIDDEN.get(arguments.backend, ()), fresh=True)
    # The reference: the model whole, at its declared defaults, on the answers the service gave;
    # then the same rows computed as the service computes them: one sequence a row at a time,
    # with the kernels a tuning search chooses, on the reference's exact tokens.
    logits = (harness.EXECUTABLE, "logits", "--artifacts", "{artifacts}", "--fixtures", f"{{runs}}/served/{served.ANSWERS}",
              "--output", "{run_dir}", "--build-identity", "{build_file}", "--cache-dir", "{engine_cache}/kernels",
              "--device", arguments.backend)
    answers = Step("logits", (*logits, "--tuning-defaults"))
    decoded = Step("logits", (*logits, "--rendered", "{runs}/answers/rendered.jsonl", "--batch", "1"))
    judge = Step("judge", ("{venv}/bin/python", "-c", JUDGE, "{runs}/answers", "{runs}/decoded", "{run_dir}/judged.json"))
    return Plan(builds=(BUILD, *kl.BUILDS), jobs=(("served", job), ("answers", answers), ("decoded", decoded), ("judge", judge)),
                device="device",
                acquires=Weights(arguments.model, headers=True), files=(arguments.fixture,))


def judged(served_run: Path, judge_run: Path) -> tuple[list[str], bool]:
    """Each answer's judgement, and whether every answer was judged and passed."""
    answers = [json.loads(line) for line in (served_run / served.ANSWERS).read_text().splitlines()]
    unjudged = json.loads((served_run / served.RESULTS).read_text()).get("unjudged", [])
    judgement = json.loads((judge_run / "judged.json").read_text())
    verdicts, bound = judgement["answers"], judgement["bound"]
    at = judgement["at"]
    where = (f" (at {at['fixture_id']} position {at['position']}, token {at['token']}: {at['reference']:.4f} below its "
             f"row's top against {at['decoded']:.4f})") if at else ""
    lines = [f"largest move between the reference and the service-shaped computation of a margin below a row's top, "
             f"among the tokens competing with it, {bound:.4f}{where}: an answered token may sit under {bound:.4f} "
             f"nats below its row's top",
             *(f"{name}: unjudged, it generated nothing" for name in unjudged)]
    failed = {*unjudged}
    if bound == 0.0:
        lines.insert(0, "the service-shaped computation moved no margin of a token competing with its row's top: it bounds nothing, "
                        "so nothing is judged")
        failed.add("bound")
    for answer in answers:
        name, given = answer["id"], answer["served"]
        verdict = verdicts.get(name)
        # The service's count is the answer's tokens and, where it stopped, its end of turn.
        counts = (given["tokens"],) if given["finish"] == "length" else (given["tokens"], given["tokens"] - 1)
        if verdict is None or verdict["rows"] not in counts:
            lines.append(f"{name}: unjudged, the harness's tokens of it are not the service's "
                         f"({verdict and verdict['rows']} rows, {given['tokens']} served tokens, {given['finish']})")
            failed.add(name)
            continue
        passed = verdict["over"] == 0 and verdict["nonfinite"] == 0
        if not passed:
            failed.add(name)
        worst = "" if verdict["worst_row"] is None else f" at token {verdict['worst_row']}"
        over = f", {verdict['over']} past {bound:.4f}" if verdict["over"] else ""
        nonfinite = f", {verdict['nonfinite']} non-finite" if verdict["nonfinite"] else ""
        lines.append(f"{name}: {verdict['top']}/{verdict['rows']} tokens their row's top, worst {verdict['worst']:.3f} nats"
                     f"{worst}{over}{nonfinite}{'' if passed else ' FAILED'}")
    if any(name.startswith("resident-") for name in failed):
        lines.insert(0, "the judge fails the resident load's own answers: it judges nothing")
    return lines, not failed


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["served"]
    written = result.directory / served.RESULTS
    if not written.exists():
        return Answer((f"served exited {result.code}", *result.output()[-25:]), False)
    selected = [line for log in sorted(result.directory.glob("service-*.log"))
                for line in log.read_text(errors="replace").splitlines() if "serving device selected" in line]
    wrong = [line for line in selected if f'backend="{arguments.backend}"' not in strip(line)]
    report = tuple(json.loads(written.read_text())["report"])
    if wrong:
        return Answer((f"the service did not select {arguments.backend}: {strip(wrong[0])[-120:]}", *report), False)
    for name in ("answers", "decoded", "judge"):
        if name not in results or results[name].code != 0:
            ended = f"exited {results[name].code}" if name in results else "did not run"
            output = results[name].output()[-15:] if name in results else []
            return Answer((*report, f"the answers' judgement: {name} {ended}", *output), False)
    lines, passed = judged(result.directory, results["judge"].directory)
    return Answer((*report, *lines), result.code == 0 and passed)


INSTRUMENT = Instrument(
    name="served",
    kind="correctness",
    question="Does a load work through the real service, its measurement job beside it?",
    observable="the service's API answers and logs",
    selectors=("backend", "host", "build", "model", "fixture", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
)
