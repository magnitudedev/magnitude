"""`mrig ab INSTRUMENT --base A --candidate B`: is B better or worse than A, by that instrument?

Both builds are synchronized and built on one host, then measured under one hold. An
instrument with its own pairing (`kl`) decides its jobs and its judgement. A measuring
instrument runs in `--rounds` alternating rounds, base first on even rounds and candidate first
on odd ones, so drift such as heat affects both alike; each measure's side value is the median
over its rounds, and its noise the larger of the widest interval any round reported and half
the spread across rounds. A change counts only beyond both sides' noise; anything smaller is
reported as below what the instrument resolved, never as a difference.
"""

import argparse
import statistics

from . import engine, selectors, sources
from .instruments import REGISTRY, Answer, Instrument, JobResult, Measure
from .results import perform

SIDES = ("base", "candidate")
SHOWN = 12


def configure(parser: argparse.ArgumentParser) -> None:
    instruments = parser.add_subparsers(dest="instrument", required=True, metavar="INSTRUMENT")
    for name, instrument in REGISTRY.items():
        if not (instrument.comparable or instrument.pairing):
            continue
        command = instruments.add_parser(name, help=instrument.question)
        selectors.add(command, tuple(s for s in instrument.selectors if s != "build"))
        command.add_argument("--base", type=selectors.build, required=True, help="a git ref, or worktree")
        command.add_argument("--candidate", type=selectors.build, default=sources.WORKTREE,
                             help="worktree (default) or a git ref")
        if instrument.pairing is None:
            command.add_argument("--rounds", type=int, default=3, help="alternating rounds (default 3)")
        instrument.configure(command)
        command.set_defaults(handler=lambda arguments, i=instrument: run(i, arguments))


def _jobs(instrument: Instrument, arguments, plans: dict) -> tuple[engine.Job, ...]:
    if instrument.pairing is not None:
        return tuple(engine.Job(side, name, step) for side, name, step in instrument.pairing.jobs(arguments, *SIDES))
    jobs = []
    for index in range(arguments.rounds):
        for side in SIDES if index % 2 == 0 else SIDES[::-1]:
            steps = instrument.alternate(arguments, side, index) if instrument.alternate else plans[side].jobs
            for name, step in steps:
                jobs.append(engine.Job(side, f"{side}-r{index}-{name}", step))
    return tuple(jobs)


def run(instrument: Instrument, arguments) -> int:
    if arguments.base == arguments.candidate:
        raise SystemExit("ab: the base and the candidate are the same source")
    with sources.opened(arguments.base) as base, sources.opened(arguments.candidate) as candidate:
        return _run(instrument, arguments, {"base": base, "candidate": candidate})


def _run(instrument: Instrument, arguments, resolved: dict[str, sources.Source]) -> int:
    plans = {side: instrument.plan(arguments, resolved[side]) for side in SIDES}
    first = plans["base"]
    request = engine.Request(
        instrument=f"ab-{instrument.name}",
        purpose=arguments.purpose,
        estimate=arguments.estimate,
        backend=arguments.backend,
        sides=tuple(engine.Side(side, resolved[side], plans[side].builds) for side in SIDES),
        jobs=_jobs(instrument, arguments, plans),
        device=first.device,
        host=arguments.host,
        acquires=first.acquires,
        files=first.files,
    )
    if instrument.pairing is not None:
        return perform(arguments, request, lambda results: instrument.pairing.answer(arguments, results))
    return perform(arguments, request, lambda results: compare(instrument, arguments, results, first.jobs))


def compare(instrument: Instrument, arguments, results: dict[str, JobResult], jobs) -> Answer:
    rounds: dict[str, list[dict[str, Measure]]] = {side: [] for side in SIDES}
    for index in range(arguments.rounds):
        for side in SIDES:
            answer = instrument.answer(
                arguments, {name: results[f"{side}-r{index}-{name}"] for name, _ in jobs}
            )
            if not answer.ok:
                return Answer((f"{side} round {index}:", *answer.lines), False)
            rounds[side].append(answer.measures)
    names = [name for name in rounds["base"][0] if all(name in measures for side in SIDES for measures in rounds[side])]
    rows = []
    for name in names:
        values = {side: [measures[name] for measures in rounds[side]] for side in SIDES}
        sample = values["base"][0]
        median = {side: statistics.median(m.value for m in values[side]) for side in SIDES}
        noise = {side: _noise(values[side]) for side in SIDES}
        delta = median["candidate"] - median["base"]
        tolerance = noise["base"] + noise["candidate"]
        if abs(delta) <= tolerance:
            verdict = "below resolution"
        else:
            verdict = "better" if (delta < 0) == sample.lower_is_better else "worse"
        relative = delta / median["base"] if median["base"] else 0.0
        rows.append((name, sample.unit, median, tolerance, delta, relative, verdict))
    rows.sort(key=lambda row: (row[0] != next(iter(names), None), -abs(row[4]) / (row[3] or 1e-12)))
    lines = [f"{arguments.base} → {arguments.candidate} · {arguments.rounds} alternating rounds"]
    for name, unit, median, tolerance, delta, relative, verdict in rows[:SHOWN]:
        lines.append(
            f"  {name[:46]:<47}{median['base']:>10.3f} → {median['candidate']:<10.3f}{unit:<3}"
            f"{100 * relative:+6.1f}% (±{tolerance:.3f})  {verdict}"
        )
    if len(rows) > SHOWN:
        lines.append(f"  … {len(rows) - SHOWN} more measures, all in the job results")
    worse = any(row[6] == "worse" for row in rows)
    return Answer(tuple(lines), not worse)


def _noise(measures: list[Measure]) -> float:
    widest = max((m.resolution or 0.0) for m in measures)
    values = [m.value for m in measures]
    return max(widest, (max(values) - min(values)) / 2)
