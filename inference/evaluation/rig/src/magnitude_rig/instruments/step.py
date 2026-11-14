"""`step`: how fast is the model's step at this context, prefill and decode, and where does the
decode step's time go?

The harness `time` at one context and `--sequences` decoded together (one by default), without
drafting (the drafter is `draft`'s), on the host's kernel cache (so stored tuning is
reused, never searched afresh between runs): the median
step with its 95% interval from the measured steps, the launches per step, and the device
time per entry from `--attribution` further steps, each launch timed alone; and the prefill of
the context, the median of `--prefill-samples` warm samples (after a cold one) with its
interval, each, like the decode window, run with a streamed load's expert bank at rest (the
harness sets the others apart), and (with attribution) one further prefill's device time per entry.

`--weight-cap BYTES` (repeatable) also times the model streamed under each cap, in the same
hold after the resident load: each streamed load reads its routed experts from storage (`fresh`
jobs), and its decode is judged against the performance model predicted from the resident
load's decode and the load's own read line and conversion, and against what assessment
estimates from the executor costs the measurement job (`magnitude-assess` on the model) stores
and the generic miss model. Each
streamed cell reports its speed against resident, the model's prediction and their ratio, the
estimate and its ratio, its misses and how they were anticipated, and whether its reads were
served from the file cache (its timing is then void). Each also reports its performance bar, the
best step any implementation could reach at the cap on this machine (the oracle bank's misses
served at host memory's measured rate beside the layers' hits), with the gaps measured − model
(implementation) and model − bar (design, by term). What a streamed load holds against its
charge is `memory`'s.

`--resident RUN` takes the resident cell from earlier step run RUN's recorded result instead of
timing it again, when the question is only the streamed cells': the streamed cells are predicted
from, and the ratios and verdict read, that record. Only a record of the same resident cell
serves (`_accept_resident`): the same host, model, backend, build identity, context, sequences,
lookahead, timing, tuning budget and fixture set; any other refuses the host, so the run goes to the host the
record was measured on, or fails naming what differs.

`--streamed-only` asks only the streamed cells' absolute terms (their steps, the window's device
busy and idle, its boundaries' held and queued time, the bank's reads, the estimate), so it times
no resident cell: no streamed cell is priced against one (no model, no bar, no tokens compared),
and its answer holds no × resident ratio, by construction and as an invariant it checks.

Each cell reports whether its loads tuned to completion: a load whose tuning refused work for
want of budget voids its cell. `--tuning-budget` gives each load's first tuning that many seconds
(a budget is part of every stored result's key, so the host tunes each kernel once per budget).

A cell's numbers should depend on no search. `--tuning-record FILE` keeps the configurations a run
chose in FILE on this machine: a reference, recorded once per hardware class (a run whose search
stopped at its budget still records, and is not voided: the file is its point).
`--tuning-replay FILE` runs them, and `--tuning-defaults` every kernel's declared default: neither
walks, searches or measures anything for tuning, nothing voids its cell, and each cell reports the
units the pin did not hold, which ran at their defaults: how good the reference is for this cell.

Under `mrig ab step` every round runs the base's first tuning (`harness`), so the comparison is
of the code, never of two tunings; `--tuning own` lets each side tune, when tuning is what
changed.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .. import store
from ..engine import Refused, Step, Weights
from . import Answer, Instrument, JobResult, Measure, Plan, harness, lacking, unrecorded
from .assess import preceded, priced
from .resolution import median_interval

SHOWN = 12
UNMEASURED = "magnitude-assess: not measured: "


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequences", type=int, default=1,
                        help="sequences prefilled and decoded together, each at the context (default 1)")
    parser.add_argument("--warm", type=int, default=8,
                        help="decode steps before the measured ones (default 8)")
    parser.add_argument("--steps", type=int, default=64, help="measured decode steps (default 64)")
    parser.add_argument("--prefill-samples", type=int, default=6,
                        help="warm prefill samples of the context, after one cold (default 6)")
    parser.add_argument("--attribution", type=int, default=3,
                        help="further steps attributed by entry, each launch timed alone (default 3; 0 skips)")
    parser.add_argument("--lookahead", choices=("on", "off"), default="on",
                        help="queue each step's successor before it completes, as served")
    parser.add_argument("--weight-cap", type=int, action="append", metavar="BYTES",
                        help="also time the model streamed under this weight cap, its experts read from storage (repeatable)")
    parser.add_argument("--tuning", choices=("replay-base", "own"), default="replay-base",
                        help="under ab: every round runs the base's first tuning (default), or each side its own")
    parser.add_argument("--untimed", action="store_true",
                        help="the question counts, not times (a routing trace): hold the device without the "
                             "thermal gate, which only timed measurements need, and measure no overlap; its "
                             "times are not judged")
    parser.add_argument("--resident", metavar="RUN",
                        help="take the resident cell from step run RUN's recorded result, not timed again "
                             "(with --weight-cap; refused unless RUN measured the same host, model, backend, "
                             "build, context, sequences, lookahead, timing, tuning budget and fixture set)")
    parser.add_argument("--streamed-only", action="store_true",
                        help="with --weight-cap: time no resident cell; report the streamed cells' absolute "
                             "terms alone, never a ratio to resident")
    parser.add_argument("--fresh-tuning", action="store_true",
                        help="the resident load searches afresh, ignoring stored tuning results: "
                             "its trace is then the whole search, for a curve of tuning over time")
    parser.add_argument("--tuning-budget", type=float, metavar="SECONDS",
                        help="seconds each load's first tuning may spend (default the engine's served budget); "
                             "each cell reports whether its tuning completed within it")
    pin = parser.add_mutually_exclusive_group()
    pin.add_argument("--tuning-defaults", action="store_true",
                     help="run every kernel's declared default configuration and tune nothing, so a cell's "
                          "numbers depend on no search: comparable across hosts' tunings and across builds")
    pin.add_argument("--tuning-record", type=Path, metavar="FILE",
                     help="keep the configurations this run tuned to, in FILE on this machine; refused if it exists")
    pin.add_argument("--tuning-replay", type=Path, metavar="FILE",
                     help="run the configurations of a recorded FILE and tune nothing; a unit it lacks runs at "
                          "its default and is reported")


def time(arguments, family: str, *extra: str) -> tuple[str, ...]:
    return (
        harness.EXECUTABLE, "time",
        "--artifacts", "{artifacts}", "--fixtures", f"{{inputs}}/{family}.jsonl",
        "--output", "{run_dir}", "--build-identity", "{build_file}",
        "--cache-dir", "{engine_cache}/kernels", "--device", arguments.backend,
        "--contexts", str(arguments.context), "--concurrency", str(arguments.sequences),
        "--warm", str(arguments.warm), "--steps", str(arguments.steps),
        "--attribution-steps", str(arguments.attribution),
        *(("--prefill-attribution",) if arguments.attribution else ()),
        "--prefill-samples", str(arguments.prefill_samples + 1),
        "--lookahead", arguments.lookahead,
        *(("--untimed",) if arguments.untimed else ()),
        *(("--tuning-budget", str(arguments.tuning_budget)) if arguments.tuning_budget else ()),
        *_pin(arguments), *extra,
    )  # fmt: skip


def _pin(arguments) -> tuple[str, ...]:
    """The harness's arguments for the tuning pin `arguments` names."""
    if arguments.tuning_defaults:
        return ("--tuning-defaults",)
    if arguments.tuning_record:
        return harness.RECORD
    if arguments.tuning_replay:
        return ("--tuning-replay", f"{{inputs}}/{arguments.tuning_replay.name}")
    return ()


def jobs(arguments, prefix: str, *pinning: str) -> tuple[tuple[str, Step], ...]:
    """The resident step (unless taken from a record, `--resident`), then, under weight caps, the
    measurement and each streamed step, predicted from the resident step's result
    (`{runs}/<prefix>step`, or the record's in the inputs)."""
    family = arguments.fixture.stem
    if arguments.streamed_only:
        resident, against = (), ()
    elif arguments.resident is None:
        fresh = ("--fresh-tuning",) if arguments.fresh_tuning else ()
        resident = (("step", Step("time", time(arguments, family, *pinning, *fresh))),)
        against = ("--resident-decode", f"{{runs}}/{prefix}step/result.json")
    else:
        resident, against = (), ("--resident-decode", f"{{inputs}}/{_recorded(arguments).name}")
    if not arguments.weight_cap:
        return resident
    caps = tuple(
        (harness.streamed(cap), Step("time", time(
            arguments, family, *pinning, "--weight-cap", str(cap), *against,
        ), fresh=True))
        for cap in arguments.weight_cap
    )  # fmt: skip
    return (*resident, *caps)


def plan(arguments, source, host) -> Plan:
    if arguments.streamed_only and not arguments.weight_cap:
        raise SystemExit("step: --streamed-only times streamed cells; give --weight-cap")
    if arguments.streamed_only and arguments.resident is not None:
        raise SystemExit("step: --streamed-only takes no resident cell, recorded or timed; drop --resident")
    if arguments.resident is not None:
        if not arguments.weight_cap:
            raise SystemExit("step: --resident serves streamed cells; give --weight-cap")
        _accept_resident(arguments, _wanted(arguments, source, host))
    if arguments.tuning_record and arguments.tuning_record.exists():
        raise SystemExit(f"step: {arguments.tuning_record} is recorded already; a reference is recorded once")
    if arguments.tuning_replay and not arguments.tuning_replay.is_file():
        raise SystemExit(f"step: {arguments.tuning_replay} is not a recorded tuning")
    return priced(Plan(
        builds=(harness.BUILD,),
        jobs=jobs(arguments, ""),
        device="device" if arguments.untimed else "timed",
        acquires=Weights(arguments.model),
        files=(arguments.fixture, *(() if arguments.resident is None else (_recorded(arguments),)),
               *((arguments.tuning_replay,) if arguments.tuning_replay else ())),
    ), bool(arguments.weight_cap), arguments.backend)


def _recorded(arguments) -> Path:
    """Where run `--resident`'s resident result is recorded."""
    return store.results() / arguments.resident / "step" / "result.json"


def _option(command: list, flag: str) -> str | None:
    return command[command.index(flag) + 1] if flag in command else None


def _identity(record: dict) -> dict:
    """What a harness `time` record's cells were measured as: where, of what and how."""
    command = record["command"]
    replay = _option(command, "--tuning-replay")
    return {
        "host": record["host"]["alias"], "model": record["model"], "build": record["build"],
        "backend": _option(command, "--device"), "context": _option(command, "--contexts"),
        "sequences": _option(command, "--concurrency"), "lookahead": _option(command, "--lookahead"),
        "untimed": "--untimed" in command, "tuning_budget": _option(command, "--tuning-budget"),
        "tuning_defaults": "--tuning-defaults" in command,
        "tuning_replay": replay and Path(replay).name,
        "fixtures": record["fixture_set"]["digest"],
    }


def _wanted(arguments, source, host: str) -> dict:
    """`_identity` of the cells this run would measure on `host`."""
    return {
        "host": host, "model": arguments.model, "build": source.identity.to_json(),
        "backend": arguments.backend, "context": str(arguments.context), "sequences": str(arguments.sequences),
        "lookahead": arguments.lookahead, "untimed": arguments.untimed,
        "tuning_budget": str(arguments.tuning_budget) if arguments.tuning_budget else None,  # as `time` passes it
        "tuning_defaults": arguments.tuning_defaults,
        "tuning_replay": arguments.tuning_replay and arguments.tuning_replay.name,
        "fixtures": hashlib.sha256(arguments.fixture.read_bytes()).hexdigest(),
    }


def _accept_resident(arguments, wanted: dict) -> None:
    """Whether run `--resident`'s recorded resident cell can serve this run: raises Refused
    unless the step succeeded and was measured as `wanted` (`_identity`), naming each
    difference. The one judge of a recorded resident step."""
    path = _recorded(arguments)
    if not path.is_file() or not (path.parent.parent / "rig.json").is_file():
        raise Refused(f"run {arguments.resident} has no recorded resident step ({path})")
    result = _recorded_step(arguments)
    if result.code != 0:
        raise Refused(f"run {arguments.resident}'s resident step exited {result.code}")
    recorded = _identity(json.loads(path.read_text()))
    differs = [f"{name} {recorded[name]!r}, not {wanted[name]!r}" for name in wanted if recorded[name] != wanted[name]]
    if differs:
        raise Refused(f"run {arguments.resident}'s resident step was measured at another " + "; ".join(differs))


def _recorded_step(arguments) -> JobResult:
    directory = _recorded(arguments).parent
    code = json.loads((directory.parent / "rig.json").read_text())["codes"].get("step")
    return JobResult(directory, directory.parent / "logs" / "job-step.log", code)


def alternate(arguments, side: str, index: int) -> tuple[tuple[str, Step], ...]:
    if arguments.resident is not None:
        raise SystemExit("ab step: each side's resident step is timed in its own rounds; --resident is step's alone")
    if arguments.tuning_record:
        raise SystemExit("ab step: a reference is recorded by step; under ab give --tuning-replay")
    if arguments.tuning == "own" or _pin(arguments):
        pinning = ()
    elif side == "base" and index == 0:
        pinning = harness.RECORD
    else:
        pinning = harness.REPLAY
    return preceded(jobs(arguments, f"{side}-r{index}-", *pinning), bool(arguments.weight_cap), arguments.backend)


def cell(result: JobResult, phase: str) -> dict:
    record = json.loads((result.directory / "result.json").read_text())
    return next(cell for cell in record["measurements"]["speed"] if cell["phase"] == phase)


# What a step answer measures, which every build's record must hold (`lacking`).
DECODE_MEASURED = ("detail.steps",)
PREFILL_MEASURED = ("detail.samples_ms", "detail.rows")
STREAMED_MEASURED = ("detail.window.wall_ms_per_step",)
# The optional sections' fields, each rendered only when a build records all of them.
ENTRY_FIELDS = ("entry", "device_ms_per_step", "launches_per_step", "share")
MODEL_FIELDS = (
    "miss_shares.reading", "miss_shares.unanticipated", "figures.overhead_ms",
    "predicted_tok_s", "ratio", "misses_per_step", "all_hit_steps", "window_steps",
)
PAGE_FIELDS = ("expert_pages",)
GROUP_FIELDS = ("layers", "groups", "sizes", "steps", "measured_ms", "bar_ms", "ratio", "bar_terms", "ceilings",
                "achieved")
LAYER_FIELDS = ("union", "conveyed", "pulled", "seeded", "parked_ms")


def _unmeasured(result: JobResult, *decode: str) -> str | None:
    """Why the job's record holds no measurement this answer can read, if it does not."""
    record = json.loads((result.directory / "result.json").read_text())
    phases = {cell["phase"]: cell for cell in record["measurements"]["speed"]}
    missing = [f"{phase} cell" for phase in ("decode", "prefill") if phase not in phases]
    if not missing:
        missing = (lacking(phases["decode"], *DECODE_MEASURED, *decode)
                   + lacking(phases["prefill"], *PREFILL_MEASURED))
    if not missing and any(lacking(step, "interval_ms") for step in phases["decode"]["detail"]["steps"][1:]):
        missing = ["decode step interval_ms"]  # each step's time from the completion before it
    return f"this build's harness records no {', '.join(missing)}" if missing else None


def _entries(detail: dict) -> list[dict] | str:
    """The attribution's entries, slowest first, or why this build's record has none to show."""
    attribution = detail.get("attribution") or {}
    entries = attribution.get("entries", [])
    missing = sorted({path for row in entries for path in lacking(row, *ENTRY_FIELDS)})
    if missing:
        return unrecorded("attribution", missing)
    return sorted(entries, key=lambda row: -row["device_ms_per_step"])


def _table(title: str, entries: list[dict] | str, unit: str) -> list[str]:
    if isinstance(entries, str):
        return [entries]
    lines = [f"  {title:<44}{unit:>9}{'launches':>10}{'share':>8}"]
    for row in entries[:SHOWN]:
        lines.append(
            f"  {row['entry'][:43]:<44}{row['device_ms_per_step']:>9.3f}"
            f"{row['launches_per_step']:>10.1f}{100 * row['share']:>7.1f}%"
        )
    if len(entries) > SHOWN:
        rest = sum(row["device_ms_per_step"] for row in entries[SHOWN:])
        lines.append(f"  {f'{len(entries) - SHOWN} more entries':<44}{rest:>9.3f}")
    return lines


def _timed(arguments, result: JobResult) -> tuple[dict, dict, float, float, float, float]:
    detail = cell(result, "decode")["detail"]
    prefill = cell(result, "prefill")["detail"]
    # Each step's time from the completion before it (the first has none; `_unmeasured` refuses
    # a record whose later steps lack one).
    step_ms, half = median_interval([step["interval_ms"] for step in detail["steps"][1:]])
    prefill_ms, prefill_half = median_interval(list(prefill["samples_ms"]))
    return detail, prefill, step_ms, half, prefill_ms, prefill_half


def _unavailable(value) -> str | None:
    """Why a recorded figure is unavailable, where the harness recorded one."""
    return value.get("unavailable") if isinstance(value, dict) else None


def _group(group: dict | None) -> list[str]:
    """The prefill's last timed sample, run whole as prompt groups, against the harness's bar
    (`bar::PrefillBar`): the slowest of the groups' cold bytes read at the storage device's peak,
    the resident prefill plus the conveyed records' convert at the convert's ceiling, and the bytes
    moved over a host tier's link at the link's ceiling; or why the harness could not price it.
    Beside it, the ceilings, and what the run's own convert and tier copy achieved (the attributed
    prefill). Then what the layers took: the union per layer and group, the experts conveyed (read
    from the file), pulled (from the host tier) and seeded, and what the steps waited for experts."""
    if group is None:
        return ["  prompt groups: none (the prompt did not run whole in groups)"]
    missing = lacking(group, *GROUP_FIELDS) or sorted(
        {path for layer in group["layers"] for path in lacking(layer, *LAYER_FIELDS)}
    )
    if missing:
        return [unrecorded("prompt groups", missing)]
    layers = group["layers"]
    unions = [layer["union"] / group["groups"] for layer in layers]
    pulled = sum(layer["pulled"] for layer in layers)
    sizes = sorted(set(group["sizes"]))
    lines = [f"  {group['groups']} prompt groups of {'/'.join(map(str, sizes))} steps, {group['steps']} in all · "
             f"measured {group['measured_ms']:.1f} ms · "
             + (f"no bar ({reason})" if (reason := _unavailable(group["bar_ms"])) else
                f"bar {group['bar_ms']:.1f} ms · bar/measured {group['ratio']:.3f}")]
    terms = group["bar_terms"]
    if terms is not None:
        lines.append(f"  bar terms: read {terms['read_ms']:.1f} ms · device "
                     f"{terms['resident_ms'] + terms['convert_ms']:.1f} ms (resident {terms['resident_ms']:.1f} + "
                     f"convert {terms['convert_ms']:.1f}) · link {terms['link_ms']:.1f} ms")
    ceilings, achieved = group["ceilings"], group["achieved"]
    convert = ceilings["convert_us_per_record"]
    link = ceilings["link_bytes_per_second"]
    lines.append("  ceilings: convert " + (f"unavailable ({reason})" if (reason := _unavailable(convert)) else
                                           f"{min(convert):.1f}–{max(convert):.1f} µs/record")
                 + " · link " + ("none (unified memory)" if link is None else
                                 f"unavailable ({reason})" if (reason := _unavailable(link)) else
                                 f"{link / 1e9:.2f} GB/s"))
    if reason := _unavailable(achieved):
        lines.append(f"  achieved: unavailable ({reason})")
    else:
        copy = achieved["copy_us_per_record"]
        lines.append(f"  achieved: convert {achieved['convert_ms']:.1f} ms · copy {achieved['copy_ms']:.1f} ms"
                     + ("" if copy is None else f" ({copy:.1f} µs/record)"))
    lines.append(
        f"  union per layer and group {min(unions):.0f}–{max(unions):.0f} (mean {sum(unions) / len(unions):.1f}) · "
        f"conveyed {sum(layer['conveyed'] for layer in layers)} · pulled {pulled} · "
        f"seeded {sum(layer['seeded'] for layer in layers)} · "
        f"waited for experts {sum(layer['parked_ms'] for layer in layers):.1f} ms")
    return lines


def _lines(figures: dict) -> list[str]:
    """The lines a streamed layer's misses are served by, as the load measured them, each distinct
    one once (layers of one record form share them): a tier record read in place (`served`, and
    `shared` beside a prompt group's transfers), a layer's file misses' landings past their read
    (the line through landings of one record and of a request's at once), and the CPU site (a
    record's seconds on the device and through their one memory, and the split it chose of the
    layer's features with the CPU's request line measured there)."""
    if "tier" not in figures:
        return ["  lines not recorded by this build's harness"]
    us = 1e6
    line = lambda record: f"{record['fixed_seconds'] * us:.1f} µs + {record['record_seconds'] * us:.1f} µs/record"
    lines = []
    tiers = {(line(tier["served"]), None if tier.get("shared") is None else line(tier["shared"]))
             for tier in figures["tier"] if tier is not None}
    for served, shared in sorted(tiers, key=str):
        lines.append(f"  tier read in place {served}" + ("" if shared is None else f" · beside a prompt group {shared}"))
    if not tiers:
        lines.append("  tier none: the bank has no host tier")
    files = sorted({line(file["landing"]) for file in figures.get("file") or [] if file})
    lines += [f"  file landings past their read {landing}" for landing in files]
    def share(site: dict) -> str:
        if site["share"] == "None":
            return f"split {site['features']} of {site['features']} features"
        features = site["share"]["Features"]
        return f"split {features['extent']} of {site['features']} features, its CPU request {line(features['request'])}"
    sites = sorted({(site["device_record_seconds"], site["memory_record_seconds"], share(site))
                    for site in figures.get("site") or [] if site})
    lines += [f"  CPU site: device {device * us:.1f} µs/record, memory {memory * us:.1f} µs/record · {chosen}"
              for device, memory, chosen in sites]
    return lines


EVENT = "magnitude-harness: event "
LANDINGS = "expert landings"
COMPLETE = "the host tier keeps every expert"


def _hints(detail: dict) -> list[str]:
    """The window's hints by the engine's one recall and precision (`HintCost::recall`,
    `HintCost::precision`): recall over the misses a ring could save, its early landings among
    them; the unhintable misses (layers no ring hints) on a line of their own."""
    hints = (detail.get("reads") or {}).get("hints")
    if not hints or "recall" not in hints:
        return []
    steps = hints["layer_steps"] / len(detail["experts"]) if detail.get("experts") and hints["layer_steps"] else None
    per = (lambda count: f"{count / steps:.1f}/step") if steps else (lambda count: str(count))
    recall = "unmeasured" if hints["recall"] is None else f"{hints['recall']:.3f}"
    precision = "unmeasured" if hints["precision"] is None else f"{hints['precision']:.3f}"
    return [
        f"  hints recall {recall} (named {per(hints['named'])}, {per(hints['early'])} landed early, of "
        f"{per(hints['hintable'] + hints['early'])} hintable) · precision {precision} of "
        f"{per(hints['candidates'])} candidates · reads {per(hints['reads'])}",
        f"  unhintable misses {per(hints['unhintable'])} (layers no ring hints)",
    ]


def _bank(result: JobResult) -> list[str]:
    """What the load's expert bank did, from its log: when its host tier first kept every expert,
    each growth decision's outcome counted (the same decision is made once while its inputs hold),
    each resize that finished (its holding, sizes, bytes and how long it held the owner), and the
    landings' time account."""
    decisions: dict[str, int] = {}
    resizes = []
    account = None
    complete = []
    for text in result.output():
        if (start := text.find(EVENT)) < 0:
            continue
        event = json.loads(text[start + len(EVENT):])
        message = event["message"]
        if message.startswith(LANDINGS):
            account = message.removeprefix("expert ")
        elif message.startswith(COMPLETE):
            complete.append("  " + message)
        elif message == "expert bank growth decision":
            key = f"{event.get('holding')} {event.get('outcome')}"
            decisions[key] = decisions.get(key, 0) + 1
        elif message in ("memory grown", "memory released"):
            change = "grown" if message == "memory grown" else "released"
            amount = event.get("grown", event.get("released"))
            # A growth joins slot by slot: its `prepared` is its first slot's join, its
            # `installed` its last's, each microseconds before the event.
            joined = (message == "memory grown" and all(
                event.get(key) is not None for key in ("decided_before", "prepared_before", "installed_before")))
            resizes.append(f"  {event.get('holding')} {change} {event.get('from')} → {event.get('to')}"
                           + ("" if amount is None else f" ({amount / 1e9:.2f} GB)")
                           + ("" if event.get("micros") is None else f" in {event['micros'] / 1e6:.2f} s")
                           + ("" if not joined else
                              f", first slot joined {(event['decided_before'] - event['prepared_before']) / 1e6:.3f} s"
                              f" and last {(event['decided_before'] - event['installed_before']) / 1e6:.3f} s"
                              " after its decision")
                           + ("" if event.get("refused") is None else f", cut short: {event['refused']}"))
    lines = complete
    if decisions:
        lines.append("  growth decisions " + ", ".join(f"{key} ×{count}" for key, count in sorted(decisions.items())))
    lines += resizes
    if account is not None:
        lines.append(f"  {account}")
    return lines


ESTIMATE_FIELDS = ("slots", "tok_s", "ratio", "record.decode_steps")


def _estimate(estimate: dict | None) -> str:
    """What assessment estimates the cell at, from headers and the machine's figures, against the
    window's measured mean step, each per sequence."""
    if estimate is None:
        return "  estimate not recorded by this build's harness"
    if "unavailable" in estimate:
        return f"  estimate unavailable: {estimate['unavailable']}"
    if missing := lacking(estimate, *ESTIMATE_FIELDS):
        return unrecorded("estimate", missing)
    return (f"  estimate {estimate['tok_s']:.1f} tok/s at {estimate['slots']} slots · measured/estimate "
            f"{estimate['ratio']:.3f} (from a record of {estimate['record']['decode_steps']} decode steps)")


BAR_FIELDS = ("form", "step_ms", "tok_s", "bar_ms", "window.capacity_per_step",
              "window.compulsory_per_step", "frontier_ms", "misses_per_step.belady", "misses_per_step.ranking",
              "misses_per_step.live", "misses_per_step.design", "misses_per_step.implementation", "gaps", "router")


def _bar(bar: dict | None, bank: dict | None) -> list[str]:
    """The cell's performance bar (the harness's `bar`): the share of experts the bank holds, the
    measured step, the ceiling (Bélády over the window the cell timed, its compulsory misses priced)
    and the online frontier, each × the resident step; the window's misses a step of Bélády (its
    compulsory share), the ranking's ideal replay and the run, with the design and implementation gaps; a unified cell's terms and gap split;
    the gaps by the model's terms; and the streamed route's device time against the resident router's."""
    if bar is None:
        return ["  bar not recorded by this build's harness"]
    if "unavailable" in bar:
        return [f"  bar unavailable: {bar['unavailable']}"]
    if missing := lacking(bar, *BAR_FIELDS):
        return [unrecorded("bar", missing)]
    ms, parts, misses, window = bar["step_ms"], bar["bar_ms"], bar["misses_per_step"], bar["window"]
    resident = parts["resident"]
    times = lambda step_ms: f"{resident / step_ms:.3f}× resident ({step_ms:.3f} ms)"
    share = "unknown" if not bank else f"{100 * bank['slots'] / bank['experts']:.1f}% of experts resident"
    lines = [
        (f"  {bar['form']}: {share} · measured {times(ms['measured'])} · ceiling {times(ms['bar'])} · "
         f"frontier {times(bar['frontier_ms'])}"
         + (f" · reachable {times(bar['reachable_ms'])}" if bar.get("reachable_ms") else "")),
        (f"  window misses/step: Bélády {misses['belady']:.1f} (compulsory {window['compulsory_per_step']:.1f}) · "
         f"ranking {misses['ranking']:.1f} · "
         f"live {misses['live']:.1f} (design {misses['design']:+.1f}, implementation {misses['implementation']:+.1f})"),
    ]
    unified = bar.get("unified")
    if unified:
        split = unified["gap_split_ms"]
        eta = unified["overlap"]
        lines.append(
            f"  unified: η {'unmeasured' if isinstance(eta, dict) else f'{eta:.3f}'} · schedule "
            f"{unified['schedule_ms']:.3f} ms · sustained {unified['sustained_ms']:.3f} ms · gap ms/step: "
            + ", ".join(f"{name.replace('_', ' ')} {value:+.3f}" for name, value in split.items())
        )
    gaps = bar["gaps"]
    if "unavailable" in gaps:
        lines.append(f"  gaps unavailable: {gaps['unavailable']}")
    else:
        design = gaps["design_ms"]
        # Without an all-hit step the model's σ is unmeasured, so its route term is too.
        route = "unmeasured" if design["route"] is None else f"{design['route']:+.3f}"
        total = "none" if design["total"] is None else f"{design['total']:+.3f}"
        lines.append(
            f"  gap ms/step: implementation (measured − model) {gaps['implementation_ms']:+.3f} · design (model − bar) "
            f"{total}: route {route}, exposed {design['exposed']:+.3f}"
        )
    router = bar["router"]
    if "absent" in router:
        lines.append(f"  router per-entry times absent: {router['absent']}")
    else:
        streamed, resident = router["streamed"], router["resident"]
        lines.append(
            f"  router: streamed {streamed['us_per_step']:.1f} µs/step ({streamed['us_per_launch']:.2f} µs/launch) · "
            f"resident {resident['us_per_step']:.1f} µs/step ({resident['us_per_launch']:.2f} µs/launch)"
        )
    return lines


REFUSED = "magnitude-harness: memory "


def _failed(label: str, result: JobResult) -> list[str]:
    """A harness that exited unmeasured: refused by memory admission, with the reading the band
    was judged on, when it says so (`admission refused`); otherwise its exit and last lines."""
    for line in result.output():
        start = line.find(REFUSED)
        if start < 0:
            continue
        record = json.loads(line[start + len(REFUSED):])
        if record.get("moment") != "admission refused":
            continue
        def gb(value):
            return "unread" if value is None else f"{value / 1e9:.2f} GB"

        cause = record["distress"] or (
            f"{record['constraint']} headroom {gb(record['headroom_bytes'])} at or below the "
            f"planning reserve {gb(record['planning_bytes'])}"
        )
        lines = [
            f"{label}refused by memory admission ({record['store']}): {cause}",
            (f"  headroom {gb(record['headroom_bytes'])} · planning reserve {gb(record['planning_bytes'])} · "
             f"wired {gb(record['wired_bytes'])} · compressed {gb(record['compressed_bytes'])}"),
        ]
        if record.get("windows"):
            windows = record["windows"]
            total = lambda name: sum(window[name] for window in windows)
            lines.append(
                f"  {len(windows)} displacement windows over {total('ms') / 1000:.1f} s · "
                f"swapped out {total('swapped_out_pages')} pages in "
                f"{sum(1 for window in windows if window['swapped_out_pages'])} of them · "
                f"compressed {total('compressed_pages')} · decompressed {total('decompressed_pages')}"
            )
        return lines
    return [f"{label}harness time exited {result.code}", *result.output()[-15:]]


def _tuning(arguments, label: str, result: JobResult) -> tuple[str, bool]:
    """Whether every load of a cell tuned to completion: each load's budget event (the harness's
    `tuning.jsonl`, tuning-trace/3) refused no work for want of budget. A load that searched
    nothing, every unit stored, has no budget event. Under a pin that tunes nothing there is no
    search to be incomplete: the line says what the pin held. A recording's cell is not its
    point, and stands whatever its search reached."""
    path = result.directory / "tuning.jsonl"
    if not path.exists():
        return f"  {label}tuning not recorded by this build's harness", True
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if arguments.tuning_defaults or arguments.tuning_replay:
        choices = [event for event in events if event.get("event") == "pinned"]
        absent = [event["entry"] for event in choices if event["choice"] == "not_replayed"]
        held = len(choices) - len(absent)
        missing = f": {', '.join(sorted(set(absent)))}" if absent else ""
        return f"  {label}tuning pinned: {held} units held, {len(absent)} not in the pin ran at default{missing}", True
    budgets = [event for event in events if event.get("event") == "budget"]
    refused = [offer for event in budgets for offer in event["refused"]]
    if refused:
        spent = ", ".join(f"{event['used_seconds']:.0f} of {event['budget_seconds']:.0f} s" for event in budgets)
        units = "; ".join(f"{offer['unit']} ({offer['class']}, {offer['reason']})" for offer in refused)
        verdict = "" if arguments.tuning_record else " · VOID"
        line = f"  {label}tuning incomplete: {len(refused)} pieces of work refused ({spent}): {units}{verdict}"
        return line, bool(arguments.tuning_record)
    return f"  {label}tuning complete ({len(budgets)} searching loads)", True


def _graphs(detail: dict, prefill: dict) -> str:
    """The graphs the window's decode steps and each warm prefill recorded and updated (serving
    records none once every shape it runs has run), and the device's graph budget."""
    missing = lacking(detail, "window.graphs") or lacking(prefill, "graphs", "graph_budget")
    if missing:
        return unrecorded("graphs", missing)
    decode, warm, budget = detail["window"]["graphs"], prefill["graphs"], prefill["graph_budget"]

    def per(counts: dict, unit: str) -> str:
        if counts["units"] == 0:
            return f"no {unit}"
        return (f"{counts['recorded_per_unit']:.2f} recorded, {counts['updated_per_unit']:.1f} updated a {unit} "
                f"(of {counts['units']})")

    return (f"  graphs: decode {per(decode, 'step')} · warm prefill {per(warm, 'prefill')} · budget "
            f"{budget['graphs']} graphs, {budget['launches']} launches, "
            f"{(budget['device_bytes'] + budget['upload_bytes']) / 1e6:.0f} MB")


def _unsteady(detail: dict, prefill: dict) -> list[str]:
    """The decode windows and warm prefill samples the harness discarded, the expert bank not at
    rest across them (a tier filling, a resize): every figure is of the steady ones."""
    windows = len(detail.get("window", {}).get("unsteady") or [])
    samples = len(prefill.get("unsteady") or [])
    if not windows and not samples:
        return []
    return [f"  discarded, the bank not at rest: {windows} decode windows, {samples} warm prefill samples"]


def _streamed(arguments, cap: int, result: JobResult, resident: dict | None) -> tuple[list[str], dict, bool]:
    """A streamed cell's lines, measures and validity, against the `resident` cell's decode detail,
    or, streamed only (None), its absolute terms alone."""
    if result.code != 0:
        return _failed(f"streamed {cap}: ", result), {}, False
    if unmeasured := _unmeasured(result, *STREAMED_MEASURED):
        return [f"streamed {cap}: {unmeasured}"], {}, False
    detail, prefill, step_ms, half, prefill_ms, prefill_half = _timed(arguments, result)
    tok_s = 1000 * arguments.sequences / step_ms
    prefill_tok_s = 1000 * prefill["rows"] / prefill_ms
    model = cell(result, "decode").get("model") or {}
    # Each load runs its own tuning, so equal tokens are evidence, unequal ones not by
    # themselves a defect: `kl --weight-cap` compares the loads at the defaults.
    if resident is None:
        tokens = "tokens not compared (no resident cell)"
    elif lacking(detail, "selected_sha256") or lacking(resident, "selected_sha256"):
        tokens = "tokens not recorded"
    else:
        tokens = "same tokens" if detail["selected_sha256"] == resident["selected_sha256"] else "other tokens"
    # A streamed step's times are heavy-tailed (a step that misses waits on reads), so its
    # throughput is the window's mean step, which the model predicts, beside the median.
    mean_ms = detail["window"]["wall_ms_per_step"]
    resident_mean_ms = (None if resident is None or lacking(resident, "window.wall_ms_per_step")
                        else resident["window"]["wall_ms_per_step"])
    mean_tok_s = 1000 * arguments.sequences / mean_ms
    against = "" if resident_mean_ms is None else f"{resident_mean_ms / mean_ms:.3f}× resident · "
    lines = [
        (f"streamed {cap} · mean {mean_ms:.3f} ms · {mean_tok_s:.1f} tok/s · "
         f"{against}median {step_ms:.3f} ms (±{half:.3f}) · "
         f"{tok_s:.1f} tok/s · prefill {prefill_tok_s:.0f} tok/s · {tokens}"),
        *_idle(detail),
    ]
    if resident is None:
        lines.append("  model not priced: streamed only, no resident cell")
    elif not model:
        # A load that holds every expert streams nothing, so it has no model.
        lines.append("  model none: the load streams no expert")
    elif "unavailable" in model:
        lines.append(f"  model unavailable: {model['unavailable']}")
    elif missing := lacking(model, *MODEL_FIELDS) or (["window.wall_ms_per_step (resident)"] if resident_mean_ms is None else []):
        lines.append(unrecorded("model", missing))
    else:
        shares = model["miss_shares"]
        overhead_ms = model["figures"]["overhead_ms"]
        # Without an all-hit step, σ is unmeasured and the model prices its
        # step on the resident step alone.
        all_hit = ("no all-hit step, σ unmeasured: the model's step on C alone" if overhead_ms is None else
                   f"all-hit {(resident_mean_ms + overhead_ms) / resident_mean_ms:.3f}× resident")
        predicted = f"{model['predicted_tok_s']:.1f} tok/s · measured/model {model['ratio']:.3f}"
        # A window without misses has no shares of them.
        missed = ("" if shares["reading"] is None else
                  f" (reading {shares['reading']:.3f}, "
                  f"unanticipated {shares['unanticipated']:.3f})")
        lines.append(
            f"  model {predicted} · {model['misses_per_step']:.1f} misses/step{missed} · "
            f"{model['all_hit_steps']}/{model['window_steps']} all-hit steps, {all_hit}"
        )
        lines += _hints(detail)
        lines += _lines(model["figures"])
    lines += _bank(result)
    # What a growth begun during the window has joined of the host tier as it ends.
    host = ((detail.get("expert_bank") or {}).get("residency") or {}).get("host")
    if host is not None:
        lines.append(f"  host tier at the window's end: {host['slots']} of {host['ceiling']} host slots, "
                     f"{host['bytes'] / 1e9:.2f} GB")
    decode = cell(result, "decode")
    if model and resident is not None:
        lines += _bar(decode.get("bar"), detail.get("expert_bank"))
    if model:
        lines.append(_estimate(decode.get("estimate")))
    # The OS's count of the expert stacks' pages the file cache held as the measured steps
    # began and ended: any is a read served from the cache, and voids the cell.
    served = decode.get("cache_served")
    if lacking(detail, *PAGE_FIELDS):
        lines.append(unrecorded("expert pages", lacking(detail, *PAGE_FIELDS)))
    elif detail["expert_pages"] is not None:
        pages = detail["expert_pages"]
        lines.append(f"  expert pages cached {pages['cached_before']} before, {pages['cached_after']} after, "
                     f"of {pages['pages']}" + (" · VOID: its reads were served from the file cache" if served else ""))
    lines += _group(prefill.get("group"))
    lines += _unsteady(detail, prefill)
    lines.append(_graphs(detail, prefill))
    tuning, tuned = _tuning(arguments, f"streamed {cap} ", result)
    lines.append(tuning)
    measures = {
        f"{harness.streamed(cap)} mean_step_ms": Measure(mean_ms, "ms", None, lower_is_better=True),
        f"{harness.streamed(cap)} step_ms": Measure(step_ms, "ms", half, lower_is_better=True),
        f"{harness.streamed(cap)} prefill_ms": Measure(prefill_ms, "ms", prefill_half, lower_is_better=True),
    }
    # A cell is void when its record says its reads were cache-served, or its tuning stopped
    # at its budget.
    return lines, measures, not served and tuned


def _idle(detail) -> list[str]:
    """The decode window's device time a step, busy and idle, and the idle by cause: between
    runs (host before a submission's commit, queued after) and at timeline boundaries, where
    a run was divided (Vulkan's parks and publications): held on the host until the next queue
    submission was handed over, then awaiting the data its gate's answer landed, the keeper's
    answer latency, or a plain signal, then starting in the driver and device."""
    if lacking(detail, "window.device_busy_ms_per_step", "window.idle"):
        return []
    window = detail["window"]
    between, at = window["idle"]["between_runs"], window["idle"]["at_boundaries"]
    lines = [f"  device busy {window['device_busy_ms_per_step']:.3f} ms, idle "
             f"{window['device_idle_ms_per_step']:.3f} ms a step: between runs {between['ms_per_step']:.3f} "
             f"(host {between['host_ms_per_step']:.3f}, queued {between['queued_ms_per_step']:.3f}), "
             f"at boundaries {at['ms_per_step']:.3f}"]
    phases = between.get("host_by_phase_ms_per_step", {})
    if phases:
        lines.append("  host idle between runs by the harness thread's phase: "
                     + ", ".join(f"{phase} {ms:.3f}" for phase, ms in phases.items()) + " ms")
    if at["per_step"]:
        parks = ", ".join(f"{park} {ms:.3f}" for park, ms in at.get("awaiting_data_by_park_ms_per_step", {}).items())
        lines.append(f"  idle at {at['per_step']:.1f} boundaries a step: held {at['held_ms_per_step']:.3f}, "
                     f"awaiting data {at['awaiting_data_ms_per_step']:.3f}"
                     + (f" ({parks})" if parks else "") + ", answer latency "
                     f"{at['answer_latency_ms_per_step']:.3f}, awaiting signal {at['awaiting_signal_ms_per_step']:.3f}, "
                     f"starting {at['starting_ms_per_step']:.3f} (submitting {at['submitting_ms_per_step']:.3f}) ms")
        layers = sorted(at.get("awaiting_data_by_layer_ms_per_step", []), key=lambda entry: -entry["ms_per_step"])
        if layers:
            lines.append("  awaiting data by park and layer, largest first: " + ", ".join(
                f"{entry['park']} {entry['layer']} {entry['ms_per_step']:.3f}" for entry in layers[:SHOWN]) + " ms")
    return lines


RATIO = "× resident"


def _streamed_only(arguments, results: dict[str, JobResult]) -> Answer:
    """The streamed cells' absolute terms, no resident cell timed; a ratio to one among them is
    an invariant violated, never shown."""
    measure = results["measure"]
    if measure.code != 0:
        return Answer((f"the measurement exited {measure.code}", *measure.output()[-15:]), False)
    lines = ["streamed only: no resident cell timed; absolute streamed terms",
             *(line[line.find(UNMEASURED):] for line in measure.output() if UNMEASURED in line)]
    if arguments.untimed:
        lines.insert(0, "UNTIMED (no thermal gate): the times below are not judged")
    measures, ok = {}, True
    for cap in arguments.weight_cap:
        streamed_lines, streamed_measures, valid = _streamed(arguments, cap, results[harness.streamed(cap)], None)
        lines += streamed_lines
        measures.update(streamed_measures)
        ok = ok and valid
    if ratios := [line for line in lines if RATIO in line]:
        raise AssertionError(f"step --streamed-only: a ratio to a resident cell it never timed: {ratios[0]!r}")
    return Answer(tuple(lines), ok, measures)


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    if arguments.tuning_record:
        run = next(iter(results.values())).directory.parent
        shutil.copyfile(run / harness.PINS.removeprefix("{runs}/"), arguments.tuning_record)
    if arguments.streamed_only:
        return _streamed_only(arguments, results)
    # A record the run's placement accepted (`_accept_resident`), else the run's own resident step.
    result = results["step"] if arguments.resident is None else _recorded_step(arguments)
    if result.code != 0:
        return Answer(tuple(_failed("", result)), False)
    if unmeasured := _unmeasured(result):
        return Answer((unmeasured,), False)
    detail, prefill, step_ms, half, prefill_ms, prefill_half = _timed(arguments, result)
    entries = _entries(detail)
    prefill_entries = _entries(prefill)
    launches = round(sum(row["launches_per_step"] for row in entries)) if isinstance(entries, list) and entries else None
    together = f" × {arguments.sequences}" if arguments.sequences > 1 else ""
    tok_s = 1000 * arguments.sequences / step_ms
    cold = "" if lacking(prefill, "cold_ms") else f" · cold {prefill['cold_ms']:.1f} ms"
    lines = [
        (f"step @{arguments.context}{together} · {step_ms:.3f} ms (±{half:.3f}, ±{100 * half / step_ms:.1f}%) · "
         f"{tok_s:.1f} tok/s" + (f" · {launches} launches" if launches is not None else "")),
        (f"prefill @{arguments.context} · {prefill_ms:.1f} ms (±{prefill_half:.1f}, "
         f"±{100 * prefill_half / prefill_ms:.1f}%) · {1000 * prefill['rows'] / prefill_ms:.0f} tok/s{cold}"),
        *_idle(detail),
    ]
    if arguments.resident is not None:
        lines.insert(0, f"resident step recorded by run {arguments.resident}, not timed again")
    lines += _unsteady(detail, prefill)
    lines.append(_graphs(detail, prefill))
    if arguments.untimed:
        # Run without the thermal gate: its times are not a measurement.
        lines.insert(0, "UNTIMED (no thermal gate): the times below are not judged")
    tuning, ok = _tuning(arguments, "resident ", result)
    if entries:
        lines += _table("decode entry", entries, "ms/step")
    if prefill_entries:
        lines += _table("prefill entry", prefill_entries, "ms")
    measures = {
        "step_ms": Measure(step_ms, "ms", half, lower_is_better=True),
        "prefill_ms": Measure(prefill_ms, "ms", prefill_half, lower_is_better=True),
    }
    for prefix, rows in (("entry", entries), ("prefill entry", prefill_entries)):
        for row in rows if isinstance(rows, list) else ():
            measures[f"{prefix} {row['entry']}"] = Measure(row["device_ms_per_step"], "ms", None, True)
    if arguments.weight_cap:
        measure = results["measure"]
        if measure.code != 0:
            return Answer((f"the measurement exited {measure.code}", *measure.output()[-15:]), False)
        lines += [line[line.find(UNMEASURED):] for line in measure.output() if UNMEASURED in line]
    for cap in arguments.weight_cap or ():
        streamed_lines, streamed_measures, valid = _streamed(arguments, cap, results[harness.streamed(cap)], detail)
        lines += streamed_lines
        measures.update(streamed_measures)
        ok = ok and valid
    lines.append(tuning)
    return Answer(tuple(lines), ok, measures)


INSTRUMENT = Instrument(
    name="step",
    kind="speed",
    question="How fast is the model's step at this context (prefill and decode), and per entry?",
    observable="decode step time with per-entry attribution",
    selectors=("backend", "host", "build", "model", "context", "fixture", "purpose", "estimate"),
    configure=configure,
    plan=plan,
    answer=answer,
    comparable=True,
    alternate=alternate,
)
