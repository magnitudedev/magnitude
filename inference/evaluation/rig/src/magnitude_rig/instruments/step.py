"""`step`: how fast is the model's step at this context, prefill and decode, and where does the
decode step's time go?

The harness `time` at one context and `--sequences` decoded together (one by default), without
drafting (the drafter is `draft`'s), on the checkout's own kernel cache (so stored tuning is
reused, never searched afresh between runs): the median
step with its 95% interval from the measured steps, the launches per step, and the device
time per entry from `--attribution` further steps, each launch timed alone; and the prefill of
the context, the median of `--prefill-samples` warm samples (after a cold one) with its
interval, and (with attribution) one further prefill's device time per entry.

Under `mrig ab step` every round runs the base's first tuning (`harness`), so the comparison is
of the code, never of two tunings; `--tuning own` lets each side tune, when tuning is what
changed.
"""

import argparse
import json

from ..engine import Step, Weights
from . import Answer, Instrument, JobResult, Measure, Plan, harness
from .resolution import median_interval

SHOWN = 12


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequences", type=int, default=1,
                        help="sequences prefilled and decoded together, each at the context (default 1)")
    parser.add_argument("--steps", type=int, default=64, help="measured decode steps (default 64)")
    parser.add_argument("--prefill-samples", type=int, default=6,
                        help="warm prefill samples of the context, after one cold (default 6)")
    parser.add_argument("--attribution", type=int, default=3,
                        help="further steps attributed by entry, each launch timed alone (default 3; 0 skips)")
    parser.add_argument("--lookahead", choices=("on", "off"), default="on",
                        help="queue each step's successor before it completes, as served")
    parser.add_argument("--tuning", choices=("replay-base", "own"), default="replay-base",
                        help="under ab: every round runs the base's first tuning (default), or each side its own")


def time(arguments, family: str, *pinning: str) -> tuple[str, ...]:
    return (
        harness.EXECUTABLE, "time",
        "--artifacts", "{artifacts}", "--fixtures", f"{{inputs}}/{family}.jsonl",
        "--output", "{run_dir}", "--build-identity", "{build_file}",
        "--cache-dir", "{checkout}/kernels", "--device", arguments.backend,
        "--contexts", str(arguments.context), "--concurrency", str(arguments.sequences),
        "--warm", "8", "--steps", str(arguments.steps),
        "--attribution-steps", str(arguments.attribution),
        *(("--prefill-attribution",) if arguments.attribution else ()),
        "--prefill-samples", str(arguments.prefill_samples + 1),
        "--lookahead", arguments.lookahead, *pinning,
    )  # fmt: skip


def plan(arguments, source) -> Plan:
    family = arguments.fixture.stem
    return Plan(
        builds=(harness.BUILD,),
        jobs=(("step", Step("time", time(arguments, family))),),
        device="timed",
        acquires=Weights(arguments.model),
        files=(arguments.fixture,),
    )


def alternate(arguments, side: str, index: int) -> tuple[tuple[str, Step], ...]:
    family = arguments.fixture.stem
    if arguments.tuning == "own":
        pinning = ()
    elif side == "base" and index == 0:
        pinning = harness.RECORD
    else:
        pinning = harness.REPLAY
    return (("step", Step("time", time(arguments, family, *pinning))),)


def cell(result: JobResult, phase: str) -> dict:
    record = json.loads((result.directory / "result.json").read_text())
    return next(cell for cell in record["measurements"]["speed"] if cell["phase"] == phase)


def _entries(detail: dict) -> list[dict]:
    attribution = detail.get("attribution") or {}
    return sorted(attribution.get("entries", []), key=lambda row: -row["device_ms_per_step"])


def _table(title: str, entries: list[dict], unit: str) -> list[str]:
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


def answer(arguments, results: dict[str, JobResult]) -> Answer:
    result = results["step"]
    if result.code != 0:
        return Answer((f"harness time exited {result.code}", *result.output()[-15:]), False)
    detail = cell(result, "decode")["detail"]
    prefill = cell(result, "prefill")["detail"]
    prefill_ms, prefill_half = median_interval(list(prefill["samples_ms"]))
    step_ms, half = median_interval([step["wall_ms"] for step in detail["steps"]])
    entries = _entries(detail)
    prefill_entries = _entries(prefill)
    launches = round(sum(row["launches_per_step"] for row in entries)) if entries else None
    together = f" × {arguments.sequences}" if arguments.sequences > 1 else ""
    lines = [
        (f"step @{arguments.context}{together} · {step_ms:.3f} ms (±{half:.3f}, ±{100 * half / step_ms:.1f}%) · "
         f"{1000 * arguments.sequences / step_ms:.1f} tok/s" + (f" · {launches} launches" if launches is not None else "")),
        (f"prefill @{arguments.context} · {prefill_ms:.1f} ms (±{prefill_half:.1f}, "
         f"±{100 * prefill_half / prefill_ms:.1f}%) · {1000 * prefill['rows'] / prefill_ms:.0f} tok/s · "
         f"cold {prefill['cold_ms']:.1f} ms"),
    ]
    if entries:
        lines += _table("decode entry", entries, "ms/step")
    if prefill_entries:
        lines += _table("prefill entry", prefill_entries, "ms")
    measures = {
        "step_ms": Measure(step_ms, "ms", half, lower_is_better=True),
        "prefill_ms": Measure(prefill_ms, "ms", prefill_half, lower_is_better=True),
    }
    for row in entries:
        measures[f"entry {row['entry']}"] = Measure(row["device_ms_per_step"], "ms", None, True)
    for row in prefill_entries:
        measures[f"prefill entry {row['entry']}"] = Measure(row["device_ms_per_step"], "ms", None, True)
    return Answer(tuple(lines), True, measures)


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
