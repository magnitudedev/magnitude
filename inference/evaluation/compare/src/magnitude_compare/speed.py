"""Speed: per-cell throughput of the candidate against the base, judged against each side's noise.

A cell is one (family, phase, context, concurrency) of the harness `time` measurement. Each
round of a side gives the cell's tokens per second (the harness's median) and its own sample
spread. A side's value is the median over its rounds. The cell's noise is the largest of each
side's spread across rounds (range over median) and every round's interquartile spread of its
samples, so a single round is judged against its sampling noise rather than against zero.
The cell improves or regresses when its relative change exceeds the larger of the noise and the
bar; a noisy host widens its own tolerance instead of producing false alarms.

Per-entry attribution names the kernels behind a changed cell: each entry's device time per
step (launches summed), median over rounds per side, for entries holding at least 2% of either
side's attributed time.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

ENTRY_SHARE = 0.02
ENTRIES_SHOWN = 6
NOISY = 0.05
SpeedVerdict = Literal["improved", "regressed", "within noise", "missing"]


@dataclass(frozen=True)
class Sample:
    """One cell of one side's round."""

    side: str
    round: int
    family: str
    cell: dict  # the harness `measurements.speed` entry


def key(sample: Sample) -> tuple[str, str, int, int]:
    return (sample.family, sample.cell["phase"], sample.cell["context"], sample.cell["concurrency"])


def spread(values: list[float]) -> float:
    """Range relative to the median; zero for a single value."""
    middle = statistics.median(values)
    return (max(values) - min(values)) / middle if len(values) > 1 else 0.0


def sample_spread(samples: list[float]) -> float:
    """Interquartile range relative to the median (the full range under four samples)."""
    if len(samples) < 4:
        return spread(samples)
    quartiles = statistics.quantiles(samples, n=4)
    return (quartiles[2] - quartiles[0]) / statistics.median(samples)


def entry_times(cell: dict) -> dict[str, float]:
    """Device milliseconds per step of each attributed entry, its launches summed; none for
    a cell measured without attribution (a prefill cell, or a run without attribution steps)."""
    times: dict[str, float] = defaultdict(float)
    attribution = (cell.get("detail") or {}).get("attribution") or {}
    for row in attribution.get("entries", []):
        times[row["entry"].split("#")[0]] += row["device_ms_per_step"]
    return times


def judge_change(
    base: float, candidate: float, noise: float, bar: float
) -> tuple[float, SpeedVerdict]:
    """Relative change of a higher-is-better value and its verdict."""
    delta = candidate / base - 1
    tolerance = max(bar, noise)
    if delta > tolerance:
        return delta, "improved"
    if delta < -tolerance:
        return delta, "regressed"
    return delta, "within noise"


def _entries(per_side: dict[str, list[dict[str, float]]]) -> list[dict]:
    names = set().union(*(times for rounds in per_side.values() for times in rounds))
    totals = {
        side: statistics.median(sum(times.values()) for times in rounds)
        for side, rounds in per_side.items()
    }
    moved = []
    for name in names:
        values = {
            side: [times[name] for times in rounds if name in times]
            for side, rounds in per_side.items()
        }
        if not values["base"] or not values["candidate"]:
            moved.append({"entry": name, "only": "base" if values["base"] else "candidate"})
            continue
        before, after = statistics.median(values["base"]), statistics.median(values["candidate"])
        if before / totals["base"] >= ENTRY_SHARE or after / totals["candidate"] >= ENTRY_SHARE:
            moved.append(
                {
                    "entry": name,
                    "base_ms": before,
                    "candidate_ms": after,
                    "delta": after / before - 1,
                }
            )
    # Entries one side lacks first, then the largest moves.
    moved.sort(key=lambda row: float("-inf") if "only" in row else -abs(row["delta"]))
    return moved[:ENTRIES_SHOWN]


def compare(samples: list[Sample], bar: float) -> list[dict]:
    """One verdict row per cell measured on either side."""
    cells: dict[tuple, dict[str, list[Sample]]] = defaultdict(lambda: {"base": [], "candidate": []})
    for sample in samples:
        cells[key(sample)][sample.side].append(sample)
    rows = []
    for (family, phase, context, concurrency), sides in sorted(cells.items()):
        row = {"family": family, "phase": phase, "context": context, "concurrency": concurrency}
        if not sides["base"] or not sides["candidate"]:
            rows.append(
                {**row, "verdict": "missing", "only": "base" if sides["base"] else "candidate"}
            )
            continue
        values = {
            side: [s.cell["tokens_per_second"] for s in group] for side, group in sides.items()
        }
        noise = max(
            spread(values["base"]),
            spread(values["candidate"]),
            *(sample_spread(s.cell["samples"]) for group in sides.values() for s in group),
        )
        base, candidate = statistics.median(values["base"]), statistics.median(values["candidate"])
        delta, verdict = judge_change(base, candidate, noise, bar)
        rows.append(
            {
                **row,
                "base_tokens_per_second": base,
                "candidate_tokens_per_second": candidate,
                "delta": delta,
                "noise": noise,
                "noisy": noise > NOISY,
                "rounds": {side: len(group) for side, group in sides.items()},
                "verdict": verdict,
                "entries": _entries(
                    {side: [entry_times(s.cell) for s in group] for side, group in sides.items()}
                ),
            }
        )
    return rows


def compare_drafting(drafting: list[tuple[str, str, dict]], bar: float) -> list[dict]:
    """Drafting acceptance (tokens per verification) per family, judged like a speed cell.
    `drafting` holds (side, family, contract `drafting` measurement) per round."""
    families: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"base": [], "candidate": []})
    for side, family, measurement in drafting:
        families[family][side].append(measurement)
    rows = []
    for family, sides in sorted(families.items()):
        if not sides["base"] or not sides["candidate"]:
            rows.append(
                {
                    "family": family,
                    "verdict": "missing",
                    "only": "base" if sides["base"] else "candidate",
                }
            )
            continue
        values = {side: [m["tokens_per_verify"] for m in group] for side, group in sides.items()}
        noise = max(spread(values["base"]), spread(values["candidate"]))
        base, candidate = statistics.median(values["base"]), statistics.median(values["candidate"])
        delta, verdict = judge_change(base, candidate, noise, bar)
        rows.append(
            {
                "family": family,
                "base_tokens_per_verify": base,
                "candidate_tokens_per_verify": candidate,
                "delta": delta,
                "noise": noise,
                "verdict": verdict,
            }
        )
    return rows
