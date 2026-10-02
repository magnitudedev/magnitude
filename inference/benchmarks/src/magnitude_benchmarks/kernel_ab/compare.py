"""The baseline/candidate report of one run directory.

A cell is one measurement (a prefill size at a history, or a decode context) of one model, from
the run's measure phase, where each build replays its first tune's configurations. Its value per
side is the median over rounds of each round's median step time. Noise is the spread of
a side's round medians relative to their median. A cell regresses when the candidate is slower
than the baseline by more than the bar or by more than either side's noise, whichever is larger,
so a noisy host widens its own tolerance instead of producing false alarms; a cell whose noise
exceeds the bar is flagged so a rerun can narrow it.
"""

import json
import math
import statistics
from array import array
from collections import defaultdict
from pathlib import Path

from .runner import LOGITS_DECODES

DEFAULT_BAR = 0.0
ENTRY_SHARE = 0.02
LOGITS_KL_LIMIT = 1e-3
BUSY_CORES = 0.5


def cells(report: dict) -> dict[str, dict]:
    found = {}
    for cell in report.get("prefill", []):
        key = f"prefill {cell['rows']}" + (f" @ {cell['history']}" if cell.get("history") else "")
        found[key] = cell
    for cell in report.get("decode", []):
        found[f"decode @ {cell['context']}"] = cell
    return found


def entry_times(cell: dict) -> dict[str, float]:
    """Device milliseconds per step of each entry, its launches summed."""
    times = defaultdict(float)
    for row in (cell.get("attribution") or {}).get("entries", []):
        times[row["entry"].split("#")[0]] += row["device_ms_per_step"]
    return times


def spread(values: list[float]) -> float:
    middle = statistics.median(values)
    return (max(values) - min(values)) / middle if len(values) > 1 and middle else 0.0


def pins(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return {
        f"{pin['entry']} [{pin['bindings']}] {json.dumps(pin['statics'], sort_keys=True)}": pin[
            "params"
        ]
        for pin in json.loads(path.read_text())
    }


def logits_rows(path: Path, rows: int) -> list[array]:
    values = array("f")
    values.frombytes(path.read_bytes())
    width = len(values) // rows
    return [values[index * width : (index + 1) * width] for index in range(rows)]


def argmax(row: array) -> int:
    return max(range(len(row)), key=row.__getitem__)


def softmax(row: array) -> list[float]:
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def compare_logits(baseline: Path, candidate: Path, rows: int) -> dict:
    if not baseline.is_file() or not candidate.is_file():
        return {"status": "missing"}
    if baseline.read_bytes() == candidate.read_bytes():
        return {"status": "identical", "rows": rows, "same_top": rows, "max_kl": 0.0}
    same_top, kls = 0, []
    for left, right in zip(logits_rows(baseline, rows), logits_rows(candidate, rows), strict=True):
        same_top += argmax(left) == argmax(right)
        pairs = zip(softmax(left), softmax(right), strict=True)
        kls.append(sum(a * (math.log(a) - math.log(b)) for a, b in pairs if a > 0 and b > 0))
    return {
        "status": "pass" if same_top == rows and max(kls) <= LOGITS_KL_LIMIT else "fail",
        "rows": rows,
        "same_top": same_top,
        "max_kl": max(kls),
        "mean_kl": statistics.fmean(kls),
    }


def analyze(directory: Path, bar: float = DEFAULT_BAR) -> dict:
    run = json.loads((directory / "run.json").read_text())
    samples = defaultdict(lambda: defaultdict(list))
    entries = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tuning = defaultdict(lambda: defaultdict(list))
    failures = []
    for result in run["results"]:
        if result["status"] != "ok":
            failures.append(result)
            continue
        if result["phase"] == "tune":
            tuning[result["model"]][result["side"]].append(pins(directory / result["pins"]))
            continue
        if result["phase"] != "measure":
            continue
        report = json.loads((directory / result["output"]).read_text())
        for key, cell in cells(report).items():
            name = (result.get("mode", "own"), result["model"], key)
            samples[name][result["side"]].append(cell["median_ms"])
            for entry, ms in entry_times(cell).items():
                entries[name][entry][result["side"]].append(ms)
    rows = []
    for (mode, model, key), sides in sorted(samples.items()):
        if not sides["baseline"] or not sides["candidate"]:
            continue
        baseline = statistics.median(sides["baseline"])
        candidate = statistics.median(sides["candidate"])
        noise = max(spread(sides["baseline"]), spread(sides["candidate"]))
        delta = (candidate - baseline) / baseline
        tolerance = max(bar, noise)
        verdict = "slower" if delta > tolerance else "faster" if delta < -tolerance else "same"
        moved = []
        for entry, per_side in entries[(mode, model, key)].items():
            if not per_side["baseline"] or not per_side["candidate"]:
                moved.append(
                    {"entry": entry, "only": "baseline" if per_side["baseline"] else "candidate"}
                )
                continue
            before = statistics.median(per_side["baseline"])
            after = statistics.median(per_side["candidate"])
            if before / baseline >= ENTRY_SHARE or after / candidate >= ENTRY_SHARE:
                moved.append(
                    {
                        "entry": entry,
                        "baseline_ms": before,
                        "candidate_ms": after,
                        "delta": (after - before) / before if before else math.inf,
                    }
                )
        moved.sort(key=lambda row: -abs(row.get("delta", 0)))
        rows.append(
            {
                "mode": mode,
                "model": model,
                "cell": key,
                "baseline_ms": baseline,
                "candidate_ms": candidate,
                "delta": delta,
                "noise": noise,
                "verdict": verdict,
                "noisy": noise > bar,
                "rounds": (len(sides["baseline"]), len(sides["candidate"])),
                "entries": moved[:6],
            }
        )
    choices = {}
    for model, per_side in tuning.items():
        differing = {}
        keys = set().union(*(pin for side in per_side.values() for pin in side))
        for key in sorted(keys):
            chosen = {
                side: [pin.get(key) for pin in per_side[side]] for side in ("baseline", "candidate")
            }
            unstable = [side for side, values in chosen.items() if len(set(map(str, values))) > 1]
            if chosen["baseline"] != chosen["candidate"]:
                differing[key] = {**chosen, "unstable": unstable}
        choices[model] = differing
    logits = {}
    rows_out = LOGITS_DECODES + 1
    for model in run["models"]:
        logits[model] = compare_logits(
            directory / f"logits-baseline-{model}.f32",
            directory / f"logits-candidate-{model}.f32",
            rows_out,
        )
    foreign = [
        result["foreign_cores"]
        for result in run["results"]
        if result.get("phase") == "measure" and result.get("foreign_cores") is not None
    ]
    # Own tuning differs by tuning luck as well as code; only same-configuration cells gate.
    regressions = [row for row in rows if row["verdict"] == "slower" and row["mode"] == "same"]
    wrong = [model for model, result in logits.items() if result["status"] == "fail"]
    return {
        "run": run,
        "bar": bar,
        "cells": rows,
        "tuning": choices,
        "logits": logits,
        "failures": failures,
        "foreign_cores": max(foreign) if foreign else None,
        "pass": not regressions and not wrong and not failures,
    }


def percent(value: float) -> str:
    return f"{value * 100:+.1f}%"


def markdown(analysis: dict) -> str:
    run = analysis["run"]
    builds = run["builds"]
    lines = [
        f"# Kernel A/B: {'PASS' if analysis['pass'] else 'FAIL'}",
        "",
        f"Host `{run['host']}` ({run['platform']}), device `{run['device']}`, "
        f"KV `{run['kv_codec']}`, {run['tunes']} tunes, {run['rounds']} measured rounds, "
        f"bar {analysis['bar'] * 100:.0f}%.",
        "",
        "| Side | Spec | Commit | Uncommitted |",
        "| --- | --- | --- | --- |",
    ]
    for side in ("baseline", "candidate"):
        build = builds[side]
        lines.append(
            f"| {side} | `{build['spec']}` | `{build['commit'][:12]}` | "
            f"{build['dirty'] or 'none'} |"
        )
    busy = analysis["foreign_cores"]
    if busy is not None:
        quiet = "quiet" if busy <= BUSY_CORES else "BUSY: measurements are suspect"
        lines += [
            "",
            f"Other processes kept up to {busy:.2f} cores busy during a measurement ({quiet}).",
        ]
    titles = {
        "same": (
            "## Speed on the same configurations",
            "Both builds replay the baseline's tuned configurations, so a difference is the code's."
            " These cells decide the verdict.",
        ),
        "own": (
            "## Speed with each build's own tuning",
            "Each build replays its own first tune: what a fresh load runs. Differences include"
            " tuning luck; see the tuning section.",
        ),
    }
    for mode in ("same", "own"):
        mode_rows = [row for row in analysis["cells"] if row["mode"] == mode]
        if not mode_rows:
            continue
        title, explanation = titles[mode]
        lines += [
            "",
            title,
            "",
            explanation + " Times are median step milliseconds; lower is better.",
            "",
            "| Model | Cell | Baseline ms | Candidate ms | Change | Noise | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in mode_rows:
            flag = " (noisy)" if row["noisy"] else ""
            lines.append(
                f"| {row['model']} | {row['cell']} | {row['baseline_ms']:.2f} | "
                f"{row['candidate_ms']:.2f} | {percent(row['delta'])} | "
                f"{row['noise'] * 100:.1f}% | {row['verdict']}{flag} |"
            )
        changed = [row for row in mode_rows if row["verdict"] != "same"]
        if changed:
            lines += ["", "Kernels behind the changed cells:", ""]
            for row in changed:
                lines.append(f"**{row['model']} {row['cell']}** ({percent(row['delta'])})")
                lines.append("")
                for entry in row["entries"]:
                    if "only" in entry:
                        lines.append(f"- `{entry['entry']}`: only in {entry['only']}")
                    else:
                        lines.append(
                            f"- `{entry['entry']}`: {entry['baseline_ms']:.3f} → "
                            f"{entry['candidate_ms']:.3f} ms ({percent(entry['delta'])})"
                        )
                lines.append("")
    lines += [
        "",
        "## Correctness",
        "",
        "| Model | Logits | Same top token | Max KL |",
        "| --- | --- | --- | --- |",
    ]
    for model, result in analysis["logits"].items():
        if result["status"] == "missing":
            lines.append(f"| {model} | missing | | |")
        else:
            lines.append(
                f"| {model} | {result['status']} | {result['same_top']}/{result['rows']} | "
                f"{result['max_kl']:.2e} |"
            )
    lines += [
        "",
        "## Tuning choices that differ between builds",
        "",
        "Each list holds one build's choice per fresh tune. An entry marked unstable chose",
        "differently between tunes of the same build, so its difference is tuning noise.",
        "",
    ]
    differing = {model: keys for model, keys in analysis["tuning"].items() if keys}
    if not differing:
        lines.append("None: every entry chose the same configuration in every tune.")
    for model, keys in differing.items():
        lines.append(f"**{model}**")
        lines.append("")
        for key, chosen in keys.items():
            unstable = f" (unstable: {', '.join(chosen['unstable'])})" if chosen["unstable"] else ""
            lines.append(
                f"- `{key}`: baseline {chosen['baseline']}, candidate {chosen['candidate']}"
                f"{unstable}"
            )
        lines.append("")
    if analysis["failures"]:
        lines += ["", "## Failed invocations", ""]
        for failure in analysis["failures"]:
            lines.append(
                f"- {failure['model']} {failure['side']} {failure['invocation']}"
                f"{' round ' + str(failure['round'] + 1) if 'round' in failure else ''}: "
                f"{failure['status']} (`{failure['log']}`)"
            )
    return "\n".join(lines) + "\n"
