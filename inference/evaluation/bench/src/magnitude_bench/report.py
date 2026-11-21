"""Derived summaries with explicit eligibility, denominators, and timing boundaries."""

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

TUNING_FALLBACK_ALERT_RATE = 0.10
TUNING_GAP_MIN_RATIO = 1.25
TUNING_TOP_N = 5


def summarize(
    records: list[dict],
    status: str,
    command: str,
    run_id: str,
    planned: int,
    error: str | None = None,
) -> dict:
    """Rows per engine, section, template and concurrency. Truncation, protocol errors,
    timeouts and cancellation never enter latency or rates, but every recorded request
    enters the outcome counts and accuracy denominators."""
    measured = [r for r in records if r["phase"] == "measured"]
    groups = defaultdict(list)
    for record in measured:
        groups[
            (
                record["target"],
                record["section"],
                record["template"],
                record.get("concurrency", 1),
            )
        ].append(record)
    rows = []
    for (target, section, template, concurrency), values in groups.items():
        outcomes = Counter(r["observation"]["outcome"] for r in values)
        valid = [
            r["observation"]
            for r in values
            if r["observation"]["outcome"] == "valid"
            or (
                (section == "context" or r["key"] == "retrieve")
                and r["observation"]["outcome"] == "invalid"
            )
        ]

        def median(field, valid=valid):
            numbers = [r[field] for r in valid if r.get(field) is not None]
            return statistics.median(numbers) if numbers else None

        def p95(field, valid=valid):
            numbers = sorted(r[field] for r in valid if r.get(field) is not None)
            return numbers[math.ceil(len(numbers) * 0.95) - 1] if numbers else None

        def native_rate(tokens, duration, valid=valid):
            numbers = [
                1000 * r["terminal"]["timings"][tokens] / r["terminal"]["timings"][duration]
                for r in valid
                if r["terminal"]["timings"][duration] > 0 and r["terminal"]["timings"][tokens] > 0
            ]
            return statistics.median(numbers) if numbers else None

        prompts = [r["terminal"]["usage"]["prompt_tokens"] for r in valid]
        drafted = [r["terminal"]["timings"] for r in valid if "draft_n" in r["terminal"]["timings"]]
        rows.append(
            {
                "target": target,
                "section": section,
                "template": template,
                "context_target": values[0]["checkpoint"],
                "concurrency": concurrency,
                "observations": len(values),
                "eligible": len(valid),
                "outcomes": dict(outcomes),
                "actual_prompt_tokens": sorted(set(prompts)),
                "actual_completion_tokens": sorted(
                    {r["terminal"]["usage"]["completion_tokens"] for r in valid}
                ),
                "ttft_ms": median("ttft_ms"),
                "completion_ms": median("completed_ms"),
                "ttft_p95_ms": p95("ttft_ms"),
                "completion_p95_ms": p95("completed_ms"),
                "prefill_tokens_per_second": native_rate("prompt_n", "prompt_ms"),
                "decode_tokens_per_second": native_rate("predicted_n", "predicted_ms"),
                "draft_proposed": sum(t["draft_n"] for t in drafted) if drafted else None,
                "draft_accepted": sum(t["draft_n_accepted"] for t in drafted) if drafted else None,
                "timing_basis": values[0]["timing_basis"],
            }
        )
        if values[0]["key"] == "retrieve":
            scores = [
                r["observation"]["retrieval"]
                for r in values
                if r["observation"]["outcome"] in ("valid", "invalid")
                and r["observation"].get("retrieval") is not None
            ]
            total = sum(r["retrieval_total"] for r in values)
            correct = sum(score["correct"] for score in scores)
            exact = sum(score["exact_match"] for score in scores)
            rows[-1]["retrieval"] = {
                "requests": len(values),
                "scored": len(scores),
                "exact_matches": exact,
                "exact_accuracy": exact / len(values),
                "correct": correct,
                "total": total,
                "field_accuracy": correct / total,
                "format_valid": sum(score["format_valid"] for score in scores),
            }
    return {
        "format": 1,
        "id": run_id,
        "status": status,
        "command": command,
        "planned": planned,
        "completed": len(measured),
        "outcomes": dict(Counter(r["observation"]["outcome"] for r in measured)),
        "rows": rows,
        "error": error,
        "comparison": "product comparison of each engine's serving of one catalog model",
        "cache_policy": "disabled; shared session history does not imply retained prefix reuse",
    }


def expert_banks(samples: list[dict], banks_of: dict[str, Callable[[dict], dict | None]]) -> list[dict]:
    """Each pass's elastic expert bank over the run, from the engine's memory observations
    (``allocations.jsonl``) sampled while it served, as each target's adapter reads a bank from
    its engine's observation (``banks_of``, by target: ``Adapter.expert_bank``): its experts'
    slots at the first and last sample and their range, its floor and ceiling, the resizes seen
    of each of its holdings (its experts, each table, its host tier: each sample shows each
    holding's last, its ``last_resize``) by kind, and the most recent across them. Passes without a bank
    are left out."""
    passes = defaultdict(list)
    for sample in samples:
        passes[(sample["target"], sample["label"])].append(sample)
    banks = []
    for (target, label), sampled in passes.items():
        experts = [banks_of[target](s["observation"]) for s in sorted(sampled, key=lambda s: s["at"])]
        experts = [bank for bank in experts if bank is not None]
        if not experts:
            continue
        slots = [bank["slots"] for bank in experts]
        seen = defaultdict(dict)  # holding -> resize by when it took effect
        for bank in experts:
            for holding, resize in _holdings(bank):
                seen[holding][resize["at_unix_ms"]] = resize
        latest = max(((holding, resize) for holding, resizes in seen.items() for resize in resizes.values()),
                     key=lambda found: found[1]["at_unix_ms"], default=None)
        banks.append({
            "target": target,
            "label": label,
            "first_slots": slots[0],
            "last_slots": slots[-1],
            "min_slots": min(slots),
            "max_slots": max(slots),
            "floor": experts[-1]["floor"],
            "ceiling": experts[-1]["ceiling"],
            "resizes": {holding: dict(Counter(resize["kind"] for resize in resizes.values()))
                        for holding, resizes in sorted(seen.items())},
            "latest_resize": None if latest is None else {"holding": latest[0], **latest[1]},
        })
    return banks


def _holdings(bank: dict) -> list[tuple[str, dict]]:
    """Each holding's last resize in one observation (each holding's ``last_resize`` in
    ``ExpertResidency``), by the holding's name: ``experts``, ``table <index>``, ``host``; a holding
    not yet resized is absent."""
    host = bank["host"]
    named = [("experts", bank["last_resize"]), ("host", None if host is None else host["last_resize"]),
             *((f"table {index}", table["last_resize"]) for index, table in enumerate(bank["tables"]))]
    return [(holding, _resize(resize)) for holding, resize in named if resize is not None]


def _resize(resize: dict) -> dict:
    """A resize as the census states it (``BankResize``: ``Shrank`` by ``by`` to ``to``, or
    ``Grew`` by ``by`` from ``from``), as its kind (``Shrank`` or ``Grew``), its size before and
    after, and when it took effect."""
    ((kind, fields),) = resize.items()
    if kind == "Shrank":
        return {"kind": kind, "from": fields["to"] + fields["by"], "to": fields["to"],
                "at_unix_ms": fields["at_unix_ms"]}
    return {"kind": kind, "from": fields["from"], "to": fields["from"] + fields["by"],
            "at_unix_ms": fields["at_unix_ms"]}


def _measured_points(candidate: dict) -> dict[str, float]:
    """Point label -> median_seconds for a Measured candidate, or {} if unmeasured."""
    outcome = candidate.get("outcome")
    if not isinstance(outcome, dict) or "Measured" not in outcome:
        return {}
    return {
        point["point"]: float(point["median_seconds"])
        for point in outcome["Measured"].get("points", ())
        if point.get("median_seconds") is not None
    }


def _weighted_seconds(points: dict[str, float], weights: dict[str, float]) -> tuple[float, float, float]:
    """Return (weighted_mean_seconds, total_weight, weighted_total_seconds) over ``points``."""
    total_weight = sum(weights.get(label, 0.0) for label in points)
    if total_weight > 0.0:
        weighted_total = sum(seconds * weights.get(label, 0.0) for label, seconds in points.items())
        return weighted_total / total_weight, total_weight, weighted_total
    mean = sum(points.values()) / len(points)
    return mean, 0.0, 0.0


def _fallback_reason(event: dict) -> str:
    stop = event.get("stop")
    if stop == "Expired":
        return "Expired"
    if stop == "NoRoom":
        return "NoRoom"
    if event.get("qualification") == "unqualified_rejected":
        return "Rejected"
    return "Canonical"


def _compact_unit(unit: str, limit: int = 96) -> str:
    return unit if len(unit) <= limit else f"{unit[: limit - 1]}…"


def tuning_health(trace: Path) -> dict | None:
    """Tuning coverage, worst default-vs-best gaps, and slowest untuned/fallback units from
    ``tuning.jsonl`` (tuning-trace/3). Prefers the last ``search`` load when present (since a
    subsequent ``load`` replays stored outcomes without transient expired attempts); returns
    ``None`` when the trace has no load."""
    loads: list[dict] = []
    for raw in trace.read_text().splitlines():
        if not raw.strip():
            continue
        event = json.loads(raw)
        kind = event.get("event")
        if kind == "began":
            loads.append({"began": event, "finished": [], "budget": None, "concluded": None})
        elif loads and kind == "finished":
            loads[-1]["finished"].append(event)
        elif loads and kind == "budget":
            loads[-1]["budget"] = event
        elif loads and kind == "concluded":
            loads[-1]["concluded"] = event

    populated = [load for load in loads if load["finished"] or load["concluded"] is not None]
    if not populated:
        return None
    searches = [load for load in populated if load["began"].get("work") == "search"]
    selected = searches[-1] if searches else populated[-1]

    finished = selected["finished"]
    concluded = selected["concluded"] or {}
    if finished:
        total_units = len(finished)
        qualified = sum(1 for item in finished if item.get("qualification") == "qualified")
        canonical_fallbacks = total_units - qualified
        expired = sum(1 for item in finished if item.get("stop") == "Expired")
        unqualified_budget = sum(1 for item in finished if item.get("qualification") == "unqualified_budget")
    else:
        total_units = int(concluded.get("units", 0))
        canonical_fallbacks = int(concluded.get("canonical_fallbacks", 0))
        qualified = max(0, total_units - canonical_fallbacks)
        expired = 0
        unqualified_budget = 0

    fallback_rate = canonical_fallbacks / total_units if total_units else 0.0
    refused = len((selected["budget"] or {}).get("refused") or ())

    gaps: list[dict] = []
    fallbacks: list[dict] = []
    for item in finished:
        entry = item.get("entry", "")
        bindings = item.get("bindings", "")
        unit = f"{entry} [{bindings}]" if bindings else entry
        candidates = item.get("candidates") or ()
        if not candidates:
            continue
        default_points = _measured_points(candidates[0])
        if not default_points:
            continue
        weights = {
            point["label"]: max(0.0, float(point.get("weight", 0.0)))
            for point in (item.get("points") or ())
        }
        default_s, total_weight, weighted_total_s = _weighted_seconds(default_points, weights)

        best_s = default_s
        for candidate in candidates[1:]:
            if not candidate.get("admitted"):
                continue
            cand_points = _measured_points(candidate)
            if not all(label in cand_points for label in default_points):
                continue
            cand_s, _, _ = _weighted_seconds(
                {label: cand_points[label] for label in default_points}, weights
            )
            if 0.0 < cand_s < best_s:
                best_s = cand_s

        if best_s > 0.0 and default_s / best_s > TUNING_GAP_MIN_RATIO:
            gaps.append({
                "unit": unit,
                "entry": entry,
                "bindings": bindings,
                "default_us": default_s * 1e6,
                "best_us": best_s * 1e6,
                "ratio": default_s / best_s,
                "qualification": item.get("qualification"),
                "stop": item.get("stop"),
            })

        if item.get("qualification") != "qualified":
            fallbacks.append({
                "unit": unit,
                "entry": entry,
                "bindings": bindings,
                "reason": _fallback_reason(item),
                "median_us": default_s * 1e6,
                "weight": total_weight,
                "weighted_ms": weighted_total_s * 1e3,
            })

    gaps.sort(key=lambda row: row["ratio"], reverse=True)
    fallbacks.sort(key=lambda row: (row["weighted_ms"], row["median_us"]), reverse=True)
    return {
        "work": selected["began"].get("work"),
        "budget_seconds": selected["began"].get("budget_seconds"),
        "total_units": total_units,
        "qualified": qualified,
        "canonical_fallbacks": canonical_fallbacks,
        "fallback_rate": fallback_rate,
        "high_fallback": fallback_rate > TUNING_FALLBACK_ALERT_RATE,
        "expired": expired,
        "unqualified_budget": unqualified_budget,
        "refused": refused,
        "total_default_gaps": len(gaps),
        "worst_default_gaps": gaps[:TUNING_TOP_N],
        "slowest_fallbacks": fallbacks[:TUNING_TOP_N],
    }


def tuning_health_alert(health: dict) -> list[str]:
    """One or two concise stdout lines when tuning health has issues, or [] when clean."""
    high_fallback = health.get("high_fallback", False)
    gaps = health.get("worst_default_gaps") or ()
    if not high_fallback and not gaps:
        return []
    lines: list[str] = []
    total = health["total_units"]
    fallbacks = health["canonical_fallbacks"]
    qualified = health["qualified"]
    expired = health["expired"]
    pct = 100.0 * health["fallback_rate"]
    if high_fallback:
        slowest = (health.get("slowest_fallbacks") or (None,))[0]
        tail = ""
        if slowest is not None:
            cost = (
                f"{slowest['weighted_ms']:.1f} weighted ms"
                if slowest["weighted_ms"] > 0.0
                else f"{slowest['median_us']:.1f} µs"
            )
            tail = f"; slowest fallback: {_compact_unit(slowest['unit'], 64)} ({cost})"
        lines.append(
            f"Tuning health: WARNING {fallbacks}/{total} ({pct:.1f}%) canonical fallbacks "
            f"({qualified}/{total} qualified, {expired} expired){tail}"
        )
    if gaps:
        worst = gaps[0]
        count = health.get("total_default_gaps", len(gaps))
        lines.append(
            f"Tuning default gaps: {count} unit{'s' if count != 1 else ''} >{TUNING_GAP_MIN_RATIO:g}×; "
            f"worst: {_compact_unit(worst['unit'], 64)} "
            f"{worst['default_us']:.1f} µs → {worst['best_us']:.1f} µs ({worst['ratio']:.2f}×)"
        )
    return lines


def markdown(summary: dict) -> str:
    def number(value):
        return "—" if value is None else f"{value:,.1f}"

    def drafts(row):
        if not row["draft_proposed"]:
            return "—"
        share = 100 * row["draft_accepted"] / row["draft_proposed"]
        return f"{row['draft_accepted']}/{row['draft_proposed']} ({share:.1f}%)"

    lines = [
        "# Serving bench",
        "",
        f"Status: **{summary['status']}**",
        "",
        "```sh",
        summary["command"],
        "```",
        "",
        f"Recorded {summary['completed']} / {summary['planned']} planned measurements.",
        "",
        f"Outcomes: {summary['outcomes']}",
        "",
        "| Target | Section | Concurrency | Template | Actual prompt tokens | "
        "Eligible / recorded | "
        "TTFT ms | Completion ms | Prefill tok/s | Decode tok/s | Drafts accepted |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if host := summary.get("host"):
        memory = host["memory_bytes"]
        lines[2:2] = [
            f"Host: {host['hostname']} · {host['chip'] or 'unknown chip'} · "
            f"{'unknown' if memory is None else f'{memory / (1 << 30):g} GiB'} RAM · "
            f"{host['os']} {host['os_version']}",
            "",
        ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['target']} | {row['section']} | {row['concurrency']} | "
            f"{row['template']} | "
            f"{', '.join(map(str, row['actual_prompt_tokens'])) or '—'} | "
            f"{row['eligible']} / {row['observations']} | {number(row['ttft_ms'])} | "
            f"{number(row['completion_ms'])} | {number(row['prefill_tokens_per_second'])} | "
            f"{number(row['decode_tokens_per_second'])} | {drafts(row)} |"
        )
    retrieval_rows = [row for row in summary["rows"] if "retrieval" in row]
    if retrieval_rows:
        lines += [
            "",
            "| Target | Section | Concurrency | Template | "
            "Exact answers | Retrieved fields |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
        for row in retrieval_rows:
            score = row["retrieval"]
            lines.append(
                f"| {row['target']} | {row['section']} | {row['concurrency']} | "
                f"{row['template']} | {score['exact_matches']}/{score['requests']} "
                f"({100 * score['exact_accuracy']:.1f}%) | {score['correct']}/{score['total']} "
                f"({100 * score['field_accuracy']:.1f}%) |"
            )
    lines += [
        "",
        "Latencies and phase rates above are medians; nearest-rank p95 latency is in summary.json.",
        "",
        {
            "retrieve": "RULER-derived retrieval; strict JSON answers, not an official RULER "
            "score. All recorded requests enter accuracy denominators; unscored failures earn "
            "zero. Partial field accuracy does not penalize extra keys; exact accuracy does. "
            "Latency includes correct and incorrect protocol-complete responses.",
            "vision": "Vision image-value check; strict JSON answers checked as retrieved fields. "
            "All recorded requests enter accuracy denominators; unscored failures earn zero. "
            "Partial field accuracy does not penalize extra keys; exact accuracy does. "
            "Latency includes correct and incorrect protocol-complete responses.",
            "repeat": "Moby Dick repeat: each request repeats a Moby Dick passage back; no "
            "answer-quality scoring. EOS or reaching the output budget ends a valid "
            "measurement; actual lengths are in summary.json.",
            "twist": "Moby Dick twist: each request repeats the end of a Moby Dick passage, "
            "writes new prose, and repeats the same words again; no answer-quality scoring. "
            "EOS or reaching the output budget ends a valid measurement; actual lengths are "
            "in summary.json.",
            "code": "BigCodeBench stacked Python implementation tasks; no answer-quality "
            "scoring. EOS or reaching the output budget ends a valid measurement; actual "
            "lengths are in summary.json.",
            "tools": "BFCL-derived tool correctness; not an official BFCL leaderboard score. "
            "Context rows may include semantically invalid responses with complete protocol "
            "evidence; all other sections require valid semantics.",
        }[summary["key"]] if summary["key"] else "No template was read.",
        "",
        "Truncation, protocol errors, timeouts and cancellation are excluded from latency and "
        "rates, and retained in outcome counts and accuracy.",
        "",
        "MLX-VLM generation timing measures server token emission. Its phase rates are shown "
        "for inspection and must not be interpreted as native model-service "
        "ratios against other engines.",
        "",
        summary["comparison"],
        "",
        summary["cache_policy"],
        "",
        "Raw request streams, per-process memory samples, source identities and engine logs "
        "are retained alongside this report.",
    ]
    if summary.get("process_footprints"):
        lines += [
            "",
            "Process-tree RSS (includes loading; not isolated GPU allocation):",
            "",
            "| Target | Pass | Ready GiB | Peak GiB |",
            "| --- | ---: | ---: | ---: |",
        ]
        for item in summary["process_footprints"]:
            lines.append(
                f"| {item['target']} | {item['block'] + 1} | "
                f"{item['baseline_rss_bytes'] / 2**30:.2f} | "
                f"{item['peak_rss_bytes'] / 2**30:.2f} |"
            )
    if summary.get("expert_banks"):
        lines += [
            "",
            "Elastic expert bank (experts' slots, from the engine's memory observations while serving):",
            "",
            "| Target | Pass | Slots first → last | Range | Floor | Ceiling | Resizes seen | Latest resize |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for bank in summary["expert_banks"]:
            resizes = "; ".join(
                f"{holding} " + ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items()))
                for holding, kinds in bank["resizes"].items()
            ) or "none"
            latest = bank["latest_resize"]
            last = ("none" if latest is None else
                    f"{latest['holding']} {latest['kind']} {latest['from']} → {latest['to']}")
            lines.append(
                f"| {bank['target']} | {bank['label']} | {bank['first_slots']} → {bank['last_slots']} | "
                f"{bank['min_slots']}–{bank['max_slots']} | {bank['floor']} | {bank['ceiling']} | {resizes} "
                f"| {last} |"
            )
    if health := summary.get("tuning_health"):
        lines += _tuning_health_markdown(health)
    if summary.get("error"):
        lines += ["", "Run error:", "```text", summary["error"], "```"]
    return "\n".join(lines) + "\n"


def _tuning_health_markdown(health: dict) -> list[str]:
    total = health["total_units"]
    qualified = health["qualified"]
    fallbacks = health["canonical_fallbacks"]
    expired = health["expired"]
    pct = 100.0 * health["fallback_rate"]
    lines = ["", "## Tuning health", ""]
    if health["high_fallback"]:
        lines.append(
            f"> **WARNING: High tuning fallback rate** — {fallbacks}/{total} ({pct:.1f}%) units "
            f"fell back to canonical default ({qualified}/{total} qualified, {expired} expired)."
        )
    else:
        lines.append(
            f"Qualified: {qualified}/{total} · Canonical fallbacks: {fallbacks}/{total} "
            f"({pct:.1f}%) · Expired: {expired}."
        )
    gaps = health.get("worst_default_gaps") or ()
    if gaps:
        lines += [
            "",
            f"Worst `default` vs. `tuned` gaps (>{TUNING_GAP_MIN_RATIO:g}×; "
            f"{health.get('total_default_gaps', len(gaps))} total):",
            "",
            "| Unit | Default µs | Best µs | Ratio | Status |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for row in gaps:
            lines.append(
                f"| `{_compact_unit(row['unit'])}` | {row['default_us']:,.1f} | "
                f"{row['best_us']:,.1f} | {row['ratio']:.2f}× | {row['qualification'] or '—'} |"
            )
    slowest = health.get("slowest_fallbacks") or ()
    if slowest:
        lines += [
            "",
            "Slowest untuned / fallback units:",
            "",
            "| Unit | Reason | Median µs | Weight | Weighted ms |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for row in slowest:
            lines.append(
                f"| `{_compact_unit(row['unit'])}` | {row['reason']} | "
                f"{row['median_us']:,.1f} | {row['weight']:,.1f} | {row['weighted_ms']:,.2f} |"
            )
    return lines
