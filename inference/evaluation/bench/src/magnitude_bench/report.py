"""Derived summaries with explicit eligibility, denominators, and timing boundaries."""

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable


def summarize(
    records: list[dict],
    status: str,
    command: str,
    run_id: str,
    planned: int,
    error: str | None = None,
) -> dict:
    """Rows per engine, section, context and concurrency. Truncation, protocol errors,
    timeouts and cancellation never enter latency or rates, but every recorded request
    enters the outcome counts and accuracy denominators."""
    measured = [r for r in records if r["phase"] == "measured"]
    groups = defaultdict(list)
    for record in measured:
        groups[
            (
                record["target"],
                record["section"],
                record["checkpoint"],
                record.get("concurrency", 1),
            )
        ].append(record)
    rows = []
    for (target, section, context, concurrency), values in groups.items():
        outcomes = Counter(r["observation"]["outcome"] for r in values)
        valid = [
            r["observation"]
            for r in values
            if r["observation"]["outcome"] == "valid"
            or (
                (section == "context" or r["family"] == "retrieval")
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
                "context_target": context,
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
        if values[0]["family"] == "retrieval":
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
    holding's last, its ``last_resize``) by cause, and the most recent across them. Passes without a bank
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
            "resizes": {holding: dict(Counter(resize["cause"] for resize in resizes.values()))
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
    """A resize as the census states it (``BankResize``: ``Shrank`` by ``by`` to ``to`` for its
    cause, or ``Grew`` by ``by`` from ``from``), as its cause, its size before and after, and when
    it took effect; a growth's cause is ``Growth``."""
    ((kind, fields),) = resize.items()
    if kind == "Shrank":
        return {"cause": fields["cause"], "from": fields["to"] + fields["by"], "to": fields["to"],
                "at_unix_ms": fields["at_unix_ms"]}
    return {"cause": "Growth", "from": fields["from"], "to": fields["from"] + fields["by"],
            "at_unix_ms": fields["at_unix_ms"]}


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
        "| Target | Section | Concurrency | Context target | Actual prompt tokens | "
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
            f"{row['context_target']} | "
            f"{', '.join(map(str, row['actual_prompt_tokens'])) or '—'} | "
            f"{row['eligible']} / {row['observations']} | {number(row['ttft_ms'])} | "
            f"{number(row['completion_ms'])} | {number(row['prefill_tokens_per_second'])} | "
            f"{number(row['decode_tokens_per_second'])} | {drafts(row)} |"
        )
    retrieval_rows = [row for row in summary["rows"] if "retrieval" in row]
    if retrieval_rows:
        lines += [
            "",
            "| Target | Section | Concurrency | Context target | "
            "Exact answers | Retrieved fields |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
        for row in retrieval_rows:
            score = row["retrieval"]
            lines.append(
                f"| {row['target']} | {row['section']} | {row['concurrency']} | "
                f"{row['context_target']} | {score['exact_matches']}/{score['requests']} "
                f"({100 * score['exact_accuracy']:.1f}%) | {score['correct']}/{score['total']} "
                f"({100 * score['field_accuracy']:.1f}%) |"
            )
    lines += [
        "",
        "Latencies and phase rates above are medians; nearest-rank p95 latency is in summary.json.",
        "",
        {
            "retrieval": "RULER-derived retrieval; strict JSON answers, not an official RULER "
            "score. All recorded requests enter accuracy denominators; unscored failures earn "
            "zero. Partial field accuracy does not penalize extra keys; exact accuracy does. "
            "Latency includes correct and incorrect protocol-complete responses.",
            "prose-continue": "Prose continuation; no answer-quality scoring. EOS or reaching "
            "the output budget ends a valid measurement; actual lengths are in summary.json.",
            "prose-repeat": "Prose repeat: each request repeats a Moby Dick passage back; no "
            "answer-quality scoring. EOS or reaching the output budget ends a valid "
            "measurement; actual lengths are in summary.json.",
            "prose-shift": "Prose shift: each request repeats the end of a Moby Dick passage, "
            "writes new prose, and repeats the same words again; no answer-quality scoring. "
            "EOS or reaching the output budget ends a valid measurement; actual lengths are "
            "in summary.json.",
            "tools": "BFCL-derived tool correctness; not an official BFCL leaderboard score. "
            "Context rows may include semantically invalid responses with complete protocol "
            "evidence; all other sections require valid semantics.",
        }[summary["mode"]],
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
                f"{holding} " + ", ".join(f"{cause} {count}" for cause, count in sorted(causes.items()))
                for holding, causes in bank["resizes"].items()
            ) or "none"
            latest = bank["latest_resize"]
            last = ("none" if latest is None else
                    f"{latest['holding']} {latest['cause']} {latest['from']} → {latest['to']}")
            lines.append(
                f"| {bank['target']} | {bank['label']} | {bank['first_slots']} → {bank['last_slots']} | "
                f"{bank['min_slots']}–{bank['max_slots']} | {bank['floor']} | {bank['ceiling']} | {resizes} "
                f"| {last} |"
            )
    if summary.get("error"):
        lines += ["", "Run error:", "```text", summary["error"], "```"]
    return "\n".join(lines) + "\n"
