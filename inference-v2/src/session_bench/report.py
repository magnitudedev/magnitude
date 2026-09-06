"""Derived summaries with explicit eligibility, denominators, and timing boundaries."""

import math
import statistics
from collections import Counter, defaultdict


def summarize(
    records: list[dict],
    status: str,
    command: str,
    run_id: str,
    planned: int,
    error: str | None = None,
) -> dict:
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
            or (section == "context" and r["observation"]["outcome"] == "invalid")
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
                "ttft_ms": median("ttft_ms"),
                "completion_ms": median("completed_ms"),
                "ttft_p95_ms": p95("ttft_ms"),
                "completion_p95_ms": p95("completed_ms"),
                "prefill_tokens_per_second": native_rate("prompt_n", "prompt_ms"),
                "decode_tokens_per_second": native_rate("predicted_n", "predicted_ms"),
                "timing_basis": values[0]["timing_basis"],
            }
        )
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
        "comparison": "product comparison; no strict equivalence inferred from aliases",
        "cache_policy": "disabled; shared session history does not imply retained prefix reuse",
    }


def markdown(summary: dict) -> str:
    def number(value):
        return "—" if value is None else f"{value:,.1f}"

    lines = [
        "# Session bench",
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
        "TTFT ms | Completion ms | Prefill tok/s | Decode tok/s |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['target']} | {row['section']} | {row['concurrency']} | "
            f"{row['context_target']} | "
            f"{', '.join(map(str, row['actual_prompt_tokens'])) or '—'} | "
            f"{row['eligible']} / {row['observations']} | {number(row['ttft_ms'])} | "
            f"{number(row['completion_ms'])} | {number(row['prefill_tokens_per_second'])} | "
            f"{number(row['decode_tokens_per_second'])} |"
        )
    lines += [
        "",
        "Latencies and phase rates above are medians; nearest-rank p95 latency is in summary.json.",
        "",
        "BFCL-derived tool correctness; not an official BFCL leaderboard score.",
        "",
        "Context rows may include semantically invalid responses with complete protocol evidence. "
        "All other sections require valid semantics. "
        "Truncation and execution failures are excluded.",
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
    if summary.get("error"):
        lines += ["", "Run error:", "```text", summary["error"], "```"]
    return "\n".join(lines) + "\n"
