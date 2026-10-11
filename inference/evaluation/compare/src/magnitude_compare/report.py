"""The human report of a comparison, rendered from its result record alone."""

from collections import defaultdict

PAIRS = {
    "base-reference": "base vs reference",
    "candidate-reference": "candidate vs reference",
    "candidate-base": "candidate vs base",
}


def _percent(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _number(value: float) -> str:
    return f"{value:,.1f}"


def _build(side: dict) -> str:
    build = side["build"]
    dirty = f", uncommitted `{build['dirty_digest'][:12]}`" if build["dirty_digest"] else ""
    return f"`{side['source']}` (`{build['commit'][:12]}`{dirty})"


def _quality(lines: list[str], record: dict, comparison: dict) -> None:
    measured = defaultdict(dict)
    for entry in record["measurements"].get("quality", []):
        measured[(entry["family"], entry["context"])][entry["pair"]] = entry
    lines += [
        "",
        "## Quality",
        "",
        "KL divergence from the truth at every scored position (lower is better) and top-1 "
        f"agreement. A change counts when its 95% paired-bootstrap interval excludes zero and it "
        f"moves the metric by more than {comparison['bars']['quality'] * 100:.1f}%.",
    ]
    for row in comparison["quality"]:
        cell = (row["family"], row["context"])
        lines += ["", f"### {row['family']} @ {row['context']}", ""]
        if "skipped" in row:
            lines.append(f"Skipped: {row['skipped']}.")
            continue
        if "reference" in row:
            reference = row["reference"]
            sanity = reference["sanity"]
            state = "passed" if sanity["passed"] else "FAILED (refused)"
            lines.append(
                f"Reference `{reference['family']}` ({'cached' if reference['cached'] else 'built'}): "
                f"sanity {state}, perplexity {sanity['perplexity']:.3g}, top-1 {sanity['top1_accuracy']:.3f}."
            )
        if row.get("rendering"):
            lines.append(
                "**Finding: the candidate renders different tokens (template or tokenizer "
                "behaviour changed); its quality below is measured on the base's tokens:**"
            )
            for difference in row["rendering"]:
                lines.append(f"- `{difference}`")
        pairs = measured[cell]
        if pairs:
            lines += [
                "",
                "| Pair | Mean KL | Trimmed KL | p99 KL | Max KL | Top-1 | Positions |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for pair, entry in pairs.items():
                lines.append(
                    f"| {PAIRS[pair]} | {entry['mean_kl']:.3e} | {entry['trimmed_kl']:.3e} | "
                    f"{entry['p99_kl']:.3e} | {entry['max_kl']:.3e} | {entry['top1_agreement']:.4f} | "
                    f"{entry['positions']} |"
                )
        if "changes" in row:
            lines += [
                "",
                "| Metric | Base | Candidate | Change | 95% interval of change | Verdict |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for change in row["changes"]:
                low, high = change["interval"]
                lines.append(
                    f"| {change['metric']} | {change['base']:.4g} | {change['candidate']:.4g} | "
                    f"{_percent(change['relative'])} | [{low:.3g}, {high:.3g}] | {change['verdict']} |"
                )
        if "builds_identical" in row:
            lines += [
                "",
                "The builds' distributions are identical."
                if row["builds_identical"]
                else "The builds' distributions differ (see candidate vs base).",
            ]


def _speed(lines: list[str], comparison: dict) -> None:
    lines += [
        "",
        "## Speed",
        "",
        f"Tokens per second (higher is better), median over {comparison['rounds']} alternating "
        f"rounds per build; tuning mode `{comparison['tuning_mode']}`. A cell changes when it moves "
        f"by more than the larger of its noise and {comparison['bars']['speed'] * 100:.1f}%.",
        "",
        "| Family | Phase | Context | Concurrency | Base tok/s | Candidate tok/s | Change | Noise | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in comparison["speed"]:
        prefix = f"| {row['family']} | {row['phase']} | {row['context']} | {row['concurrency']} |"
        if row["verdict"] == "missing":
            lines.append(f"{prefix} | | | | measured on the {row['only']} only |")
            continue
        noisy = " (noisy)" if row["noisy"] else ""
        lines.append(
            f"{prefix} {_number(row['base_tokens_per_second'])} | "
            f"{_number(row['candidate_tokens_per_second'])} | {_percent(row['delta'])} | "
            f"{row['noise'] * 100:.1f}%{noisy} | {row['verdict']} |"
        )
    for differing in comparison["speed_rendering"]:
        lines.append("")
        lines.append(
            f"**{differing['family']} round {differing['round']}: the builds timed different "
            "tokens:**"
        )
        lines.extend(f"- `{difference}`" for difference in differing["differences"])
    changed = [row for row in comparison["speed"] if row["verdict"] in ("improved", "regressed")]
    if changed:
        lines += ["", "Kernels behind the changed cells (device ms per step, lower is better):", ""]
        for row in changed:
            lines.append(
                f"**{row['family']} {row['phase']} @ {row['context']} x{row['concurrency']}** "
                f"({_percent(row['delta'])})"
            )
            lines.append("")
            for entry in row["entries"]:
                if "only" in entry:
                    lines.append(f"- `{entry['entry']}`: only in the {entry['only']}")
                else:
                    lines.append(
                        f"- `{entry['entry']}`: {entry['base_ms']:.3f} → {entry['candidate_ms']:.3f} ms "
                        f"({_percent(entry['delta'])})"
                    )
            lines.append("")


def _drafting(lines: list[str], comparison: dict) -> None:
    lines += [
        "",
        "## Drafting",
        "",
        "| Family | Base tokens/verify | Candidate tokens/verify | Change | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in comparison["drafting"]:
        if row["verdict"] == "missing":
            lines.append(f"| {row['family']} | | | | measured on the {row['only']} only |")
            continue
        lines.append(
            f"| {row['family']} | {row['base_tokens_per_verify']:.3f} | "
            f"{row['candidate_tokens_per_verify']:.3f} | {_percent(row['delta'])} | {row['verdict']} |"
        )


def _tuning(lines: list[str], tuning: list[dict]) -> None:
    lines += [
        "",
        "## Tuning",
        "",
        "| Run | Budget s | Fresh | Loads (units searched/stored, tuning s; of a load that "
        "searched: s past the budget, offered pieces refused) |",
        "| --- | --- | --- | --- |",
    ]

    def budget(load: dict) -> str:
        # Only a load that searched reports what became of its budget.
        if "refused" not in load:
            return ""
        return f", {load['overrun_seconds']:.2f} s past, {load['refused']} refused"

    for entry in tuning:
        if entry["outcome"] is None and entry.get("unavailable"):
            loads = f"Unavailable: {entry['unavailable']}"
        else:
            # A build whose search runs in a Prepare worker tags each entry `search` or `load`;
            # a build whose loads searched in their own process tags none.
            loads = "; ".join(
                f"{load['work'] + ': ' if 'work' in load else ''}"
                f"{load['units']} units {load['searched']}/{load['stored']}, "
                f"{load['tuning_seconds']:.1f} s{budget(load)}"
                for load in entry["outcome"]["loads"]
            )
        lines.append(
            f"| {entry['run']} | {entry['budget_seconds']:g} | {entry['fresh']} | {loads} |"
        )


def render(record: dict) -> str:
    comparison = record["measurements"]["comparison"]
    sides = record["sides"]
    lines = [
        f"# Compare: {comparison['verdict'].upper()}",
        "",
        f"Model `{record['model']}`. Base {_build(sides['base'])}; candidate {_build(sides['candidate'])}. "
        "Both builds run the working tree's harness.",
    ]
    host = record["host"]
    if host:  # completed by the remote recorder after the run
        lines.append(f"Host `{host['alias'] or host['hostname']}` ({host['chip']}, {host['os']}).")
    if comparison["failures"]:
        lines += ["", "## Failures", ""]
        for failure in comparison["failures"]:
            log = f" (`{failure['log']}`)" if "log" in failure else ""
            lines.append(f"- **{failure['step']}**: {failure['error']}{log}")
    if comparison["quality"]:
        _quality(lines, record, comparison)
    if comparison["speed"]:
        _speed(lines, comparison)
    if comparison["drafting"]:
        _drafting(lines, comparison)
    if "tuning" in record["measurements"]:
        _tuning(lines, record["measurements"]["tuning"])
    return "\n".join(lines) + "\n"
