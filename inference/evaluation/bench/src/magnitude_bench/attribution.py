"""Per-run attribution extraction: raw per-request deltas, residency, hardware rates, and tuning."""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import record, report

SCHEMA = "bench-attribution/1"
ATTRIBUTION_FILE = "attribution.json"
#: `u64::MAX` is the engine's internal load-time probe request (`RequestId`).
INTERNAL_REQUEST_ID = (1 << 64) - 1
#: `seismic_io::disk::FULL_DEPTH * FULL_READ == WIDEST_WINDOW` (32 MiB).
STORAGE_FULL_BYTES = 32 * 1024 * 1024

_ENGAGEMENT_RE = re.compile(r"expert engagement \(request (?P<req>\d+)\): (?P<fields>[^\r\n]+)")
_KV_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<value>-?[0-9]+(?:\.[0-9]+)?)")
_PROMO_LINK_RE = re.compile(
    r"a promotion's copy costs its link time at (?P<gb_s>[0-9]+(?:\.[0-9]+)?) GB/s"
    r"(?:, and (?P<stall>[0-9]+(?:\.[0-9]+)?)% of that in device time beside it)?"
)
_STORAGE_FULL_RE = re.compile(
    r'configuration:\s*\{"storage":\s*1\},\s*point:\s*\{"full":\s*1\}\s*\}:\s*'
    r"Measured\(Timing\s*\{\s*median:\s*(?P<median>[0-9.eE+-]+)"
)
_BANK_SLOTS_RE = re.compile(r"the expert bank's growth ended at (?P<slots>\d+) slots")
_HOST_SLOTS_RE = re.compile(
    r"the host tier's (?:backing|growth) ended at (?P<host>\d+) of \d+ planned host slots "
    r"\((?P<total>\d+) every expert\)"
)
_GROWTH_REFUSED_MSG = "the expert bank's storage refused a growth"


def _int(kv: dict[str, str], key: str, default: int = 0) -> int:
    raw = kv.get(key)
    return int(raw) if raw is not None else default


def _float(kv: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = kv.get(key)
    return float(raw) if raw is not None else default


def parse_engagement_lines(log_text: str) -> list[dict]:
    """Parse ``expert engagement (request X): ...`` lines into per-request deltas.

    Supports both the structured per-request delta format (``req_*`` keys) and
    legacy cumulative logs (converting cumulative counters to per-request deltas).
    """
    entries: list[dict] = []
    prev_cum: dict[str, int] = {
        "bytes_hidden": 0,
        "bytes_exposed": 0,
        "staged_hits": 0,
        "landed_hits": 0,
        "carryover_hits": 0,
        "carryover_wasted": 0,
    }
    for match in _ENGAGEMENT_RE.finditer(log_text):
        req_id = int(match["req"])
        kv = {m["key"]: m["value"] for m in _KV_RE.finditer(match["fields"])}
        if "req_bytes_hidden" in kv:
            entries.append(
                {
                    "engine_request_id": req_id,
                    "drive_busy_fraction": _float(kv, "drive_busy_fraction"),
                    "bytes_hidden": _int(kv, "req_bytes_hidden"),
                    "bytes_exposed": _int(kv, "req_bytes_exposed"),
                    "staged_hits": _int(kv, "req_staged_hits"),
                    "landed_hits": _int(kv, "req_landed_hits"),
                    "carryover_hits": _int(kv, "req_carryover_hits"),
                    "carryover_wasted": _int(kv, "req_carryover_wasted"),
                    "dma_calls": _int(kv, "req_dma_calls"),
                    "dma_skipped_zero_miss": _int(kv, "req_dma_skipped_zero_miss"),
                    "dma_experts_landed": _int(kv, "req_dma_experts"),
                    "dma_bytes_landed": _int(kv, "req_dma_bytes"),
                    "dma_overflow_experts": _int(kv, "req_dma_overflow"),
                    "dma_host_wait_ms": _float(kv, "req_dma_wait_ms"),
                    "growth_refusals": _int(kv, "req_growth_refusals"),
                    "cpu_site_jobs": _int(kv, "req_cpu_site_jobs"),
                    "vram_slots": _int(kv, "vram_slots") if "vram_slots" in kv else None,
                    "host_slots": _int(kv, "host_slots") if "host_slots" in kv else None,
                    "total_experts": _int(kv, "total_experts") if "total_experts" in kv else None,
                }
            )
        else:
            cur = {key: _int(kv, key) for key in prev_cum}
            delta = {key: max(0, cur[key] - prev_cum[key]) for key in prev_cum}
            prev_cum = cur
            entries.append(
                {
                    "engine_request_id": req_id,
                    "drive_busy_fraction": _float(kv, "drive_busy_fraction"),
                    **delta,
                    "dma_calls": 0,
                    "dma_skipped_zero_miss": 0,
                    "dma_experts_landed": 0,
                    "dma_bytes_landed": 0,
                    "dma_overflow_experts": 0,
                    "dma_host_wait_ms": 0.0,
                    "growth_refusals": 0,
                    "cpu_site_jobs": 0,
                    "vram_slots": None,
                    "host_slots": None,
                    "total_experts": None,
                }
            )
    return entries


def _parse_line_stream_gb_s(measure_log_text: str) -> float | None:
    """Extract measured VRAM stream bandwidth (`executorCosts.weightRates`) from `magnitude-assess`."""
    for raw in reversed(measure_log_text.splitlines()):
        line = raw[9:] if raw.startswith("[stdout] ") else raw
        if not line.startswith("{") or '"executorCosts"' not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        weight_rates = (payload.get("executorCosts") or {}).get("weightRates") or ()
        fallback: float | None = None
        for entry in weight_rates:
            for rate in entry.get("rates") or ():
                bps = (rate.get("bytesPerSecond") or {}).get("value")
                if bps is None or bps <= 0:
                    continue
                gb_s = float(bps) / 1e9
                if rate.get("format") == "bf16":
                    return gb_s
                if fallback is None:
                    fallback = gb_s
        if fallback is not None:
            return fallback
    return None


def parse_hardware_rates(block_logs: list[str], measure_log_text: str | None = None) -> dict:
    """Extract measured `line_stream`, `transfer_copy`, and `storage[full]` GB/s rates."""
    line_stream_gb_s = _parse_line_stream_gb_s(measure_log_text) if measure_log_text else None
    transfer_copy_gb_s: float | None = None
    transfer_copy_stall_pct: float | None = None
    storage_full_gb_s: float | None = None

    for text in block_logs:
        for match in _PROMO_LINK_RE.finditer(text):
            transfer_copy_gb_s = float(match["gb_s"])
            if match["stall"] is not None:
                transfer_copy_stall_pct = float(match["stall"])

    search_texts = ([measure_log_text] if measure_log_text else []) + list(block_logs)
    for text in search_texts:
        for match in _STORAGE_FULL_RE.finditer(text):
            median_s = float(match["median"])
            if median_s > 0.0:
                storage_full_gb_s = STORAGE_FULL_BYTES / median_s / 1e9

    return {
        "line_stream_gb_s": line_stream_gb_s,
        "transfer_copy_gb_s": transfer_copy_gb_s,
        "transfer_copy_stall_pct": transfer_copy_stall_pct,
        "storage_full_gb_s": storage_full_gb_s,
    }


def _fallback_residency(block_logs: list[str]) -> dict:
    vram_slots: int | None = None
    host_slots: int | None = None
    total_experts: int | None = None
    for text in block_logs:
        for match in _BANK_SLOTS_RE.finditer(text):
            vram_slots = int(match["slots"])
        for match in _HOST_SLOTS_RE.finditer(text):
            host_slots = int(match["host"])
            total_experts = int(match["total"])
    return {
        "vram_slots": vram_slots,
        "host_slots": host_slots,
        "total_experts": total_experts,
    }


def _request_timing_fields(row: dict | None) -> dict:
    if row is None:
        return {
            "request_id": None,
            "phase": None,
            "outcome": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "prompt_ms": None,
            "decode_ms": None,
            "decode_tps": None,
            "draft_proposed": None,
            "draft_accepted": None,
        }
    obs = row.get("observation") or {}
    terminal = obs.get("terminal") or {}
    timings = terminal.get("timings") or {}
    usage = terminal.get("usage") or {}
    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = timings.get("predicted_n")
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = timings.get("prompt_n")
    prompt_ms = timings.get("prompt_ms")
    decode_ms = timings.get("predicted_ms")
    decode_tps = (
        1000.0 * output_tokens / decode_ms
        if output_tokens and decode_ms and decode_ms > 0
        else None
    )
    return {
        "request_id": obs.get("request_id") or row.get("session"),
        "phase": row.get("phase"),
        "outcome": obs.get("outcome"),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_ms": prompt_ms,
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
        "draft_proposed": timings.get("draft_n"),
        "draft_accepted": timings.get("draft_n_accepted"),
    }


def _enrich_ratios(entry: dict) -> dict:
    out_tok = entry.get("output_tokens")
    decode_ms = entry.get("decode_ms")
    bytes_hidden = entry.get("bytes_hidden", 0)
    bytes_exposed = entry.get("bytes_exposed", 0)
    dma_bytes = entry.get("dma_bytes_landed", 0)
    entry["bytes_hidden_per_tok"] = bytes_hidden / out_tok if out_tok and out_tok > 0 else None
    entry["bytes_exposed_per_tok"] = bytes_exposed / out_tok if out_tok and out_tok > 0 else None
    entry["dma_bytes_per_tok"] = dma_bytes / out_tok if out_tok and out_tok > 0 else None
    entry["implied_exposed_gb_s"] = (
        bytes_exposed / (decode_ms * 1e6) if decode_ms and decode_ms > 0 else None
    )
    entry["implied_total_stream_gb_s"] = (
        (bytes_exposed + bytes_hidden) / (decode_ms * 1e6)
        if decode_ms and decode_ms > 0
        else None
    )
    return entry


def _sum_group(requests: list[dict]) -> dict | None:
    if not requests:
        return None
    output_tokens = sum(r["output_tokens"] or 0 for r in requests)
    prompt_tokens = sum(r["prompt_tokens"] or 0 for r in requests)
    prompt_ms = sum(r["prompt_ms"] or 0.0 for r in requests)
    decode_ms = sum(r["decode_ms"] or 0.0 for r in requests)
    draft_vals = [r for r in requests if r.get("draft_proposed") is not None]
    draft_proposed = sum(r["draft_proposed"] for r in draft_vals) if draft_vals else None
    draft_accepted = sum(r["draft_accepted"] or 0 for r in draft_vals) if draft_vals else None
    totals = {
        "requests": len(requests),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prompt_ms": prompt_ms,
        "decode_ms": decode_ms,
        "decode_tps": 1000.0 * output_tokens / decode_ms if output_tokens > 0 and decode_ms > 0 else None,
        "draft_proposed": draft_proposed,
        "draft_accepted": draft_accepted,
        "bytes_hidden": sum(r["bytes_hidden"] for r in requests),
        "bytes_exposed": sum(r["bytes_exposed"] for r in requests),
        "staged_hits": sum(r["staged_hits"] for r in requests),
        "landed_hits": sum(r["landed_hits"] for r in requests),
        "carryover_hits": sum(r["carryover_hits"] for r in requests),
        "carryover_wasted": sum(r["carryover_wasted"] for r in requests),
        "dma_calls": sum(r["dma_calls"] for r in requests),
        "dma_skipped_zero_miss": sum(r["dma_skipped_zero_miss"] for r in requests),
        "dma_experts_landed": sum(r["dma_experts_landed"] for r in requests),
        "dma_bytes_landed": sum(r["dma_bytes_landed"] for r in requests),
        "dma_overflow_experts": sum(r["dma_overflow_experts"] for r in requests),
        "dma_host_wait_ms": sum(r["dma_host_wait_ms"] for r in requests),
        "growth_refusals": sum(r["growth_refusals"] for r in requests),
        "cpu_site_jobs": sum(r["cpu_site_jobs"] for r in requests),
    }
    return _enrich_ratios(totals)


def extract(bench_dir: Path, measure_log: Path | None = None) -> dict:
    """Build the full raw attribution record for a bench directory."""
    if measure_log is None:
        candidate = bench_dir.parent / "logs" / "job-measure.log"
        if candidate.is_file():
            measure_log = candidate

    measure_text = measure_log.read_text(errors="replace") if measure_log and measure_log.is_file() else None

    results_by_block: dict[int, list[dict]] = {}
    results_path = bench_dir / "results.jsonl"
    if results_path.is_file():
        for raw in results_path.read_text().splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("target") == "magnitude":
                results_by_block.setdefault(int(row.get("block", 0)), []).append(row)

    logs_dir = bench_dir / "logs"
    block_log_paths: dict[int, Path] = {}
    if logs_dir.is_dir():
        for path in sorted(logs_dir.glob("magnitude-block-*.log")):
            suffix = path.stem.removeprefix("magnitude-block-")
            if suffix.isdigit():
                block_log_paths[int(suffix)] = path

    all_blocks = sorted(set(results_by_block) | set(block_log_paths))
    block_texts: list[str] = []
    requests: list[dict] = []
    internal_requests: list[dict] = []
    last_residency = {"vram_slots": None, "host_slots": None, "total_experts": None}
    log_growth_refusals = 0

    for block in all_blocks:
        text = block_log_paths[block].read_text(errors="replace") if block in block_log_paths else ""
        if text:
            block_texts.append(text)
            log_growth_refusals += text.count(_GROWTH_REFUSED_MSG)
        eng_lines = parse_engagement_lines(text)
        served_eng = []
        for entry in eng_lines:
            if entry["vram_slots"] is not None:
                last_residency = {
                    "vram_slots": entry["vram_slots"],
                    "host_slots": entry["host_slots"],
                    "total_experts": entry["total_experts"],
                }
            if entry["engine_request_id"] == INTERNAL_REQUEST_ID:
                internal_requests.append({"block": block, "phase": "internal", **entry})
            else:
                served_eng.append(entry)

        rows = results_by_block.get(block, [])
        count = max(len(rows), len(served_eng))
        for idx in range(count):
            row = rows[idx] if idx < len(rows) else None
            eng = served_eng[idx] if idx < len(served_eng) else None
            timing = _request_timing_fields(row)
            if eng is None:
                eng = {
                    "engine_request_id": None,
                    "drive_busy_fraction": 0.0,
                    "bytes_hidden": 0,
                    "bytes_exposed": 0,
                    "staged_hits": 0,
                    "landed_hits": 0,
                    "carryover_hits": 0,
                    "carryover_wasted": 0,
                    "dma_calls": 0,
                    "dma_skipped_zero_miss": 0,
                    "dma_experts_landed": 0,
                    "dma_bytes_landed": 0,
                    "dma_overflow_experts": 0,
                    "dma_host_wait_ms": 0.0,
                    "growth_refusals": 0,
                    "cpu_site_jobs": 0,
                    "vram_slots": None,
                    "host_slots": None,
                    "total_experts": None,
                }
            merged = {"block": block, **timing, **eng}
            requests.append(_enrich_ratios(merged))

    if last_residency["vram_slots"] is None:
        last_residency = _fallback_residency(block_texts)

    hardware_rates = parse_hardware_rates(block_texts, measure_text)

    tuning_data: dict | None = None
    trace_path = bench_dir / record.TUNING_TRACE
    if trace_path.is_file():
        health = report.tuning_health(trace_path)
        if health is not None:
            total_units = health["total_units"]
            fallbacks = health["canonical_fallbacks"]
            tuning_data = {
                "units": total_units,
                "qualified": health["qualified"],
                "canonical_fallbacks": fallbacks,
                "canonical_fallback_pct": 100.0 * fallbacks / total_units if total_units else None,
                "expired": health["expired"],
                "refused": health["refused"],
            }

    measured = [r for r in requests if r.get("phase") == "measured"]
    totals = _sum_group(measured) or _sum_group(requests)
    if totals is not None and totals["growth_refusals"] == 0 and log_growth_refusals > 0:
        totals["growth_refusals"] = log_growth_refusals

    return {
        "schema": SCHEMA,
        "residency": last_residency,
        "hardware_rates": hardware_rates,
        "tuning": tuning_data,
        "measured_totals": totals,
        "requests": requests,
        "internal_requests": internal_requests,
    }


def write(bench_dir: Path, measure_log: Path | None = None) -> dict:
    """Extract attribution and atomically write ``attribution.json`` inside ``bench_dir``."""
    data = extract(bench_dir, measure_log=measure_log)
    target = bench_dir / ATTRIBUTION_FILE
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(target)
    return data


def _fmt_opt(value: float | None, fmt: str = ".2f", suffix: str = "") -> str:
    return "—" if value is None else f"{value:{fmt}}{suffix}"


def format_summary(data: dict) -> list[str]:
    """Compact 4-line raw attribution summary (zero hardcoded thresholds or heuristics)."""
    totals = data.get("measured_totals")
    if totals is None:
        return []
    res = data.get("residency") or {}
    rates = data.get("hardware_rates") or {}
    tuning = data.get("tuning")

    vram = "—" if res.get("vram_slots") is None else str(res["vram_slots"])
    host = "—" if res.get("host_slots") is None else str(res["host_slots"])
    total_exp = "—" if res.get("total_experts") is None else str(res["total_experts"])
    draft_part = ""
    if totals.get("draft_proposed") is not None:
        draft_part = f", drafts={totals.get('draft_accepted', 0)}/{totals['draft_proposed']}"

    line1 = (
        f"Attribution ({totals['requests']} req, {totals['output_tokens']} tok, "
        f"{totals['decode_ms']:.0f} ms, {_fmt_opt(totals.get('decode_tps'), '.1f', ' tok/s')}"
        f"{draft_part}; residency {vram}/{host}/{total_exp} vram/host/total):"
    )

    hidden_gb = totals["bytes_hidden"] / 1e9
    exposed_gb = totals["bytes_exposed"] / 1e9
    hidden_mb_tok = (
        totals["bytes_hidden_per_tok"] / 1e6 if totals.get("bytes_hidden_per_tok") is not None else None
    )
    exposed_mb_tok = (
        totals["bytes_exposed_per_tok"] / 1e6 if totals.get("bytes_exposed_per_tok") is not None else None
    )
    line2 = (
        f"  Stream: hidden {hidden_gb:.2f} GB ({_fmt_opt(hidden_mb_tok, '.1f', ' MB/tok')}), "
        f"exposed {exposed_gb:.2f} GB ({_fmt_opt(exposed_mb_tok, '.1f', ' MB/tok')}, "
        f"{_fmt_opt(totals.get('implied_exposed_gb_s'), '.2f', ' GB/s')} implied), "
        f"staged={totals['staged_hits']} landed={totals['landed_hits']} "
        f"carryover={totals['carryover_hits']}/{totals['carryover_wasted']} "
        f"cpu_jobs={totals['cpu_site_jobs']} refusals={totals['growth_refusals']}"
    )

    dma_gb = totals["dma_bytes_landed"] / 1e9
    dma_mb_tok = (
        totals["dma_bytes_per_tok"] / 1e6 if totals.get("dma_bytes_per_tok") is not None else None
    )
    line3 = (
        f"  DMA: calls={totals['dma_calls']} ({totals['dma_skipped_zero_miss']} skipped), "
        f"landed={totals['dma_experts_landed']} experts ({dma_gb:.2f} GB, "
        f"{_fmt_opt(dma_mb_tok, '.1f', ' MB/tok')}), "
        f"overflow={totals['dma_overflow_experts']}, host_wait={totals['dma_host_wait_ms']:.1f} ms"
    )

    tuning_str = (
        f"{tuning['canonical_fallbacks']}/{tuning['units']} fallbacks "
        f"({_fmt_opt(tuning.get('canonical_fallback_pct'), '.1f', '%')}), "
        f"{tuning['refused']} refused"
        if tuning is not None
        else "—"
    )
    line4 = (
        f"  Rates & Tuning: line_stream={_fmt_opt(rates.get('line_stream_gb_s'), '.1f', ' GB/s')} "
        f"transfer_copy={_fmt_opt(rates.get('transfer_copy_gb_s'), '.2f', ' GB/s')} "
        f"storage[full]={_fmt_opt(rates.get('storage_full_gb_s'), '.2f', ' GB/s')} "
        f"| tuning {tuning_str}"
    )
    return [line1, line2, line3, line4]
