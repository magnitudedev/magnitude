"""100%-Additive Floor Waterfall for `mrig step`, `mrig draft`, and `mrig inspect <run>`.

Single owner for parsing a completed run's artifacts (`result.json`, `draft.jsonl`,
`tuning.jsonl`, `host.json`, `logs/job-*.log`, `rig.json`) and formatting a concise,
zero-remainder diagnosis of every wall separating the run from its physical floor:

  1. Top-Line Wall vs Floor Summary (never prints device_ms without wall_ms and wall - dev)
  2. 100%-Additive Disjoint Wall-Time Decomposition (sums to 100.0% of wall_ms, 0.0 ms remainder)
  3. Memory, Placement & Bank-Slot Accounting Audit
  4. Tuning & Anomaly Flags (auto-ranked Next-Run Fix List)
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EVENT_PREFIX = "magnitude-harness: event "
U64_MAX = 18446744073709551615

# Kernel family mapping for step attribution entries (order matters: more specific first).
_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("routed_gate_up", ("routed_gate_up", "routed_up", "routed_gate")),
    ("routed_output", ("routed_output", "routed_down")),
    ("routed_route", ("routed_route", "routed_select", "routed_hint", "rows_topk")),
    ("routed_land", ("routed_land", "routed_convert")),
    ("attention", ("attention_", "indexer_", "rope", "kv_")),
    ("gated_delta", ("gated_delta_",)),
    ("stream_mix", ("stream_mix_",)),
    ("dense/readout", ("dense_", "readout_", "project_rows", "embed", "lm_head", "norm", "rms_")),
)

_HOT_VERIFY_PREFIXES = (
    "routed_gate_up", "routed_output", "routed_down", "routed_route", "routed_select",
    "routed_land", "attention_", "gated_delta_", "stream_mix_", "dense_", "readout_",
)


def classify_entry(entry: str) -> str:
    """Map a kernel entry name to its canonical waterfall family."""
    base = entry.split("#", 1)[0]
    for family, prefixes in _FAMILY_RULES:
        if any(base.startswith(prefix) for prefix in prefixes):
            return family
    return "other"


def _parse_some_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    m = re.match(r"^Some\((\d+)\)$", value.strip())
    return int(m.group(1)) if m else None


def _parse_kv(text: str) -> dict[str, str]:
    return dict(re.findall(r"([a-zA-Z0-9_]+)=([^\s]+)", text))


def _parse_sizing_figures(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in re.findall(r"(\w+):\s*(\d+)", text):
        out[key] = int(val)
    return out


def _gb(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    return f"{n / 1e9:.2f} GB"


@dataclass
class TuningInfo:
    total_units: int = 0
    searched: int = 0
    stored: int = 0
    refused_count: int = 0
    canonical_fallbacks: int = 0
    refused_entries: set[str] = field(default_factory=set)
    not_replayed_entries: set[str] = field(default_factory=set)
    budget_seconds: float | None = None
    used_seconds: float | None = None

    @property
    def tuned_units(self) -> int:
        return max(0, self.total_units - self.refused_count - len(self.not_replayed_entries))

    @property
    def unsearched_entries(self) -> set[str]:
        return self.refused_entries | self.not_replayed_entries

    @property
    def hot_unsearched(self) -> list[str]:
        return sorted(
            e for e in self.unsearched_entries
            if any(e.startswith(p) for p in _HOT_VERIFY_PREFIXES)
        )


@dataclass
class LogTelemetry:
    # Placement
    budget_available: int | None = None
    resident_bytes: int | None = None
    resident_short: int | None = None
    whole_bank_short: int | None = None
    # Bank & path records
    vram_slots: int | None = None
    total_experts: int | None = None
    ahead: int | None = None
    reserve_records: int | None = None
    stage_slots: int | None = None
    decode_rows_planned: int | None = None
    actual_decode_rows: int | None = None
    host_lent_bytes: int | None = None
    # Bandwidths
    pcie_gbps: float | None = None
    device_ssd_gbps: float | None = None
    # HostRam / sizing
    host_slots: int | None = None
    planned_host_slots: int | None = None
    sizing: dict[str, int] = field(default_factory=dict)
    grown_bytes: int = 0
    resizes: list[tuple[str, str, int | None, int | None, str | None, int | None]] = field(default_factory=list)
    # Engagement (per-request for draft anchors, plus summary)
    req_engagements: list[dict[str, Any]] = field(default_factory=list)
    summary_engagement: dict[str, Any] = field(default_factory=dict)
    # Landings
    demand_count: int = 0
    demand_read_ms: float = 0.0
    demand_read_n: int = 0
    demand_queued_ms: float = 0.0
    demand_converting_ms: float = 0.0
    hinted_count: int = 0
    hinted_queued_ms: float = 0.0
    hinted_converting_ms: float = 0.0
    record_waits: int = 0
    record_waits_ms: float = 0.0
    # Guarantee oscillation tracking
    guarantee_values: list[int] = field(default_factory=list)

    def oscillation(self) -> tuple[bool, str]:
        """Detect slot ping-pong (e.g. 81 <-> 82) or guarantee-driven resize oscillation."""
        if len(self.resizes) < 2:
            return False, ""
        # Check ping-pong between two slot counts on the same holding.
        by_holding: dict[str, list[tuple[str, int | None, int | None, str | None]]] = {}
        for kind, holding, frm, to, term, _ in self.resizes:
            by_holding.setdefault(holding, []).append((kind, frm, to, term))
        for holding, ops in by_holding.items():
            if len(ops) < 2:
                continue
            kinds = {op[0] for op in ops}
            if "grown" in kinds and "released" in kinds:
                slots = sorted({s for op in ops for s in (op[1], op[2]) if s is not None})
                if len(slots) >= 2 and (slots[-1] - slots[0]) <= 4:
                    terms = {op[3] for op in ops if op[3]}
                    tag = "guarantee " if "guarantee" in terms else ""
                    return True, f"{holding} {tag}oscillation {slots[0]}↔{slots[-1]} ({len(ops)} resizes)"
        # Check guarantee toggling between 0 and >0 across resizes.
        if len(self.guarantee_values) >= 2:
            has_zero = any(g == 0 for g in self.guarantee_values)
            max_g = max(self.guarantee_values)
            if has_zero and max_g >= 50_000_000 and sum(1 for r in self.resizes if r[4] == "guarantee") >= 2:
                return True, f"guarantee toggled 0↔{max_g / 1e6:.0f} MB across {len(self.resizes)} resizes"
        return False, ""


def _parse_tuning(job_dir: Path, result_data: dict[str, Any]) -> TuningInfo:
    info = TuningInfo()
    # First from result.json (if present)
    loads = (result_data.get("measurements", {}).get("tuning", {}).get("outcome") or {}).get("loads") or []
    if loads:
        # Prefer the searching load if multiple loads exist (e.g. search then warm load).
        search_loads = [ld for ld in loads if ld.get("work") == "search" or ld.get("refused", 0) > 0]
        primary = search_loads[-1] if search_loads else loads[-1]
        info.total_units = int(primary.get("units", 0))
        info.searched = int(primary.get("searched", 0))
        info.stored = int(primary.get("stored", 0))
        info.refused_count = int(primary.get("refused", 0))
        info.budget_seconds = primary.get("budget_seconds")
    # Enrich with tuning.jsonl (exact refused entry names, canonical fallbacks, pinned)
    tpath = job_dir / "tuning.jsonl"
    if tpath.is_file():
        for raw in tpath.read_text(errors="replace").splitlines():
            if not raw.strip():
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            etype = ev.get("event")
            if etype == "budget":
                info.budget_seconds = ev.get("budget_seconds", info.budget_seconds)
                info.used_seconds = ev.get("used_seconds", info.used_seconds)
                refused = ev.get("refused") or []
                if len(refused) > info.refused_count:
                    info.refused_count = len(refused)
                for offer in refused:
                    unit = offer.get("unit", "")
                    if unit:
                        info.refused_entries.add(unit.split()[0])
            elif etype == "concluded":
                if info.total_units == 0:
                    info.total_units = int(ev.get("units", 0))
                    info.searched = int(ev.get("searched", 0))
                    info.stored = int(ev.get("stored", 0))
                info.canonical_fallbacks = max(info.canonical_fallbacks, int(ev.get("canonical_fallbacks", 0)))
            elif etype == "pinned":
                if ev.get("choice") == "not_replayed" and ev.get("entry"):
                    info.not_replayed_entries.add(ev["entry"])
    return info


def _parse_log(log_path: Path | None) -> LogTelemetry:
    tel = LogTelemetry()
    if log_path is None or not log_path.is_file():
        return tel
    for line in log_path.read_text(errors="replace").splitlines():
        idx = line.find(EVENT_PREFIX)
        if idx < 0:
            continue
        try:
            ev = json.loads(line[idx + len(EVENT_PREFIX):])
        except json.JSONDecodeError:
            continue
        msg = ev.get("message", "")

        if msg == "the load's experts are placed":
            tel.budget_available = _parse_some_int(ev.get("budget_available")) or tel.budget_available
            tel.resident_bytes = _parse_some_int(ev.get("resident_bytes")) or tel.resident_bytes
            rs = _parse_some_int(ev.get("resident_short"))
            if rs is not None:
                tel.resident_short = rs
            wbs = _parse_some_int(ev.get("whole_bank_short"))
            if wbs is not None:
                tel.whole_bank_short = wbs

        elif msg == "the bank's path records are settled":
            if "slots" in ev:
                tel.vram_slots = int(ev["slots"])
            if "ahead" in ev:
                tel.ahead = int(ev["ahead"])
            if "reserve" in ev:
                tel.reserve_records = int(ev["reserve"])
            if "stage_slots" in ev:
                tel.stage_slots = int(ev["stage_slots"])
            if "decode_rows" in ev:
                tel.decode_rows_planned = int(ev["decode_rows"])

        elif msg.startswith("expert bank:"):
            m = re.search(r"expert bank:\s*(\d+)\s+of the widest routed layer's\s+(\d+)\s+experts", msg)
            if m:
                tel.vram_slots = int(m.group(1))
                tel.total_experts = int(m.group(2))

        elif msg.startswith("a promotion's copy costs its link time at "):
            m = re.search(r"at\s+([0-9.]+)\s+GB/s", msg)
            if m:
                tel.pcie_gbps = float(m.group(1))

        elif msg.startswith("the host tier's "):
            m = re.search(r"ended at\s+(\d+)\s+of\s+(\d+)\s+planned host slots\s+\((\d+)\s+every expert\)", msg)
            if m:
                tel.host_slots = int(m.group(1))
                tel.planned_host_slots = int(m.group(2))
                tel.total_experts = int(m.group(3))

        elif msg in ("memory grown", "memory released"):
            kind = "grown" if msg == "memory grown" else "released"
            holding = str(ev.get("holding", "?"))
            frm = ev.get("from")
            to = ev.get("to")
            term = ev.get("term")
            if kind == "grown" and ev.get("grown") is not None:
                tel.grown_bytes += int(ev["grown"])
            if "bank_slots" in ev:
                tel.vram_slots = int(ev["bank_slots"])
            if holding == "Host" and to is not None:
                tel.host_slots = int(to)
            figs = _parse_sizing_figures(str(ev.get("figures", "")))
            if figs:
                tel.sizing = figs
                if "guarantee" in figs:
                    tel.guarantee_values.append(figs["guarantee"])
            tel.resizes.append((kind, holding, frm, to, term, figs.get("guarantee")))

        elif msg.startswith("expert engagement (request "):
            m = re.match(r"expert engagement \(request (\d+)\):\s*(.*)", msg)
            if m:
                req_id = int(m.group(1))
                kv = _parse_kv(m.group(2))
                if req_id != U64_MAX:
                    tel.req_engagements.append(kv)
                # Always pick up latest vram/host/total metadata.
                if "vram_slots" in kv:
                    tel.vram_slots = int(kv["vram_slots"])
                if "host_slots" in kv:
                    tel.host_slots = int(kv["host_slots"])
                if "total_experts" in kv:
                    tel.total_experts = int(kv["total_experts"])

        elif msg.startswith("expert engagement:"):
            kv = _parse_kv(msg.split(":", 1)[1])
            tel.summary_engagement = kv
            if "vram_slots" in kv:
                tel.vram_slots = int(kv["vram_slots"])
            if "host_slots" in kv:
                tel.host_slots = int(kv["host_slots"])
            if "total_experts" in kv:
                tel.total_experts = int(kv["total_experts"])

        elif msg.startswith("expert landings"):
            dm = re.search(
                r"demand\s+(\d+):\s+read\s+([0-9.]+)\s+ms\s+\(mean of\s+(\d+)\),\s+"
                r"queued\s+([0-9.]+)\s+ms,\s+converting\s+([0-9.]+)\s+ms",
                msg,
            )
            if dm:
                tel.demand_count = int(dm.group(1))
                tel.demand_read_ms = float(dm.group(2))
                tel.demand_read_n = int(dm.group(3))
                tel.demand_queued_ms = float(dm.group(4))
                tel.demand_converting_ms = float(dm.group(5))
            hm = re.search(
                r"hinted\s+(\d+):\s+read\s+[0-9.]+\s+ms\s+\(mean of\s+\d+\),\s+"
                r"queued\s+([0-9.]+)\s+ms,\s+converting\s+([0-9.]+)\s+ms",
                msg,
            )
            if hm:
                tel.hinted_count = int(hm.group(1))
                tel.hinted_queued_ms = float(hm.group(2))
                tel.hinted_converting_ms = float(hm.group(3))
            wm = re.search(r"waits for a record\s+(\d+)\s+\(([0-9.]+)\s+ms in all\)", msg)
            if wm:
                tel.record_waits = int(wm.group(1))
                tel.record_waits_ms = float(wm.group(2))

        # Optional extended telemetry fields (supported forward-compatibly)
        for key in ("stage_slots", "decode_rows_planned", "bank_decode_rows", "actual_decode_rows", "host_lent_bytes"):
            if key in ev and ev[key] is not None:
                val = int(ev[key])
                if key == "stage_slots":
                    tel.stage_slots = val
                elif key in ("decode_rows_planned", "bank_decode_rows"):
                    tel.decode_rows_planned = val
                elif key == "actual_decode_rows":
                    tel.actual_decode_rows = val
                elif key == "host_lent_bytes":
                    tel.host_lent_bytes = val
    return tel


def _is_cuda(result_data: dict[str, Any], host: dict[str, Any]) -> bool:
    cmd = result_data.get("command") or []
    if "--device" in cmd:
        idx = cmd.index("--device")
        if idx + 1 < len(cmd):
            return cmd[idx + 1] == "cuda"
    gpus = host.get("gpus") or []
    return any("nvidia" in str(g.get("name", "")).lower() for g in gpus)


def _is_discrete(host: dict[str, Any]) -> bool:
    gpus = host.get("gpus") or []
    return any(g.get("memory_bytes") is not None for g in gpus)


def _format_draft_waterfall(
    label: str,
    job_dir: Path,
    result_data: dict[str, Any],
    tel: LogTelemetry,
    tuning: TuningInfo,
) -> list[str]:
    host = result_data.get("host") or {}
    if not host and (job_dir / "host.json").is_file():
        host = json.loads((job_dir / "host.json").read_text())
    model = result_data.get("model", "?")
    alias = host.get("alias", "?")
    cuda = _is_cuda(result_data, host)
    discrete = _is_discrete(host)

    drafting = result_data.get("measurements", {}).get("drafting") or {}
    tok_per_verify = float(drafting.get("tokens_per_verify", 1.0))
    proposed = int(drafting.get("proposed", 0))
    accepted = int(drafting.get("accepted", 0))
    anchors = int(drafting.get("anchors", 0))
    acc_pct = 100.0 * accepted / max(proposed, 1)

    dpath = job_dir / "draft.jsonl"
    rows = [json.loads(l) for l in dpath.read_text().splitlines() if l.strip()] if dpath.is_file() else []
    actual_verify_rows = (len(rows[0].get("proposals", [])) + 1) if rows else None

    phase_stats: dict[str, dict[str, float]] = {}
    for phase in ("plain", "draft", "verify"):
        vals = [r["attribution"][phase] for r in rows if "attribution" in r and phase in r["attribution"]]
        if vals:
            wall = statistics.median(v["wall_ms"] for v in vals)
            dev = statistics.median(v["device_ms"] for v in vals)
            subs = statistics.median(v.get("submissions", 0) for v in vals)
            phase_stats[phase] = {
                "wall_ms": wall,
                "dev_ms": dev,
                "gap_ms": max(0.0, wall - dev),
                "subs": subs,
            }

    lines = [f"=== FLOOR WATERFALL [{label}] · {model} · {alias} ==="]

    # ── Section 1: Top-Line Wall vs Floor Summary ────────────────────────────
    lines.append("1. WALL vs FLOOR (median across anchors):")
    if not phase_stats:
        lines.append("  no --attribute timings recorded in draft.jsonl (run with --attribute for wall/dev breakdown)")
        return lines

    plain = phase_stats.get("plain")
    draft = phase_stats.get("draft")
    verify = phase_stats.get("verify")

    if plain:
        w, d, g, s = plain["wall_ms"], plain["dev_ms"], plain["gap_ms"], plain["subs"]
        lines.append(
            f"  plain : wall {w:7.2f} ms ({1000 / w:5.1f} tok/s) · dev {d:7.2f} ms ({1000 / d:5.1f} tok/s) · "
            f"wall-dev {g:6.2f} ms ({100 * g / w:4.1f}%) · subs {s:.0f}"
        )
    if draft and verify:
        cycle_wall = draft["wall_ms"] + verify["wall_ms"]
        cycle_dev = draft["dev_ms"] + verify["dev_ms"]
        cycle_gap = max(0.0, cycle_wall - cycle_dev)
        dw, dd, dg, ds = draft["wall_ms"], draft["dev_ms"], draft["gap_ms"], draft["subs"]
        vw, vd, vg, vs = verify["wall_ms"], verify["dev_ms"], verify["gap_ms"], verify["subs"]
        lines.append(
            f"  draft : wall {dw:7.2f} ms · dev {dd:7.2f} ms · wall-dev {dg:6.2f} ms ({100 * dg / dw:4.1f}%) · "
            f"subs {ds:.0f} · {100 * dw / cycle_wall:4.1f}% of cycle"
        )
        lines.append(
            f"  verify: wall {vw:7.2f} ms · dev {vd:7.2f} ms · wall-dev {vg:6.2f} ms ({100 * vg / vw:4.1f}%) · "
            f"subs {vs:.0f}"
        )
        lines.append(
            f"  cycle : wall {cycle_wall:7.2f} ms ({1000 * tok_per_verify / cycle_wall:5.1f} tok/s) · "
            f"dev {cycle_dev:7.2f} ms ({1000 * tok_per_verify / cycle_dev:5.1f} tok/s) · "
            f"wall-dev {cycle_gap:6.2f} ms ({100 * cycle_gap / cycle_wall:4.1f}%) · "
            f"{tok_per_verify:.2f} tok/verify ({acc_pct:.1f}% acc, {anchors} anchors)"
        )
    elif verify:
        cycle_wall, cycle_dev = verify["wall_ms"], verify["dev_ms"]
    else:
        cycle_wall, cycle_dev = plain["wall_ms"], plain["dev_ms"]

    # ── Section 2: 100%-Additive Disjoint Wall-Time Decomposition ────────────
    # Compute per-anchor engagement medians (excluding u64::MAX prefill).
    req_dma_bytes = statistics.median(int(r.get("req_dma_bytes", 0)) for r in tel.req_engagements) if tel.req_engagements else 0.0
    req_dma_wait_ms = statistics.median(float(r.get("req_dma_wait_ms", 0.0)) for r in tel.req_engagements) if tel.req_engagements else 0.0
    req_exposed = statistics.median(int(r.get("req_bytes_exposed", 0)) for r in tel.req_engagements) if tel.req_engagements else 0.0
    req_hidden = statistics.median(int(r.get("req_bytes_hidden", 0)) for r in tel.req_engagements) if tel.req_engagements else 0.0

    # Each anchor runs 1 plain + 1 verify. Attribute anchor-level I/O to verify proportionally
    # to its share of (plain_gap + verify_gap) when both exist.
    vw = verify["wall_ms"] if verify else cycle_wall
    vd = verify["dev_ms"] if verify else cycle_dev
    vg = verify["gap_ms"] if verify else max(0.0, cycle_wall - cycle_dev)
    vs = verify["subs"] if verify else 0.0
    pg = plain["gap_ms"] if plain else 0.0
    verify_io_share = (vg / (vg + pg)) if (verify and plain and (vg + pg) > 0) else 1.0

    verify_dma_bytes = req_dma_bytes * verify_io_share
    verify_land_misses_ms = req_dma_wait_ms * verify_io_share
    verify_exposed_bytes = req_exposed * verify_io_share

    pcie_wire_ms = (verify_dma_bytes / (tel.pcie_gbps * 1e9) * 1000.0) if (tel.pcie_gbps and verify_dma_bytes > 0) else 0.0
    # Effective achieved SSD bandwidth from per-anchor exposed bytes / host land_decode_misses wall time.
    achieved_ssd_gbps: float | None = None
    if req_exposed > 0 and req_dma_wait_ms > 0:
        achieved_ssd_gbps = (req_exposed / (req_dma_wait_ms * 1e-3)) / 1e9
    nvme_wire_ms = (verify_exposed_bytes / (tel.device_ssd_gbps * 1e9) * 1000.0) if (tel.device_ssd_gbps and verify_exposed_bytes > 0) else 0.0

    lines.append(f"2. 100%-ADDITIVE CYCLE WATERFALL (total = {cycle_wall:.2f} ms, 0.00 ms unattributed):")
    buckets: list[float] = []
    hot_flag = " [DEFAULT/UNSEARCHED]" if tuning.hot_unsearched else ""

    # Bucket A: Verify GPU Device (vd) — CUDA DirectBatch::ensure_started()
    # defers the start event past cuStreamWaitEvent(DMA), so device_ms is pure
    # GPU compute and DMA wire wait lives in the wall-dev gap below.
    buckets.append(vd)
    lines.append(f"  [GPU] verify device_ms             {vd:7.2f} ms ({100 * vd / cycle_wall:5.1f}%){hot_flag}")

    # Bucket B: Verify Wall - Device Gap (vg)
    # Note: req_dma_wait_ms is host time inside land_decode_misses() (mostly serialized NVMe demand reads + claim).
    host_land_ms = min(vg, verify_land_misses_ms) if verify_land_misses_ms > 0 else 0.0
    if host_land_ms == 0.0 and nvme_wire_ms > 0:
        host_land_ms = min(vg, nvme_wire_ms)
    rem_gap = max(0.0, vg - host_land_ms)
    pcie_gap_ms = min(rem_gap, pcie_wire_ms) if (cuda and pcie_wire_ms > 0) else 0.0
    host_sync_ms = max(0.0, rem_gap - pcie_gap_ms)
    buckets.append(host_land_ms)
    nvme_note = (
        f"NVMe wire ~{nvme_wire_ms:.1f} ms, " if nvme_wire_ms > 0
        else (f"exposed {verify_exposed_bytes / 1e6:.0f} MB, " if verify_exposed_bytes > 0 else "")
    )
    lines.append(
        f"  [GAP] host_land_misses (NVMe+claim){host_land_ms:7.2f} ms ({100 * host_land_ms / cycle_wall:5.1f}%)  "
        f"<- {nvme_note}anchor land={req_dma_wait_ms:.1f} ms, q(dem={tel.demand_queued_ms:.2f}/hnt={tel.hinted_queued_ms:.2f} ms)"
    )
    if pcie_gap_ms > 0:
        buckets.append(pcie_gap_ms)
        lines.append(
            f"  [GAP] verify PCIe H2D wire (DMA)   {pcie_gap_ms:7.2f} ms ({100 * pcie_gap_ms / cycle_wall:5.1f}%)  "
            f"<- {verify_dma_bytes / 1e9:.2f} GB @ {tel.pcie_gbps:.1f} GB/s"
        )
    buckets.append(host_sync_ms)
    lines.append(
        f"  [GAP] host_sync_and_other          {host_sync_ms:7.2f} ms ({100 * host_sync_ms / cycle_wall:5.1f}%)  "
        f"<- {vs:.0f} submissions/verify"
    )

    # Bucket C: Draft Cost (if speculative)
    if draft:
        dd, dg = draft["dev_ms"], draft["gap_ms"]
        buckets.extend([dd, dg])
        lines.append(
            f"  [DFT] draft device_ms              {dd:7.2f} ms ({100 * dd / cycle_wall:5.1f}%)  "
            f"<- total draft {100 * draft['wall_ms'] / cycle_wall:.1f}% of cycle"
        )
        lines.append(
            f"  [DFT] draft wall-dev gap           {dg:7.2f} ms ({100 * dg / cycle_wall:5.1f}%)  "
            f"<- {draft['subs']:.0f} submissions/draft"
        )

    accounted = sum(buckets)
    remainder = cycle_wall - accounted
    assert abs(remainder) < 1e-4, f"waterfall remainder {remainder}"
    lines.append(f"  TOTAL ACCOUNTED                    {accounted:7.2f} ms (100.0%, remainder {remainder:+.2f} ms)")

    # ── Section 3: Memory, Placement & Bank-Slot Accounting Audit ────────────
    lines.extend(_format_memory_and_streaming(tel, host, discrete, actual_verify_rows, achieved_ssd_gbps))

    # ── Section 4: Tuning & Auto-Ranked Next-Run Fix List ────────────────────
    lines.extend(_format_tuning_and_fixes(
        tuning=tuning,
        tel=tel,
        host=host,
        discrete=discrete,
        wall_ms=cycle_wall,
        dev_ms=cycle_dev,
        gap_ms=vg,
        host_land_ms=host_land_ms,
        host_sync_ms=host_sync_ms,
        pcie_wire_ms=pcie_wire_ms if cuda else 0.0,
        subs=vs,
        actual_decode_rows=actual_verify_rows,
        top_family=None,
    ))
    return lines


def _format_step_waterfall(
    label: str,
    job_dir: Path,
    result_data: dict[str, Any],
    tel: LogTelemetry,
    tuning: TuningInfo,
) -> list[str]:
    host = result_data.get("host") or {}
    if not host and (job_dir / "host.json").is_file():
        host = json.loads((job_dir / "host.json").read_text())
    model = result_data.get("model", "?")
    alias = host.get("alias", "?")
    cuda = _is_cuda(result_data, host)
    discrete = _is_discrete(host)

    speed = result_data.get("measurements", {}).get("speed") or []
    decode_cell = next((c for c in speed if c.get("phase") == "decode"), None)
    if decode_cell is None:
        return [f"=== FLOOR WATERFALL [{label}] · no decode cell recorded ==="]

    detail = decode_cell.get("detail") or {}
    concurrency = int(decode_cell.get("concurrency", 1))
    window = detail.get("window") or {}
    wall_ms = float(window.get("wall_ms_per_step") or detail.get("median_ms") or 0.0)
    if wall_ms <= 0:
        return [f"=== FLOOR WATERFALL [{label}] · invalid decode wall_ms ==="]

    entries = (detail.get("attribution") or {}).get("entries") or []
    raw_entry_sum = sum(float(e.get("device_ms_per_step", 0.0)) for e in entries)
    dev_busy = window.get("device_busy_ms_per_step")
    dev_ms = float(dev_busy) if dev_busy is not None else min(wall_ms, raw_entry_sum)
    dev_ms = min(dev_ms, wall_ms)
    gap_ms = max(0.0, wall_ms - dev_ms)
    launches = round(sum(float(e.get("launches_per_step", 0.0)) for e in entries)) if entries else 0

    lines = [f"=== FLOOR WATERFALL [{label}] · {model} · {alias} ==="]

    # ── Section 1: Top-Line Wall vs Floor Summary ────────────────────────────
    lines.append("1. WALL vs FLOOR:")
    lines.append(
        f"  decode: wall {wall_ms:7.2f} ms ({1000 * concurrency / wall_ms:5.1f} tok/s) · "
        f"dev {dev_ms:7.2f} ms ({1000 * concurrency / max(dev_ms, 1e-6):5.1f} tok/s) · "
        f"wall-dev {gap_ms:6.2f} ms ({100 * gap_ms / wall_ms:4.1f}%) · {launches} launches (C={concurrency})"
    )

    # ── Section 2: 100%-Additive Disjoint Wall-Time Decomposition ────────────
    lines.append(f"2. 100%-ADDITIVE DECODE WATERFALL (total = {wall_ms:.2f} ms, 0.00 ms unattributed):")
    buckets: list[float] = []
    unsearched = tuning.unsearched_entries

    # Group attribution entries by canonical family, scaled proportionally to dev_ms so sum == dev_ms.
    fam_ms: dict[str, float] = {}
    fam_launches: dict[str, float] = {}
    fam_unsearched: set[str] = set()
    for e in entries:
        ename = str(e.get("entry", ""))
        fam = classify_entry(ename)
        fam_ms[fam] = fam_ms.get(fam, 0.0) + float(e.get("device_ms_per_step", 0.0))
        fam_launches[fam] = fam_launches.get(fam, 0.0) + float(e.get("launches_per_step", 0.0))
        base = ename.split("#", 1)[0]
        if base in unsearched or ename in unsearched:
            fam_unsearched.add(fam)

    top_family: tuple[str, float] | None = None
    if fam_ms and raw_entry_sum > 0:
        scale = dev_ms / raw_entry_sum
        ordered = sorted(fam_ms.items(), key=lambda kv: -kv[1])
        top_family = (ordered[0][0], ordered[0][1] * scale)
        for fam, raw_ms in ordered:
            ms = raw_ms * scale
            buckets.append(ms)
            flag = " [DEFAULT/UNSEARCHED]" if fam in fam_unsearched else ""
            lines.append(
                f"  [GPU] {fam:<20}         {ms:7.2f} ms ({100 * ms / wall_ms:5.1f}%)  "
                f"{fam_launches[fam]:5.0f} launches{flag}"
            )
    else:
        buckets.append(dev_ms)
        flag = " [DEFAULT/UNSEARCHED]" if tuning.hot_unsearched else ""
        lines.append(f"  [GPU] device_ms                    {dev_ms:7.2f} ms ({100 * dev_ms / wall_ms:5.1f}%){flag}")

    # Wall - Device gap breakdown (exact from window.idle when present).
    idle = window.get("idle") or {}
    between = idle.get("between_runs") or {}
    at_b = idle.get("at_boundaries") or {}
    subs_per_step = sum(float(s.get("submissions_per_step", 0.0)) for s in (between.get("by_submission") or []))
    cb_cuts = float(at_b.get("per_step", 0.0))

    # Extract SSD BW from step result.json if present.
    read_bps = detail.get("read_bytes_per_second")
    if read_bps and not tel.device_ssd_gbps:
        tel.device_ssd_gbps = float(read_bps) / 1e9
    reads = detail.get("reads") or {}
    demand_r = reads.get("demand") or {}
    achieved_ssd_gbps: float | None = None
    if demand_r.get("seconds", 0) > 0 and demand_r.get("bytes", 0) > 0:
        achieved_ssd_gbps = (float(demand_r["bytes"]) / float(demand_r["seconds"])) / 1e9

    awaiting_data_ms = min(gap_ms, float(at_b.get("awaiting_data_ms_per_step", 0.0)))
    queued_between_ms = min(max(0.0, gap_ms - awaiting_data_ms), float(between.get("queued_ms_per_step", 0.0)))
    host_sync_ms = max(0.0, gap_ms - awaiting_data_ms - queued_between_ms)
    buckets.extend([awaiting_data_ms, queued_between_ms, host_sync_ms])

    if awaiting_data_ms > 0 or gap_ms > 0.01:
        lines.append(
            f"  [GAP] awaiting_data (NVMe+land)    {awaiting_data_ms:7.2f} ms ({100 * awaiting_data_ms / wall_ms:5.1f}%)  "
            f"<- q(dem={tel.demand_queued_ms:.2f}/hnt={tel.hinted_queued_ms:.2f} ms)"
        )
        lines.append(
            f"  [GAP] queued_between_runs          {queued_between_ms:7.2f} ms ({100 * queued_between_ms / wall_ms:5.1f}%)"
        )
        lines.append(
            f"  [GAP] host_sync_and_other          {host_sync_ms:7.2f} ms ({100 * host_sync_ms / wall_ms:5.1f}%)  "
            f"<- subs={subs_per_step:.1f}/step, cb_cuts={cb_cuts:.1f}/step"
        )

    accounted = sum(buckets)
    remainder = wall_ms - accounted
    assert abs(remainder) < 1e-4, f"waterfall remainder {remainder}"
    lines.append(f"  TOTAL ACCOUNTED                    {accounted:7.2f} ms (100.0%, remainder {remainder:+.2f} ms)")

    # Enrich telemetry from step's expert_bank / unsteady / reads if log lacked them.
    eb = detail.get("expert_bank") or {}
    if eb:
        if tel.vram_slots is None and eb.get("slots") is not None:
            tel.vram_slots = int(eb["slots"])
        if tel.total_experts is None and eb.get("experts") is not None:
            tel.total_experts = int(eb["experts"])
        if tel.host_lent_bytes is None and isinstance(eb.get("lend"), int):
            tel.host_lent_bytes = int(eb["lend"])
        host_tier = (eb.get("residency") or {}).get("host")
        if host_tier and tel.host_slots is None:
            tel.host_slots = int(host_tier.get("slots", 0))
    unsteady_windows = window.get("unsteady") or []
    unsteady_resizes = sum((w.get("moved") or w.get("warm") or {}).get("resizes_begun", 0) for w in unsteady_windows)

    # ── Section 3: Memory, Placement & Bank-Slot Accounting Audit ────────────
    lines.extend(_format_memory_and_streaming(
        tel, host, discrete, concurrency, achieved_ssd_gbps, extra_resizes=unsteady_resizes, step_reads=reads,
    ))

    # ── Section 4: Tuning & Auto-Ranked Next-Run Fix List ────────────────────
    lines.extend(_format_tuning_and_fixes(
        tuning=tuning,
        tel=tel,
        host=host,
        discrete=discrete,
        wall_ms=wall_ms,
        dev_ms=dev_ms,
        gap_ms=gap_ms,
        host_land_ms=awaiting_data_ms,
        host_sync_ms=host_sync_ms,
        pcie_wire_ms=0.0,
        subs=subs_per_step or cb_cuts,
        actual_decode_rows=concurrency,
        top_family=top_family,
    ))
    return lines


def _format_memory_and_streaming(
    tel: LogTelemetry,
    host: dict[str, Any],
    discrete: bool,
    actual_decode_rows: int | None,
    achieved_ssd_gbps: float | None,
    *,
    extra_resizes: int = 0,
    step_reads: dict[str, Any] | None = None,
) -> list[str]:
    lines = ["3. MEMORY, PLACEMENT & STREAMING AUDIT:"]

    # VRAM / UMA line
    vram_held_str = "n/a"
    vram_reserve_str = ""
    reserve_flag = ""
    if tel.req_engagements:
        last_req = tel.req_engagements[-1]
        if "vram_held" in last_req:
            vram_held_str = last_req["vram_held"]
        if "vram_reserve" in last_req:
            vres = int(last_req["vram_reserve"])
            vram_reserve_str = f" · vram_reserve {vres}"
            if "/" in vram_held_str:
                cap = int(vram_held_str.split("/")[1])
                if cap > 0 and vres / cap > 0.10:
                    reserve_flag = f" [! RESERVE {100 * vres / cap:.1f}% >10%]"
    stage_str = ""
    if tel.stage_slots is not None:
        stage_str = f" · stage_slots {tel.stage_slots}"
        if tel.vram_slots and tel.stage_slots / tel.vram_slots > 0.10:
            stage_str += " [! >10% CAP]"

    slot_pct = (
        f"{tel.vram_slots}/{tel.total_experts} ({100 * tel.vram_slots / tel.total_experts:.1f}%)"
        if tel.vram_slots is not None and tel.total_experts
        else "n/a"
    )
    short_str = ""
    if tel.resident_short is not None:
        short_str = f" · resident_short {_gb(tel.resident_short)}"
        if 0 < tel.resident_short < 5_000_000_000:
            short_str += " [! <5 GB TO 100% RESIDENT CLIFF]"
    total_resizes = len(tel.resizes) + extra_resizes
    osc, osc_msg = tel.oscillation()
    osc_str = f" [! {osc_msg}]" if osc else ""

    tier_label = "VRAM" if discrete else "UMA "
    lines.append(
        f"  {tier_label}: slots {slot_pct} · held {vram_held_str}{vram_reserve_str}{reserve_flag}"
        f"{stage_str}{short_str} · resizes {total_resizes}{osc_str}"
    )

    # HostRam / Sizing line
    host_slot_str = (
        f"{tel.host_slots}/{tel.total_experts} ({100 * tel.host_slots / tel.total_experts:.1f}%)"
        if tel.host_slots is not None and tel.total_experts
        else "none"
    )
    sizing_parts: list[str] = []
    if tel.sizing:
        if "committed" in tel.sizing:
            sizing_parts.append(f"committed {_gb(tel.sizing['committed'])}")
        if tel.grown_bytes:
            sizing_parts.append(f"grown {_gb(tel.grown_bytes)}")
        if "others" in tel.sizing:
            sizing_parts.append(f"others {_gb(tel.sizing['others'])}")
        if "guarantee" in tel.sizing:
            sizing_parts.append(f"guarantee {tel.sizing['guarantee'] / 1e6:.0f} MB")
    rows_str = ""
    if tel.decode_rows_planned is not None and actual_decode_rows is not None:
        mismatch = " [! MISMATCH]" if tel.decode_rows_planned != actual_decode_rows else ""
        rows_str = f" · decode_rows plan={tel.decode_rows_planned} vs act={actual_decode_rows}{mismatch}"
    ahead_str = f" · ahead/res {tel.ahead}/{tel.reserve_records}" if tel.ahead is not None else ""
    lent_str = f" · host_lent {_gb(tel.host_lent_bytes)}" if tel.host_lent_bytes else ""
    sizing_str = (" · " + ", ".join(sizing_parts)) if sizing_parts else ""
    lines.append(f"  Host: slots {host_slot_str}{sizing_str}{rows_str}{ahead_str}{lent_str}")

    # Streaming health line
    eng = tel.summary_engagement
    dma_ovf = int(eng.get("dma_overflow", 0))
    cpu_exp = int(eng.get("cpu_site_experts", 0))
    b_exp = int(eng.get("bytes_exposed", 0))
    b_hid = int(eng.get("bytes_hidden", 0))
    b_tot = b_exp + b_hid
    hid_pct = (100.0 * b_hid / b_tot) if b_tot > 0 else 100.0
    staged_hits = int(eng.get("staged_hits", 0))
    landed_hits = int(eng.get("landed_hits", 0))
    hint_hits = staged_hits + landed_hits
    if tel.hinted_count > 0:
        overread = tel.hinted_count / max(hint_hits, 1)
        hint_str = f"hints {tel.hinted_count} issued / {hint_hits} hits ({overread:.1f}x overread)"
    elif step_reads and (step_reads.get("hints") or {}).get("recall") is not None:
        h = step_reads["hints"]
        hint_str = f"hints recall {h['recall']:.2f}, prec {h.get('precision') or 0.0:.2f}"
    else:
        hint_str = f"hints hits {hint_hits}"

    ssd_str = ""
    if achieved_ssd_gbps is not None or tel.device_ssd_gbps is not None:
        ach = f"{achieved_ssd_gbps:.2f}" if achieved_ssd_gbps is not None else "n/a"
        dev = f"{tel.device_ssd_gbps:.2f}" if tel.device_ssd_gbps is not None else "n/a"
        ssd_str = f" · SSD {ach}/{dev} GB/s"

    ovf_flag = " [! OVERFLOW]" if dma_ovf > 0 else ""
    lines.append(
        f"  IO  : dma_overflow {dma_ovf}{ovf_flag} · cpu_site_exp {cpu_exp} · {hint_str} · "
        f"exposed {_gb(b_exp)} / hidden {_gb(b_hid)} ({hid_pct:.1f}% hidden){ssd_str}"
    )
    return lines


def _format_tuning_and_fixes(
    *,
    tuning: TuningInfo,
    tel: LogTelemetry,
    host: dict[str, Any],
    discrete: bool,
    wall_ms: float,
    dev_ms: float,
    gap_ms: float,
    host_land_ms: float,
    host_sync_ms: float,
    pcie_wire_ms: float,
    subs: float,
    actual_decode_rows: int | None,
    top_family: tuple[str, float] | None,
) -> list[str]:
    lines = ["4. TUNING & AUTO-RANKED NEXT-RUN FIX LIST:"]
    if tuning.total_units > 0:
        ref_flag = f" [! TUNING STARVATION: {tuning.refused_count} refused]" if tuning.refused_count > 0 else ""
        fb_str = f" · {tuning.canonical_fallbacks} canonical_fallbacks" if tuning.canonical_fallbacks > 0 else ""
        lines.append(
            f"  tuning: {tuning.tuned_units}/{tuning.total_units} tuned "
            f"(stored {tuning.stored}, searched {tuning.searched}, refused {tuning.refused_count}){fb_str}{ref_flag}"
        )
    else:
        lines.append("  tuning: not recorded")

    # Rank actionable gaps by estimated ms impact (descending).
    fixes: list[tuple[float, str]] = []

    if gap_ms > 5.0 or (wall_ms > 0 and gap_ms / wall_ms > 0.15):
        causes: list[str] = []
        if host_land_ms > 2.0:
            causes.append(f"host_land_misses {host_land_ms:.1f} ms")
        if host_sync_ms > 2.0:
            causes.append(f"host_sync/other {host_sync_ms:.1f} ms ({subs:.0f} subs)")
        cause_s = ", ".join(causes) or f"{subs:.0f} subs"
        fixes.append((
            gap_ms,
            f"Wall-Dev Gap: {gap_ms:.1f} ms ({100 * gap_ms / wall_ms:.0f}% of wall) — {cause_s}",
        ))

    if pcie_wire_ms > 5.0:
        fixes.append((
            pcie_wire_ms,
            f"CUDA PCIe H2D DMA Wait (inside device_ms): {pcie_wire_ms:.1f} ms ({100 * pcie_wire_ms / wall_ms:.0f}% of wall) — "
            f"overlap H2D or raise VRAM slots ({tel.vram_slots}/{tel.total_experts})",
        ))

    if tuning.refused_count > 0:
        hot_sample = ", ".join(tuning.hot_unsearched[:4]) if tuning.hot_unsearched else "multiple units"
        # Estimate conservative 15% of dev_ms impact for unsearched hot kernels.
        est_ms = max(10.0, 0.25 * dev_ms) if tuning.hot_unsearched else 5.0
        fixes.append((
            est_ms,
            f"Tuning Starvation: {tuning.refused_count}/{tuning.total_units} units refused "
            f"(hot defaults: {hot_sample}) — increase --tuning-budget or warm kernel cache",
        ))

    if tel.resident_short is not None and 0 < tel.resident_short < 5_000_000_000:
        fixes.append((
            max(gap_ms + 5.0, 0.20 * wall_ms),
            f"Resident Cliff: only {_gb(tel.resident_short)} short of 100% resident — "
            f"reclaiming {_gb(tel.resident_short)} eliminates all streaming & NVMe stalls",
        ))

    # HostRam / staging bloat check
    ram_bytes = host.get("memory_bytes") or 0
    committed = tel.sizing.get("committed", 0)
    if discrete and ram_bytes > 0 and committed > 0.25 * ram_bytes and tel.host_slots and tel.total_experts and tel.host_slots < tel.total_experts:
        fixes.append((
            max(host_land_ms * 0.8, 15.0),
            f"HostRam Committed Bloat: {_gb(committed)} committed ({100 * committed / ram_bytes:.0f}% of RAM), "
            f"leaving only {tel.host_slots}/{tel.total_experts} host slots — trim staging/decode_rows reserve",
        ))
    elif tel.decode_rows_planned is not None and actual_decode_rows is not None and tel.decode_rows_planned != actual_decode_rows:
        fixes.append((
            15.0,
            f"HostRam Staging Mismatch: decode_rows planned={tel.decode_rows_planned} vs actual={actual_decode_rows}",
        ))

    osc, osc_msg = tel.oscillation()
    if osc:
        fixes.append((12.0, f"Bank Resize Oscillation: {osc_msg} — stabilize guarantee hysteresis"))

    dma_ovf = int(tel.summary_engagement.get("dma_overflow", 0))
    if dma_ovf > 0:
        fixes.append((
            max(10.0, 0.1 * gap_ms),
            f"DMA Staging Overflow: {dma_ovf} overflows (cpu_site_exp={tel.summary_engagement.get('cpu_site_experts', 0)}) — "
            f"widen staging ring or cut per-layer misses",
        ))

    if tel.hinted_queued_ms > 5.0:
        fixes.append((
            tel.hinted_queued_ms,
            f"Hint Queue Backlog: {tel.hinted_count} hints queued mean {tel.hinted_queued_ms:.1f} ms — "
            f"throttle hint overread",
        ))

    if top_family is not None and top_family[1] / wall_ms > 0.30:
        fixes.append((
            top_family[1] * 0.5,
            f"Dominant GPU Family `{top_family[0]}`: {top_family[1]:.1f} ms ({100 * top_family[1] / wall_ms:.0f}% of wall)",
        ))

    fixes.sort(key=lambda item: -item[0])
    if not fixes:
        lines.append("  no major anomalies detected (within expected floor envelope)")
    else:
        for rank, (_, desc) in enumerate(fixes[:5], start=1):
            lines.append(f"  {rank}. {desc}")
    return lines


def render_job(label: str, job_dir: Path, log_path: Path | None) -> list[str]:
    """Render the 100%-Additive Floor Waterfall for one `step` or `draft` job directory."""
    rpath = job_dir / "result.json"
    if not rpath.is_file():
        return []
    result_data = json.loads(rpath.read_text())
    tel = _parse_log(log_path)
    tuning = _parse_tuning(job_dir, result_data)
    measurements = result_data.get("measurements") or {}
    if "drafting" in measurements:
        return _format_draft_waterfall(label, job_dir, result_data, tel, tuning)
    if "speed" in measurements:
        return _format_step_waterfall(label, job_dir, result_data, tel, tuning)
    return []


def render_run(run_dir: Path) -> str:
    """Render the 100%-Additive Floor Waterfall for every `step` or `draft` job in `run_dir`."""
    run_dir = Path(run_dir)
    if (run_dir / "result.json").is_file():
        # Direct job directory passed.
        parent_log = run_dir.parent / "logs" / f"job-{run_dir.name}.log"
        lines = render_job(run_dir.name, run_dir, parent_log if parent_log.is_file() else None)
        return "\n".join(lines)

    # Discover job directories in canonical order (from rig.json codes if present, else sorted subdirs).
    job_names: list[str] = []
    rig_json = run_dir / "rig.json"
    if rig_json.is_file():
        try:
            codes = json.loads(rig_json.read_text()).get("codes") or {}
            job_names = [name for name, code in codes.items() if code == 0]
        except json.JSONDecodeError:
            job_names = []
    if not job_names:
        job_names = sorted(
            p.name for p in run_dir.iterdir()
            if p.is_dir() and (p / "result.json").is_file()
        )

    blocks: list[str] = []
    for name in job_names:
        job_dir = run_dir / name
        if not (job_dir / "result.json").is_file():
            continue
        log_path = run_dir / "logs" / f"job-{name}.log"
        lines = render_job(name, job_dir, log_path if log_path.is_file() else None)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
