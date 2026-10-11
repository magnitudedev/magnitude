"""The evaluation result record (contract "Result record") of a bench run.

``measurements.serving`` holds one row per measured request of every engine;
``measurements.drafting`` the catalog drafter's acceptance on our engine;
``measurements.tuning`` our engine's tuning, read from the tuning trace its launches append.
Host facts and the thermal trace are added by the remote flow's recorder around the job.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "evaluation-result/1"
TUNING_TRACE = "tuning.jsonl"
TUNING_TRACE_SCHEMA = "tuning-trace/3"


def tuning(trace: Path, budget_seconds: float | None, fresh: bool) -> dict:
    """``{budget_seconds, fresh, outcome}``: every search and every load the trace records, in
    order, each tagged with its ``work``. A ``search`` is a model's Prepare, in a worker process
    of its own, with what became of its budget: when its optional work ended, the time it ran
    past the budget and the offered pieces that did not fit. A ``load`` searched nothing (the
    harness's reading of the same trace)."""
    loads: list[dict] = []
    for number, line in enumerate(trace.read_text().splitlines(), 1):
        event = json.loads(line)
        kind = event.get("event")
        if kind == "began":
            if event.get("schema") != TUNING_TRACE_SCHEMA:
                raise ValueError(
                    f"tuning trace {trace} line {number}: schema {event.get('schema')!r}, "
                    f"not {TUNING_TRACE_SCHEMA!r}"
                )
            # `work`: a Prepare worker's `search`, or the `load`, which searched nothing.
            loads.append({"work": event["work"], "budget_seconds": event["budget_seconds"],
                          "fresh": event["fresh"]})  # fmt: skip
        elif kind in ("budget", "concluded"):
            if not loads:
                raise ValueError(f"tuning trace {trace} line {number}: {kind} precedes its load")
            load = loads[-1]
            if kind == "budget":
                load["concluded_at_seconds"] = event["used_seconds"]
                load["overrun_seconds"] = event["overrun_seconds"]
                load["refused"] = len(event["refused"])
            else:
                for key in ("units", "searched", "stored", "tuning_seconds", "seconds"):
                    load[key] = event[key]
    unfinished = [index for index, load in enumerate(loads) if "units" not in load]
    if unfinished:
        raise ValueError(f"tuning trace {trace}: load {unfinished[0]} never concluded")
    if budget_seconds is None and loads:
        budget_seconds = loads[0]["budget_seconds"]  # The engine's standard budget.
    return {"budget_seconds": budget_seconds, "fresh": fresh, "outcome": {"loads": loads}}


def serving_row(row: dict) -> dict:
    """One measured request: client latency, engine-counter rates and drafting counters.

    Rates come only from engine counters; failures keep their outcome and carry no rates.
    """
    observation = row["observation"]
    terminal = observation.get("terminal")
    timings = terminal["timings"] if terminal else {}
    usage = terminal["usage"] if terminal else {}

    def rate(tokens: str, duration: str) -> float | None:
        if not terminal or timings[duration] <= 0 or timings[tokens] <= 0:
            return None
        return 1000 * timings[tokens] / timings[duration]

    return {
        "engine": row["target"],
        "block": row["block"],
        "section": row["section"],
        "context": row["checkpoint"],
        "concurrency": row["concurrency"],
        "session": row["session"],
        "fixture_id": row["fixture_id"],
        "outcome": observation["outcome"],
        "valid": observation["outcome"] == "valid",
        "ttft_ms": observation.get("ttft_ms"),
        "completed_ms": observation.get("completed_ms"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prefill_tps": rate("prompt_n", "prompt_ms"),
        "decode_tps": rate("predicted_n", "predicted_ms"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "retrieval": observation.get("retrieval"),
        "timing_basis": row["timing_basis"],
    }


def drafting(rows: list[dict], engine: str) -> dict | None:
    """Acceptance over ``engine``'s valid measured requests that drafted.

    The serving counters report proposed and accepted draft tokens, not verification steps,
    so tokens per verification is not derivable here and is recorded as null.
    """
    drafted = [
        row for row in rows
        if row["engine"] == engine and row["valid"] and row["draft_n"] is not None
    ]
    if not drafted:
        return None
    proposed = sum(row["draft_n"] for row in drafted)
    accepted = sum(row["draft_n_accepted"] for row in drafted)
    return {
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else None,
        "tokens_per_verify": None,
        "requests": len(drafted),
    }


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
