import json

import pytest
from magnitude_bench import record


def test_tuning_measurement_reads_each_load_from_the_trace(tmp_path):
    trace = tmp_path / "tuning.jsonl"
    events = [
        {"event": "began", "schema": "tuning-trace/3", "work": "search", "budget_seconds": 30.0,
         "fresh": True},
        {"event": "progress", "completed_ms": 10, "total_ms": 30000},
        {"event": "stage", "class": "measure"},
        {"event": "published", "actual_seconds": 0.003},
        {"event": "budget", "used_seconds": 29.5, "overrun_seconds": 0.0,
         "refused": [{"unit": "a", "reason": "seconds"}]},
        {"event": "concluded", "units": 12, "searched": 9, "stored": 9, "tuning_seconds": 29.0,
         "seconds": 31.0},
        {"event": "began", "schema": "tuning-trace/3", "work": "load", "budget_seconds": 30.0,
         "fresh": False},
        {"event": "concluded", "units": 12, "searched": 0, "stored": 0, "tuning_seconds": 0.0,
         "seconds": 0.4},
    ]
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    value = record.tuning(trace, None, True)
    assert value["budget_seconds"] == 30.0 and value["fresh"] is True
    first, second = value["outcome"]["loads"]
    assert first["concluded_at_seconds"] == 29.5 and first["overrun_seconds"] == 0.0
    assert first["refused"] == 1 and first["searched"] == 9
    assert (first["work"], second["work"]) == ("search", "load")
    # A load that searched nothing reports no budget.
    assert second["fresh"] is False and second["searched"] == 0 and "refused" not in second
    trace.write_text(json.dumps(events[0]) + "\n")
    with pytest.raises(ValueError, match="never concluded"):
        record.tuning(trace, 30.0, True)
    trace.write_text(json.dumps({**events[0], "schema": "tuning-trace/2"}) + "\n")
    with pytest.raises(ValueError, match="schema 'tuning-trace/2'"):
        record.tuning(trace, 30.0, True)


def row(outcome="valid", draft=(10, 7), engine="magnitude"):
    timings = {"prompt_n": 1000, "prompt_ms": 500, "predicted_n": 100, "predicted_ms": 2000}
    if draft is not None:
        timings |= {"draft_n": draft[0], "draft_n_accepted": draft[1]}
    observation = {"outcome": outcome, "ttft_ms": 520.0, "completed_ms": 2600.0}
    if outcome == "valid":
        observation["terminal"] = {
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            "timings": timings,
        }
    return {
        "target": engine, "block": 0, "phase": "measured", "section": "context",
        "checkpoint": 16384, "concurrency": 1, "session": "context-t16384-c1",
        "fixture_id": "tools/x", "timing_basis": "native-model-service",
        "observation": observation,
    }


def test_serving_rows_take_rates_only_from_engine_counters():
    served = record.serving_row(row())
    assert served["prefill_tps"] == 2000 and served["decode_tps"] == 50
    assert served["valid"] and served["context"] == 16384
    assert served["draft_n"] == 10 and served["draft_n_accepted"] == 7
    failed = record.serving_row(row("timeout"))
    assert not failed["valid"] and failed["prefill_tps"] is None and failed["ttft_ms"] == 520.0


def test_drafting_reports_acceptance_over_valid_drafting_requests():
    rows = [record.serving_row(r) for r in (
        row(), row(draft=(20, 5)), row("timeout"), row(draft=None), row(engine="llama.cpp"),
    )]
    value = record.drafting(rows, "magnitude")
    assert value == {"proposed": 30, "accepted": 12, "acceptance": 0.4,
                     "tokens_per_verify": None, "requests": 2}
    assert record.drafting([record.serving_row(row(draft=None))], "magnitude") is None
