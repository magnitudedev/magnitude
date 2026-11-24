"""Tests for the 100%-Additive Floor Waterfall (`waterfall.py`, `mrig inspect`, `report_answer`)."""

import json
from pathlib import Path
from types import SimpleNamespace

from magnitude_rig import cli, engine, results, waterfall
from magnitude_rig.instruments import Answer


def _write_event(stream, message: str, **extra) -> None:
    payload = {"at": 1.0, "message": message, **extra}
    stream.write(f"[stderr] magnitude-harness: event {json.dumps(payload)}\n")


def test_draft_cuda_waterfall_is_100_percent_additive_and_separates_cuda_dma(tmp_path: Path):
    run = tmp_path / "20261124T000000Z-draft-test01"
    job = run / "draft"
    logs = run / "logs"
    job.mkdir(parents=True)
    logs.mkdir(parents=True)

    (run / "rig.json").write_text(json.dumps({"codes": {"draft": 0}}))
    (job / "result.json").write_text(json.dumps({
        "model": "deepseek-v4.1-flash:gguf:iq2-xxs",
        "command": ["magnitude-harness", "draft", "--device", "cuda"],
        "host": {
            "alias": "rtx5090-128g",
            "memory_bytes": 134_000_000_000,
            "gpus": [{"name": "NVIDIA GeForce RTX 5090", "memory_bytes": 34_000_000_000}],
        },
        "measurements": {
            "drafting": {"proposed": 24, "accepted": 22, "tokens_per_verify": 3.75, "anchors": 2},
            "tuning": {"outcome": {"loads": [{"work": "load", "units": 153, "searched": 0, "stored": 153, "refused": 0}]}},
        },
    }))
    row = {
        "proposals": [1, 2, 3],
        "attribution": {
            "plain": {"wall_ms": 80.0, "device_ms": 30.0, "submissions": 82},
            "draft": {"wall_ms": 20.0, "device_ms": 15.0, "submissions": 1},
            "verify": {"wall_ms": 210.0, "device_ms": 130.0, "submissions": 82},
        },
    }
    (job / "draft.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with open(logs / "job-draft.log", "w") as log:
        _write_event(log, "the load's experts are placed", resident_short="Some(120000000000)")
        _write_event(log, "the bank's path records are settled", slots=45, ahead=90, reserve=90, decode_rows=16)
        _write_event(log, "a promotion's copy costs its link time at 40.00 GB/s, and 20% of that in device time beside it")
        _write_event(
            log, "memory grown", holding="Host", bank_slots=45, **{"from": 276, "to": 281},
            grown=1_000_000_000, term="load",
            figures="SizingFigures { budget: 118000000000, supply: 2000000000, others: 10000000000, committed: 28000000000, other_cache: 0, guarantee: 0, cap: None }",
        )
        _write_event(
            log,
            "expert engagement (request 1): req_bytes_hidden=700000000 req_bytes_exposed=200000000 "
            "req_dma_bytes=3200000000 req_dma_wait_ms=100.0 vram_slots=45 host_slots=281 total_experts=384 "
            "vram_held=1560/1800 vram_reserve=200",
        )
        _write_event(
            log,
            "expert engagement: bytes_hidden=700000000 bytes_exposed=200000000 staged_hits=50 landed_hits=30 "
            "dma_overflow=5 cpu_site_experts=7 vram_slots=45 host_slots=281 total_experts=384",
        )
        _write_event(
            log,
            "expert landings, each term a landing's mean: demand 100: read 4.000 ms (mean of 80), "
            "queued 0.250 ms, converting 1.000 ms, worker time 10.0 ms; "
            "hinted 5000: read 0.000 ms (mean of 0), queued 3.500 ms, converting 1.000 ms, worker time 10.0 ms; "
            "on the device 0; waits for a record 0 (0.0 ms in all)",
        )

    out = waterfall.render_run(run)
    assert "remainder +0.00 ms" in out
    assert "[GPU] verify device_ms" in out
    assert "verify PCIe H2D wire (DMA)" in out
    assert "host_land_misses (NVMe+claim)" in out
    assert "[! RESERVE 11.1% >10%]" in out
    assert "decode_rows plan=16 vs act=4 [! MISMATCH]" in out
    assert "dma_overflow 5 [! OVERFLOW]" in out


def test_step_waterfall_flags_unsearched_families_resident_cliff_and_oscillation(tmp_path: Path):
    run = tmp_path / "20261124T000000Z-step-test02"
    job = run / "step"
    logs = run / "logs"
    job.mkdir(parents=True)
    logs.mkdir(parents=True)

    (run / "rig.json").write_text(json.dumps({"codes": {"step": 0}}))
    (job / "result.json").write_text(json.dumps({
        "model": "qwen3.8-flash-next:gguf:q2",
        "command": ["magnitude-harness", "time", "--device", "metal"],
        "host": {"alias": "m5-pro-01", "memory_bytes": 51_000_000_000, "gpus": [{"name": "Apple M5 Pro", "memory_bytes": None}]},
        "measurements": {
            "speed": [{
                "phase": "decode",
                "concurrency": 1,
                "detail": {
                    "median_ms": 40.0,
                    "read_bytes_per_second": 3_500_000_000.0,
                    "window": {
                        "wall_ms_per_step": 40.0,
                        "device_busy_ms_per_step": 35.0,
                        "device_idle_ms_per_step": 5.0,
                        "idle": {
                            "between_runs": {"queued_ms_per_step": 2.0, "by_submission": [{"submissions_per_step": 4.0}]},
                            "at_boundaries": {"per_step": 2.0, "awaiting_data_ms_per_step": 2.5},
                        },
                    },
                    "attribution": {
                        "entries": [
                            {"entry": "routed_gate_up_part", "device_ms_per_step": 15.0, "launches_per_step": 48.0, "share": 0.4},
                            {"entry": "attention_decode_k8v4", "device_ms_per_step": 10.0, "launches_per_step": 24.0, "share": 0.3},
                            {"entry": "routed_land", "device_ms_per_step": 10.0, "launches_per_step": 48.0, "share": 0.3},
                        ],
                    },
                    "expert_bank": {"slots": 481, "experts": 512},
                },
            }],
        },
    }))
    (job / "tuning.jsonl").write_text(
        json.dumps({"event": "budget", "budget_seconds": 60.0, "used_seconds": 60.5,
                    "refused": [{"unit": "attention_decode_k8v4 [M=1..2]", "class": "d", "reason": "budget"}]}) + "\n"
        + json.dumps({"event": "concluded", "units": 100, "searched": 99, "stored": 1, "canonical_fallbacks": 10}) + "\n"
    )
    with open(logs / "job-step.log", "w") as log:
        _write_event(log, "the load's experts are placed", resident_short="Some(2900000000)")
        _write_event(
            log, "memory grown", holding="Experts", **{"from": 81, "to": 82}, grown=68_000_000, term="guarantee",
            figures="SizingFigures { budget: 40000000000, supply: 1000000000, others: 5000000000, committed: 8000000000, other_cache: 0, guarantee: 0, cap: None }",
        )
        _write_event(
            log, "memory released", holding="Experts", **{"from": 82, "to": 81}, released=68_000_000, term="guarantee",
            figures="SizingFigures { budget: 40000000000, supply: 1000000000, others: 5000000000, committed: 8000000000, other_cache: 0, guarantee: 190000000, cap: None }",
        )

    out = waterfall.render_run(run)
    assert "remainder +0.00 ms" in out
    assert "attention" in out and "[DEFAULT/UNSEARCHED]" in out
    assert "[! <5 GB TO 100% RESIDENT CLIFF]" in out
    assert "oscillation 81↔82" in out
    assert "Bank Resize Oscillation" in out


def test_inspect_cli_and_report_answer_invoke_waterfall(tmp_path: Path, capsys):
    run = tmp_path / "20261124T000000Z-step-cli01"
    job = run / "step"
    job.mkdir(parents=True)
    (run / "rig.json").write_text(json.dumps({"codes": {"step": 0}}))
    (job / "result.json").write_text(json.dumps({
        "model": "qwen:q4",
        "host": {"alias": "local", "gpus": []},
        "measurements": {"speed": [{"phase": "decode", "concurrency": 1, "detail": {"median_ms": 10.0}}]},
    }))

    args = cli.parser().parse_args(["inspect", str(run), "--waterfall"])
    assert args.handler(args) == 0
    captured = capsys.readouterr().out
    assert "=== FLOOR WATERFALL [step]" in captured

    outcome = engine.Outcome(run.name, "local", run, {"step": 0}, [{"seconds": 1.0}], 60.0)
    step_args = SimpleNamespace(action="step", purpose="test", estimate=60.0)
    assert results.report_answer(step_args, Answer(("ok",), True), outcome) == 0
    captured = capsys.readouterr().out
    assert "=== FLOOR WATERFALL [step]" in captured
