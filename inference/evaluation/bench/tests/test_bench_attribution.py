"""Unit tests for per-run attribution extraction (`magnitude_bench.attribution`)."""

import json
from pathlib import Path

import pytest

from magnitude_bench import attribution


def _write_results(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_parse_new_delta_format_with_dma_residency_and_rates(tmp_path: Path):
    bench_dir = tmp_path / "bench"
    logs_dir = bench_dir / "logs"
    rig_logs = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    rig_logs.mkdir(parents=True)

    (logs_dir / "magnitude-block-0.log").write_text(
        "\n".join(
            [
                "2026-11-22T20:43:07.421Z  INFO residency: expert engagement (request 18446744073709551615): "
                "drive_busy_fraction=0.900 req_bytes_hidden=1000000 req_bytes_exposed=500000 "
                "req_staged_hits=10 req_landed_hits=5 req_carryover_hits=0 req_carryover_wasted=0 "
                "req_dma_calls=40 req_dma_skipped_zero_miss=10 req_dma_experts=25 req_dma_bytes=35000000 "
                "req_dma_overflow=0 req_dma_wait_ms=1.200 req_growth_refusals=0 req_cpu_site_jobs=3 "
                "vram_slots=61 host_slots=286 total_experts=384",
                "2026-11-22T20:43:08.025Z  INFO watcher: a promotion's copy costs its link time at 45.17 GB/s, "
                "and 25% of that in device time beside it",
                "2026-11-22T20:43:59.918Z  INFO residency: expert engagement (request 1): "
                "drive_busy_fraction=0.500 req_bytes_hidden=4000000000 req_bytes_exposed=2000000000 "
                "req_staged_hits=200 req_landed_hits=100 req_carryover_hits=15 req_carryover_wasted=5 "
                "req_dma_calls=400 req_dma_skipped_zero_miss=50 req_dma_experts=1200 req_dma_bytes=1680000000 "
                "req_dma_overflow=3 req_dma_wait_ms=12.500 req_growth_refusals=2 req_cpu_site_jobs=18 "
                "vram_slots=63 host_slots=286 total_experts=384",
                "2026-11-22T20:44:37.616Z  INFO residency: expert engagement (request 2): "
                "drive_busy_fraction=0.400 req_bytes_hidden=3000000000 req_bytes_exposed=6000000000 "
                "req_staged_hits=150 req_landed_hits=80 req_carryover_hits=10 req_carryover_wasted=4 "
                "req_dma_calls=500 req_dma_skipped_zero_miss=100 req_dma_experts=1500 req_dma_bytes=2100000000 "
                "req_dma_overflow=1 req_dma_wait_ms=14.250 req_growth_refusals=0 req_cpu_site_jobs=22 "
                "vram_slots=63 host_slots=286 total_experts=384",
            ]
        )
        + "\n"
    )

    assess_json = json.dumps(
        {
            "executorCosts": {
                "weightRates": [
                    {
                        "route": {"backend": "Cuda", "id": 0},
                        "rates": [
                            {"format": "bf16", "bytesPerSecond": {"value": 1650366762810.0}}
                        ],
                    }
                ]
            }
        }
    )
    (rig_logs / "job-measure.log").write_text(
        "\n".join(
            [
                '[stderr] magnitude-assess: 7.281 s: PointKey { subject: Volume { volume: VolumeId { device: 49, uuid: None }, os: "Linux 6.8" }, kernels: [], configuration: {"storage": 1}, point: {"full": 1} }: Measured(Timing { median: 0.003517742, interval: None, samples: 8 })',
                f"[stdout] {assess_json}",
            ]
        )
        + "\n"
    )

    _write_results(
        bench_dir / "results.jsonl",
        [
            {
                "block": 0,
                "phase": "warmup",
                "target": "magnitude",
                "session": "warmup",
                "observation": {
                    "request_id": "warmup",
                    "outcome": "valid",
                    "terminal": {
                        "timings": {
                            "prompt_n": 3826,
                            "prompt_ms": 20000.0,
                            "predicted_n": 100,
                            "predicted_ms": 10000.0,
                            "draft_n": 20,
                            "draft_n_accepted": 12,
                        },
                        "usage": {"prompt_tokens": 3826, "completion_tokens": 100},
                    },
                },
            },
            {
                "block": 0,
                "phase": "measured",
                "target": "magnitude",
                "session": "context-t4096-c1-0",
                "observation": {
                    "request_id": "context-t4096-c1-0",
                    "outcome": "valid",
                    "terminal": {
                        "timings": {
                            "prompt_n": 3864,
                            "prompt_ms": 15000.0,
                            "predicted_n": 150,
                            "predicted_ms": 20000.0,
                            "draft_n": 40,
                            "draft_n_accepted": 25,
                        },
                        "usage": {"prompt_tokens": 3864, "completion_tokens": 150},
                    },
                },
            },
        ],
    )

    (bench_dir / "tuning.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"schema": "tuning-trace/3", "event": "began", "work": "search", "budget_seconds": 60.0, "fresh": True}),
                json.dumps({"event": "budget", "used_seconds": 59.0, "overrun_seconds": 0.0, "refused": ["a", "b", "c"]}),
                json.dumps({"event": "concluded", "units": 20, "canonical_fallbacks": 5, "searched": 15, "stored": 15, "tuning_seconds": 59.0, "seconds": 60.0}),
            ]
        )
        + "\n"
    )

    data = attribution.write(bench_dir)
    assert (bench_dir / "attribution.json").is_file()
    assert data["schema"] == attribution.SCHEMA
    assert data["residency"] == {"vram_slots": 63, "host_slots": 286, "total_experts": 384}
    assert data["hardware_rates"]["line_stream_gb_s"] == pytest.approx(1650.36676281)
    assert data["hardware_rates"]["transfer_copy_gb_s"] == pytest.approx(45.17)
    assert data["hardware_rates"]["transfer_copy_stall_pct"] == pytest.approx(25.0)
    assert data["hardware_rates"]["storage_full_gb_s"] == pytest.approx(33_554_432 / 0.003517742 / 1e9)
    assert data["tuning"] == {
        "units": 20,
        "qualified": 15,
        "canonical_fallbacks": 5,
        "canonical_fallback_pct": pytest.approx(25.0),
        "expired": 0,
        "refused": 3,
    }

    assert len(data["internal_requests"]) == 1
    assert data["internal_requests"][0]["engine_request_id"] == attribution.INTERNAL_REQUEST_ID
    assert len(data["requests"]) == 2
    warmup, measured = data["requests"]
    assert warmup["phase"] == "warmup"
    assert warmup["dma_experts_landed"] == 1200
    assert warmup["growth_refusals"] == 2
    assert measured["phase"] == "measured"
    assert measured["output_tokens"] == 150
    assert measured["decode_ms"] == pytest.approx(20000.0)
    assert measured["draft_accepted"] == 25
    assert measured["draft_proposed"] == 40
    assert measured["bytes_exposed"] == 6_000_000_000
    assert measured["bytes_exposed_per_tok"] == pytest.approx(40_000_000.0)
    assert measured["implied_exposed_gb_s"] == pytest.approx(0.3)
    assert measured["dma_calls"] == 500
    assert measured["dma_skipped_zero_miss"] == 100
    assert measured["dma_experts_landed"] == 1500
    assert measured["dma_bytes_landed"] == 2_100_000_000
    assert measured["dma_overflow_experts"] == 1
    assert measured["dma_host_wait_ms"] == pytest.approx(14.25)
    assert measured["cpu_site_jobs"] == 22

    lines = attribution.format_summary(data)
    assert len(lines) == 4
    assert "residency 63/286/384 vram/host/total" in lines[0]
    assert "drafts=25/40" in lines[0]
    assert "cpu_jobs=22 refusals=0" in lines[1]
    assert "landed=1500 experts" in lines[2]
    assert "overflow=1, host_wait=14.2 ms" in lines[2]
    assert "line_stream=1650.4 GB/s" in lines[3]
    assert "transfer_copy=45.17 GB/s" in lines[3]
    assert "storage[full]=9.54 GB/s" in lines[3]
    assert "tuning 5/20 fallbacks (25.0%), 3 refused" in lines[3]


def test_parse_legacy_cumulative_logs_converts_to_deltas():
    log = "\n".join(
        [
            "INFO residency: expert engagement (request 18446744073709551615): drive_busy_fraction=0.930 bytes_hidden=1000 bytes_exposed=2000 staged_hits=10 landed_hits=5 carryover_hits=0 carryover_wasted=0",
            "INFO residency: expert engagement (request 1): drive_busy_fraction=0.586 bytes_hidden=5000 bytes_exposed=9000 staged_hits=50 landed_hits=25 carryover_hits=2 carryover_wasted=1",
            "INFO residency: expert engagement (request 2): drive_busy_fraction=0.497 bytes_hidden=8000 bytes_exposed=20000 staged_hits=60 landed_hits=45 carryover_hits=5 carryover_wasted=3",
        ]
    )
    entries = attribution.parse_engagement_lines(log)
    assert len(entries) == 3
    assert entries[0]["bytes_hidden"] == 1000
    assert entries[1]["bytes_hidden"] == 4000
    assert entries[1]["bytes_exposed"] == 7000
    assert entries[1]["staged_hits"] == 40
    assert entries[2]["bytes_hidden"] == 3000
    assert entries[2]["bytes_exposed"] == 11000
    assert entries[2]["staged_hits"] == 10
    assert entries[2]["landed_hits"] == 20
    assert entries[2]["carryover_hits"] == 3
    assert entries[2]["carryover_wasted"] == 2


@pytest.mark.parametrize(
    "run_name",
    [
        "20261122T203219Z-bench-2a8127",
        "20261122T194425Z-bench-567e45",
    ],
)
def test_real_rig_run_logs_extract_cleanly(run_name: str):
    # Resolve results in the main checkout if present locally; skip cleanly in isolated CI.
    results_dir = Path("/Users/anerli/magnitude/inference/evaluation/results") / run_name / "bench"
    if not results_dir.is_dir():
        pytest.skip(f"local run {run_name} not present")
    data = attribution.extract(results_dir)
    assert data["measured_totals"] is not None
    assert data["measured_totals"]["output_tokens"] > 0
    assert data["hardware_rates"]["transfer_copy_gb_s"] is not None
    assert data["hardware_rates"]["storage_full_gb_s"] is not None
    assert data["hardware_rates"]["line_stream_gb_s"] is not None
    lines = attribution.format_summary(data)
    assert len(lines) == 4
