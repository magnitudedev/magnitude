#!/usr/bin/env python3
"""Summarize independent measurements without inventing native primitive costs.

Loop body counts are source counts. A positive slope is necessary, not sufficient,
for a primitive mapping: native loop/control/conversion effects still contribute.
"""
import argparse
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("observations", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
raw = json.loads(a.observations.read_text())
groups = {}
for row in raw["observations"]:
    if not row["finite"] or any(t <= 0 for t in row["gpuSeconds"]):
        raise SystemExit(f"invalid observation: {row['name']}")
    groups.setdefault((row["name"], row["threads"], row["operationsPerIteration"]), []).append(row)
results = []
for (name, threads, operations), rows in groups.items():
    rows.sort(key=lambda r: r["iterations"])
    if name == "device_copy":
        seconds = statistics.median(rows[0]["gpuSeconds"])
        results.append({"name": name, "bytes_read_and_written": threads * 8,
                        "median_gpu_seconds": seconds,
                        "effective_bytes_per_second": threads * 8 / seconds})
        continue
    if name == "control":
        continue
    lo, hi = rows
    delta = hi["iterations"] - lo["iterations"]
    slope = (statistics.median(hi["gpuSeconds"]) - statistics.median(lo["gpuSeconds"])) / delta
    # This interval separates timing noise from a genuine iteration-dependent cost.
    lower = (min(hi["gpuSeconds"]) - max(lo["gpuSeconds"])) / delta
    upper = (max(hi["gpuSeconds"]) - min(lo["gpuSeconds"])) / delta
    results.append({"name": name, "threads": threads, "operations_per_iteration": operations,
                    "seconds_per_chain_iteration": slope,
                    "observed_interval": [lower, upper],
                    "positive_iteration_cost": lower > 0,
                    "qualified_primitive_mapping": False,
                    "reason": "compound chain includes loop overhead and possible native transformations"})
marginals = []
for name, threads in sorted({(r["name"], r.get("threads")) for r in results if r["name"] != "device_copy"}):
    pair = sorted((r for r in results if r["name"] == name and r.get("threads") == threads), key=lambda r: r["operations_per_iteration"])
    if len(pair) != 2:
        continue
    low, high = pair
    count = high["operations_per_iteration"] - low["operations_per_iteration"]
    slope = (high["seconds_per_chain_iteration"] - low["seconds_per_chain_iteration"]) / count
    interval = [(high["observed_interval"][0] - low["observed_interval"][1]) / count,
                (high["observed_interval"][1] - low["observed_interval"][0]) / count]
    marginals.append({"name": name, "threads": threads,
                      "marginal_seconds": slope, "observed_interval": interval,
                      "positive_marginal_cost": interval[0] > 0,
                      "qualified_primitive_mapping": False})
a.output.write_text(json.dumps({"device": raw["device"], "measurements": results, "marginals": marginals,
    "status": "independent native evidence; primitive mappings not yet qualified; not a model benchmark"}, indent=2) + "\n")
