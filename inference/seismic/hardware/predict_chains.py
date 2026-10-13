#!/usr/bin/env python3
"""Test an additive source-chain hypothesis against independent held-out probes.

The fitted coefficients are diagnostic predictions, never primitive contracts.
No held-out observation participates in fitting or compiler selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def groups(report):
    result = {}
    for row in report["observations"]:
        if row["name"] in ("control", "device_copy"):
            continue
        if not row["finite"] or not row["gpuSeconds"] or any(
            not math.isfinite(t) or t <= 0 for t in row["gpuSeconds"]
        ):
            raise ValueError(f"invalid observation: {row['pipelineID']}")
        result.setdefault((row["name"], row["threads"]), []).append(row)
    return result


def fit(rows):
    counts = sorted({r["operationsPerIteration"] for r in rows})
    if len(counts) != 2:
        raise ValueError("calibration needs two distinct source body counts")
    lines = []
    for count in counts:
        pair = sorted((r for r in rows if r["operationsPerIteration"] == count), key=lambda r: r["iterations"])
        if len(pair) != 2 or pair[0]["iterations"] >= pair[1]["iterations"]:
            raise ValueError("each calibration body count needs two distinct iteration counts")
        lo, hi = pair
        slope = (median(hi["gpuSeconds"]) - median(lo["gpuSeconds"])) / (hi["iterations"] - lo["iterations"])
        lines.append((slope, median(lo["gpuSeconds"]) - lo["iterations"] * slope))
    operation = (lines[1][0] - lines[0][0]) / (counts[1] - counts[0])
    loop = lines[0][0] - counts[0] * operation
    return {"dispatch_seconds": sum(line[1] for line in lines) / 2,
            "loop_seconds": loop, "source_operation_seconds": operation,
            "calibration_intercept_difference_seconds": abs(lines[1][1] - lines[0][1])}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("held_out", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_bytes())
    held_out = json.loads(args.held_out.read_bytes())
    for field in ("device", "registryID", "operatingSystem"):
        if calibration[field] != held_out[field]:
            raise ValueError(f"device conditions differ: {field}")
    train, test = groups(calibration), groups(held_out)
    if train.keys() != test.keys():
        raise ValueError("held-out operation/thread cases differ from calibration")
    results = []
    for key, rows in sorted(train.items()):
        model = fit(rows)
        seen = {(r["operationsPerIteration"], r["iterations"]) for r in rows}
        predictions = []
        for row in test[key]:
            if (row["operationsPerIteration"], row["iterations"]) in seen:
                raise ValueError("held-out data overlaps calibration")
            predicted = model["dispatch_seconds"] + row["iterations"] * (
                model["loop_seconds"] + row["operationsPerIteration"] * model["source_operation_seconds"])
            observed = median(row["gpuSeconds"])
            predictions.append({"pipeline_id": row["pipelineID"], "iterations": row["iterations"],
                "source_operations_per_iteration": row["operationsPerIteration"],
                "predicted_seconds": predicted, "observed_seconds": observed,
                "observed_sample_range": [min(row["gpuSeconds"]), max(row["gpuSeconds"])],
                "relative_error": abs(predicted - observed) / observed,
                "within_observed_sample_range": min(row["gpuSeconds"]) <= predicted <= max(row["gpuSeconds"])})
        results.append({"name": key[0], "threads": key[1], "fitted_hypothesis": model,
            "predictions": predictions, "qualified_primitive_mapping": False})
    controls = []
    for name in ("control", "device_copy"):
        for threads in sorted({r["threads"] for r in calibration["observations"] if r["name"] == name}):
            samples = [[t for r in report["observations"] if r["name"] == name and r["threads"] == threads
                        for t in r["gpuSeconds"]] for report in (calibration, held_out)]
            if not all(samples):
                raise ValueError("missing control observation")
            before, after = map(median, samples)
            controls.append({"name": name, "threads": threads, "calibration_median_seconds": before,
                "held_out_median_seconds": after, "held_out_to_calibration_ratio": after / before})
    report = {"status": "held-out test of an additive source-chain hypothesis; not a hardware contract",
        "equation": "dispatch + iterations * (loop + source_operations_per_iteration * source_operation)",
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "held_out_sha256": hashlib.sha256(args.held_out.read_bytes()).hexdigest(),
        "calibration_path": str(args.calibration), "held_out_path": str(args.held_out),
        "predictor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": calibration["device"], "results": results, "control_comparison": controls,
        "limitations": ["Coefficients include compiler transformations, loop behavior and device concurrency.",
            "Successful prediction of these chains would not establish primitive mappings, occupancy, or mixed-workload performance.",
            "Three timing samples are an observed range, not a statistical confidence interval.",
            "Separate runs may differ in device operating state; control/copy observations expose drift but do not identify its cause.",
            "No prediction or measurement scores Seismic candidates."]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    predictions = [p for r in results for p in r["predictions"]]
    print(json.dumps({"predictions": len(predictions), "median_relative_error": median(p["relative_error"] for p in predictions),
        "max_relative_error": max(p["relative_error"] for p in predictions),
        "within_observed_sample_range": sum(p["within_observed_sample_range"] for p in predictions)}))


if __name__ == "__main__":
    main()
