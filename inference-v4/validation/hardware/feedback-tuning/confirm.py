"""Compare campaign selections in randomized blocks under one observer process."""
import argparse
import json
import random
import statistics
from pathlib import Path
from tune import Worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("results"))
    parser.add_argument("--worker", type=Path, default=Path("./metal-worker"))
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    campaigns = []
    for path in sorted(args.directory.glob("*.jsonl")):
        if path.name.startswith("smoke-"):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        results = [r for r in rows if r.get("kind") == "result"]
        if results:
            campaigns.append(dict(results[-1], file=path.name))
    rng = random.Random(271828)
    reports = []
    raw = args.directory / "confirmation-observations.jsonl"
    with raw.open("w") as output:
        for fixture in sorted({r["fixture"] for r in campaigns}):
            runs = [r for r in campaigns if r["fixture"] == fixture]
            points = sorted({tuple(r["selected"]) for r in runs})
            reference = next(r for r in runs if r["algorithm"] == "exhaustive")
            samples = {p: [] for p in points}
            worker = Worker(args.worker.resolve(), fixture)
            try:
                # Compile and warm every point before comparison order begins.
                for point in points:
                    result = worker.observe(point, trials=2, target_ms=5)
                    output.write(json.dumps(dict(kind="warmup", fixture=fixture, **result)) + "\n")
                for block in range(args.rounds):
                    order = list(points)
                    rng.shuffle(order)
                    for point in order:
                        result = worker.observe(point, trials=8, target_ms=6)
                        output.write(json.dumps(dict(kind="comparison", block=block, fixture=fixture, **result)) + "\n")
                        output.flush()
                        samples[point].append(statistics.mean(result["samples_ms"]))
                means = {p: statistics.mean(v) for p, v in samples.items()}
                ref_point = tuple(reference["selected"])
                best_point = min(points, key=means.get)
                for run in runs:
                    point = tuple(run["selected"])
                    ratios = [a / b - 1 for a, b in zip(samples[point], samples[ref_point])]
                    reports.append(dict(**run, confirmed_ms=means[point],
                                        reference_ms=means[ref_point],
                                        regret_percent=100 * (means[point] / means[ref_point] - 1),
                                        paired_regret_samples_percent=[100 * v for v in ratios],
                                        reference_screen_complete=reference["unique_points"] == reference["total_points"],
                                        best_confirmed_point=list(best_point),
                                        best_confirmed_ms=means[best_point]))
                print(json.dumps(dict(fixture=fixture, points=len(points), best_ms=means[best_point],
                                      reference_ms=means[ref_point])), flush=True)
            finally:
                worker.close()
    (args.directory / "confirmation-summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
