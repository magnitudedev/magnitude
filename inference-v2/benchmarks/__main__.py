"""Run one declared experiment in a fresh, time-bounded child process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.loading import digest, load
from benchmarks.runner import Run, environment, measure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", help="Python module:experiment_name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-digest", help=argparse.SUPPRESS)
    args = parser.parse_args()
    experiment = load(args.case)
    if args.describe:
        print(json.dumps(experiment.record(), indent=2))
        return 0
    if args.output is None:
        parser.error("--output is required for a measured run")
    if args.output.exists():
        parser.error("output already exists; choose a new evidence path")
    root = Path(__file__).resolve().parent.parent
    if args.worker:
        if args.expected_digest != digest(experiment):
            raise ValueError("experiment changed between supervisor and worker")
        result = measure(
            experiment,
            environment(root, isolated=True),
        )
        result.write(args.output)
        print(json.dumps(result.record()["statistics_ms"]))
        return 1 if result.rejections else 0
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks",
                args.case,
                "--output",
                str(args.output.resolve()),
                "--worker",
                "--expected-digest",
                digest(experiment),
            ],
            cwd=root,
            timeout=experiment.timeout_seconds,
        )
        reason = f"benchmark process exited with {process.returncode}"
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        reason, returncode = "benchmark exceeded declared timeout", 1
    if not args.output.exists():
        Run(experiment, environment(root), rejections=[reason]).write(args.output)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
