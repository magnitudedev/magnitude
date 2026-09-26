"""Sequential GPU campaigns; do not run alongside another GPU benchmark."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
jobs = [("chain", "evolution", 1), ("reduce", "evolution", 1)]
# Alternate order to expose algorithm/thermal order effects rather than give
# every later algorithm the same position. OS compiler caches are not reset.
jobs += [("stencil", algorithm, seed) for seed in [0, 1, 2]
         for algorithm in (["evolution", "random"] if seed != 1 else ["random", "evolution"])]
for fixture, algorithm, seed in jobs:
    output = root / "results" / f"{fixture}-{algorithm}-{seed}.jsonl"
    command = [sys.executable, str(root / "tune.py"), "--worker", str(root / "metal-worker"),
               "--fixture", fixture, "--algorithm", algorithm, "--seed", str(seed),
               "--budget", "60", "--output", str(output)]
    print("START", output.name, flush=True)
    subprocess.run(command, check=True)
for fixture in ["chain", "reduce", "stencil"]:
    output = root / "results" / f"{fixture}-exhaustive.jsonl"
    command = [sys.executable, str(root / "tune.py"), "--worker", str(root / "metal-worker"),
               "--fixture", fixture, "--algorithm", "exhaustive", "--seed", "19",
               "--budget", "240", "--trials", "3", "--target-ms", "2", "--finalists", "32",
               "--output", str(output)]
    print("START", output.name, flush=True)
    subprocess.run(command, check=True)
