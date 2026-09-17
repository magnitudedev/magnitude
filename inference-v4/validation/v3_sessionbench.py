#!/usr/bin/env python3
"""Run the unchanged V3 session benchmark in a frozen existing environment.

Only preparation/launch, explicit startup timeout and result placement are adapted: no uv sync, no model
or engine policy changes, and no writes to the source checkout. Execute with the
source checkout's .venv/bin/python. Source, dependencies and artifacts retain V3's
normal verification. This is a V3 baseline, not a V4 adapter or parity claim.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--suite", default="single")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    artifact = args.artifact.resolve(strict=True)
    if args.context <= 0 or args.repeat <= 0 or args.startup_timeout <= 0:
        parser.error("context, repeat and startup timeout must be positive")
    # Do not resolve the venv executable symlink: its parent identifies the venv.
    if Path(sys.prefix) != source / ".venv":
        parser.error("run with SOURCE/.venv/bin/python to preserve the frozen environment")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "src"))
    os.environ.update(PYTHONPATH=os.pathsep.join([str(source / "src"),str(source)]), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from benchmark_fixtures.ruler import RulerFixture
    from session_bench import runner
    from session_bench.engines import base
    from session_bench.engines.base import installed_versions, runtime_digest
    base.STARTUP_TIMEOUT_SECONDS = args.startup_timeout
    from session_bench.engines.magnitude import Magnitude
    from session_bench.models import Target
    from session_bench.results import RunStore, atomic_json
    from session_bench.suites import SECTIONS

    sections = tuple(args.suite.split(","))
    if not sections or any(section not in SECTIONS for section in sections):
        parser.error(f"suite must be one or more of {SECTIONS}")

    class ExternalStore(RunStore):
        def __init__(self, root, command, selection):
            super().__init__(args.results.resolve(), command, selection)
            self.root = root
            exact = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
            (self.path / "command.txt").write_text(exact + "\n")
            atomic_json(self.path / "frozen-launcher.json", {
                "source": str(source), "python": sys.executable,
                "startup_timeout_seconds": args.startup_timeout,
                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "preparation": "verify existing frozen environment; no package synchronization",
                "baseline": "V3; original fixture generation, tokenization, requests and measurement",
            })

    class FrozenMagnitude(Magnitude):
        async def prepare(self):
            self.identity = self.store.snapshot(self.target.id, self.runtime)
            self.source_identity = runtime_digest(self.runtime)
            self.identity["dependencies"] = installed_versions(self.runtime)
            self.verify()

        def argv(self, *parameters):
            original = super().argv(*parameters)
            if original[:5] != ["uv", "run", "--frozen", "--no-sync", "python"]:
                raise RuntimeError("V3 launcher changed; review frozen adapter")
            return [sys.executable, *original[5:]]

    runner.RunStore = ExternalStore
    runner.ADAPTERS = {**runner.ADAPTERS, "magnitude": FrozenMagnitude}
    result = asyncio.run(runner.run(
        source, [Target(engine="magnitude", reference=str(artifact))], sections,
        (args.context,), (), args.repeat, None,
        lambda message: print(message, file=sys.stderr, flush=True),
        retrieval=RulerFixture(seed=42, variant="single", haystack="records", queries=1),
    ))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
