#!/usr/bin/env python3
"""Run the unchanged V3 Session Bench against the native Inference V4 server.

Use the V3 checkout's existing Session Bench virtual environment. Each local GGUF is measured in
its own run, in argument order. The V3 target key remains ``magnitude`` because
its serialized target schema is fixed; runtime evidence identifies this adapter
and binary as Inference V4.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_evidence(v4_root: Path, launcher: Path) -> dict[str, str]:
    files: set[Path] = set()
    for directory, subdirs, names in os.walk(v4_root):
        subdirs[:] = sorted(name for name in subdirs if name not in {
            "target", "results", ".git", ".venv"
        })
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            if (name.endswith((".rs", ".seismic", ".metal"))
                    or name in ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml")
                    or (directory_path == v4_root / "validation" and name.endswith(".py"))
                    or path.is_relative_to(v4_root / "engine" / "templates" / "native")):
                files.add(path)
    files.add(launcher)
    return {
        str(path.relative_to(v4_root)): file_hash(path)
        for path in sorted(files)
    }


def runner_source_evidence(source: Path) -> dict[str, str]:
    """Identify the V3 runner, client, report, fixtures, and their lock inputs."""
    files = {source / "pyproject.toml", source / "uv.lock"}
    for package in ("session_bench", "benchmark_fixtures"):
        package_root = source / "src" / package
        files.update(path for path in package_root.rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json"})
    return {
        str(path.relative_to(source)): file_hash(path)
        for path in sorted(files)
    }


def source_hash(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="V3 Session Bench checkout")
    parser.add_argument("--binary", type=Path, required=True, help="built V4 magnitude-engine")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="local GGUF; repeat to run models sequentially")
    parser.add_argument("--suite", default="single")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--workload", choices=("retrieval", "prose-continue", "prose-repeat"), default="retrieval"
    )
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--device", default="auto", help="V4 device selector, such as cpu or metal")
    parser.add_argument("--cache-dir", type=Path, help="persistent V4 kernel and tuning cache")
    parser.add_argument("--method", choices=("auto", "plain", "mtp"), default="auto")
    parser.add_argument("--mtp-proposals", type=int,
                        help="MTP proposal width (default: the engine's)")
    args = parser.parse_args()
    if args.mtp_proposals is not None and args.method != "mtp":
        parser.error("--mtp-proposals requires --method mtp")
    if any(value <= 0 for value in (
        args.context, args.repeat, args.startup_timeout
    )):
        parser.error("context, repeat, and startup timeout must be positive")
    source = args.source.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    cache_dir = args.cache_dir.expanduser().absolute() if args.cache_dir else None
    # Keep the GGUF filename when a snapshot entry is a symlink into a
    # content-addressed cache. resolve() would replace its suffix with the
    # blob hash even though the same local file is opened and verified.
    artifacts = [path.expanduser().absolute() for path in args.artifact]
    if Path(sys.prefix) != source / ".venv":
        parser.error("run with SOURCE/.venv/bin/python to identify the runner environment")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must be an executable native magnitude-engine")
    if any(not path.is_file() or path.suffix.lower() != ".gguf" for path in artifacts):
        parser.error("each --artifact must name a local GGUF file")
    launcher = Path(__file__).resolve()
    v4_root = launcher.parent.parent
    initial_files = source_evidence(v4_root, launcher)
    initial_source_hash = source_hash(initial_files)
    initial_runner_files = runner_source_evidence(source)
    initial_runner_source_hash = source_hash(initial_runner_files)
    initial_binary_hash = file_hash(binary)

    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "src"))
    os.environ.update(
        PYTHONPATH=os.pathsep.join((str(source / "src"), str(source))),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    import httpx
    from benchmark_fixtures.ruler import RulerFixture
    from session_bench import runner
    from session_bench.engines import base
    from session_bench.engines.base import Adapter, installed_versions
    from session_bench.models import Target
    from session_bench.results import RunStore, atomic_json
    from session_bench.suites import SECTIONS

    sections = tuple(args.suite.split(","))
    if not sections or any(section not in SECTIONS for section in sections):
        parser.error(f"suite must be one or more of {SECTIONS}")
    base.STARTUP_TIMEOUT_SECONDS = args.startup_timeout
    initial_versions = installed_versions(source)

    class V4Store(RunStore):
        def __init__(self, root, command, selection):
            super().__init__(args.results.resolve(), command, selection)
            self.root = root
            (self.path / "command.txt").write_text(
                shlex.join([sys.executable, str(launcher), *sys.argv[1:]]) + "\n"
            )
            atomic_json(self.path / "v4-launcher.json", {
                "adapter": "magnitude-v4-native",
                "source": str(source),
                "python": sys.executable,
                "runner_dependencies": initial_versions,
                "runner_environment": "existing V3 Session Bench venv; no package synchronization",
                "v4_source": str(v4_root),
                "engine_source_sha256": initial_source_hash,
                "engine_source_files": initial_files,
                "runner_source_sha256": initial_runner_source_hash,
                "runner_source_files": initial_runner_files,
                "binary": str(binary),
                "binary_sha256": initial_binary_hash,
                "binary_bytes": binary.stat().st_size,
                "launcher_sha256": file_hash(launcher),
                "suite": sections,
                "requested_context_tokens": args.context,
                "workload": args.workload,
                "repeat": args.repeat,
                "startup_timeout_seconds": args.startup_timeout,
                "device": args.device,
                "cache_dir": str(cache_dir) if cache_dir else None,
                "method": args.method,
                "mtp_proposals": args.mtp_proposals,
                "count_endpoint": "/v1/count",
                "target_schema_key": "magnitude",
                "measurement": "unchanged V3 Session Bench fixtures, client, runner, and report",
            })

    class V4Magnitude(Adapter):
        def __init__(self, *parameters):
            super().__init__(*parameters)
            self.identity = {
                "adapter": "magnitude-v4-native",
                "binary": str(binary),
                "binary_sha256": initial_binary_hash,
                "engine_source_sha256": initial_source_hash,
                "runner_source_sha256": initial_runner_source_hash,
                "model_path": str(self.artifact.path),
                "model_sha256": next(iter(self.artifact.files)).sha256,
                "count_endpoint": "/v1/count",
            }

        async def prepare(self):
            if self.artifact.kind != "gguf" or self.artifact.path != Path(self.target.reference):
                raise ValueError("V4 Session Bench requires a local GGUF target")
            self.verify()

        def verify(self):
            self.artifact.verify_unchanged()
            if file_hash(binary) != initial_binary_hash:
                raise ValueError("V4 binary changed during benchmark")
            if source_hash(source_evidence(v4_root, launcher)) != initial_source_hash:
                raise ValueError("V4 engine source changed during benchmark")
            if source_hash(runner_source_evidence(source)) != initial_runner_source_hash:
                raise ValueError("Session Bench runner or fixture source changed during benchmark")
            if installed_versions(source) != initial_versions:
                raise ValueError("Session Bench runner dependencies changed during benchmark")

        def argv(self, port, context, parallel, directory):
            return [
                str(binary), "--model", str(self.artifact.path), "--no-projector",
                "--host", "127.0.0.1", "--port", str(port),
                "--served-model", self.served_model(),
                "--context-tokens", str(context),
                "--device", args.device,
                *(["--cache-dir", str(cache_dir)] if cache_dir else []),
                "--max-batch", str(parallel),
                "--output-capacity", "256",
                "--method", args.method,
                *(["--mtp-proposals", str(args.mtp_proposals)]
                  if args.mtp_proposals is not None else []),
            ]

        def verify_ready(self, data, context, parallel):
            if data.get("ready") is not True or data.get("model") != self.served_model():
                raise ValueError(f"V4 readiness identity mismatch: {data}")
            limit = data.get("context_tokens")
            if type(limit) is not int or limit != context:
                raise ValueError(f"V4 readiness context mismatch: {data}")
            if type(data.get("vocabulary")) is not int or data["vocabulary"] <= 0:
                raise ValueError(f"V4 readiness vocabulary missing: {data}")

        async def count_requests(self, endpoint, requests):
            counts = {}
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                for request in requests:
                    response = await client.post(
                        endpoint + "/v1/count", json=request.body(self.served_model())
                    )
                    response.raise_for_status()
                    count = response.json().get("prompt_tokens")
                    if type(count) is not int or count <= 0:
                        raise ValueError(f"V4 /v1/count has no authoritative prompt_tokens: {response.text}")
                    counts[request.id] = count
            return counts

        @contextlib.asynccontextmanager
        async def context_counter(self):
            async with self.launch(args.context, 1, "fixture-count") as engine:
                async def count(context):
                    plan = self.context_plan(context)
                    return (await self.count_requests(
                        engine.endpoint, plan.prepared_requests
                    ))["fixture-sizing"]
                yield count

        async def prompt_counts(self, plan):
            async with self.launch(args.context, 1, "prompt-count") as engine:
                return await self.count_requests(engine.endpoint, plan.prepared_requests)

    runner.RunStore = V4Store
    runner.ADAPTERS = {**runner.ADAPTERS, "magnitude": V4Magnitude}

    async def all_runs():
        results = []
        for artifact in artifacts:
            result = await runner.run(
                source, [Target(engine="magnitude", reference=str(artifact))], sections,
                (args.context,), (), args.repeat, None,
                lambda message: print(message, file=sys.stderr, flush=True),
                prose=None if args.workload == "retrieval" else args.workload,
                retrieval=None if args.workload != "retrieval" else RulerFixture(
                    seed=42, variant="single", haystack="records", queries=1
                ),
            )
            results.append({"artifact": str(artifact), "result": result})
            if result.get("status") != "completed":
                break
        return results

    results = asyncio.run(all_runs())
    print(json.dumps(results, indent=2))
    if any(entry["result"].get("status") != "completed" for entry in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
