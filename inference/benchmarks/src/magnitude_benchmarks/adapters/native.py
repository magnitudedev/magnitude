"""The native Magnitude engine: a built ``magnitude-engine`` binary serving one local GGUF."""

import contextlib
import hashlib
import os
from pathlib import Path

import httpx

from ..session_bench.models import file_hash
from .base import Adapter

# Counting is host-only and may inspect inputs up to the artifact capability, so preparation
# launches never allocate numerical state for the measured context.
COUNT_CONTEXT_TOKENS = 4096
OUTPUT_CAPACITY = 256
SOURCE_SUFFIXES = (
    ".rs",
    ".seismic",
    ".metal",
    ".cu",
    ".cuh",
    ".comp",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".m",
    ".mm",
)
SOURCE_NAMES = ("Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml")
# The management service shares the Cargo workspace but is not linked into the engine binary.
EXCLUDED_DIRECTORIES = {
    "target",
    ".agent-targets",
    "benchmarks",
    "validation",
    "service",
    "results",
    ".git",
    ".venv",
}


def engine_sources(workspace: Path) -> dict[str, str]:
    """Per-file hashes of the Rust workspace inputs that build the engine binary."""
    files = []
    for directory, subdirectories, names in os.walk(workspace):
        subdirectories[:] = sorted(d for d in subdirectories if d not in EXCLUDED_DIRECTORIES)
        files += [
            Path(directory) / name
            for name in names
            if name.endswith(SOURCE_SUFFIXES) or name in SOURCE_NAMES
        ]
    return {str(path.relative_to(workspace)): file_hash(path) for path in sorted(files)}


def sources_digest(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(f"{name}\0{value}\n".encode())
    return digest.hexdigest()


class Native(Adapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.native = self.options.native
        # The engine workspace is the benchmarks project's parent: ``inference/``.
        self.workspace = self.root.parent

    async def prepare(self):
        binary = self.native.binary
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"native engine binary is not an executable file: {binary}")
        sources = engine_sources(self.workspace)
        self.identity = {
            "adapter": "magnitude-native",
            "binary": str(binary),
            "binary_sha256": file_hash(binary),
            "binary_bytes": binary.stat().st_size,
            "workspace": str(self.workspace),
            "engine_source_sha256": sources_digest(sources),
            "engine_source_files": sources,
            "options": self.native.model_dump(mode="json"),
            "count_endpoint": "/v1/count",
        }

    def verify(self):
        super().verify()
        if file_hash(self.native.binary) != self.identity["binary_sha256"]:
            raise ValueError("native engine binary changed after preparation")
        if sources_digest(engine_sources(self.workspace)) != self.identity["engine_source_sha256"]:
            raise ValueError("native engine source changed after preparation")

    def argv(self, port, context, parallel, directory):
        native = self.native
        return [
            str(native.binary),
            "--model",
            str(self.artifact.path),
            "--no-projector",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model",
            self.served_model(),
            "--context-tokens",
            str(context),
            "--device",
            native.device,
            *(["--cache-dir", str(native.cache_dir)] if native.cache_dir else []),
            "--output-capacity",
            str(OUTPUT_CAPACITY),
            "--method",
            native.method,
            *(
                ["--mtp-proposals", str(native.mtp_proposals)]
                if native.mtp_proposals is not None
                else []
            ),
        ]

    def verify_ready(self, data, context, parallel):
        if data.get("ready") is not True or data.get("model") != self.served_model():
            raise ValueError(f"native engine readiness identity mismatch: {data}")
        if type(data.get("context_tokens")) is not int or data["context_tokens"] != context:
            raise ValueError(f"native engine readiness context mismatch: {data}")
        if type(data.get("vocabulary")) is not int or data["vocabulary"] <= 0:
            raise ValueError(f"native engine readiness vocabulary missing: {data}")

    def count_context(self) -> int:
        return min(COUNT_CONTEXT_TOKENS, self.artifact.context_limit)

    async def render_counts(self, requests, engine) -> dict[str, int]:
        counts = {}
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            for request in requests:
                response = await client.post(
                    engine.endpoint + "/v1/count", json=request.body(engine.model)
                )
                response.raise_for_status()
                count = response.json()["prompt_tokens"]
                if type(count) is not int or count <= 0:
                    raise ValueError(f"/v1/count returned no prompt token count: {response.text}")
                counts[request.id] = count
        return counts

    @contextlib.asynccontextmanager
    async def context_counter(self):
        async with self.launch(self.count_context(), 1, "fixture-prepare") as engine:

            async def count(context):
                plan = self.context_plan(context)
                return (await self.render_counts(plan.prepared_requests, engine))[
                    "fixture-sizing"
                ]

            yield count

    async def prompt_counts(self, plan):
        async with self.launch(self.count_context(), 1, "prepare") as engine:
            return await self.render_counts(plan.prepared_requests, engine)
