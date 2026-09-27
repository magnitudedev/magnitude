"""Single-request text benchmark bridge to an unchanged local TileLang PoC."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from .base import Adapter, installed_versions, runtime_digest


class TileLangPoc(Adapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.runtime = Path(
            os.environ.get("TILELANG_POC_ROOT", str(Path.home() / "lab/tilelang-poc"))
        ).resolve()
        self.count_only = False
        self.prepared = None

    async def prepare(self):
        # Preserve the PoC's existing compiler/environment; never uv-sync it.
        if not (self.runtime / ".venv/bin/python").is_file():
            raise ValueError(f"PoC Python environment missing: {self.runtime}")
        if not (self.runtime / "src/tilelang_poc/model.py").is_file():
            raise ValueError(f"PoC source missing: {self.runtime}")
        self.identity = self.store.snapshot(self.target.id, self.runtime)
        self.source_identity = runtime_digest(self.runtime)
        self.identity["dependencies"] = installed_versions(self.runtime)
        self.identity["benchmark_policy"] = {
            "concurrency": 1,
            "workload": "prose",
            "prefill": "unchanged PoC block size",
            "warmup": "full prepared requests before readiness; history reset afterward",
            "timing": "completed prefill; selection plus completed decode steps; excludes SSE",
            "prefix_cache": False,
        }

    def environment(self):
        env = super().environment()
        env["PYTHONPATH"] = str(self.runtime / "src")
        return env

    def argv(self, port, context, parallel, directory):
        if parallel != 1:
            raise ValueError("The PoC benchmark bridge supports concurrency 1 only")
        args = [
            str(self.runtime / ".venv/bin/python"),
            "-P",  # Do not let sibling tokenize.py shadow Python's standard library.
            str(Path(__file__).with_name("poc_native.py")),
            "--model",
            str(self.artifact.path),
            "--port",
            str(port),
            "--context",
            str(context),
            "--evidence",
            str(directory),
        ]
        if self.count_only:
            return [*args, "--count-only"]
        if self.prepared is None:
            raise ValueError("PoC request preparation must precede measurement")
        requests = directory / "warmup-requests.json"
        requests.write_text(
            json.dumps(
                [request.body(self.served_model()) for request in self.prepared.prepared_requests]
            )
        )
        return [*args, "--warmup-requests", str(requests)]

    def verify_ready(self, data, context, parallel):
        expected = {
            "ready": True,
            "served_model": self.served_model(),
            "context_capacity": context,
            "parallel_sequences": parallel,
            "count_only": self.count_only,
            "prefix_cache": False,
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"PoC readiness mismatch: {key}")

    @asynccontextmanager
    async def context_counter(self):
        self.count_only = True
        try:
            async with self.launch(4096, 1, "fixture-prepare") as engine:
                async with httpx.AsyncClient(timeout=120, trust_env=False) as client:

                    async def count(context):
                        request = self.context_plan(context).requests[0]
                        response = await client.post(
                            engine.endpoint + "/count", json=request.body(engine.model)
                        )
                        response.raise_for_status()
                        return response.json()["prompt_tokens"]

                    yield count
        finally:
            self.count_only = False

    async def prompt_counts(self, plan):
        if plan.parallel_sequences != 1 or any(
            request.workload != "prose" for request in plan.prepared_requests
        ):
            raise ValueError("The PoC benchmark bridge supports single-request prose only")
        self.prepared = plan
        self.count_only = True
        try:
            async with self.launch(4096, 1, "prepare") as engine:
                async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                    counts = {}
                    for request in plan.prepared_requests:
                        response = await client.post(
                            engine.endpoint + "/count", json=request.body(engine.model)
                        )
                        response.raise_for_status()
                        counts[request.id] = response.json()["prompt_tokens"]
                    return counts
        finally:
            self.count_only = False
