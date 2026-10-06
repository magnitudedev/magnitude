"""Ollama as a product: its own server, runner selection and timing counters.

``ollama`` imports a local GGUF into Ollama's store unchanged (``FROM <path>``), which Ollama
serves with its bundled llama.cpp runner. ``ollama-mlx`` and ``ollama-registry`` serve a model
already pulled from Ollama's registry: in safetensors form, which Ollama serves with its MLX
runner, and in GGUF form, which it serves with its llama.cpp runner.
"""

import json
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from ..session_bench.models import file_hash
from .base import Adapter, command
from .ollama_native.translate import prompt_identity

# Ollama picks the runner from the stored model format; the engine fixes which is expected.
MODEL_FORMATS = {"ollama": "gguf", "ollama-mlx": "safetensors", "ollama-registry": "gguf"}
RUNNERS = {"ollama": "llama-server", "ollama-mlx": "mlx", "ollama-registry": "llama-server"}


def store_path(configured: Path | None) -> Path:
    if configured is not None:
        return configured.expanduser().absolute()
    if "OLLAMA_MODELS" not in os.environ:
        raise ValueError("Ollama targets require --ollama-models or OLLAMA_MODELS")
    return Path(os.environ["OLLAMA_MODELS"]).expanduser().absolute()


class Ollama(Adapter):
    async def prepare(self):
        selected = self.options.ollama
        executable = str(selected.binary) if selected.binary else shutil.which("ollama")
        if not executable:
            raise ValueError("ollama is required on PATH or as --ollama-binary")
        if not Path(executable).is_file():
            raise ValueError(f"ollama executable does not exist: {executable}")
        self.executable = executable
        self.store_directory = store_path(selected.models)
        self.identity = {
            "adapter": "ollama-native",
            "executable": executable,
            "sha256": file_hash(Path(executable)),
            "store": str(self.store_directory),
            "expected_model_format": MODEL_FORMATS[self.target.engine],
            "expected_runner": RUNNERS[self.target.engine],
            "options": selected.model_dump(mode="json"),
            # `ollama --version` reports the client; no server runs yet, which it also says.
            "version": await command(
                [executable, "--version"],
                self.root,
                self.store.path / "logs" / f"{self.target.id}-version.log",
            ),
        }

    def verify(self):
        super().verify()
        if file_hash(Path(self.executable)) != self.identity["sha256"]:
            raise ValueError("ollama executable changed after preparation")

    def expected_path(self) -> Path:
        return self.store.path / f"{self.target.id}-expected-prompt-tokens.json"

    def headroom(self, context: int) -> int:
        """Tokens allocated beyond ``context``; a launch at the model's limit has none to add."""
        return min(self.options.ollama.context_headroom, self.artifact.context_limit - context)

    def argv(self, port, context, parallel, directory):
        selected = self.options.ollama
        source = (
            ["--gguf", str(self.artifact.path)]
            if self.artifact.kind == "gguf"
            else ["--registry-model", self.artifact.reference.removeprefix("ollama:")]
        )
        return [
            sys.executable,
            "-m",
            "magnitude_benchmarks.adapters.ollama_native.server",
            "--ollama",
            self.executable,
            "--store",
            str(self.store_directory),
            *source,
            "--served-model",
            self.served_model(),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--base-path",
            str(directory),
            "--max-concurrent-requests",
            str(parallel),
            "--context-capacity",
            str(context),
            *(["--kv-cache-type", selected.kv_cache_type] if selected.kv_cache_type else []),
            "--flash-attention",
            selected.flash_attention,
            "--speculation",
            selected.speculation,
            "--context-headroom",
            str(self.headroom(context)),
            *(["--answer-prefill"] if selected.answer_prefill else []),
            *(
                ["--expected-prompt-tokens", str(self.expected_path())]
                if self.expected_path().is_file()
                else []
            ),
        ]

    def ready_path(self):
        return "/magnitude/benchmark/readiness"

    def verify_ready(self, data, context, parallel):
        expected = {
            "ready": True,
            "served_model": self.served_model(),
            "context_capacity": context,
            "max_concurrent_requests": parallel,
            "model_format": MODEL_FORMATS[self.target.engine],
            "runner": RUNNERS[self.target.engine],
            "speculation": self.options.ollama.speculation,
            "context_headroom": self.headroom(context),
            "answer_prefill": self.options.ollama.answer_prefill,
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"Ollama readiness mismatch: {key} is {data.get(key)!r}")
        if RUNNERS[self.target.engine] == "llama-server":
            # The runner's own load lines, one per model (the target, then any draft model).
            # The sizes Ollama reports are not evidence: for a tag with a separate draft model
            # they cover a fraction of the load.
            layers = data.get("gpu_layers")
            if not layers or any(placed != total or total <= 0 for placed, total in layers):
                raise ValueError(f"Ollama did not place every layer on the GPU: {layers}")
            return
        size, accelerated = data.get("size_bytes"), data.get("size_vram_bytes")
        if data.get("mlx_device") != "gpu" or type(size) is not int or size <= 0:
            raise ValueError(f"Ollama's MLX runner is not on the GPU: {data.get('mlx_device')}")
        if accelerated != size:
            raise ValueError(
                f"Ollama did not place the whole model on the GPU: {accelerated} of {size} bytes"
            )

    def counting_capacity(self, plan) -> int:
        """Context for the counting launch: what the largest request is sized to need.

        Ollama counts a prompt by evaluating it, so the launch must hold the prompt, and a
        prompt that turns out larger is still counted (from the runner's rejection or Ollama's
        truncation warning). The tokenisation allowance other engines launch with can be the
        model's whole context limit, which a large model does not fit on the GPU.
        """
        needed = max(r.checkpoint + r.output_limit for r in plan.prepared_requests)
        return min(self.provisional_capacity(plan), needed)

    async def prompt_counts(self, plan):
        async with self.launch(
            self.counting_capacity(plan), plan.parallel_sequences, "prepare"
        ) as engine:
            counts = await self.render_counts(plan, engine)
        # Measured launches refuse a response whose evaluated prompt differs from these.
        expected = {
            prompt_identity(request.body(self.served_model())): counts[request.id]
            for request in plan.prepared_requests
        }
        self.expected_path().write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return counts

    @asynccontextmanager
    async def context_counter(self):
        capacity = min(self.options.ollama.sizing_context, self.artifact.context_limit)
        async with self.launch(capacity, 1, "fixture-prepare") as engine:

            async def count(context):
                return (await self.render_counts(self.context_plan(context), engine))[
                    "fixture-sizing"
                ]

            yield count

    async def render_counts(self, plan, engine):
        counts = {}
        async with httpx.AsyncClient(timeout=1800, trust_env=False) as client:
            for request in plan.prepared_requests:
                response = await client.post(
                    engine.endpoint + "/session-bench/count", json=request.body(engine.model)
                )
                response.raise_for_status()
                counts[request.id] = response.json()["prompt_tokens"]
        return counts
