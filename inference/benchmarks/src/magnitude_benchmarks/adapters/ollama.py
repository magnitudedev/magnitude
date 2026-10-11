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


from ..session_bench.models import file_hash
from .base import Adapter, command
from .counting import magnitude_count, tokenizer_json_count
from .ollama_native.prompts import PromptCounter, PromptFormat, Tag
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
        self.tag = (Tag.read(self.artifact.path, self.artifact.reference.removeprefix("ollama:"))
                    if self.artifact.kind == "registry" else None)
        self.format = PromptFormat.of(self.tag, RUNNERS[self.target.engine]) if self.tag else None
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

    @asynccontextmanager
    async def counter(self):
        """Ollama's renderer reproduced and the tag's own tokenizer, without a model load
        (``ollama_native.prompts``)."""
        if self.tag is None or self.format is None:
            raise ValueError(
                "Ollama renders an imported GGUF with a Go template, which is not reproduced"
            )
        if self.format.runner == "llama-server":
            binary = self.options.ollama.count_binary
            if binary is None:
                raise ValueError("counting a GGUF tag's prompts needs --ollama-count-binary")
            log = self.store.path / "logs" / f"{self.target.id}-count.log"
            async with magnitude_count(binary, "text", self.tag.weights(), log) as encode:
                yield self.counts(PromptCounter(self.format, encode, self.options.ollama.answer_prefill))
            return
        tokens = tokenizer_json_count(self.tag.layer(name="tokenizer.json"))

        async def encode(text):
            return tokens(text)

        yield self.counts(PromptCounter(self.format, encode, self.options.ollama.answer_prefill))

    def counts(self, prompts: PromptCounter):
        async def count(requests):
            return {r.id: await prompts.count(r.body(self.served_model())) for r in requests}

        return count

    async def prompt_counts(self, plan):
        counts = await super().prompt_counts(plan)
        expected = {
            prompt_identity(request.body(self.served_model())): counts[request.id]
            for request in plan.prepared_requests
        }
        self.expected_path().write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return counts
