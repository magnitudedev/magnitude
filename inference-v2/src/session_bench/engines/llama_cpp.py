import shutil
from pathlib import Path

import httpx

from ..models import file_hash
from .base import Adapter, command


class LlamaCpp(Adapter):
    extensions = {"cache_prompt": False}

    async def prepare(self):
        executable = shutil.which("llama-server")
        if not executable:
            raise ValueError("upstream llama-server is required on PATH for --engine llama.cpp")
        self.executable = executable
        self.identity = {"executable": executable, "sha256": file_hash(Path(executable))}
        self.identity["version"] = await command(
            [self.executable, "--version"],
            self.root,
            self.store.path / "logs" / "llama-version.log",
        )

    def verify(self):
        super().verify()
        if file_hash(Path(self.executable)) != self.identity["sha256"]:
            raise ValueError("llama-server changed after preparation")

    def argv(self, port, context, parallel, directory):
        return [
            self.executable,
            "--model",
            str(self.artifact.path),
            "--alias",
            self.served_model(),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(context * parallel),
            "--parallel",
            str(parallel),
            "--jinja",
            "--flash-attn",
            "on",
            "--cont-batching",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--no-cache-prompt",
            "--cache-ram",
            "0",
            "--offline",
            "--no-context-shift",
        ]

    def ready_path(self):
        return "/props"

    def verify_ready(self, data, context, parallel):
        if data.get("total_slots") != parallel:
            raise ValueError("llama.cpp slot capacity differs from the requested capacity")
        settings = data.get("default_generation_settings", {})
        if settings.get("n_ctx") != context:
            raise ValueError(
                f"llama.cpp per-slot context is {settings.get('n_ctx')}, expected {context}"
            )

    async def prompt_counts(self, plan):
        counts = {}
        async with self.launch(
            self.provisional_capacity(plan), plan.parallel_sequences, "prepare"
        ) as engine:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                for request in plan.prepared_requests:
                    response = await client.post(
                        engine.endpoint + "/apply-template", json=request.body(engine.model)
                    )
                    response.raise_for_status()
                    prompt = response.json()["prompt"]
                    response = await client.post(
                        engine.endpoint + "/tokenize",
                        json={"content": prompt, "add_special": True, "parse_special": True},
                    )
                    response.raise_for_status()
                    counts[request.id] = len(response.json()["tokens"])
        return counts
