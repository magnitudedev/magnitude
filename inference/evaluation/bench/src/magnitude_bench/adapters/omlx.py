"""Pinned oMLX, plain decoding, with native phase timings (``omlx_native``).

It counts its own prompts: the MLX snapshot's template and tokenizer are not the GGUF's and
no host-only renderer of ours reads MLX snapshots, while oMLX's chat renderer lives in its
loaded engine. A counting launch therefore serves ``/magnitude/benchmark/count``, which
renders with oMLX's own renderer and tokenizer without generating.
"""

import contextlib

import httpx

from ..policy import RUNTIMES
from .base import Adapter

COUNTING_CONTEXT = 4096


class Omlx(Adapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.runtime = RUNTIMES / "omlx"
        self.countings = 0

    def argv(self, port, context, parallel, directory):
        return [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "python",
            "-m",
            "magnitude_bench.adapters.omlx_native.server",
            "--model",
            str(self.artifact.path),
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
        ]

    def ready_path(self):
        return "/magnitude/benchmark/readiness"

    def verify_ready(self, data, context, parallel):
        expected = {
            "ready": True,
            "served_model": self.served_model(),
            "loaded": True,
            "context_capacity": context,
            "max_concurrent_requests": parallel,
            "speculative_backend": "none",
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"oMLX readiness mismatch: {key}")

    @contextlib.asynccontextmanager
    async def counter(self):
        context = min(COUNTING_CONTEXT, self.artifact.context_limit)
        self.countings += 1
        async with self.launch(context, 1, f"counting-{self.countings}") as engine:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:

                async def lengths(bodies):
                    counts = []
                    for body in bodies:
                        response = await client.post(
                            engine.endpoint + "/magnitude/benchmark/count",
                            json={**body, "model": engine.model},
                        )
                        response.raise_for_status()
                        counts.append(response.json()["prompt_tokens"])
                    return counts

                yield lengths
