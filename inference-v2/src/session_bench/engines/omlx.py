import httpx

from .base import Adapter


class Omlx(Adapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.runtime = self.root / "session-bench-runtimes" / "omlx"

    def argv(self, port, context, parallel, directory):
        return [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "python",
            "-m",
            "session_bench.engines.omlx_native.server",
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

    async def prompt_counts(self, plan):
        counts = {}
        async with self.launch(
            self.provisional_capacity(plan), plan.parallel_sequences, "prepare"
        ) as engine:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                for request in plan.prepared_requests:
                    response = await client.post(
                        engine.endpoint + "/session-bench/count", json=request.body(engine.model)
                    )
                    response.raise_for_status()
                    counts[request.id] = response.json()["prompt_tokens"]
        return counts
