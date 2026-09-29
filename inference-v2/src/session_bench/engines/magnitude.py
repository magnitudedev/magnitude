from ..policy import OUTPUT_CAPACITY, PREFILL_TOKENS
from .base import Adapter


class Magnitude(Adapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.runtime = self.root
        import psutil

        total = psutil.virtual_memory().total
        self.memory_limit = max(1, total - max(8 << 30, total // 5))

    def argv(self, port, context, parallel, directory):
        return [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "python",
            "-m",
            "magnitude_engine.serving",
            "--target",
            str(self.artifact.path),
            "--model",
            self.served_model(),
            "--port",
            str(port),
            "--context-tokens",
            str(context),
            "--max-active",
            str(parallel),
            "--prefill-tokens",
            str(PREFILL_TOKENS),
            "--output-capacity",
            str(OUTPUT_CAPACITY),
            "--retained-prefixes",
            "0",
            "--memory-bytes",
            str(self.memory_limit),
        ]

    def verify_ready(self, data, context, parallel):
        expected = {
            "status": "ready",
            "model": self.served_model(),
            "context_tokens": context,
            "parallel_sequences": parallel,
            "speculative_backend": "none",
            "retained_prefixes": 0,
            "prefill_tokens": PREFILL_TOKENS,
            "output_capacity": OUTPUT_CAPACITY,
            "memory_bytes": self.memory_limit,
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(
                    f"Magnitude readiness {key}: expected {value}, got {data.get(key)}"
                )
        import hashlib

        canonical = data.get("composition_json")
        if not isinstance(canonical, str) or hashlib.sha256(
            canonical.encode()
        ).hexdigest() != data.get("composition_digest"):
            raise ValueError("engine composition digest missing or inconsistent")

    async def prompt_counts(self, plan):
        return await self.tokenizer_counts(plan, "magnitude")
