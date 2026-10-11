"""Stock MLX-VLM serving an MLX snapshot of the catalog model.

It counts its own prompts: the snapshot's template and tokenizer are not the GGUF's, and no
host-only renderer of ours reads MLX snapshots, so its own processor renders them in its
locked runtime (``mlx_vlm_count``), loading only the processor, not the model.
"""

import json
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from magnitude_fixtures.canonical import encoded

from ..policy import PREFILL_TOKENS, RUNTIMES, SERVER_MAX_OUTPUT_TOKENS
from .base import Adapter, command


class MlxVlm(Adapter):
    timing_basis = "server-token-emission"

    def __init__(self, *args):
        super().__init__(*args)
        self.runtime = RUNTIMES / "mlx-vlm"

    def served_model(self):
        return str(self.artifact.path)

    def argv(self, port, context, parallel, directory):
        return [
            "uv",
            "run",
            "--frozen",
            "--no-sync",
            "mlx_vlm.server",
            "--model",
            str(self.artifact.path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--prefill-step-size",
            str(PREFILL_TOKENS),
            "--max-num-seqs",
            str(parallel),
            "--max-tokens",
            str(SERVER_MAX_OUTPUT_TOKENS),
            "--max-kv-size",
            str(context),
            "--log-progress-interval",
            "0",
        ]

    def environment(self):
        env = super().environment()
        env.update(APC_ENABLED="0", MLX_VLM_ENABLE_THINKING="0", MLX_TRUST_REMOTE_CODE="false")
        # Ambient server authentication must not change this private listener.
        for key in ("MLX_VLM_API_KEY", "MLX_VLM_MANAGEMENT_API_KEY"):
            env.pop(key, None)
        return env

    def verify_ready(self, data, context, parallel):
        if data.get("loaded_model") is None:
            return False
        expected = {
            "status": "healthy",
            "loaded_model": self.served_model(),
            "effective_context_limit": context,
            "apc_enabled": False,
            "continuous_batching_enabled": True,
        }
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"MLX-VLM readiness mismatch: {key}")
        if data.get("loaded_adapter") is not None:
            raise ValueError("MLX-VLM unexpectedly loaded an adapter")

    @asynccontextmanager
    async def counter(self):
        yield self.render_counts

    async def render_counts(self, bodies):
        """Stock MLX-VLM has no count route; render with its own processor in its runtime."""
        with tempfile.TemporaryDirectory(prefix="render-", dir=self.store.path / "logs") as work:
            source = Path(work) / "bodies.jsonl"
            output = Path(work) / "counts.json"
            source.write_text("".join(encoded(body) + "\n" for body in bodies))
            await command(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "--no-sync",
                    "python",
                    "-m",
                    "magnitude_bench.adapters.mlx_vlm_count",
                    str(self.artifact.path),
                    str(source),
                    str(output),
                ],
                self.runtime,
                self.store.path / "logs" / f"{self.target.id}-tokenize.log",
                env=self.environment(),
            )
            return json.loads(output.read_text())
