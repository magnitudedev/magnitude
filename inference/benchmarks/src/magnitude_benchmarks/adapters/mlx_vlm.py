import json
import tempfile
from pathlib import Path

from ..session_bench.policy import MAX_OUTPUT_TOKENS, PREFILL_TOKENS
from ..session_bench.sessions import encoded
from .base import Adapter, command


class MlxVlm(Adapter):
    timing_basis = "server-token-emission"

    def __init__(self, *args):
        super().__init__(*args)
        self.runtime = self.root / "runtimes" / "mlx-vlm"

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
            str(MAX_OUTPUT_TOKENS),
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

    async def prompt_counts(self, plan):
        """Stock MLX-VLM has no count route; render with its own processor in its runtime."""
        with tempfile.TemporaryDirectory(prefix="fixture-tokenize-", dir=self.store.path) as work:
            source = Path(work) / "requests.jsonl"
            output = Path(work) / "counts.json"
            source.write_text(
                "".join(encoded(r.model_dump(mode="json")) + "\n" for r in plan.prepared_requests)
            )
            await command(
                [
                    "uv",
                    "run",
                    "--frozen",
                    "--no-sync",
                    "python",
                    "-m",
                    "magnitude_benchmarks.adapters.mlx_vlm_count",
                    str(self.artifact.path),
                    str(source),
                    str(output),
                ],
                self.runtime,
                self.store.path / "logs" / f"{self.target.id}-tokenize.log",
                env=self.environment(),
            )
            return json.loads(output.read_text())
