from ..policy import MAX_OUTPUT_TOKENS, PREFILL_TOKENS
from .base import Adapter


class MlxVlm(Adapter):
    timing_basis = "server-token-emission"

    def __init__(self, *args):
        super().__init__(*args)
        self.runtime = self.root / "session-bench-runtimes" / "mlx-vlm"

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
        return await self.tokenizer_counts(plan, "mlx-vlm")
