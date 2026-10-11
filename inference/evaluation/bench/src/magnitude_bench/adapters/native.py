"""Our engine: a ``magnitude-engine`` binary serving the catalog target with its drafter.

By default the binary is this tree's build; any other (for example origin's) is measured the
same way over HTTP and recorded by its bytes and declared source build.

Prompts are counted, and fixtures sized, host-only by the engine's own serving render path
(``magnitude-harness render``, run by the runner for every GGUF engine). The engine tunes on its first load under the run's budget
and appends its tuning trace to the run directory's ``tuning.jsonl``. Like a user's
installation it keeps a persistent kernel cache (per model and engine binary under the
kernel cache root, or exactly ``cache_dir``), so later launches reuse stored tuning unless
the run tunes afresh.
"""

import os

import httpx

from ..models import file_hash
from ..record import TUNING_TRACE
from .base import Adapter

OUTPUT_CAPACITY = 256


class Native(Adapter):
    def __init__(self, *args):
        super().__init__(*args)
        self.magnitude = self.options.magnitude
        self.launches = 0
        #: The cache the engine is launched with; a root's is resolved by `prepare`.
        self.kernel_cache = self.magnitude.cache_dir

    @property
    def drafting(self) -> str | None:
        drafter = self.acquisition.drafter
        return drafter.method if self.options.drafting and drafter is not None else None

    async def prepare(self):
        binary = self.magnitude.binary
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"magnitude-engine is unavailable: {binary}")
        binary_sha256 = file_hash(binary)
        if self.magnitude.kernel_cache_root is not None:
            # One persistent cache per model and engine binary, kept across runs.
            model = self.acquisition.model.replace(":", "-")
            self.kernel_cache = self.magnitude.kernel_cache_root / model / binary_sha256[:16]
        self.identity = {
            "adapter": "magnitude",
            "binary": str(binary),
            "binary_sha256": binary_sha256,
            "binary_bytes": binary.stat().st_size,
            "kernel_cache": None if self.kernel_cache is None else str(self.kernel_cache),
            "drafting": self.drafting,
            "options": self.magnitude.model_dump(mode="json"),
        }

    def verify(self):
        super().verify()
        if file_hash(self.magnitude.binary) != self.identity["binary_sha256"]:
            raise ValueError("magnitude-engine changed after preparation")

    async def memory_observation(self, engine):
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.get(engine.endpoint + "/v1/memory")
            response.raise_for_status()
            return response.json()

    def generation(self) -> list[str]:
        """The catalog drafter, or plain decoding when drafting is off or none is declared."""
        drafter = self.acquisition.drafter
        if self.drafting is None or drafter is None:
            return ["--method", "plain"]
        if self.acquisition.embedded_drafter:
            arguments = ["--method", drafter.method]
        else:
            arguments = ["--method", drafter.method, "--draft", str(drafter.path)]
        if self.magnitude.mtp_proposals is not None:
            arguments += ["--mtp-proposals", str(self.magnitude.mtp_proposals)]
        if self.magnitude.draft_budget is not None:
            arguments += ["--draft-budget", self.magnitude.draft_budget]
        return arguments

    def tuning(self) -> list[str]:
        """The run's tuning budget and trace, each only when requested; only the run's first
        launch tunes afresh."""
        arguments = []
        if self.magnitude.tuning_trace:
            arguments += ["--tuning-trace", str(self.store.path / TUNING_TRACE)]
        if self.magnitude.tuning_seconds is not None:
            arguments += ["--tuning-budget", str(self.magnitude.tuning_seconds)]
        if self.magnitude.fresh_tuning and self.launches == 0:
            arguments += ["--fresh-tuning"]
        return arguments

    def argv(self, port, context, parallel, directory):
        magnitude = self.magnitude
        arguments = [
            str(magnitude.binary),
            "--model", str(self.artifact.path),
            "--no-projector",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--served-model", self.served_model(),
            "--context-tokens", str(context),
            "--device", magnitude.device,
            *(["--cache-dir", str(self.kernel_cache)] if self.kernel_cache is not None else []),
            "--output-capacity", str(OUTPUT_CAPACITY),
            *(
                ["--prefill-tokens", str(magnitude.prefill_tokens)]
                if magnitude.prefill_tokens is not None
                else []
            ),
            *self.generation(),
            *self.tuning(),
        ]
        self.launches += 1
        return arguments

    def verify_ready(self, data, context, parallel):
        if data.get("ready") is not True or data.get("model") != self.served_model():
            raise ValueError(f"magnitude readiness identity mismatch: {data}")
        if type(data.get("context_tokens")) is not int or data["context_tokens"] != context:
            raise ValueError(f"magnitude readiness context mismatch: {data}")
        if type(data.get("vocabulary")) is not int or data["vocabulary"] <= 0:
            raise ValueError(f"magnitude readiness vocabulary missing: {data}")
