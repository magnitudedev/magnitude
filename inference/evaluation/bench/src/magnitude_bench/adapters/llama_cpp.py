"""Upstream llama-server serving the catalog target, with the catalog drafter when it
supports the drafter's method.

Its prompts are counted host-only by ``magnitude-harness render`` over the same GGUF
template and tokenizer; llama.cpp's terminal counts stay authoritative for every request.
"""

import re
import shutil
from pathlib import Path

from ..models import file_hash
from .base import Adapter, command

# llama.cpp logs each per-sequence memory buffer as it allocates it (at
# verbosity 4): the attention KV cache and, for hybrid models, the recurrent
# state. Their sizes are what one slot of the configured context holds.
STATE_BUFFER = re.compile(r"\b(KV|RS) buffer size = +([0-9.]+) MiB")
#: Catalog drafting method → llama.cpp ``--spec-type``. DFlash2 and LiLiCorr drafters run on
#: llama.cpp's DFlash implementation, which reads a LiLiCorr draft's corrector from its GGUF.
SPECULATION = {
    "mtp": "draft-mtp",
    "dflash": "draft-dflash",
    "dflash2": "draft-dflash",
    "lilicorr": "draft-dflash",
    "dspark": "draft-dspark",
}


def executable(configured: Path | None) -> str:
    found = str(configured) if configured else shutil.which("llama-server")
    if not found:
        raise ValueError("llama-server is required on PATH or as --llama-binary")
    if not Path(found).is_file():
        raise ValueError(f"llama-server executable does not exist: {found}")
    return found


class LlamaCpp(Adapter):
    extensions = {"cache_prompt": False}

    @property
    def drafting(self) -> str | None:
        drafter = self.acquisition.drafter
        return drafter.method if self.options.drafting and drafter is not None else None

    async def memory_observation(self, engine):
        kv = recurrent = 0.0
        for kind, mib in STATE_BUFFER.findall(engine.log.read_text(errors="replace")):
            if kind == "KV":
                kv += float(mib)
            else:
                recurrent += float(mib)
        if kv == 0 and recurrent == 0:
            return None
        return {
            "state": {
                "kv_bytes": round(kv * 1024 * 1024),
                "recurrent_bytes": round(recurrent * 1024 * 1024),
            }
        }

    async def prepare(self):
        if self.drafting is not None and self.drafting not in SPECULATION:
            raise ValueError(
                f"llama.cpp cannot serve the catalog's {self.drafting} drafter; "
                "measure it with --drafting off"
            )
        self.executable = executable(self.options.llama.binary)
        self.identity = {
            "executable": self.executable,
            "sha256": file_hash(Path(self.executable)),
            "drafting": self.drafting,
            "version": await command(
                [self.executable, "--version"],
                Path(self.executable).parent,
                self.store.path / "logs" / "llama-version.log",
            ),
        }

    def verify(self):
        super().verify()
        if file_hash(Path(self.executable)) != self.identity["sha256"]:
            raise ValueError("llama-server changed after preparation")

    def speculation(self) -> list[str]:
        drafter = self.acquisition.drafter
        if self.drafting is None or drafter is None:
            return []
        arguments = [
            "--spec-type", SPECULATION[drafter.method],
            "--spec-draft-n-max", str(self.options.llama.draft_proposals),
        ]
        if not self.acquisition.embedded_drafter:
            arguments += [
                "--model-draft", str(drafter.path),
                "--n-gpu-layers-draft", str(self.options.llama.gpu_layers),
            ]
        return arguments

    def headroom(self, context: int) -> int:
        """Tokens per slot beyond ``context``; a launch at the model's limit has none to add."""
        return min(self.options.llama.context_headroom, self.artifact.context_limit - context)

    def argv(self, port, context, parallel, directory):
        return [
            self.executable,
            "--model", str(self.artifact.path),
            "--alias", self.served_model(),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str((context + self.headroom(context)) * parallel),
            "--parallel", str(parallel),
            "--n-gpu-layers", str(self.options.llama.gpu_layers),
            *self.speculation(),
            "--jinja",
            "--flash-attn", "on",
            "--cont-batching",
            "--cache-type-k", self.options.llama.cache_type_k,
            "--cache-type-v", self.options.llama.cache_type_v,
            "--no-cache-prompt",
            "--cache-ram", "0",
            "--offline",
            "--no-context-shift",
            "--log-verbosity", "4",
        ]

    def ready_path(self):
        return "/props"

    def verify_ready(self, data, context, parallel):
        if data.get("total_slots") != parallel:
            raise ValueError("llama.cpp slot capacity differs from the requested capacity")
        settings = data.get("default_generation_settings", {})
        expected = context + self.headroom(context)
        if settings.get("n_ctx") != expected:
            raise ValueError(
                f"llama.cpp per-slot context is {settings.get('n_ctx')}, expected {expected}"
            )
