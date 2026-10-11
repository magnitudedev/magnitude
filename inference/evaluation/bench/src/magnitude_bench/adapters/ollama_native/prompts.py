"""The prompt Ollama's MLX runner evaluates for a benchmark request, known without running
the model.

Ollama renders a chat request with the Go renderer its tag's config names (``render``) and
hands the text to the MLX runner, which tokenizes it as is with the safetensors tag's
``tokenizer.json`` and adds BOS when ``tokenizer_config.json`` sets ``add_bos_token``
(``mlxrunner/pipeline.go``, ``mlxrunner/tokenizer``). GGUF runners are counted by the harness.

A format without a thinking-off switch opens the reply at its answer header (``render``).
Ollama's chat route cannot write that header, so such a request goes through the raw route
with exactly the text the chat route would hand the runner, plus the header; Ollama passes
raw text to the runner unchanged.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import render, translate

# Layers that would change what Ollama renders: Modelfile messages and system prompt.
RENDERING_LAYERS = {
    "application/vnd.ollama.image.messages",
    "application/vnd.ollama.image.system",
}


@dataclass(frozen=True)
class Tag:
    """A pulled registry tag, read from its manifest."""

    name: str
    config: dict[str, Any]
    layers: list[dict[str, Any]]
    blobs: Path

    @classmethod
    def read(cls, manifest: Path, name: str) -> "Tag":
        document = json.loads(manifest.read_text())
        blobs = manifest.parents[4] / "blobs"
        config = json.loads((blobs / document["config"]["digest"].replace(":", "-")).read_text())
        return cls(name, config, document["layers"], blobs)

    def blob(self, layer: dict[str, Any]) -> Path:
        return self.blobs / layer["digest"].replace(":", "-")

    def layer(self, *, media_type: str | None = None, name: str | None = None) -> Path:
        found = [
            item
            for item in self.layers
            if (media_type is None or item.get("mediaType") == media_type)
            and (name is None or item.get("name") == name)
        ]
        if len(found) != 1:
            raise ValueError(f"Ollama tag {self.name} has {len(found)} {media_type or name} layers")
        return self.blob(found[0])

    def renderer(self) -> str:
        if self.config.get("model_family") == "qwen3":
            # Ollama filters think tags out of earlier qwen3 turns before rendering.
            raise ValueError("qwen3-family think-tag filtering is not reproduced")
        rendering = sorted({item.get("mediaType") for item in self.layers} & RENDERING_LAYERS)
        if rendering:
            raise ValueError(f"Ollama tag {self.name} carries {rendering}, which are not reproduced")
        configured = self.config.get("renderer") or ""
        if not configured:
            raise ValueError(f"Ollama tag {self.name} names no renderer; its template is not reproduced")
        return render.renderer(configured, self.name, self.config.get("model_type") or "")


def mlx_adds_bos(tag: Tag) -> bool:
    """The MLX runner's AddBOS: ``add_bos_token`` from ``tokenizer_config.json``, off unless set,
    and only with a BOS token to add."""
    config = json.loads(tag.layer(name="tokenizer_config.json").read_text())
    if config.get("add_bos_token") is not True:
        return False
    if not config.get("bos_token"):
        raise ValueError(f"Ollama tag {tag.name} adds BOS but declares no BOS token")
    return True


@dataclass(frozen=True)
class PromptFormat:
    """How Ollama turns a request for one tag into the text its runner evaluates."""

    renderer: str
    adds_bos: bool

    @classmethod
    def of(cls, tag: Tag) -> "PromptFormat":
        return cls(tag.renderer(), mlx_adds_bos(tag))

    @property
    def answer_header(self) -> str:
        """The header the reply opens at with thinking off; empty when the format has a switch."""
        return render.answer_header(self.renderer)

    def chat_text(self, chat: dict[str, Any]) -> str:
        """The text Ollama's chat route hands the runner for a native chat request."""
        if chat.get("tools"):
            raise ValueError("Ollama rendering is reproduced without tools")
        return render.render(self.renderer, chat["messages"], chat.get("think"))

    def runner_text(self, chat: dict[str, Any]) -> str:
        """The text the runner evaluates: the chat route's, opened at the answer header."""
        return self.chat_text(chat) + self.answer_header


@dataclass(frozen=True)
class PromptCounter:
    """Counts the prompt tokens Ollama's runner evaluates for a benchmark request body."""

    format: PromptFormat
    encode: Callable[[str], Awaitable[int]]
    answer_prefill: bool = False

    async def count(self, body: dict[str, Any]) -> int:
        chat = translate.chat_request(body, "counted", 0)
        text = self.format.runner_text(chat) if self.answer_prefill else self.format.chat_text(chat)
        return await self.encode(text) + int(self.format.adds_bos)
