"""Engine launch selection and supervision bounds chosen on the command line."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from ..fixtures.records import Record
from .policy import project_root


def default_native_binary() -> Path:
    """The release build of the sibling engine workspace: ``inference/target/release``."""
    return project_root().parent / "target" / "release" / "magnitude-engine"


#: Native methods that run a separate draft model.
SEPARATE_DRAFTS = ("dflash", "dspark", "dflash2")


class NativeOptions(Record):
    """Launch selection for the native engine; recorded in the reproduction command."""

    binary: Path = Field(default_factory=default_native_binary)
    device: str = "auto"
    cache_dir: Path | None = None
    method: Literal["auto", "plain", "mtp", "dflash", "dspark", "dflash2"] = "auto"
    mtp_proposals: int | None = Field(default=None, gt=0)
    prefill_tokens: int | None = Field(default=None, gt=0)
    #: A separate draft model (DFlash, DSpark, DFlash2) for the target; the
    #: engine refuses a draft that is not the requested method's.
    draft: Path | None = None
    #: Kernel error classes admitted for the model (the engine's
    #: `--admit-error-class`); none by default.
    error_classes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def proposals_need_a_drafter(self) -> Self:
        if self.mtp_proposals is not None and self.method not in ("mtp", *SEPARATE_DRAFTS):
            raise ValueError(
                "--native-mtp-proposals requires --native-method mtp, dflash, dspark or dflash2"
            )
        if self.method in SEPARATE_DRAFTS and self.draft is None:
            raise ValueError(f"--native-method {self.method} requires --native-draft")
        return self

    def arguments(self) -> list[str]:
        """Public CLI flags that reproduce this selection."""
        args = ["--native-binary", str(self.binary), "--native-device", self.device]
        if self.cache_dir is not None:
            args += ["--native-cache-dir", str(self.cache_dir)]
        args += ["--native-method", self.method]
        if self.mtp_proposals is not None:
            args += ["--native-mtp-proposals", str(self.mtp_proposals)]
        if self.prefill_tokens is not None:
            args += ["--native-prefill-tokens", str(self.prefill_tokens)]
        if self.draft is not None:
            args += ["--native-draft", str(self.draft)]
        for error_class in self.error_classes:
            args += ["--native-error-class", error_class]
        return args


# llama.cpp KV cache element types.
KvCacheType = Literal["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]


class LlamaOptions(Record):
    """Upstream llama-server selection recorded with each benchmark run."""

    binary: Path | None = None
    draft: Path | None = None
    draft_method: Literal["dflash", "dspark"] = "dflash"
    draft_proposals: int = Field(default=3, gt=0)
    gpu_layers: int = Field(default=99, ge=0)
    cache_type_k: KvCacheType = "f16"
    cache_type_v: KvCacheType = "f16"

    def arguments(self) -> list[str]:
        args = []
        if self.binary is not None:
            args += ["--llama-binary", str(self.binary)]
        if self.draft is not None:
            args += ["--llama-draft", str(self.draft)]
            args += ["--llama-draft-method", self.draft_method]
            args += ["--llama-draft-proposals", str(self.draft_proposals)]
        args += ["--llama-gpu-layers", str(self.gpu_layers)]
        args += ["--llama-cache-type-k", self.cache_type_k]
        args += ["--llama-cache-type-v", self.cache_type_v]
        return args


class OllamaOptions(Record):
    """Ollama selection recorded with each benchmark run.

    Unset values leave Ollama's own defaults in force: the product is measured as shipped
    except for the context, which the workload fixes.
    """

    binary: Path | None = None
    #: Ollama's model store (``OLLAMA_MODELS``): imports are written here, pulls are read here.
    models: Path | None = None
    #: ``OLLAMA_KV_CACHE_TYPE``; Ollama's default is f16.
    kv_cache_type: Literal["f16", "q8_0", "q4_0"] | None = None
    #: ``OLLAMA_FLASH_ATTENTION``; auto leaves the choice to Ollama.
    flash_attention: Literal["auto", "on", "off"] = "auto"
    #: Ollama's MLX runner drafts whenever a model ships a draft head. ``off`` parks it by
    #: requesting logprobs, the only switch Ollama exposes; ``default`` leaves it on.
    speculation: Literal["off", "default"] = "off"
    #: Tokens allocated beyond the benchmark context. Ollama's llama.cpp runner ends a drafting
    #: request before a context that fits the prompt and output exactly is full.
    context_headroom: int = Field(default=0, ge=0)
    #: Context of the launch that sizes the fixture. A prompt larger than it is counted without
    #: being evaluated; a larger value costs evaluations but avoids small-context launches.
    sizing_context: int = Field(default=4096, gt=0)
    #: Open the reply with the model's answer header. Only for a family whose format has no
    #: switch for thinking (Muse Glimmer): with ``think`` false the model still reasons first
    #: and Ollama discards that text. The prompt Ollama renders is sent through its raw
    #: completion route with the header appended; the adapter refuses any other family.
    answer_prefill: bool = False

    def arguments(self) -> list[str]:
        args = []
        if self.binary is not None:
            args += ["--ollama-binary", str(self.binary)]
        if self.models is not None:
            args += ["--ollama-models", str(self.models)]
        if self.kv_cache_type is not None:
            args += ["--ollama-kv-cache-type", self.kv_cache_type]
        args += ["--ollama-flash-attention", self.flash_attention]
        args += ["--ollama-speculation", self.speculation]
        if self.context_headroom:
            args += ["--ollama-context-headroom", str(self.context_headroom)]
        if self.sizing_context != 4096:
            args += ["--ollama-sizing-context", str(self.sizing_context)]
        if self.answer_prefill:
            args += ["--ollama-answer-prefill"]
        return args


class Watchdog(Record):
    """Bounds on a measured pass. Exceeding either retires the engine and fails the run.

    ``stall_seconds`` bounds time without progress: a request starting or finishing, a
    streamed event, or new engine output. ``request_seconds`` bounds each request's elapsed
    time. Both are disabled unless selected.
    """

    stall_seconds: float | None = Field(default=None, gt=0)
    request_seconds: float | None = Field(default=None, gt=0)

    @property
    def enabled(self) -> bool:
        return self.stall_seconds is not None or self.request_seconds is not None

    def arguments(self) -> list[str]:
        args = []
        if self.stall_seconds is not None:
            args += ["--stall-seconds", str(self.stall_seconds)]
        if self.request_seconds is not None:
            args += ["--request-seconds", str(self.request_seconds)]
        return args


class Schedule(Record):
    """How many measured passes each target gets.

    Unset, the schedule is balanced: every target is measured once per rotation of the target
    order, at least twice. ``passes`` fixes the number of passes instead; with one pass each
    target is measured once, in the order given, with no rotation.
    """

    passes: int | None = Field(default=None, gt=0)

    def blocks(self, targets: int, repeat: int) -> int:
        return self.passes if self.passes is not None else max(2, targets) * repeat

    def arguments(self) -> list[str]:
        return [] if self.passes is None else ["--passes", str(self.passes)]


class EngineOptions(Record):
    native: NativeOptions = Field(default_factory=NativeOptions)
    llama: LlamaOptions = Field(default_factory=LlamaOptions)
    ollama: OllamaOptions = Field(default_factory=OllamaOptions)
    watchdog: Watchdog = Field(default_factory=Watchdog)
    schedule: Schedule = Field(default_factory=Schedule)


DEFAULT_ENGINE_OPTIONS = EngineOptions()
