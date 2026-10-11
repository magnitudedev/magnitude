"""Engine launch selection and supervision bounds chosen on the command line.

The model, its drafter and its artifacts are never options: they are the catalog's
(``models``). Options choose how each engine serves them.
"""

from pathlib import Path
from typing import Literal

from magnitude_fixtures.canonical import Record
from pydantic import Field, model_validator

from .policy import INFERENCE


class SourceBuild(Record):
    """A build identity (contract ``build``): commit plus uncommitted digest."""

    commit: str
    dirty_digest: str | None


class MagnitudeOptions(Record):
    """Our engine (``magnitude-engine``) and the harness that renders with its serving path.

    Any ``magnitude-engine`` binary can be measured, including one built from another commit:
    it is a black box over HTTP, identified by its bytes and, when given, its source build.
    A binary that predates the tuning options is launched without them (``tuning_trace``
    off, no budget or fresh tuning): it tunes with its own fixed budget and leaves no trace.
    """

    binary: Path = INFERENCE / "target" / "release" / "magnitude-engine"
    #: The binary's source build when it is not this tree's.
    build: SourceBuild | None = None
    harness: Path = INFERENCE / "target" / "release" / "magnitude-harness"
    device: str = "auto"
    #: The engine's kernel cache (compiled kernels and stored tuning) exactly as given.
    cache_dir: Path | None = None
    #: A persistent root under which each model and engine binary keeps its own kernel cache
    #: (``<root>/<model>/<binary sha256 prefix>``), as a user's installation keeps one.
    kernel_cache_root: Path | None = None
    prefill_tokens: int | None = Field(default=None, gt=0)
    #: Proposals per draft for the catalog drafter; the engine's own when unset.
    mtp_proposals: int | None = Field(default=None, gt=0)
    #: Exact engine draft-budget JSON, or ``adaptive``, preserved in run identity.
    draft_budget: str | None = None
    #: The engine draft method that drafts with the target's own head (``mtp``) in place of
    #: the catalog drafter.
    method: Literal["mtp"] | None = None
    #: The numerical profile's α (how far an admitted kernel may move a result from its
    #: default's); the engine's standard when unset. A value near zero admits only kernels that
    #: move almost nothing.
    alpha: float | None = Field(default=None, gt=0)
    #: Seconds the first load spends tuning; the engine's standard budget when unset.
    tuning_seconds: float | None = Field(default=None, gt=0)
    #: Tune the run's first launch afresh, ignoring stored results (compiled kernels are kept).
    fresh_tuning: bool = False
    #: Ask the engine for its tuning trace (``--tuning-trace``).
    tuning_trace: bool = True

    @model_validator(mode="after")
    def _tuning_requests_need_the_trace(self):
        if not self.tuning_trace and (self.tuning_seconds is not None or self.fresh_tuning):
            raise ValueError("a tuning budget or fresh tuning needs the tuning trace to record it")
        if self.cache_dir is not None and self.kernel_cache_root is not None:
            raise ValueError("give either a kernel cache directory or a kernel cache root")
        return self


# llama.cpp KV cache element types.
KvCacheType = Literal["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]


class LlamaOptions(Record):
    """Upstream llama-server."""

    binary: Path | None = None
    draft_proposals: int = Field(default=3, gt=0)
    gpu_layers: int = Field(default=99, ge=0)
    cache_type_k: KvCacheType = "f16"
    cache_type_v: KvCacheType = "f16"
    #: Tokens allocated per slot beyond the benchmark context. The context is sized from the
    #: harness's rendering; a template that renders longer in llama.cpp would otherwise end a
    #: request at the context limit. Terminal counts stay authoritative.
    context_headroom: int = Field(default=0, ge=0)


class OllamaOptions(Record):
    """Ollama as shipped, except for the context, which the workload fixes."""

    binary: Path | None = None
    #: Ollama's model store (``OLLAMA_MODELS``): imports are written here, pulls are read here.
    models: Path | None = None
    #: ``OLLAMA_KV_CACHE_TYPE``; Ollama's default is f16.
    kv_cache_type: Literal["f16", "q8_0", "q4_0"] | None = None
    #: ``OLLAMA_FLASH_ATTENTION``; auto leaves the choice to Ollama.
    flash_attention: Literal["auto", "on", "off"] = "auto"
    #: Ollama drafts with its own draft head or draft model, never the catalog drafter.
    #: ``off`` parks its MLX runner's drafting by requesting logprobs, the only switch Ollama
    #: exposes; ``default`` leaves Ollama's drafting on.
    speculation: Literal["off", "default"] = "off"
    #: Tokens allocated beyond the benchmark context. Ollama's llama.cpp runner ends a drafting
    #: request before a context that fits the prompt and output exactly is full.
    context_headroom: int = Field(default=0, ge=0)
    #: Open the reply with the model's answer header. Only for a family whose format has no
    #: switch for thinking (Muse Glimmer): with ``think`` false the model still reasons first
    #: and Ollama discards that text. The prompt Ollama renders is sent through its raw
    #: completion route with the header appended; the adapter refuses any other family.
    answer_prefill: bool = False


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


class Schedule(Record):
    """How many measured passes each engine gets.

    Unset, the schedule is balanced: every engine is measured once per rotation of the
    engine order, at least twice. ``passes`` fixes the number of passes instead; with one
    pass each engine is measured once, in the order given.
    """

    passes: int | None = Field(default=None, gt=0)

    def blocks(self, engines: int) -> int:
        return self.passes if self.passes is not None else max(2, engines)


class EngineOptions(Record):
    #: Serve with the catalog drafter where the engine supports its method.
    drafting: bool = True
    magnitude: MagnitudeOptions = Field(default_factory=MagnitudeOptions)
    llama: LlamaOptions = Field(default_factory=LlamaOptions)
    ollama: OllamaOptions = Field(default_factory=OllamaOptions)
    watchdog: Watchdog = Field(default_factory=Watchdog)
    schedule: Schedule = Field(default_factory=Schedule)
