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


class EngineOptions(Record):
    native: NativeOptions = Field(default_factory=NativeOptions)
    llama: LlamaOptions = Field(default_factory=LlamaOptions)
    watchdog: Watchdog = Field(default_factory=Watchdog)


DEFAULT_ENGINE_OPTIONS = EngineOptions()
