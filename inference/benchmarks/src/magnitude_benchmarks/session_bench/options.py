"""Engine launch selection and supervision bounds chosen on the command line."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from ..fixtures.records import Record
from .policy import project_root


def default_native_binary() -> Path:
    """The release build of the sibling engine workspace: ``inference/target/release``."""
    return project_root().parent / "target" / "release" / "magnitude-engine"


class NativeOptions(Record):
    """Launch selection for the native engine; recorded in the reproduction command."""

    binary: Path = Field(default_factory=default_native_binary)
    device: str = "auto"
    cache_dir: Path | None = None
    method: Literal["auto", "plain", "mtp", "dflash"] = "auto"
    mtp_proposals: int | None = Field(default=None, gt=0)
    #: A separate draft model (DFlash, DSpark) for the target.
    draft: Path | None = None

    @model_validator(mode="after")
    def proposals_need_a_drafter(self) -> Self:
        if self.mtp_proposals is not None and self.method not in ("mtp", "dflash"):
            raise ValueError("--native-mtp-proposals requires --native-method mtp or dflash")
        if self.method == "dflash" and self.draft is None:
            raise ValueError("--native-method dflash requires --native-draft")
        return self

    def arguments(self) -> list[str]:
        """Public CLI flags that reproduce this selection."""
        args = ["--native-binary", str(self.binary), "--native-device", self.device]
        if self.cache_dir is not None:
            args += ["--native-cache-dir", str(self.cache_dir)]
        args += ["--native-method", self.method]
        if self.mtp_proposals is not None:
            args += ["--native-mtp-proposals", str(self.mtp_proposals)]
        if self.draft is not None:
            args += ["--native-draft", str(self.draft)]
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
    watchdog: Watchdog = Field(default_factory=Watchdog)


DEFAULT_ENGINE_OPTIONS = EngineOptions()
