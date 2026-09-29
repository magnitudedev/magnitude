"""A checkpoint source identity; reading metadata never materializes model arrays."""

from dataclasses import dataclass
from pathlib import Path

from .tensors import read_json


@dataclass(frozen=True)
class LocalArtifact:
    path: str

    @property
    def directory(self) -> Path:
        return Path(self.path)

    def configuration(self) -> dict:
        return read_json(self.directory / "config.json")
