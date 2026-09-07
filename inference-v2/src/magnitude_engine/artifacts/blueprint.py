from pathlib import Path

from magnitude_engine.composition import Blueprint, blueprint

from .source import LocalArtifact


@blueprint
class Local(Blueprint[LocalArtifact]):
    path: str

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise ValueError("model artifacts require an absolute local path")

    @staticmethod
    def implementation() -> type[LocalArtifact]:
        from .source import LocalArtifact

        return LocalArtifact
