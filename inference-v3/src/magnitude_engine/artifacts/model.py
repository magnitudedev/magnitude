"""Owned immutable GGUF artifact, with portable directory and content identity."""

from pathlib import Path

from magnitude_engine.artifacts.gguf import read_directory
from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.platform.storage import FileSource


class GGUFArtifact:
    def __init__(self, path: str):
        self.source = FileSource(Path(path))
        try:
            self.directory = read_directory(self.source)
            self.identity = ArtifactIdentity(self.source.digest())
        except BaseException:
            self.source.close()
            raise

    def close(self) -> None:
        self.source.close()
