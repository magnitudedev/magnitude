"""Host-safe prepared media and the model's CPU preparation boundary.

The transport carries validated tensors. Each model adapter accepts its exact
schema and turns it into architecture-owned operands before generation sees it.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

from magnitude_engine.artifacts.source import LocalArtifact

if TYPE_CHECKING:
    import numpy as np

DTYPES = {"float32": 4, "int32": 4, "int64": 8, "uint8": 1, "bool": 1}


@dataclass(frozen=True)
class PreparedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.dtype not in DTYPES
            or (
                not isinstance(self.shape, tuple)
                or not 1 <= len(self.shape) <= 5
                or any(type(n) is not int or not 0 < n < 2**31 for n in self.shape)
                or not isinstance(self.data, bytes)
            )
        ):
            raise ValueError("invalid prepared tensor metadata")
        if prod(self.shape) * DTYPES[self.dtype] != len(self.data):
            raise ValueError("prepared tensor byte count differs from its geometry")

    @classmethod
    def from_array(cls, name: str, array: np.ndarray) -> PreparedTensor:
        import numpy as np

        array = np.asarray(array)
        return cls(name, str(array.dtype), tuple(array.shape), array.tobytes(order="C"))

    def array(self) -> np.ndarray:
        import numpy as np

        return np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape)


@dataclass(frozen=True)
class PreparedMedia:
    processor: str
    tensors: tuple[PreparedTensor, ...]

    def __post_init__(self):
        if (
            not isinstance(self.processor, str)
            or len(self.processor) != 64
            or (
                not isinstance(self.tensors, tuple)
                or not 1 <= len(self.tensors) <= 16
                or any(not isinstance(t, PreparedTensor) for t in self.tensors)
            )
        ):
            raise ValueError("prepared media requires a processor identity and bounded tensors")
        if len({t.name for t in self.tensors}) != len(self.tensors):
            raise ValueError("prepared media repeats tensor names")

    @property
    def buffers(self) -> tuple[bytes, ...]:
        return tuple(t.data for t in self.tensors)

    def encode(self) -> dict:
        """Small control metadata; numerical bytes travel in exact binary buffers."""
        return {
            "processor": self.processor,
            "tensors": [
                {"name": t.name, "dtype": t.dtype, "shape": list(t.shape)} for t in self.tensors
            ],
        }

    @classmethod
    def decode(cls, value: dict, buffers: tuple[bytes, ...]) -> PreparedMedia:
        if (
            not isinstance(value, dict)
            or set(value) != {"processor", "tensors"}
            or not isinstance(value["tensors"], list)
            or not 1 <= len(value["tensors"]) <= 16
            or len(value["tensors"]) != len(buffers)
        ):
            raise ValueError("prepared media fields differ from the protocol")
        tensors = []
        for t, data in zip(value["tensors"], buffers, strict=True):
            if (
                not isinstance(t, dict)
                or set(t) != {"name", "dtype", "shape"}
                or not isinstance(t["shape"], list)
            ):
                raise ValueError("invalid prepared tensor encoding")
            tensors.append(PreparedTensor(t["name"], t["dtype"], tuple(t["shape"]), data))
        return cls(value["processor"], tuple(tensors))

    def identity(self) -> bytes:
        digest = hashlib.sha256(self.processor.encode())
        for t in self.tensors:
            for field in (t.name.encode(), t.dtype.encode(), repr(t.shape).encode(), t.data):
                digest.update(len(field).to_bytes(8, "little"))
                digest.update(field)
        return digest.digest()


class ImagePreparation(ABC):
    """One immutable artifact binding, shared by a template's CPU request work."""

    def __init__(self, artifact: LocalArtifact):
        from transformers.models.auto.image_processing_auto import AutoImageProcessor

        from magnitude_engine.artifacts.identity import processor_identity
        from magnitude_engine.components import component_id

        self.artifact = artifact
        self.processor = AutoImageProcessor.from_pretrained(
            artifact.path, local_files_only=True, trust_remote_code=False, backend="pil"
        )
        self.identity = processor_identity(artifact.directory, component_id(self))

    @abstractmethod
    def process(self, text: str, images: list[Any], tokenizer: Any) -> tuple[str, PreparedMedia]:
        """Interpret images and expand exactly their placeholders in the rendered template."""
