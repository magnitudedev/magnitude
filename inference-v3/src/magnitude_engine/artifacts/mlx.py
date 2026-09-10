"""MLX affine Safetensors artifacts. No MLX execution dependency."""

import hashlib
import json
import math
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.artifacts.weights import WeightDescriptor
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.platform.storage import FileSource


@dataclass(frozen=True)
class StoredTensor:
    source: FileSource
    offset: int
    spec: TensorSpec

    def read(self) -> bytes:
        return self.source.read(self.offset, self.spec.nbytes)


class MLXArtifact:
    """Own an immutable directory snapshot and validate encoded tensor ranges.

    Artifact interpretation is host work; all numerical conversion and execution
    uses the selected TileLang device. Floating tensors retain their stored dtype.
    """

    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        with ExitStack() as cleanup:
            config = FileSource(self.path / "config.json")
            cleanup.callback(config.close)
            content = config.read(0, config.size)
            self.config = TypeAdapter(dict[str, object]).validate_json(content)
            quantization = self.config.get("quantization")
            if quantization != {"group_size": 64, "bits": 4, "mode": "affine"}:
                raise ValueError("MLX artifact requires uniform affine Q4 group-64 weights")
            digest = hashlib.sha256(content)
            self.tensors: dict[str, StoredTensor] = {}
            for tensor_path in sorted(self.path.glob("*.safetensors")):
                source = FileSource(tensor_path)
                cleanup.callback(source.close)
                digest.update(tensor_path.name.encode() + b"\x00" + source.digest().encode())
                header_size = struct.unpack("<Q", source.read(0, 8))[0]
                if header_size > 64 * 1024**2 or 8 + header_size > source.size:
                    raise ValueError("invalid Safetensors header extent")
                header = json.loads(source.read(8, header_size))
                ranges = []
                for name, entry in header.items():
                    if name == "__metadata__":
                        continue
                    if name in self.tensors:
                        raise ValueError(f"duplicate Safetensors tensor {name}")
                    dtype = {"U32": DType.U32, "BF16": DType.BF16, "F32": DType.F32}.get(
                        entry["dtype"]
                    )
                    if dtype is None:
                        raise ValueError(
                            f"unsupported Safetensors dtype for {name}: {entry['dtype']}"
                        )
                    shape = tuple(entry["shape"])
                    if any(type(n) is not int or n <= 0 for n in shape):
                        raise ValueError(f"invalid Safetensors shape for {name}")
                    spec = TensorSpec(shape, dtype)
                    start, end = entry["data_offsets"]
                    if (
                        type(start) is not int
                        or type(end) is not int
                        or start < 0
                        or end - start != spec.nbytes
                        or end > source.size - 8 - header_size
                    ):
                        raise ValueError(f"invalid Safetensors range for {name}")
                    ranges.append((start, end))
                    self.tensors[name] = StoredTensor(source, 8 + header_size + start, spec)
                ordered = sorted(ranges)
                if any(b[0] < a[1] for a, b in zip(ordered, ordered[1:], strict=False)):
                    raise ValueError("overlapping Safetensors ranges")
            if not self.tensors:
                raise ValueError("MLX artifact has no tensors")
            self.identity = ArtifactIdentity(digest.hexdigest())
            self._cleanup = cleanup.pop_all()

    def descriptor(self, name: str, shape: tuple[int, ...]) -> WeightDescriptor:
        stored = self.tensors[name]
        if stored.spec.dtype == DType.U32:
            if len(shape) != 2 or shape[1] % 64:
                raise ValueError(f"invalid affine geometry: {name}")
            n, k = shape
            if stored.spec.shape != (n, k // 8) or not name.endswith(".weight"):
                raise ValueError(f"affine code shape differs: {name}")
            for suffix in ("scales", "biases"):
                if self.tensors[name.removesuffix("weight") + suffix].spec != TensorSpec(
                    (n, k // 64), DType.BF16
                ):
                    raise ValueError(f"affine parameter shape/dtype differs: {name}")
        elif math.prod(stored.spec.shape) != math.prod(shape):
            raise ValueError(f"floating parameter shape differs: {name}")
        return WeightDescriptor(name=name, shape=shape)

    def close(self) -> None:
        self._cleanup.close()
