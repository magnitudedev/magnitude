"""Translate storage encodings into logical tensors before choosing residency."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .tensors import DTYPE_BYTES, FileSlice, TensorCatalog, TensorRegion


@dataclass(frozen=True)
class LogicalTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    pieces: tuple[FileSlice, ...]

    def __post_init__(self) -> None:
        if (
            self.dtype not in DTYPE_BYTES
            or any(n < 0 for n in self.shape)
            or self.nbytes != sum(piece.size for piece in self.pieces)
        ):
            raise ValueError("logical tensor geometry disagrees with physical storage")

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]

    def row(self, index: int) -> tuple[FileSlice, ...]:
        if not self.shape or not 0 <= index < self.shape[0]:
            raise IndexError("logical tensor row is out of bounds")
        width = self.nbytes // self.shape[0]
        start, end = index * width, (index + 1) * width
        position, result = 0, []
        for piece in self.pieces:
            lo, hi = max(start, position), min(end, position + piece.size)
            if lo < hi:
                result.append(piece.slice(lo - position, hi - lo))
            position += piece.size
            if position >= end:
                break
        return tuple(result)


@dataclass(frozen=True)
class ExpertSchema:
    """Architecture-owned names and component order, without a transport implementation."""

    layer_prefix: str
    module: str
    components: tuple[tuple[str, str], ...]
    expert_count: int

    def match(self, name: str) -> re.Match[str] | None:
        return re.fullmatch(
            rf"{re.escape(self.layer_prefix)}\.(?P<layer>[0-9]+)\."
            rf"{re.escape(self.module)}\.(?P<projection>[^.]+)\.(?P<component>[^.]+)",
            name,
        )


def logical_tensors(
    catalog: TensorCatalog, *, declaration: dict | None, expert_schema: ExpertSchema | None = None
) -> dict[str, LogicalTensor]:
    split_pattern = re.compile(
        r"(?P<base>.+)\.experts\.(?P<expert>0|[1-9][0-9]*)\."
        r"(?P<projection>[^.]+)\.(?P<component>[^.]+)"
    )
    expert_major = declaration is not None
    if expert_major:
        required = {
            "format": "expert_major_safetensors",
            "version": 1,
            "expert_axis": 0,
            "key_template": "{base}.experts.{expert}.{projection}.{component}",
            "physical_order": "layer_expert_component",
            "group_must_not_cross_shards": True,
        }
        if (
            expert_schema is None
            or not isinstance(declaration, dict)
            or any(
                type(declaration.get(key)) is not type(value) or declaration.get(key) != value
                for key, value in required.items()
            )
        ):
            raise ValueError("unsupported expert-major declaration or missing architecture schema")
    for metadata in catalog.metadata.values():
        marker, version = (
            metadata.get("expert_storage_format"),
            metadata.get("expert_storage_version"),
        )
        if (marker is not None or version is not None) and (
            not expert_major or marker != "expert_major_safetensors" or version != "1"
        ):
            raise ValueError("artifact and shard storage declarations disagree")
    logical: dict[str, LogicalTensor] = {}
    groups: dict[str, dict[int, dict[tuple[str, str], TensorRegion]]] = {}
    for name, region in catalog.tensors.items():
        split = split_pattern.fullmatch(name)
        logical_name = (
            f"{split['base']}.{split['projection']}.{split['component']}" if split else name
        )
        routed = expert_schema is not None and expert_schema.match(logical_name) is not None
        if not expert_major:
            if split and routed:
                raise ValueError("expert-major tensors require an explicit format declaration")
            logical[name] = LogicalTensor(name, region.shape, region.dtype, (region.storage,))
        elif split:
            if not routed:
                raise ValueError("split expert tensor is outside the architecture schema")
            groups.setdefault(split["base"], {}).setdefault(int(split["expert"]), {})[
                split["projection"], split["component"]
            ] = region
        else:
            if routed:
                raise ValueError("mixed stacked and split expert tensors")
            logical[name] = LogicalTensor(name, region.shape, region.dtype, (region.storage,))
    if not expert_major:
        return logical
    assert expert_schema is not None
    if not groups:
        raise ValueError("expert-major artifact contains no declared routed tensors")
    for base, experts in groups.items():
        if set(experts) != set(range(expert_schema.expert_count)):
            raise ValueError("missing or nonconsecutive expert IDs")
        for expert in experts.values():
            if set(expert) != set(expert_schema.components):
                raise ValueError("incomplete expert record")
            ordered = [expert[component].storage for component in expert_schema.components]
            for left, right in zip(ordered[:-1], ordered[1:], strict=True):
                if left.path != right.path or left.offset + left.size != right.offset:
                    raise ValueError("expert record must be contiguous within one shard")
        for projection, component in expert_schema.components:
            pieces = [
                experts[index][projection, component] for index in range(expert_schema.expert_count)
            ]
            first = pieces[0]
            if any(piece.shape != first.shape or piece.dtype != first.dtype for piece in pieces):
                raise ValueError("experts disagree on component geometry")
            name = f"{base}.{projection}.{component}"
            logical[name] = LogicalTensor(
                name,
                (len(pieces), *first.shape),
                first.dtype,
                tuple(piece.storage for piece in pieces),
            )
    return logical
