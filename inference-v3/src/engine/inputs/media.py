"""Immutable host tensors crossing preparation and execution ownership.

Numerical bytes stay binary. No processor, image object or device resource is
retained by these records, and their NumPy views cannot mutate the payload.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import prod

import numpy as np

_DTYPES = {"float32": "<f4", "int32": "<i4", "int64": "<i8", "uint8": "u1"}
MAX_PREPARED_BYTES = 512 << 20


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
            or self.dtype not in _DTYPES
            or not isinstance(self.shape, tuple)
            or not 1 <= len(self.shape) <= 5
            or any(type(n) is not int or not 0 < n < 2**31 for n in self.shape)
            or not isinstance(self.data, bytes)
        ):
            raise ValueError("invalid prepared tensor metadata")
        size = prod(self.shape) * np.dtype(_DTYPES[self.dtype]).itemsize
        if size != len(self.data) or size > MAX_PREPARED_BYTES:
            raise ValueError("prepared tensor bytes differ from geometry or exceed capacity")

    @classmethod
    def from_array(cls, name: str, value: np.ndarray):
        value = np.asarray(value)
        dtype = value.dtype.name
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported prepared tensor dtype: {value.dtype}")
        return cls(name, dtype, tuple(value.shape), value.astype(_DTYPES[dtype]).tobytes())

    def array(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype=_DTYPES[self.dtype]).reshape(self.shape)


@dataclass(frozen=True)
class PreparedMedia:
    processor: str
    tensors: tuple[PreparedTensor, ...]

    def __post_init__(self):
        if (
            not isinstance(self.processor, str)
            or len(self.processor) != 64
            or any(c not in "0123456789abcdef" for c in self.processor)
            or not isinstance(self.tensors, tuple)
            or not 1 <= len(self.tensors) <= 16
            or any(not isinstance(t, PreparedTensor) for t in self.tensors)
        ):
            raise ValueError("prepared media requires a processor identity and bounded tensors")
        if len({t.name for t in self.tensors}) != len(self.tensors):
            raise ValueError("prepared media repeats tensor names")
        if self.nbytes > MAX_PREPARED_BYTES:
            raise ValueError("prepared media exceeds capacity")

    @property
    def nbytes(self) -> int:
        return sum(len(t.data) for t in self.tensors)

    @property
    def identity(self) -> str:
        digest = sha256(self.processor.encode())
        for tensor in self.tensors:
            for field in (
                tensor.name.encode(),
                tensor.dtype.encode(),
                repr(tensor.shape).encode(),
                tensor.data,
            ):
                digest.update(len(field).to_bytes(8, "little"))
                digest.update(field)
        return digest.hexdigest()
