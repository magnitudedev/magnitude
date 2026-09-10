"""Independent host-side storage rounding for benchmark references only."""

import numpy as np

from magnitude_engine.platform.execution import DType


def encode(values: np.ndarray, dtype: DType) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    if dtype == DType.F32:
        return values.tobytes()
    if dtype != DType.BF16:
        raise ValueError("reference storage dtype is unsupported")
    bits = values.view(np.uint32)
    # Round to nearest, ties to even. Preserve NaNs when payload low bits vanish.
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    rounded = np.where(np.isnan(values), (bits >> 16).astype(np.uint16) | 0x40, rounded)
    return rounded.astype(np.uint16).tobytes()


def decode(content: bytes, dtype: DType) -> np.ndarray:
    if dtype == DType.F32:
        return np.frombuffer(content, np.float32)
    if dtype != DType.BF16:
        raise ValueError("reference storage dtype is unsupported")
    return (np.frombuffer(content, np.uint16).astype(np.uint32) << 16).view(np.float32)


def rounded(values: np.ndarray, dtype: DType) -> np.ndarray:
    return decode(encode(values, dtype), dtype).reshape(values.shape)
