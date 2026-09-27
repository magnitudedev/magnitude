"""The single MLX construction boundary; plans derive signatures and launch bindings."""

from functools import lru_cache
from hashlib import sha256
from typing import Any

import mlx.core as mx


@lru_cache(maxsize=256)
def kernel_name(
    source: str, inputs: tuple[str, ...], outputs: tuple[str, ...], header: str = ""
) -> str:
    identity = repr((source, inputs, outputs, header))
    return "magnitude_generated_" + sha256(identity.encode()).hexdigest()[:20]


@lru_cache(maxsize=256)
def generated_kernel(
    source: str, inputs: tuple[str, ...], outputs: tuple[str, ...], header: str = ""
) -> Any:
    return mx.fast.metal_kernel(
        name=kernel_name(source, inputs, outputs, header),
        input_names=list(inputs),
        output_names=list(outputs),
        source=source,
        header=header,
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
    )


def execution_context() -> str:
    # MLX 0.32 streams compare by value but inherit an object-identity hash. The
    # public representation includes device and stream index; qualify it with the
    # export adapter rather than using private fields or the inconsistent hash.
    return str(mx.default_stream(mx.default_device()))
