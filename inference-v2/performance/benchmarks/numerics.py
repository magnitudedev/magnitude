"""Independent checks and observation digests, outside timed device work."""

import hashlib

from performance.records import Observation


def compare(actual, expected, *, atol=1e-5, rtol=1e-5):
    import mlx.core as mx
    import numpy as np

    actual = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected = expected if isinstance(expected, (tuple, list)) else (expected,)
    h, errors = hashlib.sha256(), []
    for value, reference in zip(actual, expected, strict=True):
        if value.shape != reference.shape:
            raise ValueError("output and reference shapes differ")
        if value.size == 0:
            h.update(str(value.shape).encode())
            continue
        error = mx.max(mx.abs(value.astype(mx.float32) - reference.astype(mx.float32))).item()
        if not mx.allclose(value, reference, atol=atol, rtol=rtol).item():
            raise ValueError(f"numerical contract rejected output: max error {error}")
        errors.append(error)
        h.update(np.asarray(value.astype(mx.float32)).tobytes())
    return Observation(h.hexdigest(), {"max_error": max(errors, default=0)})
