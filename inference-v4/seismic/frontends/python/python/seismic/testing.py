"""Explicit host assertions using Seismic's numerical comparison owner."""

import json

import numpy as np


def assert_close(
    actual,
    expected,
    *,
    rtol=0.0,
    atol=0.0,
    relative_floor=0.0,
    ulps=None,
    check_dtype=True,
    equal_nan=False,
    preserve_signed_zero=True,
):
    from . import _NUMPY_DTYPES, Precision, Tensor, Tolerance, _invoke, _native

    tolerance = Tolerance(
        atol=atol, rtol=rtol, relative_floor=relative_floor, ulps=ulps
    )
    config = json.dumps({"precision": Precision.bounded(tolerance)._config()})

    def compare(a, e, path):
        if isinstance(a, tuple) or isinstance(e, tuple):
            if not isinstance(a, tuple) or not isinstance(e, tuple) or len(a) != len(e):
                raise AssertionError(f"{path}: tuple structure mismatch")
            for i, (x, y) in enumerate(zip(a, e)):
                compare(x, y, f"{path}[{i}]")
            return
        if a is None or e is None:
            if a is not None or e is not None:
                raise AssertionError(f"{path}: unit mismatch")
            return
        if isinstance(a, range) or isinstance(e, range):
            if type(a) is not type(e) or (a.start, a.stop, a.step) != (
                e.start,
                e.stop,
                e.step,
            ):
                raise AssertionError(f"{path}: range mismatch")
            return
        a = a.numpy() if isinstance(a, Tensor) else np.asarray(a)
        e = e.numpy() if isinstance(e, Tensor) else np.asarray(e)
        context = f"{path}: actual {a.dtype} {a.shape}, expected {e.dtype} {e.shape}"
        if a.shape != e.shape:
            raise AssertionError(context + "; shape mismatch")
        if check_dtype and a.dtype != e.dtype:
            raise AssertionError(context + "; dtype mismatch")
        if not a.size:
            return
        if a.dtype.kind in "biu" or e.dtype.kind in "biu":
            # Python scalar equality preserves full integer precision, including mixed signedness.
            mask = np.array(
                [x.item() == y.item() for x, y in zip(a.flat, e.flat)]
            ).reshape(a.shape)
            metrics = "exact integer/boolean comparison"
        else:
            if a.dtype.kind not in "f" and a.dtype != _NUMPY_DTYPES["bf16"]:
                raise TypeError(context + "; unsupported values")
            if e.dtype.kind not in "f" and e.dtype != _NUMPY_DTYPES["bf16"]:
                raise TypeError(context + "; unsupported reference")
            name = next((n for n, d in _NUMPY_DTYPES.items() if d == a.dtype), None)
            if ulps is not None and (name is None or a.dtype != e.dtype):
                raise TypeError(
                    context
                    + "; ULP comparison requires matching supported storage dtypes"
                )
            rows = _invoke(
                _native.compare,
                a.astype(float).ravel().tolist(),
                e.astype(float).ravel().tolist(),
                name or "f32",
                config,
                equal_nan,
                preserve_signed_zero,
            )
            mask = np.array([r[0] for r in rows]).reshape(a.shape)
            metrics = (
                f"max absolute={max(r[1] for r in rows)}, relative={max(r[2] for r in rows)}, "
                f"ULPs={max(r[3] for r in rows) if name else 'unavailable'}, specials={sum(r[4] for r in rows)}"
            )
        if not np.all(mask):
            flat = int(np.flatnonzero(~mask)[0])
            coordinate = np.unravel_index(flat, a.shape)
            raise AssertionError(
                f"{context}; mismatched={np.count_nonzero(~mask)}/{a.size}, first={coordinate}; {metrics}"
            )

    compare(actual, expected, "result")


from dataclasses import dataclass


class UnsupportedCheck(RuntimeError):
    pass


class CheckResourceLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckReport:
    status: str
    diagnostic: str
    work_units: int

    def assert_passed(self):
        if self.status == "passed":
            return
        if self.status == "failed":
            raise AssertionError(self.diagnostic)
        if self.status == "resource_limit":
            raise CheckResourceLimit(self.diagnostic)
        raise UnsupportedCheck(self.diagnostic)


def check(
    kernel,
    *,
    args=(),
    kwargs=None,
    comparison=None,
    memory_bytes=268435456,
    work_limit=1000000,
):
    from . import Kernel, Precision, _invoke, _natural

    if not isinstance(kernel, Kernel):
        raise TypeError("check requires a prepared kernel")
    policy = comparison or (Precision.exact() if kernel.native else kernel.precision)
    if policy.kind == "unconstrained":
        raise ValueError("check requires an explicit exact or bounded comparison")
    encoded = kernel._arguments(args, kwargs or {})
    config = json.dumps({"precision": policy._config()}, allow_nan=False)
    return CheckReport(
        *_invoke(
            kernel._inner.check,
            encoded,
            config,
            _natural(memory_bytes),
            _natural(work_limit),
        )
    )
