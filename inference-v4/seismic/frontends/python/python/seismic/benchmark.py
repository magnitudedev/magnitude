"""Measure synchronous, prepared Seismic calls."""

import statistics
from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class BenchmarkResult:
    samples: tuple[float, ...]
    warmup: int
    function: str
    device: str
    build_profile: str
    endpoint: str = "host call through native completion and Python result construction"

    @property
    def repeat(self):
        return len(self.samples)

    @property
    def median(self):
        return statistics.median(self.samples)

    @property
    def quartiles(self):
        return tuple(float(x) for x in np.quantile(self.samples, [0.25, 0.75]))


def benchmark(kernel, *, args=(), kwargs=None, setup=None, warmup=3, repeat=20):
    from . import Kernel, Tensor, _Move, _natural

    if not isinstance(kernel, Kernel):
        raise TypeError("benchmark requires a prepared kernel")
    warmup, repeat = _natural(warmup), _natural(repeat)
    if repeat == 0:
        raise ValueError("repeat must be positive")
    kwargs = {} if kwargs is None else kwargs
    if setup is not None and (args or kwargs):
        raise ValueError("setup is the sole argument provider")

    def writable(ty):
        return (ty["kind"] == "tensor" and ty["access"] != "shared") or (
            ty["kind"] == "tuple" and any(writable(t) for t in ty["items"])
        )

    if setup is None and any(
        writable(p["type"]) for p in kernel.function._schema["parameters"]
    ):
        raise ValueError(
            "mutable and owned inputs require setup to reset each invocation"
        )

    def metadata(v):
        if isinstance(v, _Move):
            return ("move", metadata(v.tensor))
        if isinstance(v, Tensor):
            if not v._inner.same_device(kernel.device._inner):
                raise ValueError("benchmark tensor device mismatch")
            return ("tensor", v.shape, v.element)
        if isinstance(v, tuple):
            return tuple(metadata(x) for x in v)
        return (type(v), repr(v))

    samples, baseline = [], None
    for i in range(warmup + repeat):
        call_args, call_kwargs = setup() if setup is not None else (args, kwargs)
        bound = kernel.signature.bind(*call_args, **call_kwargs)
        current = tuple((n, metadata(v)) for n, v in bound.arguments.items())
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise ValueError("argument metadata changed across benchmark samples")
        start = perf_counter()
        result = kernel(*call_args, **call_kwargs)
        elapsed = perf_counter() - start
        del result
        if i >= warmup:
            samples.append(elapsed)
    return BenchmarkResult(
        tuple(samples),
        warmup,
        kernel.__name__,
        kernel.device.name,
        kernel.preparation_report["build_profile"],
    )
