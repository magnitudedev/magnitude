"""Prepared recurrence and resulting state, shared across production implementations."""

from magnitude_engine import components as c
from performance.assembly import Binding, inspect_component
from performance.benchmarks.numerics import compare
from performance.runner import recording


def benchmark(
    component,
    *,
    geometry=None,
    query_tokens=1,
    batch_size=1,
    dtype="bfloat16",
    profile=None,
    output=None,
    warmup=2,
    repetitions=7,
):
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_kernel

    from magnitude_engine.models.recurrence.inputs import DeltaInputs

    if isinstance(component, Binding):
        binding = component
        geometry = geometry or binding.node.parameters
    else:
        if not isinstance(geometry, c.RecurrentGeometry):
            raise TypeError("standalone recurrence requires RecurrentGeometry")
        binding = inspect_component(component, context=geometry).at("component")
    if not isinstance(geometry, c.RecurrentGeometry):
        raise TypeError("recurrence requires RecurrentGeometry")
    if geometry.element_bytes != (4 if dtype == "float32" else 2):
        raise ValueError("recurrence dtype differs from bound geometry")
    workload = {
        "query_tokens": query_tokens,
        "batch_size": batch_size,
        "geometry": geometry.model_dump(mode="json"),
        "dtype": dtype,
        "fixture": "synthetic.normal",
        "seed": 73,
        "numerical_contract": "upstream-delta-atol1e-5-rtol1e-5",
    }
    with recording(
        binding,
        benchmark="recurrence.advance",
        workload=workload,
        profile=profile,
        output=output,
        warmup=warmup,
        repetitions=repetitions,
    ) as run:
        hk, hv, dk, dv = (
            geometry.key_heads,
            geometry.value_heads,
            geometry.key_width,
            geometry.value_width,
        )
        mx.random.seed(73)
        q = mx.random.normal((batch_size, query_tokens, hk, dk)).astype(getattr(mx, dtype)) / dk
        k = mx.random.normal(q.shape).astype(getattr(mx, dtype)) / dk**0.5
        v = mx.random.normal((batch_size, query_tokens, hv, dv)).astype(getattr(mx, dtype))
        decay = mx.random.uniform(shape=(batch_size, query_tokens, hv))
        beta = mx.random.uniform(shape=decay.shape).astype(getattr(mx, dtype))
        state = mx.random.normal((batch_size, hv, dv, dk)) * 0.1
        inputs = DeltaInputs(q, k, v, decay, beta)
        reference = gated_delta_kernel(q, k, v, decay, beta, state)
        mx.eval(q, k, v, decay, beta, state, reference)
        run.measure(
            lambda: binding.instance.advance(inputs, state),
            complete=lambda value: mx.eval(*value),
            validate=lambda value: compare(value, reference),
            deterministic=True,
        )
    return run


def decode(component, *, geometry=None, **options):
    return benchmark(component, geometry=geometry, query_tokens=1, **options)


def lengths(component, *, geometry=None, tokens=(1, 3, 512), **options):
    return [benchmark(component, geometry=geometry, query_tokens=q, **options) for q in tokens]
