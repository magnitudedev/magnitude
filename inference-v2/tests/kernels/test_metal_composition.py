from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import pytest

from magnitude_engine.kernels import computation, operation
from magnitude_engine.kernels.core import metal

SOURCE = str(Path(__file__).parent / "metal/composition.metal")


def contraction(fold):
    @operation
    def linear(x, weight):
        return x @ weight.T

    @linear.metal(source=SOURCE, function="row_fold" if fold else "row_dot")
    def bind(x, weight):
        if x.ndim != 2 or weight.ndim != 2 or x.dtype != mx.float32:
            return metal.UNSUPPORTED
        domain = metal.Domain(rows=x.shape[0], columns=weight.shape[0])
        row, column = domain.indices
        args = {"x": metal.ReadOnly(x[row, :])}
        if not fold:
            args["weight"] = metal.ReadOnly(weight[column, :])
        args.update(K=metal.UInt(x.shape[1]), lane=metal.Lane())
        result = metal.Replicated((row, column), mx.float32)
        if not fold:
            return metal.TileCall(domain, metal.SIMDGroup(), args, result)
        iteration = metal.Iteration("k", x.shape[1], metal.Lane())
        sample = metal.Sample("input", x[row, iteration.index])
        state = metal.State("sum", mx.float32)
        return metal.OrderedFold(
            domain,
            metal.SIMDGroup(),
            args,
            iteration,
            (sample,),
            (state,),
            {state: metal.Float(0)},
            metal.Call("dot_step", state, sample, metal.Load(weight[column, iteration.index])),
            metal.Call("dot_finish", state),
            result,
        )

    return linear


@pytest.mark.parametrize("fold", [False, True])
def test_automatic_parallel_tiles_scalar_fanout_and_warm_compile(fold, tmp_path):
    linear = contraction(fold)

    @computation
    def region(x, w, u):
        g = linear(x, w)
        h = linear(x, u)
        return g, g * mx.sigmoid(g) * h

    x = mx.random.normal((3, 64), key=mx.random.key(1))
    w = mx.random.normal((8, 64), key=mx.random.key(2))
    u = mx.random.normal((8, 64), key=mx.random.key(3))
    bound = region.specialize(x, w, u)
    assert len(bound.call.regions) == 1
    assert "2 Metal tiles" in bound.explain()
    if fold:
        assert "1 shared ordered drivers" in bound.explain()
        assert bound.call.regions[0].call.source.count("= row_fold(") == 1
    g, actual = bound(x, w, u)
    expected = (x @ w.T) * mx.sigmoid(x @ w.T) * (x @ u.T)
    assert mx.allclose(g, x @ w.T, atol=1e-5).item()
    assert mx.allclose(actual, expected, atol=1e-4).item()
    compiled = mx.compile(bound)
    mx.eval(compiled(x, w, u))
    with patch("magnitude_engine.kernels.core.computation.capture", side_effect=AssertionError):
        mx.eval(compiled(x + 1, w, u))
    import re

    def direct(*arrays):
        values = {v.name: a for v, a in zip(bound.graph.inputs, arrays, strict=True)}
        return bound.call.regions[0].call(
            *(values[v.name] for v in bound.call.regions[0].graph.inputs)
        )

    exported = []
    for call in (compiled, mx.compile(direct)):
        path = tmp_path / "graph.dot"
        mx.export_to_dot(str(path), *call(x, w, u))
        ids = {}

        # DOT uses pointer-derived node IDs; only these addresses are normalized.
        def normalize(match, ids=ids):
            value = match.group(0)
            return ids.setdefault(value, f"node{len(ids)}")

        exported.append(re.sub(r"\b[0-9]{7,}\b", normalize, path.read_text()))
    assert exported[0] == exported[1]


def test_new_function_composes_without_planner_registration():
    @operation
    def bump(x):
        return x + 2

    @bump.metal(source=SOURCE, function="bump")
    def bind(x):
        domain = metal.Domain(item=x.size)
        (i,) = domain.indices
        return metal.TileCall(
            domain, metal.Thread(), {"x": metal.Load(x[i])}, metal.Replicated((i,), x.dtype)
        )

    @computation
    def region(x):
        return mx.tanh(bump(x))

    x = mx.arange(17).astype(mx.float32)
    bound = region.specialize(x)
    assert len(bound.call.regions) == 1
    assert mx.allclose(bound(x), mx.tanh(x + 2)).item()


def test_dependent_dots_retain_global_boundary():
    linear = contraction(False)

    @computation
    def region(x, w):
        return linear(linear(x, w), w)

    x = mx.ones((3, 64))
    w = mx.eye(64)
    bound = region.specialize(x, w)
    assert len(bound.call.regions) == 2
    assert mx.array_equal(bound(x, w), x).item()


def test_local_exchange_and_mlx_between_custom_calls():
    @operation
    def bump(x):
        return x + 2

    @operation
    def paired(x):
        return x - x.reshape(-1, 2)[:, ::-1].reshape(x.shape)

    @bump.metal(source=SOURCE, function="bump")
    def bump_binding(x):
        domain = metal.Domain(item=x.size)
        (i,) = domain.indices
        return metal.TileCall(
            domain, metal.SIMDGroup(), {"x": metal.Load(x[i])}, metal.Distributed((i,), x.dtype)
        )

    @paired.metal(source=SOURCE, function="pair_subtract")
    def paired_binding(x):
        domain = metal.Domain(item=x.size)
        (i,) = domain.indices
        return metal.TileCall(
            domain,
            metal.SIMDGroup(),
            {"x": metal.Load(x[i]), "partner": metal.Load(x[i ^ 1])},
            metal.Distributed((i,), x.dtype),
        )

    @computation
    def region(x):
        return paired(mx.tanh(bump(x)))

    x = mx.arange(64).astype(mx.float32) / 31
    bound = region.specialize(x)
    assert len(bound.call.regions) == 1
    assert "1 local exchanges" in bound.explain()
    expected = mx.tanh(x + 2)
    expected = expected - expected.reshape(-1, 2)[:, ::-1].reshape(x.shape)
    assert mx.allclose(bound(x), expected, atol=1e-6).item()


def test_unavailable_primitive_retains_original_computation_without_recursive_capture():
    @computation
    def child(x):
        return mx.sin(x)

    @computation
    def region(x):
        return child(x).sum(axis=-1)

    x = mx.ones((2, 32))
    bound = region.specialize(x)
    assert "Original MLX computation retained" in bound.explain()
    assert mx.allclose(bound(x), mx.sin(x).sum(axis=-1)).item()


def test_invalid_index_and_collective_layout_rejected_before_launch():
    @operation
    def bump(x):
        return x + 2

    @bump.metal(source=SOURCE, function="bump")
    def binding(x):
        domain = metal.Domain(item=x.size)
        (i,) = domain.indices
        return metal.TileCall(
            domain, metal.Thread(), {"x": metal.Load(x[i + 1])}, metal.Replicated((i,), x.dtype)
        )

    with pytest.raises(ValueError, match="tensor domain"):
        bump(mx.ones(32))
    domain = metal.Domain(item=31)
    with pytest.raises(ValueError, match="32-element"):
        metal.TileCall(domain, metal.SIMDGroup(), {}, metal.Distributed(domain.indices, mx.float32))


def test_plan_artifacts_and_execution_stream_specialization():
    import json

    from performance.assembly import source_key

    linear = contraction(False)

    @computation
    def region(x, w):
        return mx.tanh(linear(x, w))

    x, w = mx.ones((2, 64)), mx.eye(64)
    first = region.specialize(x, w)
    artifact = first.artifact()
    assert artifact["regions"][0]["source"]
    assert artifact["regions"][0]["sources"]
    assert "safe" in json.dumps(artifact)
    key, sources = source_key((first,))
    assert key and any(path.endswith(".metal") for path in sources)
    stream = mx.new_stream(mx.gpu)
    with mx.stream(stream):
        second = region.specialize(x, w)
        assert second is not first
        assert mx.allclose(second(x, w), mx.tanh(x)).item()
        with pytest.raises(ValueError, match="stream"):
            first(x, w)


def test_default_planner_respects_its_resource_budget():
    from magnitude_engine.kernels.core.scheduling import Automatic

    linear = contraction(False)

    @computation
    def region(x, w):
        return linear(x, w) * mx.sigmoid(linear(x + 1, w))

    x, w = mx.ones((2, 64)), mx.eye(64)
    bound = region.with_schedule(Automatic(max_connections=1, max_region_nodes=2)).specialize(x, w)
    assert mx.allclose(bound(x, w), x * mx.sigmoid(x + 1)).item()


def test_explicit_inference_and_new_cooperative_driver_use_same_planner():
    @operation
    def transform(x):
        raise AssertionError("explicit output inference must not run the reference")

    @transform.infer
    def infer(x):
        return metal.Tensor(x.shape, x.dtype)

    @transform.metal(source=SOURCE, function="row_transform")
    def bind(x):
        return metal.RowTransform(
            metal.Tensor(x.shape, x.dtype),
            x.value,
            metal.Threadgroup(32),
            dict(row=metal.GroupPosition("y"), tid=metal.ThreadPosition()),
            (x.shape[-1], 32),
        )

    @computation
    def region(x):
        return mx.tanh(transform(x + 3))

    x = mx.random.normal((2, 64), key=mx.random.key(93))
    assert mx.allclose(transform(x), x * 2 + 1).item()
    bound = region.specialize(x)
    assert len(bound.call.regions) == 1
    assert mx.allclose(bound(x), mx.tanh((x + 3) * 2 + 1)).item()
    with pytest.raises(RuntimeError, match="before specialization"):
        transform.infer(infer)


def test_interface_result_must_match_declared_operation():
    @operation
    def bump(x):
        return x + 2

    @bump.metal(source=SOURCE, function="bump")
    def bind(x):
        domain = metal.Domain(item=x.size // 2)
        (i,) = domain.indices
        return metal.TileCall(
            domain, metal.Thread(), {"x": metal.Load(x[i])}, metal.Replicated((i,), x.dtype)
        )

    with pytest.raises(ValueError, match="output differs"):
        bump(mx.ones(32))


def test_prebound_computation_expands_into_enclosing_graph():
    @computation
    def child(x):
        return mx.tanh(x)

    x = mx.ones((32,))
    prepared = child.specialize(x)

    @computation
    def parent(x):
        return prepared(x) * x

    bound = parent.specialize(x)
    assert len(bound.call.regions) == 1
    assert mx.array_equal(bound(x), mx.tanh(x) * x).item()


def test_export_serialization_failure_retains_original_mlx():
    @computation
    def region(x):
        return mx.contiguous(x) + 2

    x = mx.ones((2, 32)).T
    bound = region.specialize(x)
    assert "Original MLX computation retained" in bound.explain()
    assert mx.array_equal(bound(x), x + 2).item()
