"""The public compilation boundary delegates call semantics to MLX itself."""

from functools import partial
from unittest.mock import patch

import mlx.core as mx
import pytest

from magnitude_engine.kernels.core.compiler import artifact, compile
from magnitude_engine.kernels.reductions.normalization import rms_norm


def test_native_callable_and_unknown_primitive_with_single_trace():
    traced = []

    @compile
    def f(x, w):
        traced.append(1)
        y = rms_norm(x + 1, w)
        return {"scan": mx.cumsum(y, axis=-1), "norm": y, "label": "kept"}

    x, w = mx.ones((2, 128)), mx.ones(128)
    result = f(x, w)
    mx.eval(result)
    assert type(f) is type(mx.compile(lambda x: x))
    assert traced == [1]
    assert [r["backend"] for r in artifact(f)["regions"]] == ["METAL", "MLX"]
    with patch("magnitude_engine.kernels.core.compiler.snapshot", side_effect=AssertionError):
        second = f(x + 1, w)
        mx.eval(second)
    assert traced == [1]
    assert second["label"] == "kept"
    assert mx.allclose(second["scan"], mx.cumsum(second["norm"], axis=-1)).item()


def test_captured_state_and_side_effects_are_not_retraced():
    state = {"value": mx.zeros((1, 128)), "calls": 0}

    @partial(compile, inputs=state, outputs=state)
    def step(x, w):
        state["calls"] += 1
        state["value"] = rms_norm(state["value"] + x, w)
        return state["value"], mx.cumsum(state["value"], axis=-1)

    x, w = mx.ones((1, 128)), mx.ones(128)
    for _ in range(2):
        value, scan = step(x, w)
        mx.eval(value, scan, state)
        assert mx.array_equal(value, state["value"]).item()
    assert state["calls"] == 1


def test_shapeless_and_python_trees_match_mlx():
    def f(x, *, scale=2):
        return {"value": mx.sin(x) * scale, "extra": (None, "constant")}

    ours, native = compile(f, shapeless=True), mx.compile(f, shapeless=True)
    for size in (3, 9):
        x = mx.arange(size).astype(mx.float32)
        assert mx.array_equal(ours(x)["value"], native(x)["value"]).item()
        assert ours(x)["extra"] == native(x)["extra"]


def test_nested_compile_expands_owned_calls():
    inner = compile(lambda x, w: rms_norm(x, w))
    outer = compile(lambda x, w: inner(x + 1, w))
    mx.eval(outer(mx.ones((1, 128)), mx.ones(128)))
    assert len(artifact(outer)["regions"]) == 1


def test_invalid_contract_raises_once():
    calls = []

    @compile
    def f(x, w):
        calls.append(1)
        return rms_norm(x, w)

    with pytest.raises(ValueError, match="weights"):
        f(mx.ones((1, 128)), mx.ones(64))
    assert len(calls) == 1


def test_disable_compile_executes_directly_and_reenable_traces_once():
    calls = []

    @compile
    def f(x, w):
        calls.append(1)
        return rms_norm(x, w)

    x, w = mx.ones((1, 128)), mx.ones(128)
    mx.disable_compile()
    try:
        mx.eval(f(x, w))
        mx.eval(f(x, w))
        assert len(calls) == 2
    finally:
        mx.enable_compile()
    mx.eval(f(x, w))
    mx.eval(f(x, w))
    assert len(calls) == 3


def test_unknown_multioutput_and_repeated_inputs_remain_native():
    @compile
    def f(x, w):
        y = rms_norm(x, w)
        q, r = mx.linalg.qr(y, stream=mx.cpu)
        return q, r, mx.concatenate([y, y], axis=0)

    x = mx.random.normal((128, 128), key=mx.random.key(15))
    w = mx.ones(128)
    q, r, both = f(x, w)
    assert mx.allclose(q @ r, rms_norm(x, w), atol=1e-4).item()
    assert mx.array_equal(both[:128], both[128:]).item()


def test_complex_neighbor_remains_mlx():
    f = compile(lambda x, w: rms_norm(x, w).astype(mx.complex64) * (1 + 2j))
    x, w = mx.ones((1, 128)), mx.ones(128)
    assert mx.allclose(f(x, w), rms_norm(x, w).astype(mx.complex64) * (1 + 2j)).item()
