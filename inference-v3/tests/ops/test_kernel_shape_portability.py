"""Numerical boundary cases that must not inherit physical schedule extents."""

from contextlib import contextmanager

import gguf
import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from engine.weights.descriptor import WeightDescriptor
from engine.weights.formats.gguf import Encoding
from engine.weights.tensor_residency import TensorWeights
from ops.lab.ownership import exclusive_measurement
from tests.ops.definitions import implement
from tests.ops.test_tilelang_runtime import _packed, _QuantizedFormat


@pytest.fixture
def device():
    backend = "cuda" if torch.cuda.is_available() else "metal"
    if backend == "metal" and not torch.backends.mps.is_available():
        pytest.skip("requires a native TileLang device")
    with (
        exclusive_measurement(),
        ops.DeviceRuntime.open(
            DevicePlan.discover(backend=backend, maximum_bytes=1 << 29)
        ) as result,
    ):
        yield result


@contextmanager
def packed_weight(device, shape):
    width = shape[-1]
    rows = int(np.prod(shape[:-1]))
    content = _packed(Encoding.Q8_0, rows, width)
    logical = gguf.dequantize(content.reshape(rows, -1), gguf.GGMLQuantizationType.Q8_0).reshape(
        shape
    )
    weights = TensorWeights(_QuantizedFormat(content.tobytes(), Encoding.Q8_0), device)
    resource = device.resolve(
        weights.bind(WeightDescriptor(name="weight", shape=shape), ops.DType.F16)
    )
    try:
        yield resource, logical
    finally:
        resource.close()
        weights.close()


def check(device, function, arrays, *, packed=None, mode="prefill", atol=3e-5, rtol=3e-4):
    packed = packed or {}
    specs = tuple(
        packed[i].spec
        if i in packed
        else ops.TensorSpec(
            a.shape,
            ops.DType.I32
            if a.dtype == np.int32
            else ops.DType.F16
            if a.dtype == np.float16
            else ops.DType.F32,
        )
        for i, a in enumerate(arrays)
    )
    signature = ops.Signature(tuple(ops.Argument(s, f"v{i}") for i, s in enumerate(specs)))
    graph = ops.trace(function, signature)
    expected = ops.evaluate_reference(graph, {f"v{i}": a for i, a in enumerate(arrays)}).outputs
    inputs = []
    compiled = execution = None
    try:
        for i, (spec, value) in enumerate(zip(specs, arrays, strict=True)):
            inputs.append(packed[i].fork() if i in packed else device.upload(spec, value.tobytes()))
        compiled = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode=mode),
        )
        execution = compiled.submit(*inputs)
        execution.completion.wait()
        assert len(execution.outputs) == len(expected)
        for actual, reference in zip(execution.outputs, expected, strict=True):
            np.testing.assert_allclose(actual.native.cpu().numpy(), reference, atol=atol, rtol=rtol)
    finally:
        if execution:
            for output in execution.outputs:
                output.close()
        if compiled:
            compiled.close()
        for resource in inputs:
            resource.close()


@pytest.mark.device
@pytest.mark.parametrize(
    "width,offsets",
    [(6, [0, 1, 4]), (6, [0, 0, 4]), (6, [0, 3, 3]), (1025, [0, 1]), (1025, [0, 0, 1])],
)
def test_recurrent_prepare_covers_channels_and_empty_sequences(device, width, offsets):
    rng = np.random.default_rng(81)
    rows, batch, channels = offsets[-1], len(offsets) - 1, 4 * width
    shapes = (
        (rows, channels),
        (channels, 3),
        (batch, channels, 2),
        (rows, 2),
        (rows, 2),
        (2,),
        (2,),
    )
    arrays = tuple(rng.normal(0, 0.1, s).astype(np.float32) for s in shapes) + (
        np.array(offsets, np.int32),
    )
    check(
        device,
        lambda *x: ops.recurrent_prepare(
            *x, key_heads=1, value_heads=2, width=width, convolution_width=3, epsilon=1e-6
        ),
        arrays,
    )


@pytest.mark.device
@pytest.mark.parametrize("width,axis", [(4095, -1), (4097, -1), (8193, 0)])
def test_softmax_empty_partitions_are_neutral(device, width, axis):
    x = np.full((2, width), -np.inf, np.float32)
    x[0, -1] = 0
    x[1, 1] = 1
    x[1, -1] = -1
    if axis == 0:
        x = x.T.copy()
    check(device, lambda x: ops.softmax(x, axis=axis), (x,), atol=1e-6)


@pytest.mark.device
@pytest.mark.parametrize("shape", [(17,), (2, 17), (2, 3, 17)])
def test_dense_normalization_and_projection_accept_leading_ranks(device, shape):
    rng = np.random.default_rng(93)
    x = rng.normal(0, 0.1, shape).astype(np.float32)
    w = rng.normal(0, 0.1, (9, 17)).astype(np.float32)
    check(device, lambda x: ops.rms_norm(x), (x,))
    check(device, lambda x, w: ops.linear(x, w), (x, w))


@pytest.mark.device
@pytest.mark.parametrize("shape", [(32,), (1, 64), (8, 128), (2, 3, 288)])
def test_packed_projection_alignment_is_not_the_vector_work_tile(device, shape):
    x = np.random.default_rng(19).normal(0, 0.1, shape).astype(np.float32)
    with packed_weight(device, (17, shape[-1])) as (weight, logical):
        check(device, lambda x, w: ops.linear(x, w), (x, logical), packed={1: weight})


@pytest.mark.device
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (1, 1, 2)])
def test_packed_embedding_accepts_index_rank(device, shape):
    ids = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    with packed_weight(device, (4, 32)) as (weight, logical):
        check(
            device,
            lambda ids, table: ops.embedding(ids, table),
            (ids, logical),
            packed={1: weight},
            atol=2e-3,
            rtol=2e-3,
        )


@pytest.mark.device
@pytest.mark.parametrize("rows,experts", [(1, 256), (16, 256), (32, 256), (64, 512)])
def test_moe_prefill_covers_sparse_routes_and_large_banks(device, rows, experts):
    from contextlib import ExitStack

    rng = np.random.default_rng(74)
    width, selected = 256, 8
    x = rng.normal(0, 0.02, (rows, width)).astype(np.float16)
    routes = rng.integers(0, experts, (rows, selected), dtype=np.int32)
    scores = np.full((rows, selected), 1 / selected, np.float32)
    with ExitStack() as stack:
        banks = [
            stack.enter_context(packed_weight(device, (experts, width, width))) for _ in range(3)
        ]
        check(
            device,
            lambda *x: ops.routed_experts(*x),
            (x, routes, scores, *(b[1] for b in banks)),
            packed={i + 3: b[0] for i, b in enumerate(banks)},
            atol=3e-3,
            rtol=3e-2,
        )


@pytest.mark.device
@pytest.mark.parametrize(
    "experts,scoring,normalize",
    [
        (1025, "softmax", True),
        (2049, "softmax", False),
        (1025, "sigmoid", True),
        (1025, "sigmoid", False),
    ],
)
def test_routing_covers_more_experts_than_threads(device, experts, scoring, normalize):
    x = np.random.default_rng(541).normal(0, 1, (2, experts)).astype(np.float32)
    x[1] = 0  # Stable cutoff ties select the largest IDs, in ascending order.
    check(device, lambda x: ops.route_topk(x, k=3, scoring=scoring, normalize=normalize), (x,))


@pytest.mark.device
@pytest.mark.parametrize(
    "width,encoded", [(16, False), (16, True), (96, True), (192, True), (256, True)]
)
def test_kv_word_ownership_is_independent_of_channel_ownership(device, width, encoded):
    from ops.kv import dense_kv
    from ops.kv_codecs import decode_kv_reference, encode_kv_reference

    rep = (
        ops.affine_k8_uniform_v4(width, width) if encoded else dense_kv(width, width, ops.DType.F16)
    )
    state_spec = ops.kv_state_spec(5, 2, ops.DType.F16, rep)
    rng = np.random.default_rng(409)
    history = rng.normal(0, 0.3, state_spec.shape).astype(np.float32)
    physical = encode_kv_reference(history, rep)
    history = decode_kv_reference(physical, state_spec)
    k, v = [rng.normal(0, 0.3, (3, 2, width)).astype(np.float16) for _ in range(2)]
    destinations = np.array([0, -1, 4], np.int32)
    specs = (
        ops.TensorSpec(k.shape, ops.DType.F16),
        ops.TensorSpec(v.shape, ops.DType.F16),
        ops.TensorSpec(destinations.shape, ops.DType.I32),
    )
    signature = ops.Signature(
        (
            ops.Argument(state_spec, "state", ops.ValueKind.RESOURCE),
            *(
                ops.Argument(s, n)
                for s, n in zip(specs, ("keys", "values", "destinations"), strict=True)
            ),
        )
    )

    def function(state, keys, values, destinations):
        return ops.kv_append(state, keys, values, destinations)

    graph = ops.trace(function, signature)
    expected = ops.evaluate_reference(
        graph, dict(state=history, keys=k, values=v, destinations=destinations)
    ).outputs[0]
    inputs = []
    state = program = execution = None
    try:
        state = device.upload(state_spec, physical)
        inputs = [
            device.upload(s, a.tobytes()) for s, a in zip(specs, (k, v, destinations), strict=True)
        ]
        program = ops.compile(
            function,
            signature=signature,
            device=device,
            constants={},
            options=ops.CompileOptions(mode="prefill"),
        )
        execution = program.submit(*inputs, resources={"state": state})
        execution.completion.wait()
        actual = decode_kv_reference(device.read(execution.outputs[0]), state_spec)
        np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
    finally:
        if execution:
            for output in execution.outputs:
                output.close()
        if program:
            program.close()
        for resource in inputs:
            resource.close()
        if state:
            state.close()


@pytest.mark.device
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_dense_expert_banks_cover_uneven_widths(device, mode):
    rng = np.random.default_rng(449)
    rows, experts, width, intermediate, selected = 3, 8, 17, 31, 2
    arrays = (
        rng.normal(0, 0.1, (rows, width)).astype(np.float32),
        rng.integers(0, experts, (rows, selected), dtype=np.int32),
        np.full((rows, selected), 0.5, np.float32),
        *(
            rng.normal(0, 0.1, shape).astype(np.float32)
            for shape in (
                (experts, intermediate, width),
                (experts, intermediate, width),
                (experts, width, intermediate),
            )
        ),
    )
    check(device, lambda *x: ops.routed_experts(*x), arrays, mode=mode)


def _recurrent_output(mixed, norm, gate, weight):
    rows, heads, width = mixed.shape
    normalized = ops.reshape(ops.rms_norm(mixed, norm), (rows, heads * width))
    return ops.linear(normalized * ops.silu(gate), weight)


_recurrent_output = implement(_recurrent_output, ops.operation_bodies.recurrent_mixer)


@pytest.mark.device
@pytest.mark.parametrize("width", [256, 1280])
def test_fused_recurrent_output_covers_every_channel(device, width):
    rng = np.random.default_rng(818)
    arrays = (
        rng.normal(0, 0.1, (2, 1, width)).astype(np.float32),
        rng.normal(1, 0.1, (width,)).astype(np.float32),
        rng.normal(0, 0.1, (2, width)).astype(np.float32),
    )
    with packed_weight(device, (17, width)) as (weight, logical):
        check(device, _recurrent_output, (*arrays, logical), packed={3: weight})
