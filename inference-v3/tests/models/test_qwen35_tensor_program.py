
from tests.ops.target_fixture import matrix_query
from dataclasses import replace

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from engine.models.qwen35.description import (
    AttentionWeights,
    BlockWeights,
    DenseDescription,
    DenseFeedForwardWeights,
    ExpertGeometry,
    Geometry,
    HeadMapping,
    MixerKind,
    RecurrentWeights,
    RoutedFeedForwardWeights,
)
from engine.models.qwen35.tensor_program import InvocationSpecs, TensorProgram
from engine.weights.descriptor import WeightDescriptor
from engine.weights.identity import ArtifactIdentity
from tests.ops.test_compiler import Runtime


def _weight(name, shape):
    return WeightDescriptor(name=name, shape=shape)


def _description():
    g = Geometry(
        activation_dtype=ops.DType.F16,
        hidden=8,
        intermediate=16,
        vocabulary=32,
        context_limit=16,
        layers=(MixerKind.ATTENTION, MixerKind.RECURRENT),
        attention_heads=2,
        kv_heads=1,
        attention_width=4,
        rotary_width=4,
        rotary_base=10_000,
        rotary_sections=(1, 1, 0, 0),
        epsilon=1e-6,
        convolution_width=3,
        recurrent_key_heads=1,
        recurrent_value_heads=2,
        recurrent_width=4,
        recurrent_head_mapping=HeadMapping.TILED,
    )

    def dense(prefix):
        return DenseFeedForwardWeights(
            gate=_weight(prefix + ".gate", (16, 8)),
            up=_weight(prefix + ".up", (16, 8)),
            down=_weight(prefix + ".down", (8, 16)),
        )

    return DenseDescription(
        artifact_identity=ArtifactIdentity("0" * 64),
        geometry=g,
        embedding=_weight("embedding", (32, 8)),
        output_norm=_weight("output_norm", (8,)),
        output=_weight("output", (32, 8)),
        blocks=(
            BlockWeights(
                input_norm=_weight("a.input_norm", (8,)),
                mixer=AttentionWeights(
                    query_gate=_weight("a.q", (16, 8)),
                    key=_weight("a.k", (4, 8)),
                    value=_weight("a.v", (4, 8)),
                    query_norm=_weight("a.q_norm", (4,)),
                    key_norm=_weight("a.k_norm", (4,)),
                    output=_weight("a.out", (8, 8)),
                ),
                feedforward_norm=_weight("a.ff_norm", (8,)),
                feedforward=dense("a.ff"),
            ),
            BlockWeights(
                input_norm=_weight("r.input_norm", (8,)),
                mixer=RecurrentWeights(
                    query_key_value=_weight("r.qkv", (16, 8)),
                    gate=_weight("r.gate", (8, 8)),
                    alpha=_weight("r.alpha", (2, 8)),
                    beta=_weight("r.beta", (2, 8)),
                    convolution=_weight("r.conv", (16, 3)),
                    decay=_weight("r.decay", (2,)),
                    time_bias=_weight("r.bias", (2,)),
                    norm=_weight("r.norm", (4,)),
                    output=_weight("r.out", (8, 8)),
                ),
                feedforward_norm=_weight("r.ff_norm", (8,)),
                feedforward=dense("r.ff"),
            ),
        ),
    )


class Residency:
    def __init__(self, device):
        self.device = device
        self.resources = []

    def bind(self, descriptor, dtype):
        resource = self.device.allocate(ops.TensorSpec(descriptor.shape, dtype))
        self.resources.append(resource)
        return resource


class ArrayResidency:
    def __init__(self, device):
        self.device = device
        self.resources = []
        self.arrays = {}
        self.rng = np.random.default_rng(44)

    def bind(self, descriptor, dtype):
        numpy_dtype = np.dtype(dtype.value)
        if descriptor.name.endswith("norm"):
            array = np.ones(descriptor.shape, dtype=numpy_dtype)
        elif descriptor.name.endswith("decay"):
            array = np.full(descriptor.shape, -0.1, dtype=numpy_dtype)
        else:
            array = self.rng.normal(0, 0.05, descriptor.shape).astype(numpy_dtype)
        resource = self.device.upload(ops.TensorSpec(descriptor.shape, dtype), array.tobytes())
        self.resources.append(resource)
        self.arrays[descriptor.name] = array
        return resource


def _array_spec(value):
    dtype = {
        np.dtype(np.int32): ops.DType.I32,
        np.dtype(np.float16): ops.DType.F16,
        np.dtype(np.float32): ops.DType.F32,
    }[value.dtype]
    return ops.TensorSpec(value.shape, dtype)


def test_whole_hybrid_step_is_one_prebound_native_submission():
    native = Runtime()
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
    residency = Residency(device)
    program = TensorProgram(_description(), device, residency)
    specs = InvocationSpecs(
        batch=1,
        tokens=ops.TensorSpec((2,), ops.DType.I32),
        coordinates=ops.TensorSpec((2, 3), ops.DType.I32),
        recurrent_offsets=ops.TensorSpec((2,), ops.DType.I32),
        output_rows=ops.TensorSpec((1,), ops.DType.I32),
        draws=ops.TensorSpec((1, 6), ops.DType.U32),
        destinations=(ops.TensorSpec((2,), ops.DType.I32),),
        visible=(ops.TensorSpec((2, 2), ops.DType.I32),),
        attention_state=(ops.TensorSpec((2, 16, 1, 4), ops.DType.F16),),
        convolution_state=(ops.TensorSpec((1, 16, 2), ops.DType.F16),),
        delta_state=(ops.TensorSpec((1, 2, 4, 4), ops.DType.F32),),
    )
    compiled = program.specialize("prefill", specs, precision="reference")
    preparation = [node for node in compiled.graph.nodes if node.operation == "recurrent_prepare"]
    assert preparation
    assert all(node.attributes["epsilon"] == pytest.approx(4e-6) for node in preparation)
    assert len(native.programs) == 1
    assert len(compiled.diagnostics.submissions) == 1
    assert native.executables[0].bound.static

    dynamic = [
        device.allocate(specs.tokens),
        device.allocate(specs.coordinates),
        device.allocate(specs.recurrent_offsets),
        device.allocate(specs.output_rows),
        device.allocate(specs.draws),
    ]
    resources = {
        name: device.allocate(spec)
        for name, spec in (
            ("attention.0.destinations", specs.destinations[0]),
            ("attention.0.visible", specs.visible[0]),
            ("attention.0.state", specs.attention_state[0]),
            ("recurrent.0.0.convolution", specs.convolution_state[0]),
            ("recurrent.0.0.delta", specs.delta_state[0]),
        )
    }
    execution = compiled.submit(*dynamic, resources=resources)
    assert len(native.executables[0].bound.calls) == 1
    execution.completion.wait()
    for output in execution.outputs:
        output.close()
    for resource in resources.values():
        resource.close()
    for resource in dynamic:
        resource.close()
    program.close()
    for resource in residency.resources:
        resource.close()
    device.close()


def test_whole_moe_prefill_rejects_dense_toy_expert_storage():
    base = _description()
    geometry = base.geometry.model_copy(
        update={
            "experts": ExpertGeometry(
                count=4,
                selected=2,
                intermediate=16,
                shared_intermediate=16,
            )
        }
    )
    routed = RoutedFeedForwardWeights(
        router=_weight("moe.router", (4, 8)),
        shared_router=_weight("moe.shared_router", (8,)),
        expert_gate=_weight("moe.gate", (4, 16, 8)),
        expert_up=_weight("moe.up", (4, 16, 8)),
        expert_down=_weight("moe.down", (4, 8, 16)),
        shared_gate=_weight("moe.shared_gate", (16, 8)),
        shared_up=_weight("moe.shared_up", (16, 8)),
        shared_down=_weight("moe.shared_down", (8, 16)),
    )
    blocks = tuple(block.model_copy(update={"feedforward": routed}) for block in base.blocks)
    description = base.model_copy(update={"geometry": geometry, "blocks": blocks})
    native = Runtime()
    native.compiler_target = replace(
        native.compiler_target,
        matrix_query=matrix_query((ops.MatrixTile(8, 8, 8, ops.DType.F16, ops.DType.F32),)),


    )
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
    residency = Residency(device)
    program = TensorProgram(description, device, residency)
    specs = InvocationSpecs(
        batch=1,
        tokens=ops.TensorSpec((8,), ops.DType.I32),
        coordinates=ops.TensorSpec((8, 3), ops.DType.I32),
        recurrent_offsets=ops.TensorSpec((2,), ops.DType.I32),
        output_rows=ops.TensorSpec((1,), ops.DType.I32),
        draws=ops.TensorSpec((1, 6), ops.DType.U32),
        destinations=(ops.TensorSpec((8,), ops.DType.I32),),
        visible=(ops.TensorSpec((8, 2), ops.DType.I32),),
        attention_state=(ops.TensorSpec((2, 16, 1, 4), ops.DType.F16),),
        convolution_state=(ops.TensorSpec((1, 16, 2), ops.DType.F16),),
        delta_state=(ops.TensorSpec((1, 2, 4, 4), ops.DType.F32),),
    )
    with pytest.raises(ValueError, match="no legal realization"):
        program.specialize("prefill", specs, precision="reference")
    program.close()
    for resource in residency.resources:
        resource.close()
    device.close()


@pytest.mark.device
@pytest.mark.parametrize("mode,rows", (("prefill", 2), ("decode", 1)))
def test_whole_hybrid_qwen_step_matches_semantic_graph_on_metal(mode, rows):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal whole-model qualification requires MPS")
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 26))
    residency = ArrayResidency(device)
    program = TensorProgram(_description(), device, residency)
    specs = InvocationSpecs(
        batch=1,
        tokens=ops.TensorSpec((rows,), ops.DType.I32),
        coordinates=ops.TensorSpec((rows, 3), ops.DType.I32),
        recurrent_offsets=ops.TensorSpec((2,), ops.DType.I32),
        output_rows=ops.TensorSpec((1,), ops.DType.I32),
        draws=ops.TensorSpec((1, 6), ops.DType.U32),
        destinations=(ops.TensorSpec((rows,), ops.DType.I32),),
        visible=(ops.TensorSpec((rows, 2), ops.DType.I32),),
        attention_state=(ops.TensorSpec((2, 16, 1, 4), ops.DType.F16),),
        convolution_state=(ops.TensorSpec((1, 16, 2), ops.DType.F16),),
        delta_state=(ops.TensorSpec((1, 2, 4, 4), ops.DType.F32),),
    )
    dynamic_arrays = {
        "tokens": np.arange(1, rows + 1, dtype=np.int32),
        "coordinates": np.repeat(np.arange(rows, dtype=np.int32)[:, None], 3, axis=1),
        "recurrent_offsets": np.asarray([0, rows], dtype=np.int32),
        "output_rows": np.asarray([rows - 1], dtype=np.int32),
        "draws": np.zeros((1, 6), dtype=np.uint32),
    }
    state_arrays = {
        "attention.0.destinations": np.arange(rows, dtype=np.int32),
        "attention.0.visible": np.asarray(
            [[0, index + 1] for index in range(rows)], dtype=np.int32
        ),
        "attention.0.state": np.zeros((2, 16, 1, 4), dtype=np.float16),
        "recurrent.0.0.convolution": np.zeros((1, 16, 2), dtype=np.float16),
        "recurrent.0.0.delta": np.zeros((1, 2, 4, 4), dtype=np.float32),
    }
    dynamic = resources = {}
    compiled = execution = None
    try:
        compiled = program.specialize(mode, specs, precision="reference")
        assert len(compiled.diagnostics.submissions) == 1
        bindings = {**residency.arrays, **dynamic_arrays, **state_arrays}
        expected = ops.evaluate_reference(compiled.graph, bindings).outputs
        dynamic = {
            name: device.upload(spec, dynamic_arrays[name].tobytes())
            for name, spec in (
                ("tokens", specs.tokens),
                ("coordinates", specs.coordinates),
                ("recurrent_offsets", specs.recurrent_offsets),
                ("output_rows", specs.output_rows),
                ("draws", specs.draws),
            )
        }
        resources = {
            name: device.upload(_array_spec(value), value.tobytes())
            for name, value in state_arrays.items()
        }
        execution = compiled.submit(*dynamic.values(), resources=resources)
        execution.completion.wait()
        assert len(execution.outputs) == len(expected)
        for actual, reference in zip(execution.outputs, expected, strict=True):
            value = actual.native.cpu().numpy()
            if np.issubdtype(reference.dtype, np.integer):
                np.testing.assert_array_equal(value, reference)
            else:
                np.testing.assert_allclose(value, reference, rtol=8e-2, atol=8e-2)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        for resource in resources.values():
            resource.close()
        for resource in dynamic.values():
            resource.close()
        program.close()
        for resource in residency.resources:
            resource.close()
        device.close()
