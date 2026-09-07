import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from magnitude_engine.artifacts.layouts import logical_tensors
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.tensors import TensorCatalog
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.experts.bank import ExpertBank, ExpertSource, ProjectionSource
from magnitude_engine.models.experts.computation import (
    ExpertWeights,
    GatedExpertMath,
    QuantizedProjection,
    ResidentExperts,
)
from magnitude_engine.models.experts.streaming import StreamedExperts
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader


def fixture(tmp_path, dtype, bits):
    library = SwitchGLU(64, 64, 4)
    library.set_dtype(dtype)
    nn.quantize(library, group_size=32, bits=bits)
    library.eval()
    arrays = {}
    projections = []
    encoding = AffineEncoding(bits, 32)
    for name in ("up", "gate", "down"):
        projection = getattr(library, f"{name}_proj")
        for component in ("weight", "scales", "biases"):
            arrays[f"{name}.{component}"] = getattr(projection, component)
        projections.append(
            QuantizedProjection(projection.weight, projection.scales, projection.biases, encoding)
        )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), arrays)
    tensors = logical_tensors(TensorCatalog.inspect(tmp_path), declaration=None)
    sources = [
        ProjectionSource(
            *(tensors[f"{name}.{component}"] for component in ("weight", "scales", "biases"))
        )
        for name in ("up", "gate", "down")
    ]
    source = ExpertSource(*sources, encoding)
    math = GatedExpertMath(library.activation)
    return library, source, ResidentExperts(ExpertWeights(*projections), math)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("tokens", [1, 8, 64])
def test_resident_and_streamed_operation_preserve_library_assignment_outputs(
    tmp_path, dtype, bits, tokens
):
    library, source, resident = fixture(tmp_path, dtype, bits)
    budget, reader = MemoryBudget(16 << 20), PositionalReader(workers=4)
    bank = ExpertBank(source, 2, budget, owner="decode.bank")
    scratch = ExpertBank(source, 4, budget, owner="prefill.scratch")
    streamed = StreamedExperts(source, resident.math, bank=bank, scratch=scratch, reader=reader)
    owner = ExecutionOwner()
    hidden = mx.random.normal((1, tokens, 64), key=mx.random.key(54)).astype(dtype)
    assignments = (mx.arange(tokens * 2, dtype=mx.int32) % 4).reshape(1, tokens, 2)
    scores = mx.softmax(mx.random.normal(assignments.shape).astype(dtype), axis=-1)
    expected = (library(hidden, assignments) * scores[..., None]).sum(axis=-2)
    with owner.scope() as scope:
        resident_output = resident.compute(hidden, assignments, scores, scope)
        streamed_output = streamed.compute(hidden, assignments, scores, scope)
        pending = scope.seal(resident_output, streamed_output, expected)
    if tokens == 1:
        with pytest.raises(RuntimeError, match="retire"):
            bank.close()
    pending.complete()
    assert mx.array_equal(resident_output, expected).item()
    assert mx.array_equal(streamed_output, expected).item()
    assert streamed_output.shape == (1, tokens, 64)
    bank.close()
    scratch.close()
    reader.close()
    owner.close()
    assert budget.snapshot().reserved == 0


def test_shared_prefill_scratch_retires_its_consumer_inside_a_model_scope(tmp_path):
    _, source, resident = fixture(tmp_path, mx.bfloat16, 4)
    budget, reader = MemoryBudget(16 << 20), PositionalReader()
    banks = [ExpertBank(source, 2, budget, owner=f"layer.{layer}") for layer in range(2)]
    scratch = ExpertBank(source, 4, budget, owner="shared.scratch")
    operators = [
        StreamedExperts(source, resident.math, bank=bank, scratch=scratch, reader=reader)
        for bank in banks
    ]
    owner = ExecutionOwner()
    with owner.scope() as scope:
        hidden = mx.ones((1, 8, 64), dtype=mx.bfloat16)
        routes = (mx.arange(16) % 4).reshape(1, 8, 2)
        for operator in operators:
            coefficients = mx.full(routes.shape, 0.5, dtype=hidden.dtype)
            hidden = operator.compute(hidden, routes, coefficients, scope)
        pending = scope.seal(hidden)
    pending.complete()
    assert mx.isfinite(hidden).all().item()
    for bank in banks:
        bank.close()
    scratch.close()
    reader.close()
    owner.close()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("width", [512, 1024])
def test_fused_selected_experts_match_independent_upstream_equation(dtype, bits, rows, width):
    from magnitude_engine.models.experts import metal

    mx.random.seed(485)
    library = SwitchGLU(width, 512, 4)
    library.set_dtype(dtype)
    nn.quantize(library, group_size=64, bits=bits)
    library.eval()
    projections = []
    for name in ("up", "gate", "down"):
        op = getattr(library, name + "_proj")
        projections.append(
            QuantizedProjection(op.weight, op.scales, op.biases, AffineEncoding(bits, 64))
        )
    weights = ExpertWeights(*projections)
    hidden = mx.random.normal((1, rows, width)).astype(dtype)
    assignments = (mx.arange(rows * 2) % 4).reshape(1, rows, 2)
    scores = mx.softmax(mx.random.normal(assignments.shape).astype(dtype), axis=-1)
    assert metal.supported(weights, hidden, assignments)
    actual = metal.apply(weights, hidden, assignments, scores)
    expected = (library(hidden, assignments) * scores[..., None]).sum(axis=-2)
    # Existing neural-region qualification bound, against stock MLX quantized gathers.
    assert mx.allclose(actual, expected, atol=0.002, rtol=0.002).item()
    assert actual.shape == hidden.shape


def test_weighted_expert_reduction_preserves_small_bf16_contributions():
    from types import SimpleNamespace

    from mlx_lm.models.switch_layers import SwiGLU

    from magnitude_engine.models.experts import metal
    from performance.benchmarks.references import experts

    def projection(output, width):
        # Every output selects the first input exactly, removing matvec rounding
        # from the test of how small weighted contributions are accumulated.
        weight = mx.concatenate(
            [
                mx.ones((8, output, 1), mx.uint32),
                mx.zeros((8, output, width // 8 - 1), mx.uint32),
            ],
            axis=-1,
        )
        shape = (8, output, width // 64)
        return QuantizedProjection(
            weight, mx.ones(shape, mx.bfloat16), mx.zeros(shape, mx.bfloat16), AffineEncoding(4, 64)
        )

    weights = ExpertWeights(projection(512, 1024), projection(512, 1024), projection(1024, 512))
    operation = SimpleNamespace(weights=weights, math=SimpleNamespace(activation=SwiGLU()))
    hidden = mx.full((1, 1, 1024), 2, mx.bfloat16)
    indices = mx.arange(8).reshape(1, 1, 8)
    scores = mx.array([1, *([2**-10] * 7)], mx.bfloat16).reshape(1, 1, 8)
    expected = experts(operation)(hidden, indices, scores)
    actual = metal.apply(weights, hidden, indices, scores)
    assert mx.array_equal(actual, expected).item()


def test_sorted_expert_boundary_uses_upstream_matrix_reduction():
    from mlx_lm.models.switch_layers import SwiGLU

    from magnitude_engine.models.experts import metal

    mx.random.seed(495)
    library = SwitchGLU(1024, 512, 8)
    library.set_dtype(mx.bfloat16)
    nn.quantize(library, group_size=64, bits=4)
    library.eval()
    weights = ExpertWeights(
        *(
            QuantizedProjection(
                getattr(library, name + "_proj").weight,
                getattr(library, name + "_proj").scales,
                getattr(library, name + "_proj").biases,
                AffineEncoding(4, 64),
            )
            for name in ("up", "gate", "down")
        )
    )
    hidden = mx.random.normal((1, 8, 1024)).astype(mx.bfloat16)
    indices = (mx.arange(64) * 3 % 5).reshape(1, 8, 8)
    scores = mx.softmax(mx.random.normal(indices.shape).astype(mx.bfloat16), axis=-1)
    assert not metal.supported(weights, hidden, indices)
    actual = GatedExpertMath(SwiGLU()).apply(weights, hidden, indices, scores)
    expected = (library(hidden, indices) * scores[..., None]).sum(axis=-2)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("bits", [4, 8])
def test_joint_shared_and_routed_experts_preserve_complete_mixture(dtype, bits):
    from mlx_lm.models.qwen3_next import Qwen3NextMLP

    from magnitude_engine.models.experts import metal
    from magnitude_engine.models.experts.computation import affine_mlp

    mx.random.seed(938)
    routed = SwitchGLU(1024, 512, 8)
    shared = Qwen3NextMLP(1024, 512)
    for layer in (routed, shared):
        layer.set_dtype(dtype)
        nn.quantize(layer, group_size=64, bits=bits)
        layer.eval()
    weights = ExpertWeights(
        *(
            QuantizedProjection(p.weight, p.scales, p.biases, AffineEncoding(bits, 64))
            for p in (routed.up_proj, routed.gate_proj, routed.down_proj)
        )
    )
    shared_weights = affine_mlp(shared)
    assert shared_weights is not None
    hidden = mx.random.normal((1, 1, 1024), key=mx.random.key(940)).astype(dtype)
    routes = mx.array([[[0, 3, 1, 7, 2, 0, 6, 4]]])
    scores = mx.softmax(mx.arange(8).astype(dtype)).reshape(routes.shape)
    coefficient = mx.array([[[0.3]]], dtype)
    assert metal.shared_supported(weights, shared_weights, hidden, routes)
    actual = metal.apply(
        weights, hidden, routes, scores, shared=shared_weights, shared_score=coefficient
    )
    expected = (routed(hidden, routes) * scores[..., None]).sum(-2) + shared(hidden) * coefficient
    assert mx.allclose(actual, expected, atol=0.002, rtol=0.002).item()
