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
    expected = library(hidden, assignments)
    with owner.scope() as scope:
        resident_output = resident.compute(hidden, assignments, scope)
        streamed_output = streamed.compute(hidden, assignments, scope)
        pending = scope.seal(resident_output, streamed_output, expected)
    if tokens == 1:
        with pytest.raises(RuntimeError, match="retire"):
            bank.close()
    pending.complete()
    assert mx.array_equal(resident_output, expected).item()
    assert mx.array_equal(streamed_output, expected).item()
    assert streamed_output.shape == (1, tokens, 2, 64)
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
            output = operator.compute(hidden, routes, scope)
            # Weighted combination remains the architecture's responsibility.
            hidden = output.mean(axis=-2)
        pending = scope.seal(hidden)
    pending.complete()
    assert mx.isfinite(hidden).all().item()
    for bank in banks:
        bank.close()
    scratch.close()
    reader.close()
    owner.close()
