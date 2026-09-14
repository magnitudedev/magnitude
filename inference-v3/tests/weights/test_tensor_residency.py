import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from engine.weights.descriptor import StoredDense, WeightDescriptor
from engine.weights.identity import ArtifactIdentity
from engine.weights.formats.gguf import Encoding, quantization
from engine.weights.tensor_residency import TensorWeights, describe_binding


class Format:
    identity = ArtifactIdentity("1" * 64)

    def __init__(self, array, dtype=ops.DType.F32):
        self.array, self.dtype = array, dtype

    def stored(self, descriptor):
        return StoredDense(self.dtype, ops.MemorySource(self.array.tobytes()), 0, self.array.nbytes)

    def close(self):
        pass


@pytest.mark.parametrize("encoding,bytes_per_block", [(Encoding.Q4_K, 192), (Encoding.Q5_K, 224), (Encoding.Q6_K, 256)])
@pytest.mark.parametrize("residency", [ops.Residency.RESIDENT, ops.Residency.STREAMED])
def test_hierarchical_import_prepares_coefficients_without_expanding_codes(encoding, bytes_per_block, residency, monkeypatch):
    from engine.weights.descriptor import StoredQuantized
    from ops.kernels.packed import packet_format
    from ops.runtime.imports import ImportActionKind, plan_import

    representation, codec = quantization(encoding)
    source = ops.MemorySource(bytes(codec.block_bytes * 2))

    class QuantizedFormat:
        identity = ArtifactIdentity("2" * 64)

        def stored(self, descriptor):
            return StoredQuantized(representation, codec, source, 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("execution representation selection is metadata only")

    monkeypatch.setattr(ops.MemorySource, "read", forbidden)
    binding = describe_binding(QuantizedFormat(), WeightDescriptor(name="weight", shape=(2, 256)),
                               ops.DType.BF16, residency)
    assert binding.spec.representation.code == representation.code
    assert binding.spec.representation.group == representation.group
    assert binding.spec.representation.coefficients.scale_dtype == ops.DType.F32
    assert binding.spec.storage_nbytes == bytes_per_block * 2
    assert binding.planes[0].span.length == codec.block_bytes * 2
    assert packet_format(binding.spec).name == "affine-direct-grouped"
    assert ops.execution_representation(binding.spec.representation) == binding.spec.representation
    plan = plan_import(binding, chunk_bytes=codec.block_bytes)
    assert sum(action.bytes for action in plan.actions if action.kind == ImportActionKind.PUBLISH) == binding.spec.storage_nbytes


@pytest.mark.parametrize("residency", [ops.Residency.RESIDENT, ops.Residency.STREAMED])
def test_artifact_binding_description_needs_no_live_device_or_payload_read(monkeypatch, residency):
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata description must not open a device or read its source")

    monkeypatch.setattr(ops.DeviceRuntime, "open", forbidden)
    monkeypatch.setattr(ops.MemorySource, "read", forbidden)
    format = Format(np.zeros((2, 4), np.float32))
    descriptor = WeightDescriptor(name="projection", shape=(2, 4))
    binding = describe_binding(format, descriptor, ops.DType.BF16, residency)
    assert binding.spec == ops.TensorSpec((2, 4), ops.DType.BF16)
    assert binding.residency == residency
    assert binding.planes[0].span.length == 32
    assert binding.value_identity == describe_binding(format, descriptor, ops.DType.BF16).value_identity


@pytest.mark.device
def test_dense_artifact_weights_are_converted_by_tilelang_to_model_precision():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal residency conversion requires MPS")
    values = np.asarray([[0.1, -0.3], [1.25, -2.5]], dtype=np.float32)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    weights = TensorWeights(Format(values), device)
    resident = None
    try:
        resident = device.resolve(weights.bind(WeightDescriptor(name="weight", shape=(2, 2)), ops.DType.F16))
        assert resident.spec == ops.TensorSpec((2, 2), ops.DType.F16)
        torch.testing.assert_close(
            resident.native.cpu().float(),
            torch.from_numpy(values).to(torch.float16).float(),
            rtol=0,
            atol=0,
        )
    finally:
        if resident is not None:
            resident.close()
        weights.close()
        device.close()


@pytest.mark.device
def test_bfloat16_artifact_storage_decodes_without_a_bfloat_device_type():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal bfloat storage decoding requires MPS")
    values = np.asarray([[0.1, -0.3], [1.25, -2.5]], dtype=np.float32)
    encoded = (values.view(np.uint32) >> 16).astype(np.uint16)
    rounded = (encoded.astype(np.uint32) << 16).view(np.float32)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20))
    weights = TensorWeights(Format(encoded, ops.DType.BF16), device)
    resident = None
    try:
        resident = device.resolve(weights.bind(WeightDescriptor(name="weight", shape=(2, 2)), ops.DType.F16))
        assert resident.spec == ops.TensorSpec((2, 2), ops.DType.F16)
        torch.testing.assert_close(
            resident.native.cpu().float(),
            torch.from_numpy(rounded).to(torch.float16).float(),
            rtol=0,
            atol=0,
        )
    finally:
        if resident is not None:
            resident.close()
        weights.close()
        device.close()
