import numpy as np
import pytest
import torch

import magnitensor as mt
from magnitude_engine.weights.descriptor import StoredDense, WeightDescriptor
from magnitude_engine.weights.identity import ArtifactIdentity
from magnitude_engine.weights.tensor_residency import TensorWeights


class Source:
    def __init__(self, content):
        self.content = content

    def read(self, offset, length):
        return self.content[offset : offset + length]


class Format:
    identity = ArtifactIdentity("1" * 64)

    def __init__(self, array, dtype=mt.DType.F32):
        self.array, self.dtype = array, dtype

    def stored(self, descriptor):
        return StoredDense(self.dtype, Source(self.array.tobytes()), 0, self.array.nbytes)

    def close(self):
        pass


@pytest.mark.device
def test_dense_artifact_weights_are_converted_by_tilelang_to_model_precision():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal residency conversion requires MPS")
    values = np.asarray([[0.1, -0.3], [1.25, -2.5]], dtype=np.float32)
    device = mt.device("metal", budget_bytes=1 << 20)
    weights = TensorWeights(Format(values), device)
    try:
        resident = weights.resident(WeightDescriptor(name="weight", shape=(2, 2)), mt.DType.F16)
        assert resident.spec == mt.TensorSpec((2, 2), mt.DType.F16)
        torch.testing.assert_close(
            resident.native.cpu().float(),
            torch.from_numpy(values).to(torch.float16).float(),
            rtol=0,
            atol=0,
        )
    finally:
        weights.close()
        device.close()


@pytest.mark.device
def test_bfloat16_artifact_storage_decodes_without_a_bfloat_device_type():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal bfloat storage decoding requires MPS")
    values = np.asarray([[0.1, -0.3], [1.25, -2.5]], dtype=np.float32)
    encoded = (values.view(np.uint32) >> 16).astype(np.uint16)
    rounded = (encoded.astype(np.uint32) << 16).view(np.float32)
    device = mt.device("metal", budget_bytes=1 << 20)
    weights = TensorWeights(Format(encoded, mt.DType.BF16), device)
    try:
        resident = weights.resident(WeightDescriptor(name="weight", shape=(2, 2)), mt.DType.F16)
        assert resident.spec == mt.TensorSpec((2, 2), mt.DType.F16)
        torch.testing.assert_close(
            resident.native.cpu().float(),
            torch.from_numpy(rounded).to(torch.float16).float(),
            rtol=0,
            atol=0,
        )
    finally:
        weights.close()
        device.close()
