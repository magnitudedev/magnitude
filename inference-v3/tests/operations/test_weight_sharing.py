import os

import gguf
import numpy as np
import pytest

from magnitude_engine.kernels.precision import REFERENCE_F32
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.platform.host.machine import open_context
from magnitude_engine.weights.descriptor import WeightDescriptor
from magnitude_engine.weights.formats.gguf import GGUFFormat
from tests.support import bind


@pytest.mark.device
def test_tied_embedding_and_readout_share_storage_and_survive_owner_retirement(tmp_path):
    weights = np.random.default_rng(123).normal(size=(7, 256)).astype(np.float32)
    path = tmp_path / "weights.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_tensor("token_embd.weight", weights)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    context = open_context(backend, 1024**2, 0)
    artifact = GGUFFormat(str(path))
    binding = bind(artifact, context, REFERENCE_F32)
    descriptor = WeightDescriptor(name="token_embd.weight", shape=weights.shape)
    embedding = binding.operations.embedding(descriptor)
    readout = binding.operations.linear(descriptor)
    assert context.allocated_bytes == weights.nbytes
    ids = np.array([6, 0, 6], np.int32)
    tokens = context.upload(TensorSpec((3,), DType.I32), ids.tobytes())
    embedded = context.allocate(TensorSpec((3, 256), DType.F32))
    logits = context.allocate(TensorSpec((3, 7), DType.F32))
    commands = [*embedding.prepare(tokens, embedded), *readout.prepare(embedded, logits)]
    binding.close()
    artifact.close()
    ticket = context.submit(commands)
    tokens.close()
    embedded.close()
    actual = np.frombuffer(context.read(logits, after=ticket), np.float32).reshape(3, 7)
    expected = weights[ids].astype(np.float64) @ weights.astype(np.float64).T
    bound = np.abs(weights[ids].astype(np.float64)) @ np.abs(weights.astype(np.float64)).T
    assert np.all(np.abs(actual - expected) <= 2e-6 * bound + 1e-6)
    assert context.allocated_bytes == logits.spec.nbytes
    logits.close()
    context.close()
