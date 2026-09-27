"""Native verification of public readout with and without generation sampling."""
import numpy as np
import pytest

import ops
from engine import DevicePlan
from engine.models.qwen35.runtime import DenseRuntime
from magnitude import ModelRequest, TokenId
from tests.models.test_forced_advance_numerics import description
from tests.models.test_qwen35_tensor_program import ArrayResidency


@pytest.mark.device
def test_logit_readout_matches_sampled_execution_on_metal():
    desc = description()
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend="metal", maximum_bytes=128 << 20)
    ) as device:
        residency = ArrayResidency(device)
        residency.identity = desc.artifact_identity
        model = DenseRuntime(desc, device, residency, max_sequences=2)
        source = model.text_input((TokenId(1),))
        left, right = source.open(), source.open()
        try:
            results = []
            for sequence, draws in ((left, None), (right, (0, 0, 0, 0, 0, 0))):
                batch = model.prepare((ModelRequest(sequence, (TokenId(1),), draw_words=draws),))
                try:
                    results.append(np.asarray(batch.advances[0].read_logits()))
                    assert (batch.advances[0].read_sample() is None) == (draws is None)
                    batch.advances[0].commit()
                finally:
                    batch.close()
            np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-5)
        finally:
            left.close()
            right.close()
            source.close()
            model.close()
            for resource in residency.resources:
                resource.close()
