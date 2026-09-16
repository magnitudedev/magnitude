import pytest

import ops
from engine.models.qwen35.runtime import DenseRuntime
from magnitude import LogitsSelection, ModelRequest, TokenId
from tests.models.test_qwen35_runtime import ModelResidency, _description
from tests.ops.test_compiler import Runtime


@pytest.mark.parametrize(
    "vocabulary,options",
    [
        ((), {}),
        ((0, 0), {}),
        ((-1,), {}),
        ((999999,), {}),
        ((True,), {}),
        ((0,), {"selection": LogitsSelection.NONE}),
        ((0,), {"draw_words": (0, 0, 0, 0, 0, 0)}),
    ],
)
def test_invalid_readout_does_not_reserve_state(vocabulary, options):
    device = ops.DeviceRuntime(Runtime(), budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(_description(), device, residency)
    source = model.text_input((TokenId(1),))
    sequence = source.open()
    try:
        with pytest.raises(ValueError):
            model.prepare(
                (ModelRequest(sequence, (TokenId(1),), vocabulary=vocabulary, **options),)
            )
        assert sequence.pending is None
        assert sequence.position == 0 and model.states.occupied_rows == 0
    finally:
        source.close()
        model.close()
        for resource in residency.resources:
            resource.close()
        device.close()
