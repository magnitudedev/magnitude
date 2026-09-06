from types import SimpleNamespace

import mlx.core as mx
import pytest

from benchmarks.subjects.context.runtime import ContextTrace
from magnitude_engine.models.inputs import ModelInputs


@pytest.mark.parametrize("mode,expected", [("replay", [7, 8, 9]), ("generate", [3, 5])])
def test_replay_uses_fixture_while_generation_feeds_predictions_and_stops(mode, expected):
    seen = []
    accepted = []

    def forward(sequence, inputs, request):
        tokens = inputs.tokens.reshape(-1).tolist() if isinstance(inputs, ModelInputs) else inputs
        seen.extend(tokens)
        output_token = 5 if len(seen) == 1 else 6
        logits = mx.array([[[float(i == output_token) for i in range(10)]]])
        return SimpleNamespace(
            output=SimpleNamespace(logits=logits), accept=accepted.append, complete=lambda: None
        )

    trace = ContextTrace.__new__(ContextTrace)
    trace.mode, trace.limit, trace.first = mode, 3, 3
    trace.eos, trace.replay = {6}, (7, 8, 9)
    trace.sequence = object()
    trace.model = SimpleNamespace(forward=forward)
    trace.generated, trace.consumed = [], 0
    trace.invoke()
    assert seen == expected
    assert accepted == [1] * len(expected)
    assert trace.consumed == len(expected)
    assert trace.generated == ([5, 6] if mode == "generate" else [])
