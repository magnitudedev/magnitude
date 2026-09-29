"""Numerical continuation isolation and ordered selected readout on real kernels."""

import numpy as np
import pytest

import ops
from engine import DevicePlan
from engine.models.qwen35.runtime import DenseRuntime
from magnitude import ModelRequest, TokenId
from tests.models.test_forced_advance_numerics import description
from tests.models.test_qwen35_tensor_program import ArrayResidency


@pytest.mark.device
def test_shared_branches_match_independent_histories_and_selected_readout():
    desc = description()
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend="metal", maximum_bytes=256 << 20)
    ) as device:
        residency = ArrayResidency(device)
        residency.identity = desc.artifact_identity
        model = DenseRuntime(desc, device, residency, max_sequences=8)
        source = model.text_input(tuple(map(TokenId, (1, 2))))
        sequences = []
        checkpoint = None

        def open_sequence():
            state = source.open()
            sequences.append(state)
            return state

        def advance(sequence, tokens, **options):
            batch = model.prepare((ModelRequest(sequence, tuple(map(TokenId, tokens)), **options),))
            try:
                values = np.asarray(batch.advances[0].read_logits())
                batch.advances[0].commit()
                return values
            finally:
                batch.close()

        try:
            parent = open_sequence()
            prefix_logits = advance(parent, (1, 2))
            checkpoint = parent.checkpoint()
            a, b, selected = (checkpoint.fork() for _ in range(3))
            sequences.extend((a, b, selected))
            assert model.states.occupied_rows == 2
            advance(parent, (3, 4))
            # Advance both descendants together, sharing prefix reads across queries.
            batch = model.prepare(
                (ModelRequest(a, (TokenId(3), TokenId(4))), ModelRequest(b, (TokenId(5),)))
            )
            try:
                actual = [np.asarray(item.read_logits()) for item in batch.advances]
                batch.advances[0].commit()
            finally:
                batch.close()
            assert model.states.occupied_rows == 6
            assert a.position == 4 and b.position == 2
            for tokens, row in zip(((3, 4), (5,)), actual, strict=True):
                independent = open_sequence()
                advance(independent, (1, 2))
                expected = advance(independent, tokens)
                np.testing.assert_allclose(row, expected, rtol=3e-3, atol=3e-3)
            # Next token crosses multiple separately owned physical history ranges.
            independent = open_sequence()
            advance(independent, (1, 2))
            advance(independent, (3, 4))
            np.testing.assert_allclose(
                advance(a, (6,)), advance(independent, (6,)), rtol=3e-3, atol=3e-3
            )
            ids = (TokenId(7), TokenId(1), TokenId(4))
            projected = advance(selected, (3, 4), vocabulary=ids)
            np.testing.assert_allclose(projected, actual[0][:, ids], rtol=3e-3, atol=3e-3)
            # Reclaiming with live owners must preserve their arena and bindings.
            model.reclaim()
            np.testing.assert_allclose(
                advance(a, (7,)), advance(independent, (7,)), rtol=3e-3, atol=3e-3
            )
            for sequence in sequences:
                sequence.close()
            sequences.clear()
            checkpoint.close()
            checkpoint = None
            assert model.states.idle
            before = device.allocated_bytes
            released = model.reclaim()
            assert released > 0 and device.allocated_bytes == before - released
            # The same geometry must bind the newly allocated arena after reclaim.
            restarted = open_sequence()
            np.testing.assert_allclose(
                advance(restarted, (1, 2)), prefix_logits, rtol=3e-3, atol=3e-3
            )
        finally:
            for sequence in sequences:
                sequence.close()
            if checkpoint is not None:
                checkpoint.close()
            source.close()
            model.close()
            for resource in residency.resources:
                resource.close()
