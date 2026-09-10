"""Shared model projections with separate causal histories and acceptance."""

import os
from contextlib import ExitStack

import numpy as np
import pytest
from test_qwen35_model import fixture, reference

from magnitude_engine.artifacts.model import GGUFArtifact
from magnitude_engine.models.qwen35.artifact import inspect_dense
from magnitude_engine.models.qwen35.inputs import InputPlan
from magnitude_engine.models.qwen35.runtime import DenseRuntime, LogitsSelection
from magnitude_engine.models.sequence import ModelRequest
from magnitude_engine.operations.factory import ResidentOperations
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.machine import open_context


@pytest.mark.device
@pytest.mark.parametrize("lengths", [(5, 2), (17, 9)])
def test_packed_model_shares_projections_and_preserves_sequence_acceptance(
    tmp_path, monkeypatch, lengths
):
    path = tmp_path / "packed.gguf"
    weights = fixture(path)
    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")), 64 * 1024**2, 0
        )
        cleanup.callback(context.close)
        artifact = GGUFArtifact(str(path))
        cleanup.callback(artifact.close)
        provider = ResidentOperations(artifact, context)
        cleanup.callback(provider.close)
        model = DenseRuntime(inspect_dense(artifact.directory, artifact.identity), provider)
        cleanup.callback(model.close)
        prompts = tuple(
            tuple((index * 7 + row + 1) % 32 for index in range(length))
            for row, length in enumerate(lengths)
        )
        sequences = tuple(model.create(InputPlan.text(prompt)) for prompt in prompts)
        states = tuple(sequence.state for sequence in sequences)
        for state in states:
            cleanup.callback(state.close)
        projected_rows = []
        projection = model.program.blocks[0].gate
        original = projection.prepare

        def observe(inputs, output):
            projected_rows.append(inputs.spec.shape[0])
            return original(inputs, output)

        monkeypatch.setattr(projection, "prepare", observe)

        def execute(
            chunks,
            selections=(LogitsSelection.LAST, LogitsSelection.LAST),
            accepted=(True, True),
            cancel_before_submission=False,
        ):
            projected_rows.clear()
            requests = tuple(
                ModelRequest(sequence, tokens, selection)
                for sequence, tokens, selection in zip(sequences, chunks, selections, strict=True)
            )
            previous_binding = model.program._binding
            forward = model.prepare(requests)
            try:
                if cancel_before_submission:
                    forward.advances[1].close()
                ticket = context.submit(forward.commands)
                forward.submitted(ticket)
                actual = np.concatenate(
                    [
                        np.frombuffer(
                            context.read(output.logits, after=ticket), np.float32
                        ).reshape(-1, model.geometry.vocabulary)
                        for output in forward.advances
                        if output.logits is not None and not output.closed
                    ]
                )
                for output, accept in zip(forward.advances, accepted, strict=True):
                    if accept:
                        output.commit()
                    else:
                        output.close()
                assert model.program._binding is not None
                if model.program._binding is previous_binding:
                    assert projected_rows == []
                else:
                    assert projected_rows == [sum(map(len, chunks))]
                return actual
            finally:
                forward.close()

        expected = [reference(weights, prompt)[0] for prompt in prompts]
        actual = execute(prompts, selections=(LogitsSelection.ALL, LogitsSelection.LAST))
        np.testing.assert_allclose(
            actual, np.concatenate((expected[0], expected[1][-1:])), rtol=2e-4, atol=3e-5
        )
        assert tuple(state.position for state in states) == lengths

        # One decode row and a multi-token continuation share projections, but
        # accepting one candidate cannot advance or overwrite its rejected peer.
        suffixes = ((2,), (4, 8, 12))
        expected = [
            reference(weights, prompt + suffix)[0][-1]
            for prompt, suffix in zip(prompts, suffixes, strict=True)
        ]
        actual = execute(suffixes, accepted=(True, False))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5)
        assert tuple(state.position for state in states) == (lengths[0] + 1, lengths[1])
        suffixes = ((7,), (5,))
        histories = (prompts[0] + (2,), prompts[1])
        expected = [
            reference(weights, history + suffix)[0][-1]
            for history, suffix in zip(histories, suffixes, strict=True)
        ]
        np.testing.assert_allclose(execute(suffixes), expected, rtol=2e-4, atol=3e-5)
        positions = tuple(state.position for state in states)
        suffixes = ((6,), (9,))
        histories = (histories[0] + (7,), histories[1] + (5,))
        expected = reference(weights, histories[0] + (6,))[0][-1:]
        actual = execute(suffixes, accepted=(True, False), cancel_before_submission=True)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5)
        assert tuple(state.position for state in states) == (positions[0] + 1, positions[1])

        # Validation and late preparation failure must leave every candidate at
        # its old accepted boundary, including the request prepared first.
        requests = tuple(ModelRequest(sequence, (1,)) for sequence in sequences)
        # The injected failure targets preparation. Cached execution must not
        # rebuild projections on every decode, so retire that binding first.
        model.program.release_binding()
        baseline = context.allocated_bytes
        with pytest.raises(ValueError, match="distinct"):
            model.prepare((requests[0], requests[0]))
        with pytest.raises(ValueError, match="token IDs"):
            model.prepare((requests[0], ModelRequest(sequences[1], (32,))))

        def fail(*args):
            raise RuntimeError("injected batch preparation failure")

        with monkeypatch.context() as patch:
            patch.setattr(projection, "prepare", fail)
            with pytest.raises(RuntimeError, match="injected batch"):
                model.prepare(requests)
        assert all(state.pending is None for state in states)
        assert all(sequence.pending is None for sequence in sequences)
        assert context.allocated_bytes == baseline
        for state in states:
            state.close()
        model.close()
        provider.close()
        assert context.allocated_bytes == 0
