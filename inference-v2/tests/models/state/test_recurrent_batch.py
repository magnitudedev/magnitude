import mlx.core as mx
import pytest

from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest
from magnitude_engine.models.state.recurrent import read_batch
from tests.models.architectures.qwen35.test_hybrid_model import setup


def forward(runtime, rows, width=1):
    return runtime.forward_batch(
        rows,
        (ModelInputs.from_tokens(tuple(range(4, 4 + width))),) * len(rows),
        ForwardRequest(committed_inputs=width),
    )


def test_stable_recurrent_batch_reuses_tensors_and_charges_physical_images_through_churn():
    _, runtime, arena, budget = setup()
    rows = tuple(runtime.create() for _ in range(3))
    size = sum(layout.nbytes for layout in runtime.states.layouts)
    checkpoint = branch = None
    try:
        for row in rows:
            runtime.prefill(row, (1, 2, 3))
        advances = forward(runtime, rows)
        for advance in advances:
            advance.accept_all_lazily()
        advances[0].complete()
        image = rows[0].state.recurrent.image
        assert image.width == 3
        assert all(row.state.recurrent.image is image for row in rows)
        for layer in range(len(runtime.states.layouts)):
            slots = tuple(row.state.slots[layer] for row in rows)
            assert read_batch(slots) is image.read(layer)
        assert budget.snapshot().owners["recurrent-state"] == 3 * size

        advances = forward(runtime, rows)
        for advance in advances:
            advance.accept_all_lazily()
        # Logical commit must not release buffers still borrowed by submitted work.
        assert not image.closed
        assert budget.snapshot().owners["recurrent-state"] == 6 * size
        # Only execution pins remain on the old image. Its Python graphs can
        # retire now, but its physical memory charge cannot retire before completion.
        assert image._pins > 0 and image._users == 0
        with pytest.raises(RuntimeError, match="unavailable"):
            image.read(0)
        with pytest.raises(ValueError, match="unavailable"):
            image.acquire(0)
        advances[0].complete()
        assert image.closed
        image = rows[0].state.recurrent.image
        rows[0].close()
        assert not image.closed
        assert budget.snapshot().owners["recurrent-state"] == 3 * size

        # One row advances alone while its peer retains the old physical batch.
        step = runtime.forward(rows[1], (6,), ForwardRequest(committed_inputs=1))
        step.accept(1)
        assert rows[1].state.recurrent.image.width == 1
        assert rows[2].state.recurrent.image is image
        assert budget.snapshot().owners["recurrent-state"] == 4 * size
        checkpoint = rows[2].checkpoint()
        assert checkpoint.recurrent.image.width == 1
        rows[2].close()
        assert image.closed
        assert budget.snapshot().owners["recurrent-state"] == 2 * size
        branch = runtime.create(checkpoint)
        assert branch.state.recurrent.image is not checkpoint.recurrent.image
        runtime.prefill(branch, (7,))
    finally:
        if branch is not None:
            branch.close()
        if checkpoint is not None:
            checkpoint.close()
        for row in rows:
            row.close()
        runtime.owner.close()
        arena.close()
    assert budget.snapshot().reserved == 0


def test_recurrent_batch_partial_admission_failure_releases_shared_destination():
    _, runtime, arena, budget = setup()
    rows = tuple(runtime.create() for _ in range(3))
    try:
        for row in rows:
            runtime.prefill(row, (1, 2))
        before = budget.snapshot().reserved
        state_bytes = sum(layout.nbytes for layout in runtime.states.layouts)
        trace_bytes = sum(layout.trace_bytes_per_token for layout in runtime.states.layouts)
        # Destination for every row fits, but only the first row's trace fits.
        budget.limit = before + 3 * state_bytes + trace_bytes
        with pytest.raises(MemoryError, match="recurrent-advance"):
            forward(runtime, rows)
        assert budget.snapshot().reserved == before
        assert all(row.state.active is None and not row.failed for row in rows)
        budget.limit = 8 << 20
        for advance in forward(runtime, rows):
            advance.accept(1)
    finally:
        budget.limit = 8 << 20
        for row in rows:
            row.close()
        runtime.owner.close()
        arena.close()
    assert budget.snapshot().reserved == 0


def test_independent_recurrent_acceptance_retains_only_its_owned_image():
    model, runtime, arena, budget = setup()
    rows = tuple(runtime.create() for _ in range(3))
    try:
        for row in rows:
            runtime.prefill(row, (1, 2, 3))
        initial = tuple(row.state.recurrent.image for row in rows)
        block = (4, 5, 6, 7)
        advances = runtime.forward_batch(rows, (ModelInputs.from_tokens(block),) * 3)
        candidate = rows[0].state.slots[0].destination.image
        for row, advance, accepted in zip(rows, advances, (0, 2, 4), strict=True):
            advance.accept(accepted)
            expected = model(mx.array([[1, 2, 3, *block[:accepted], 9]]), cache=model.make_cache())[
                :, -1:
            ]
            follow = runtime.forward(row, (9,))
            follow.complete()
            assert mx.allclose(follow.output.logits, expected, atol=1e-4).item()
            # Keep the committed acceptance boundary; discard this comparison probe.
            follow.accept(0)
        assert rows[0].state.recurrent.image is initial[0]
        assert rows[1].state.recurrent.image.width == 1
        assert rows[2].state.recurrent.image is candidate
        rows[2].close()
        assert candidate.closed
    finally:
        for row in rows:
            row.close()
        runtime.owner.close()
        arena.close()
    assert budget.snapshot().reserved == 0


def test_repair_allocation_failure_does_not_publish_or_change_peer_state():
    _, runtime, arena, budget = setup()
    rows = tuple(runtime.create() for _ in range(3))
    try:
        for row in rows:
            runtime.prefill(row, (1, 2, 3))
        initial = rows[0].state.recurrent.image
        advances = runtime.forward_batch(rows, (ModelInputs.from_tokens((4, 5, 6)),) * 3)
        advances[0].complete()
        budget.limit = budget.snapshot().reserved
        with pytest.raises(MemoryError, match="recurrent-state"):
            advances[0].accept(1)
        assert rows[0].failed and rows[0].state.position == 3
        assert rows[0].state.recurrent.image is initial
        assert all(not row.failed and row.state.position == 3 for row in rows[1:])
        rows[0].close()
        budget.limit = 8 << 20
        advances[1].accept(3)
        advances[2].accept(0)
        assert (rows[1].state.position, rows[2].state.position) == (6, 3)
    finally:
        budget.limit = 8 << 20
        for row in rows:
            row.close()
        runtime.owner.close()
        arena.close()
    assert budget.snapshot().reserved == 0
