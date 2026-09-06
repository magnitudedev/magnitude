import mlx.core as mx
import pytest

from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from tests.generation.test_mtp import setup


@pytest.mark.parametrize("budget_ns", [None, 1])
@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("speculative", [False, True])
def test_plain_and_speculative_single_and_multi_session_execution(rows, speculative, budget_ns):
    target, method, budget, _, head_calls, _ = setup()
    projection_shapes = []
    def project(hidden):
        projection_shapes.append(hidden.shape)
        candidate = mx.where(hidden.astype(mx.int32) % 5 == 0, (hidden + 7) % 128, hidden)
        return -mx.square(mx.arange(128) - candidate) * 3
    method.project = project
    runtime = GenerationRuntime(target, method if speculative else PlainMethod())
    sequences = [runtime.create(tuple(range(1, start + 1)), SamplingPolicy(temperature=0), 15)
                 for start in range(1, rows + 1)]
    actual = [[] for _ in sequences]
    target_shapes = []
    forward = target.program._forward
    def capture(tokens, caches, request):
        target_shapes.append(tokens.shape)
        return forward(tokens, caches, request)
    target.program._forward = capture
    acceptance = []
    cohorts = set()
    for _ in range(300):
        active = [i for i, row in enumerate(sequences) if not row.finished]
        if not active:
            break
        measured = runtime.step_many(
            tuple(sequences[i] for i in active), (5,) * len(active), budget_ns=budget_ns
        )
        for index, service in zip(active, measured, strict=True):
            if service.outcome is None:
                continue
            assert not isinstance(service.outcome, BaseException)
            actual[index].extend(service.outcome.tokens)
            if service.outcome.proposed:
                acceptance.append(service.outcome.accepted)
            if sequences[index].model.state.batch is not None:
                cohorts.add(id(sequences[index].model.state.batch))
    assert actual == [list(range(start + 1, start + 16)) for start in range(1, rows + 1)]
    assert all(row.finished for row in sequences)
    if speculative:
        assert len(set(acceptance)) > 1
        if rows > 1:
            assert any(call.shape[0] > 1 for call in head_calls), "draft forwards were serialized"
            assert any(b > 1 for b, _, _ in projection_shapes), "draft projections were serialized"
            assert any(b > 1 and t > 1 for b, t in target_shapes), "verification was serialized"
    else:
        assert not head_calls
    if rows > 1:
        assert any(b > 1 for b, _ in target_shapes)
        assert len(cohorts) == 1, "acceptance/width changes rebuilt persistent target storage"
    for row in sequences:
        row.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("boundary", range(1, 16))
def test_cancelling_suspended_round_does_not_cancel_shared_peer(boundary):
    target, method, budget, _, _, _ = setup()
    runtime = GenerationRuntime(target, method)
    rows = [runtime.create((1, 2), SamplingPolicy(temperature=0), 20) for _ in range(2)]
    for _ in range(boundary):
        runtime.step_many(tuple(rows), (5, 5), budget_ns=1)
        assert not target.owner._pending, "returning service must leave safe device lifetimes"
    rows[0].close()
    while not rows[1].finished:
        result = runtime.step_many((rows[1],), (5,), budget_ns=1)[0]
        assert not isinstance(result.outcome, BaseException)
    assert rows[1].context == list(range(1, 23))
    rows[1].close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize('rows', [1, 3])
@pytest.mark.parametrize('speculative', [False, True])
def test_dependency_yields_preserve_submission_pipeline_without_per_forward_fences(
    rows, speculative
):
    from magnitude_engine.models.execution import MLXCompletion

    class Completion(MLXCompletion):
        submitted = drained = completed = 0

        def submit(self, arrays):
            self.submitted += 1
            super().submit(arrays)

        def complete(self, arrays):
            self.completed += 1
            super().complete(arrays)

        def drain(self):
            self.drained += 1
            super().drain()

    target, method, budget, _, _, _ = setup()
    completion = target.owner.backend = Completion()
    runtime = GenerationRuntime(target, method if speculative else PlainMethod())
    sequences = tuple(
        runtime.create((1, 2), SamplingPolicy(temperature=0), 30) for _ in range(rows)
    )
    runtime.step_many(sequences, (1,) * rows)  # Establish draft seeds outside the measured round.
    completion.submitted = completion.drained = completion.completed = 0
    services = runtime.step_many(sequences, (5,) * rows)
    assert all(service.outcome is not None for service in services)
    assert completion.submitted >= 3
    assert completion.drained == 1, 'dependent operations must share a retirement fence'
    assert completion.completed == (1 if speculative else 0)
    for row in sequences:
        row.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0


def test_publishable_result_returns_before_another_requests_draft_chain_finishes():
    target, method, budget, _, head_calls, _ = setup()
    runtime = GenerationRuntime(target, method)
    ready = runtime.create((1, 2), SamplingPolicy(temperature=0), 12)
    drafting = runtime.create((1, 2), SamplingPolicy(temperature=0), 12)
    drafting.step(1)
    outcomes = runtime.step_many((ready, drafting), (4, 4))
    assert outcomes[0].outcome.tokens == (3,)
    assert outcomes[1].outcome is None
    assert len(head_calls) == 1  # Only catch-up; the dependency chain has not completed.
    assert not target.owner._pending
    with pytest.raises(ValueError, match='reserved output allowance'):
        drafting.step(1)
    while not ready.finished or not drafting.finished:
        rows = tuple(row for row in (ready, drafting) if not row.finished)
        runtime.step_many(rows, (4,) * len(rows))
    assert ready.context == drafting.context == list(range(1, 15))
    ready.close()
    drafting.close()
    target.owner.close()
    assert budget.snapshot().reserved == 0
