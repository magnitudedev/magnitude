"""Prompt batching uses the same model/state contract as generation rounds."""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.runtime import GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget


def test_prefill_exposes_preparation_as_progress_before_consuming_decoder_inputs():
    from contextlib import ExitStack

    from magnitude_engine.models.operations import Complete, compute
    from tests.generation.test_computation import Work

    budget = MemoryBudget(1 << 20)
    owner = ExecutionOwner()
    calls = []

    class Inputs:
        cache_hits = 0

        def __init__(self):
            self.ready, self.closed = False, False

        def boundary(self, position):
            return True

        def prepare(self, position, count):
            if not self.ready:
                result = yield from compute(Work(owner, 2, calls))
                yield Complete(result.execution)
                self.ready = True

        def assemble(self, inputs, position):
            assert self.ready
            return inputs

        def acquire(self, position, count):
            return ExitStack()

        def batch_key(self, position, count):
            return None

        def close(self):
            self.closed = True

    def call(tokens, caches):
        values = tokens.astype(mx.float32)[:, None, :, None]
        caches[0].update_and_fetch(values, values)
        return values[:, 0]

    model = ModelRuntime(
        LibraryProgram(call),
        LibraryStateStore(
            lambda: [KVCache()],
            budget,
            lambda n, q: 4096,
        ),
        owner,
    )
    generation = GenerationRuntime(model, PlainMethod())
    row = generation.prepare((1, 2, 3), SamplingPolicy(temperature=0), 1)
    context = Inputs()
    row.model.inputs = context
    first = generation.prefill_many((row,), (2,))[0]
    assert first.outcome is None and first.preparation_ns > 0
    assert row.prefilled == row.model.position == 0
    second = generation.prefill_many((row,), (2,))[0]
    assert second.outcome == 2 and second.preparation_ns == 0
    assert row.prefilled == row.model.position == 2
    assert calls == [(2,)]
    row.close()
    owner.close()
    assert context.closed and budget.snapshot().reserved == 0


@pytest.mark.parametrize("lengths", [(8,), (8, 8, 8), (8, 4, 8)])
@pytest.mark.parametrize("batched", [False, True])
def test_prefill_groups_compatible_work_and_preserves_each_prompt(lengths, batched, monkeypatch):
    budget = MemoryBudget(1 << 20)
    calls = []

    def call(tokens, caches):
        calls.append(tokens.shape)
        values = tokens.astype(mx.float32)[:, None, :, None]
        caches[0].update_and_fetch(values, values)
        return values[:, 0]

    program = LibraryProgram(call)
    if not batched:
        monkeypatch.setattr(program, "forward_batch", None)
    model = ModelRuntime(
        program,
        LibraryStateStore(
            lambda: [KVCache()],
            budget,
            lambda n, q: 4096,
        ),
        ExecutionOwner(),
    )
    generation = GenerationRuntime(model, PlainMethod())
    rows = tuple(
        generation.prepare(
            tuple(range(length + 1)),
            SamplingPolicy(temperature=0),
            2,
        )
        for length in lengths
    )
    try:
        groups = generation.prefill_groups(rows)
        assert groups == ((rows,) if batched else tuple((row,) for row in rows))
        results = generation.prefill_many(rows, lengths)
        assert [r.outcome for r in results] == list(lengths)
        if batched:
            assert calls == [(lengths.count(n), n) for n in dict.fromkeys(lengths)]
        else:
            assert sorted(calls) == sorted((1, n) for n in lengths)
        for row, length in zip(rows, lengths, strict=True):
            assert row.prefill_remaining == 0
            assert row.target_position == length
            cache = row.model.state.caches[0]
            assert cache.offset == length
            assert cache.keys[0, 0, :length, 0].tolist() == list(range(length))
            assert row.model.pending is None
        assert not model.owner._pending
    finally:
        for row in rows:
            row.close()
        model.owner.close()
    assert budget.snapshot().reserved == 0


def test_prompt_batch_allocation_failure_splits_without_losing_committed_peer_state():
    budget = MemoryBudget(3 * 4096)
    shapes = []

    def call(tokens, caches):
        shapes.append(tokens.shape)
        values = tokens.astype(mx.float32)[:, None, :, None]
        caches[0].update_and_fetch(values, values)
        return values[:, 0]

    model = ModelRuntime(
        LibraryProgram(call),
        LibraryStateStore(
            lambda: [KVCache()],
            budget,
            lambda n, q: 4096,
        ),
        ExecutionOwner(),
    )
    runtime = GenerationRuntime(model, PlainMethod())
    rows = tuple(
        runtime.prepare((i, i + 1, i + 2), SamplingPolicy(temperature=0), 1) for i in range(3)
    )
    try:
        for row in rows:
            row.reserve_prompt()
        # All singleton reservations fit; a replacement physical batch cannot
        # coexist with them. The dispatcher must retain and execute each row.
        results = runtime.prefill_many(rows, (2,) * 3)
        assert [r.outcome for r in results] == [2] * 3
        assert shapes == [(1, 2)] * 3
        for i, row in enumerate(rows):
            assert not row.failed and row.prefill_remaining == 0
            assert row.model.state.caches[0].keys[0, 0, :2, 0].tolist() == [i, i + 1]
    finally:
        for row in rows:
            row.close()
        model.owner.close()
    assert budget.snapshot().reserved == 0


def test_model_batches_combine_output_demands_without_combining_causal_commitments():
    from magnitude_engine.generation.execution import execute
    from magnitude_engine.models.inputs import ModelInputs
    from magnitude_engine.models.operations import complete, forward
    from magnitude_engine.models.runtime import ForwardRequest, ModelOutput

    budget = MemoryBudget(1 << 20)
    calls = []

    class Program:
        features = frozenset({"hidden", "residual"})
        conditioning = frozenset()

        def forward(self, inputs, state, request, scope):
            return self.forward_batch((inputs,), (state,), request, scope)

        def forward_batch(self, inputs, states, request, scope):
            calls.append((len(states), request))
            values = mx.concatenate([row.tokens for row in inputs]).astype(mx.float32)
            for row, state in zip(inputs, states, strict=True):
                keys = row.tokens[:, None, :, None].astype(mx.float32)
                state.caches[0].update_and_fetch(keys, keys)
            return ModelOutput(
                values[..., None] if request.logits else None,
                {name: values[..., None] for name in request.features},
            )

    model = ModelRuntime(
        Program(),
        LibraryStateStore(
            lambda: [KVCache()],
            budget,
            lambda n, q: 4096,
        ),
        ExecutionOwner(),
    )
    rows = tuple(model.create() for _ in range(4))
    requests = (
        ForwardRequest(False, frozenset({"hidden"}), 1),
        ForwardRequest(True, frozenset({"residual"}), 1),
        ForwardRequest(False, committed_inputs=1),
        ForwardRequest(True, committed_inputs=0),
    )

    def task(row, request):
        advance = yield from forward(row, ModelInputs.from_tokens((3,)), request)
        yield from complete(advance)
        assert not request.logits or advance.output.logits is not None
        assert request.features <= advance.output.features.keys()
        advance.accept(1)
        return 1

    try:
        results = execute(tuple(task(row, req) for row, req in zip(rows, requests, strict=True)))
        assert [result.result for result in results] == [1] * 4
        assert calls == [
            (3, ForwardRequest(True, frozenset({"hidden", "residual"}), 1)),
            (1, ForwardRequest(True, committed_inputs=0)),
        ]
    finally:
        for row in rows:
            row.close()
        model.owner.close()
    assert budget.snapshot().reserved == 0
