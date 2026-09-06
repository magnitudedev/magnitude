import pytest

from magnitude_engine.engine.delivery import Finished
from magnitude_engine.engine.prefixes.radix import Radix
from magnitude_engine.engine.prefixes.retention import LeastRecentlyUsed
from magnitude_engine.engine.requests import GenerationRequest
from magnitude_engine.engine.runtime import Engine
from magnitude_engine.engine.scheduler.time_shared import TimeShared
from magnitude_engine.generation.constraint_spec import ConstraintSpec
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod
from magnitude_engine.generation.runtime import GenerationResult, GenerationRuntime
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.attention.metal import MetalPagedAttention
from tests.engine.test_engine import available, tokens
from tests.generation.test_constraints import compiler
from tests.generation.test_generation_runtime import collect
from tests.models.architectures.qwen35.test_hybrid_model import setup


@pytest.mark.parametrize("method", [PlainMethod(), SuffixMethod(1, 3)])
def test_batched_generation_matches_request_local_sampling_and_grammar(method):
    _, model, arena, budget = setup(
        moe=True, bits=4, attention=MetalPagedAttention(), head_width=32
    )
    runtime = GenerationRuntime(model, method, compiler(64))
    policies = tuple(
        SamplingPolicy(temperature=0.7, seed=i + 73, frequency_penalty=0.2) for i in range(4)
    )
    prompts = ((1,), (2, 3, 2, 3), (1, 2, 3, 1, 2, 3), (4, 5))
    specs = (
        None,
        ConstraintSpec('start: ("bc" | "de")+ "f"'),
        ConstraintSpec('start: "bcdefghijk"'),
        ConstraintSpec('start: "bcdefghijk"'),
    )
    expected = []
    for prompt, policy, spec in zip(prompts, policies, specs, strict=True):
        row = runtime.create(prompt, policy, 11, (63,), constraint=spec)
        expected.append(collect(row, 3)[0])
        row.close()
    rows = tuple(
        runtime.create(prompt, policy, 11, (63,), constraint=spec)
        for prompt, policy, spec in zip(prompts, policies, specs, strict=True)
    )
    actual = [[] for _ in rows]
    batches = []
    for _ in range(15):
        active = tuple(i for i, row in enumerate(rows) if not row.finished)
        if not active:
            break
        measured = runtime.step_many(tuple(rows[i] for i in active), (4,) * len(active))
        for index, service in zip(active, measured, strict=True):
            if service.outcome is None:
                continue
            assert isinstance(service.outcome, GenerationResult)
            actual[index].extend(service.outcome.tokens)
            batches.append(service.batch_size)
    assert [tuple(row) for row in actual] == expected
    assert max(batches) >= 2  # The two forced rows share target work without sharing matchers.
    for row in rows:
        assert row.finished and row.model.state.position == len(row.context) - 1
        row.close()
    model.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0


def test_engine_physically_batches_ready_rows_and_excludes_cancelled_or_blocked_consumers():
    _, model, arena, budget = setup(attention=MetalPagedAttention(), head_width=32)
    generation = GenerationRuntime(model, PlainMethod())
    engine = Engine(
        generation,
        namespace=b"physical-batch",
        prefixes=Radix(retention=LeastRecentlyUsed(0, None)),
        scheduler=TimeShared(max_active=4, decode_tokens=1),
    )
    policy = SamplingPolicy(temperature=0.7, seed=81)
    expected = []
    for prompt in ((1,), (2,), (3,), (4,)):
        row = generation.create(prompt, policy, 6)
        expected.append(collect(row, 0)[0])
        row.close()
    shapes = []
    output = model.program.output

    def record(hidden):
        shapes.append(hidden.shape)
        return output(hidden)

    model.program.output = record
    handles = tuple(
        engine.submit(GenerationRequest((i + 1,), policy, 6), output_capacity=1 if i < 2 else 8)
        for i in range(4)
    )
    first = engine.tick()
    assert shapes == [(4, 1, 64)]
    assert all(m.batch_size == 4 for m in first)
    handles[1].cancel()
    events = [[], [], available(handles[2]), available(handles[3])]
    for _ in range(6):
        measured = engine.tick()
        assert all(m.request_id not in (handles[0].identity, handles[1].identity) for m in measured)
        events[2].extend(available(handles[2]))
        events[3].extend(available(handles[3]))
        if handles[2].delivery.finish and handles[3].delivery.finish:
            break
    assert shapes[1:] and all(s[0] == 2 for s in shapes[1:])
    assert tokens(events[2]) == expected[2] and tokens(events[3]) == expected[3]
    assert isinstance(events[2][-1], Finished)
    assert handles[0].delivery.finish is None
    assert handles[1].delivery.finish.reason == "cancelled"
    handles[0].cancel()
    engine.tick()
    assert handles[0].delivery.finish.generated_tokens == 1
    engine.close()
    model.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0
