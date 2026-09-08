import pytest

from magnitude_engine.generation.execution import run
from magnitude_engine.generation.methods.plain.runtime import PlainMethod
from magnitude_engine.generation.methods.suffix.runtime import SuffixMethod, SuffixSession
from magnitude_engine.generation.sampling_policy import SamplingPolicy
from magnitude_engine.models.prompt import InputSpan, Prompt
from tests.generation.test_generation_runtime import collect, make_generation


def test_suffix_matching_and_proposals_stop_at_non_language_barriers():
    session = SuffixSession(2, 3, None)
    # Matching text before an image may propose the adjacent text, but never
    # placeholders or text stitched across an image.
    assert run(session.propose([1, 2, 3, None, 9, 1, 2], 5)).host() == (3,)
    session.close()
    session = SuffixSession(2, 3, None)
    assert run(session.propose([1, None, 2, 3, 1, 2], 5)).host() == ()
    session.close()


@pytest.mark.parametrize("method", [PlainMethod(), SuffixMethod(1, 3)])
@pytest.mark.parametrize("width", [1, 2, 8])
def test_final_dependent_unit_predicts_from_last_output(method, width):
    runtime, budget = make_generation(method)
    prompt = Prompt((0, 1, 2, 3, 4), (InputSpan(1, 5, b"dependent", True),))
    calls = []
    call = runtime.model.program.call

    def record(tokens, caches):
        calls.append(tokens.shape[1])
        return call(tokens, caches)

    runtime.model.program.call = record
    row = runtime.create(prompt, SamplingPolicy(temperature=0), 7, chunk_size=width)
    assert calls == [1] and row.prefill_remaining == 0
    first = row.step(width)
    assert first.tokens == (5,) and first.evaluated_inputs == 4
    assert calls == [1, 4]
    assert first.proposed == 0 and row.model.position == 5
    assert collect(row, width)[0] == (6, 0, 1, 2, 3, 4)
    row.close()
    assert budget.snapshot().reserved == 0


def test_soft_prefill_allowance_makes_legal_progress_and_checkpoint_identity_is_semantic():
    runtime, budget = make_generation(SuffixMethod(1, 3))
    prompt = Prompt((0, 1, 2, 3, 4, 5), (InputSpan(1, 5, b"image-a", True),))
    row = runtime.prepare(prompt, SamplingPolicy(temperature=0), 3)
    assert row.prefill(2) == 1
    assert row.prefill(1) == 4
    checkpoint = row.checkpoint()
    assert checkpoint.prompt == prompt.prefix(5)
    changed = Prompt(prompt.tokens, (InputSpan(1, 5, b"image-b", True),))
    with pytest.raises(ValueError, match="match"):
        runtime.prepare(changed, SamplingPolicy(temperature=0), 3, checkpoint=checkpoint)
    warm = runtime.create(prompt, SamplingPolicy(temperature=0), 3, checkpoint=checkpoint)
    assert collect(warm, 3)[0] == collect(row, 3)[0]
    checkpoint.close()
    row.close()
    warm.close()
    assert budget.snapshot().reserved == 0
