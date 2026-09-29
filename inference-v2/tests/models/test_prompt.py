import pytest
from hypothesis import given
from hypothesis import strategies as st

from magnitude_engine.models.prompt import InputSpan, Prompt


def test_soft_allowances_preserve_dependency_spans_without_starving_them():
    prompt = Prompt(tuple(range(15)), (InputSpan(3, 11, b"features", indivisible=True),))
    assert prompt.advance(0, 5) == 3
    assert prompt.advance(3, 1) == 11
    assert prompt.advance(11, 2) == 13
    assert prompt.advance(13, 7) == 15
    assert prompt.advance(15, 7) == 15
    with pytest.raises(ValueError, match="boundaries"):
        prompt.advance(4, 4)
    with pytest.raises(ValueError, match="boundaries"):
        prompt.advance(0, 2, end=8)
    with pytest.raises(ValueError, match="unresolved"):
        prompt.prefix(8)


def test_final_input_is_a_legal_unit_even_when_the_prompt_ends_in_conditioning():
    prompt = Prompt((1, 2, 3, 3, 3), (InputSpan(2, 5, b"features", indivisible=True),))
    assert prompt.anchor_start == 2
    assert prompt.extend((7,)).anchor_start == 5
    assert Prompt((1,)).anchor_start == 0
    with pytest.raises(ValueError, match="nonempty"):
        _ = Prompt(()).anchor_start


def test_causal_input_transitions_preserve_reusable_prefixes_without_atomicity():
    prompt = Prompt(tuple(range(15)), (InputSpan(3, 11, b"features"),))
    assert prompt.advance(0, 15) == 15
    assert prompt.advance(3, 2) == 5
    assert prompt.advance(5, 15) == 15
    assert prompt.advance(11, 15) == 15
    assert prompt.retention_boundaries == (3, 11)


def test_reuse_identity_distinguishes_content_processing_and_positions():
    tokens = (1, 2, 3, 3, 4)
    first = Prompt(tokens, (InputSpan(2, 4, b"image:a:processor:1"),))
    changed = Prompt(tokens, (InputSpan(2, 4, b"image:b:processor:1"),))
    resized = Prompt(tokens, (InputSpan(2, 4, b"image:a:processor:2"),))
    assert first.identities()[:2] == changed.identities()[:2] == resized.identities()[:2]
    assert first.identities()[2:4] != changed.identities()[2:4]
    assert first.identities()[2:4] != resized.identities()[2:4]
    assert first.identities()[2] != first.identities()[3]
    assert first.prefix(3).identities() == first.identities()[:3]
    assert first.extend((8, 9)).identities()[:5] == first.identities()
    assert first.language_tokens() == (1, 2, 4)


@given(st.integers(0, 20), st.integers(1, 30), st.integers(0, 20), st.integers(1, 25))
def test_advancement_partitions_the_prompt_at_closed_boundaries(before, width, after, allowance):
    prompt = Prompt(
        tuple(range(before + width + after)),
        (InputSpan(before, before + width, b"unit", indivisible=True),),
    )
    position = 0
    while position < len(prompt.tokens):
        end = prompt.advance(position, allowance)
        assert position < end <= len(prompt.tokens)
        assert prompt.boundary(end)
        if end - position > allowance:
            assert (position, end) == (before, before + width)
        position = end


@pytest.mark.parametrize("tokens", [(True,), (-1,), (2**31,), (1.0,)])
def test_rejects_invalid_token_values(tokens):
    with pytest.raises(ValueError):
        Prompt(tokens)


def test_rejects_overlapping_or_out_of_bounds_semantics():
    with pytest.raises(ValueError, match="ordered"):
        Prompt((1, 2, 3), (InputSpan(0, 2, b"a"), InputSpan(1, 3, b"b")))
    with pytest.raises(ValueError, match="inside"):
        Prompt((1, 2, 3), (InputSpan(0, 4, b"a"),))
