import pytest

from magnitude_engine.inputs.layout import BoundaryRule, InputLayout, InputSpan
from magnitude_engine.models.qwen35.inputs import InputPlan, InputState


def test_semantic_chunk_boundaries_and_language_history():
    layout = InputLayout(
        count=14,
        spans=(
            InputSpan(start=2, end=7, identity="image-a", boundaries=BoundaryRule.INDIVISIBLE),
            InputSpan(start=9, end=13, identity="image-b"),
        ),
    )
    assert layout.chunk_end(0, 14, 5) == 2
    assert layout.chunk_end(2, 14, 1) == 7
    assert layout.chunk_end(7, 14, 3) == 10
    assert not layout.boundary(5)
    assert layout.boundary(10)
    assert layout.boundary(100)
    assert [i for i in range(16) if layout.language(i)] == [0, 1, 7, 8, 13, 14, 15]
    with pytest.raises(ValueError, match="legal"):
        layout.chunk_end(2, 6, 1)
    with pytest.raises(ValueError):
        InputLayout(count=3, spans=(InputSpan(start=1, end=4, identity="bad"),))
    assert InputLayout.model_validate_json(layout.model_dump_json()) == layout


def test_text_continuation_checks_prompt_and_preserves_position_meaning():
    plan = InputPlan.text((1, 2, 3))
    state = InputState(plan, 0, (), 8)
    assert state.assemble((1, 2)).coordinates == ((0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="bound prompt"):
        state.assemble((1, 3))
    following = state.after(2)
    state.close()
    assert following.assemble((3, 5, 8)).coordinates == ((2, 2, 2), (3, 3, 3), (4, 4, 4))
    with pytest.raises(ValueError, match="already released"):
        following.after(1)
    following.close()
