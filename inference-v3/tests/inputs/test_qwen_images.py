import numpy as np
import pytest

from engine.data import TokenId
from engine.inputs.media import PreparedMedia, PreparedTensor
from engine.models.qwen35.preparation import ImageGeometry, interpret

GEOMETRY = ImageGeometry(
    channels=1,
    temporal_patch=1,
    patch=1,
    merge=2,
    image_token=TokenId(99),
    start_token=TokenId(98),
    end_token=TokenId(100),
)
PROCESSOR = "a" * 64


def media(grids=((1, 2, 4),), offset=0):
    count = sum(t * h * w for t, h, w in grids)
    return PreparedMedia(
        PROCESSOR,
        (
            PreparedTensor.from_array(
                "pixel_values", np.arange(count, dtype=np.float32)[:, None] + offset
            ),
            PreparedTensor.from_array("image_grid_thw", np.asarray(grids, dtype=np.int64)),
        ),
    )


def test_image_coordinates_continuation_and_history_are_not_token_offsets():
    tokens = tuple(map(TokenId, (7, 98, 99, 99, 100, 8)))
    result = interpret(tokens, media(), processor=PROCESSOR, geometry=GEOMETRY)
    assert result.plan.coordinates == (
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (2, 2, 3),
        (4, 4, 4),
        (5, 5, 5),
    )
    assert result.plan.rotary(6, 2) == ((6, 6, 6), (7, 7, 7))
    square = interpret(
        tuple(map(TokenId, (7, 98, 99, 99, 99, 99, 100, 8))),
        media(((1, 4, 4),)),
        processor=PROCESSOR,
        geometry=GEOMETRY,
    )
    assert square.plan.continuation == 6
    assert square.plan.rotary(8, 1) == ((6, 6, 6),)
    assert square.plan.layout.boundary(4)  # Causal image can be chunked.
    assert not square.plan.layout.language(4)
    assert square.plan.layout.language(8)


def test_changed_pixels_cannot_alias_the_same_placeholder_prefix():
    tokens = tuple(map(TokenId, (98, 99, 99, 100)))
    first = interpret(tokens, media(), processor=PROCESSOR, geometry=GEOMETRY)
    changed = interpret(tokens, media(offset=1), processor=PROCESSOR, geometry=GEOMETRY)
    assert first.plan.layout.spans[0].identity != changed.plan.layout.spans[0].identity
    with pytest.raises(ValueError, match="processor"):
        interpret(tokens, media(), processor="b" * 64, geometry=GEOMETRY)


@pytest.mark.parametrize(
    "tokens", [(99, 99, 100), (98, 99, 100), (98, 99, 99, 8), (98, 99, 99, 100, 99), (1, 2, 3)]
)
def test_exact_placeholder_alignment_is_required(tokens):
    with pytest.raises(ValueError):
        interpret(tuple(map(TokenId, tokens)), media(), processor=PROCESSOR, geometry=GEOMETRY)


def test_each_image_retains_its_own_patch_slice_and_order():
    tokens = tuple(map(TokenId, (98, 99, 99, 100, 7, 98, 99, 100)))
    result = interpret(
        tokens, media(((1, 2, 4), (1, 2, 2))), processor=PROCESSOR, geometry=GEOMETRY
    )
    assert [(s.start, s.end) for s in result.plan.layout.spans] == [(1, 3), (6, 7)]
    np.testing.assert_array_equal(result.images[0].pixels.array().ravel(), np.arange(8))
    np.testing.assert_array_equal(result.images[1].pixels.array().ravel(), np.arange(8, 12))


def test_grid_products_cannot_overflow_into_a_valid_payload():
    supplied = PreparedMedia(
        PROCESSOR,
        (
            PreparedTensor.from_array("pixel_values", np.zeros((4, 1), dtype=np.float32)),
            PreparedTensor.from_array(
                "image_grid_thw", np.array([[1, 2**32, 2**32]], dtype=np.int64)
            ),
        ),
    )
    with pytest.raises(ValueError, match="cover"):
        interpret(
            (TokenId(98), TokenId(99), TokenId(100)),
            supplied,
            processor=PROCESSOR,
            geometry=GEOMETRY,
        )
