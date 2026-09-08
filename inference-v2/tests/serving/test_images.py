import base64
import io

import pytest
from PIL import Image

from magnitude_engine.serving.images import extract_images
from magnitude_engine.serving.template import normalize_messages


def data_url(color):
    output = io.BytesIO()
    Image.new("RGB", (7, 5), color).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_image_parts_preserve_order_and_historical_message_association():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "image_url", "image_url": {"url": data_url("red")}},
                {"type": "text", "text": "Then"},
                {"type": "image_url", "image_url": {"url": data_url("blue")}},
            ],
        },
        {"role": "assistant", "content": "I see two images."},
        {"role": "user", "content": "Compare them."},
    ]
    prepared = normalize_messages(messages, allow_images=True)
    images = extract_images(prepared)
    assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]
    assert prepared[0]["content"] == [
        {"type": "text", "text": "First"},
        {"type": "image"},
        {"type": "text", "text": "Then"},
        {"type": "image"},
    ]
    assert messages[0]["content"][1]["type"] == "image_url"
    assert prepared[1:] == messages[1:]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://example.com/image.png",
        "data:image/png,raw",
        "data:image/png;base64,broken!!",
        "data:image/png;base64," + base64.b64encode(b"text").decode(),
    ],
)
def test_invalid_or_disallowed_sources_fail_before_model_preparation(url):
    with pytest.raises(ValueError):
        extract_images(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ]
        )


def test_text_only_composition_rejects_images_instead_of_discarding_them():
    with pytest.raises(ValueError, match="media-aware"):
        normalize_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url("red")}},
                    ],
                }
            ]
        )


def test_required_tool_instruction_preserves_ordered_system_content_and_sources():
    from magnitude_engine.serving.tool_choice import ToolSelection

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Use the reference."},
                {"type": "image_url", "image_url": {"url": data_url("red")}},
            ],
        },
        {"role": "user", "content": "Describe it using the tool."},
    ]
    prepared = normalize_messages(messages, allow_images=True)
    instructed = ToolSelection((), True, "describe").instruct(prepared, parallel=False)
    assert instructed[0]["content"][:2] == messages[0]["content"]
    assert instructed[0]["content"][-1] == {
        "type": "text",
        "text": "Call the supplied tool 'describe' to answer this request.",
    }
    assert len(messages[0]["content"]) == 2
    assert extract_images(instructed)[0].getpixel((0, 0)) == (255, 0, 0)
