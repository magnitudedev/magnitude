import base64
import io

import numpy as np
import pytest
from PIL import Image

from engine.inputs.media import PreparedMedia, PreparedTensor
from engine.serving.images import resolve_images


def source(color):
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(output, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        },
    }


def test_prepared_bytes_are_lossless_immutable_and_endian_independent():
    values = np.array([[1.25, -4.5]], dtype=">f4")
    tensor = PreparedTensor.from_array("pixels", values)
    values[:] = 0
    np.testing.assert_array_equal(tensor.array(), [[1.25, -4.5]])
    assert not tensor.array().flags.writeable
    with pytest.raises(ValueError):
        tensor.array()[0, 0] = 2
    media = PreparedMedia("a" * 64, (tensor,))
    assert media.identity != PreparedMedia("b" * 64, (tensor,)).identity
    assert media.nbytes == 8


def test_tensor_geometry_and_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="geometry"):
        PreparedTensor("pixels", "float32", (2,), b"1234")
    tensor = PreparedTensor.from_array("pixels", np.ones((2,), dtype=np.float32))
    with pytest.raises(ValueError, match="repeats"):
        PreparedMedia("a" * 64, (tensor, tensor))


def test_source_order_and_caller_ownership():
    red, blue = source("red"), source("blue")
    messages = [{"role": "user", "content": [red, {"type": "text", "text": "then"}, blue]}]
    normalized, images = resolve_images(messages)
    assert [im.getpixel((0, 0)) for im in images] == [(255, 0, 0), (0, 0, 255)]
    assert normalized[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "then"},
        {"type": "image"},
    ]
    assert messages[0]["content"][0] is red
    assert red["type"] == "image_url"


@pytest.mark.parametrize(
    "url", ["https://example.com/image.png", "file:///tmp/image.png", "data:image/png;base64,!!!"]
)
def test_sources_do_not_trigger_external_io_or_accept_malformed_bytes(url):
    with pytest.raises(ValueError):
        resolve_images([{"content": [{"type": "image_url", "image_url": {"url": url}}]}])


def test_aggregate_image_limits(monkeypatch):
    from engine.serving import images

    monkeypatch.setattr(images, "MAX_PIXELS", 300)
    with pytest.raises(ValueError, match="pixel limits"):
        resolve_images([{"content": [source("red"), source("blue")]}])
