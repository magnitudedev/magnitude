import numpy as np
import pytest

from magnitude_engine.models.preparation import PreparedMedia, PreparedTensor


def test_prepared_media_roundtrip_is_exact_and_identity_covers_geometry_and_processing():
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, ::-1]
    tensor = PreparedTensor.from_array("pixels", value)
    media = PreparedMedia("a" * 64, (tensor,))
    restored = PreparedMedia.decode(media.encode(), media.buffers)
    assert restored == media and np.array_equal(restored.tensors[0].array(), value)
    assert restored.identity() == media.identity()
    assert PreparedMedia("b" * 64, (tensor,)).identity() != media.identity()
    reshaped = PreparedTensor(tensor.name, tensor.dtype, (6, 4), tensor.data)
    assert PreparedMedia(media.processor, (reshaped,)).identity() != media.identity()


@pytest.mark.parametrize(
    "shape,dtype,data",
    [
        ((2,), "float32", b"\x00" * 4),
        ((True,), "float32", b"\x00" * 4),
        ((0,), "uint8", b""),
        ((1,), "object", b"\x00" * 8),
        ((2**30,), "uint8", b""),
    ],
)
def test_prepared_tensor_rejects_invalid_geometry_before_creating_array(shape, dtype, data):
    with pytest.raises(ValueError):
        PreparedTensor("pixels", dtype, shape, data)


def test_media_rejects_duplicate_fields_and_unbound_buffers():
    tensor = PreparedTensor.from_array("pixels", np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="repeats"):
        PreparedMedia("a" * 64, (tensor, tensor))
    encoded = PreparedMedia("a" * 64, (tensor,)).encode()
    encoded["tensors"][0]["data"] = "not-base64!"
    with pytest.raises(ValueError):
        PreparedMedia.decode(encoded, (tensor.data,))
