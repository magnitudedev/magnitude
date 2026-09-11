import struct

import pytest

from magnitude_engine.weights.formats.gguf import ByteOrder, Encoding, InvalidGGUF, read_directory


class Source:
    def __init__(self, content: bytes):
        self.content = content

    @property
    def size(self):
        return len(self.content)

    def read(self, offset: int, length: int):
        return self.content[offset : offset + length]


def container(*, order="<", entries=None, metadata=(), alignment=32):
    def string(text):
        raw = text.encode()
        return struct.pack(order + "Q", len(raw)) + raw

    if entries is None:
        entries = [("weight", (256, 2), Encoding.Q4_K, 0)]
    header = b"GGUF" + struct.pack(order + "IQQ", 3, len(entries), len(metadata))
    for name, kind, value in metadata:
        header += string(name) + struct.pack(order + "I", kind) + value
    size = 0
    for name, dimensions, encoding, offset in entries:
        header += string(name) + struct.pack(order + "I", len(dimensions))
        header += struct.pack(order + "Q" * len(dimensions), *dimensions)
        header += struct.pack(order + "IQ", encoding, offset)
        size = max(size, offset + 288)
    return header + bytes((-len(header)) % alignment) + bytes(size)


@pytest.mark.parametrize("order,expected", [("<", ByteOrder.LITTLE), (">", ByteOrder.BIG)])
def test_directory_is_portable_and_reverses_ggml_axes(order, expected):
    directory = read_directory(Source(container(order=order)))
    tensor = directory.tensor("weight")
    assert directory.byte_order == expected
    assert tensor.shape == (2, 256)
    assert tensor.nbytes == 288
    assert tensor.encoding == Encoding.Q4_K
    assert directory.data_offset % 32 == 0
    assert type(directory).model_validate_json(directory.model_dump_json()) == directory


def test_metadata_types_and_arrays_are_preserved():
    metadata = [
        ("signed", 11, struct.pack("<q", -4)),
        ("flag", 7, b"\x01"),
        ("values", 9, struct.pack("<IQfff", 6, 3, 0.5, 1.0, -2.0)),
    ]
    directory = read_directory(Source(container(metadata=metadata)))
    assert directory.value("signed") == -4
    assert directory.value("flag") is True
    assert directory.value("values") == (0.5, 1.0, -2.0)


@pytest.mark.parametrize("length", [0, 3, 7, 8, 23, 35, 69, 351])
def test_truncated_container(length):
    with pytest.raises(InvalidGGUF):
        read_directory(Source(container()[:length]))


@pytest.mark.parametrize(
    "entries",
    [
        [("weight", (255, 2), Encoding.Q4_K, 0)],
        [("weight", (0, 2), Encoding.Q4_K, 0)],
        [("weight", (256, 2), Encoding.Q4_K, 1)],
        [("weight", (256, 2), 999, 0)],
        [("weight", (256, 2), Encoding.Q4_K, 0), ("weight", (256, 2), Encoding.Q4_K, 288)],
        [("first", (256, 2), Encoding.Q4_K, 0), ("second", (256, 2), Encoding.Q4_K, 32)],
    ],
)
def test_reject_invalid_tensor_directory(entries):
    with pytest.raises(InvalidGGUF):
        read_directory(Source(container(entries=entries)))


@pytest.mark.parametrize("value", [0, 3, 17])
def test_alignment_must_be_power_of_two(value):
    raw = container(metadata=[("general.alignment", 4, struct.pack("<I", value))])
    with pytest.raises(InvalidGGUF, match="alignment"):
        read_directory(Source(raw))


def test_duplicate_metadata_and_invalid_boolean():
    for metadata in [[("flag", 7, b"\x02")], [("a", 7, b"\x00"), ("a", 7, b"\x01")]]:
        with pytest.raises(InvalidGGUF):
            read_directory(Source(container(metadata=metadata)))


def test_oversized_metadata_is_bounded_before_read():
    raw = b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 2**64 - 1) + bytes(100)
    with pytest.raises(InvalidGGUF):
        read_directory(Source(raw))


def test_directory_order_is_independent_of_physical_order():
    entries = [("second", (256, 2), Encoding.Q4_K, 288), ("first", (256, 2), Encoding.Q4_K, 0)]
    directory = read_directory(Source(container(entries=entries)))
    assert tuple(t.name for t in directory.tensors) == ("second", "first")
