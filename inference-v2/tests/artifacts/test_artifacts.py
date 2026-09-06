import json
import struct

import mlx.core as mx
import pytest

from magnitude_engine.artifacts.tensors import TensorCatalog, inspect_shard


def raw_shard(path, header, data=b""):
    encoded = header.encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.mark.parametrize(
    "header,data",
    [
        ('{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}', b"0000"),
        ('{"x":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}', b"000"),
        ('{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}', b"0"),
        ('{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b"00"),
        ('{"x":{},"x":{}}', b""),
        ('{"__metadata__":{"bad":3}}', b""),
    ],
)
def test_invalid_shards_are_rejected_before_tensor_allocation(tmp_path, header, data):
    path = tmp_path / "model.safetensors"
    raw_shard(path, header, data)
    with pytest.raises(ValueError):
        inspect_shard(path)


def test_index_must_exactly_describe_shards_and_cannot_traverse_paths(tmp_path):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"a": mx.ones((2, 3))})
    index = tmp_path / "model.safetensors.index.json"
    for mapping in ({"a": "../model.safetensors"}, {"missing": "model.safetensors"}):
        index.write_text(json.dumps({"weight_map": mapping}))
        with pytest.raises(ValueError):
            TensorCatalog.inspect(tmp_path)
    index.write_text(json.dumps({"weight_map": {"a": "model.safetensors"}}))
    catalog = TensorCatalog.inspect(tmp_path)
    assert catalog.tensors["a"].row(1).size == 12
    with pytest.raises(IndexError):
        catalog.tensors["a"].row(2)
