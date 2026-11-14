"""Every shard of a split file is readable on its own at the file's alignment, and one shard can be rewritten alone,
every other shard's header beside it."""

import numpy as np
import pytest
from conversion.base import gguf

from magnitude_conversion.check import shard_facts
from magnitude_conversion.native import ALIGNMENT, add_file_metadata

TENSORS = {f"t{i}": np.arange(1000 + i, dtype=np.float32) for i in range(3)}


def _write(path, shards=None):
    writer = gguf.GGUFWriter(path, "llama", split_max_tensors=1)
    add_file_metadata(writer, "https://example.org/source", shards)
    for name, data in TENSORS.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_every_shard_carries_the_alignment(tmp_path):
    _write(tmp_path / "m.gguf")
    paths = sorted(tmp_path.glob("m-*-of-00003.gguf"))
    assert len(paths) == 3
    assert len(shard_facts(paths[0])) == 3
    for path in paths:
        reader = gguf.GGUFReader(path)
        assert reader.get_field(gguf.Keys.General.ALIGNMENT).contents() == ALIGNMENT
        (t,) = reader.tensors
        assert t.data_offset % ALIGNMENT == 0
        assert np.array_equal(np.asarray(t.data), TENSORS[t.name])


def test_one_shard_alone_is_the_whole_writes_shard(tmp_path):
    (tmp_path / "whole").mkdir()
    (tmp_path / "one").mkdir()
    _write(tmp_path / "whole" / "m.gguf")
    _write(tmp_path / "one" / "m.gguf", frozenset({2}))
    name = "m-00002-of-00003.gguf"
    assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()


def test_the_other_shards_are_their_headers_at_full_size(tmp_path):
    """Each unwritten shard holds the whole write's header bytes, then nothing, at the whole shard's size: every
    shard's facts read in the one shard's directory."""
    (tmp_path / "whole").mkdir()
    (tmp_path / "one").mkdir()
    _write(tmp_path / "whole" / "m.gguf")
    _write(tmp_path / "one" / "m.gguf", frozenset({2}))
    assert len(shard_facts(tmp_path / "one" / "m-00001-of-00003.gguf")) == 3
    for i in (1, 3):
        name = f"m-0000{i}-of-00003.gguf"
        whole, one = (tmp_path / "whole" / name).read_bytes(), (tmp_path / "one" / name).read_bytes()
        (t,) = gguf.GGUFReader(tmp_path / "whole" / name).tensors
        assert len(one) == len(whole)
        assert one[:t.data_offset] == whole[:t.data_offset]
        assert not any(one[t.data_offset:])


def test_no_shard_whole_is_the_files_headers(tmp_path):
    """With no shard written whole, every shard is its header: the file's plan, its tensors named, none read."""
    _write(tmp_path / "m.gguf", frozenset())
    readers = shard_facts(tmp_path / "m-00001-of-00003.gguf")
    assert sorted(t.name for r in readers for t in r.tensors) == sorted(TENSORS)
    assert all(not any(np.asarray(t.data).view(np.uint8)) for r in readers for t in r.tensors)


def test_a_written_shard_is_never_overwritten_by_a_header(tmp_path):
    _write(tmp_path / "m.gguf")
    with pytest.raises(ValueError, match=r"m-00001-of-00003\.gguf holds a shard's data; --shards 2"):
        _write(tmp_path / "m.gguf", frozenset({2}))
