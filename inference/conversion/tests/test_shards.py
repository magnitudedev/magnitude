"""Every shard of a split file is readable on its own at the file's alignment, and one shard can be rewritten alone."""

import numpy as np
from conversion.base import gguf

from magnitude_conversion.check import shard_facts
from magnitude_conversion.deepseek41 import ALIGNMENT, _add_file_metadata

TENSORS = {f"t{i}": np.arange(1000 + i, dtype=np.float32) for i in range(3)}


def _write(path, shards=None):
    writer = gguf.GGUFWriter(path, "llama", split_max_tensors=1)
    _add_file_metadata(writer, "https://example.org/source", shards)
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
    assert [p.name for p in (tmp_path / "one").iterdir()] == ["m-00002-of-00003.gguf"]
    name = "m-00002-of-00003.gguf"
    assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()
