"""Every shard of a split file is readable on its own at the file's alignment."""

import numpy as np
from conversion.base import gguf

from magnitude_conversion.deepseek41 import ALIGNMENT, _add_file_metadata


def test_every_shard_carries_the_alignment(tmp_path):
    writer = gguf.GGUFWriter(tmp_path / "m.gguf", "llama", split_max_tensors=1)
    _add_file_metadata(writer, "https://example.org/source")
    tensors = {f"t{i}": np.arange(1000 + i, dtype=np.float32) for i in range(3)}
    for name, data in tensors.items():
        writer.add_tensor(name, data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    paths = sorted(tmp_path.glob("m-*-of-00003.gguf"))
    assert len(paths) == 3
    for path in paths:
        reader = gguf.GGUFReader(path)
        assert reader.get_field(gguf.Keys.General.ALIGNMENT).contents() == ALIGNMENT
        (t,) = reader.tensors
        assert t.data_offset % ALIGNMENT == 0
        assert np.array_equal(np.asarray(t.data), tensors[t.name])
