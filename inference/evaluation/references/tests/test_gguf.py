from pathlib import Path

import numpy as np
from gguf.constants import GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize as ggml_dequantize

from magnitude_references.gguf import Package, TensorInfo, Value, Writer, dequantize, quantize


def test_writer_parser_round_trip(tmp_path: Path) -> None:
    """Every value type and several tensor encodings survive a write and a read."""
    rng = np.random.default_rng(0)
    metadata = {
        "general.architecture": Value(GGUFValueType.STRING, "synthetic"),
        "synthetic.count": Value(GGUFValueType.UINT32, 7), "synthetic.offset": Value(GGUFValueType.INT64, -3),
        "synthetic.scale": Value(GGUFValueType.FLOAT32, 0.5), "synthetic.flag": Value(GGUFValueType.BOOL, True),
        "synthetic.ids": Value(GGUFValueType.ARRAY, [1, 2, 3], GGUFValueType.INT32),
        "synthetic.words": Value(GGUFValueType.ARRAY, ["a", "", "ç"], GGUFValueType.STRING),
        "synthetic.none": Value(GGUFValueType.ARRAY, [], GGUFValueType.STRING),
    }
    kinds = [GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16,
             GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_0]
    tensors = {f"t.{kind.name}": rng.standard_normal((3, 64)).astype(np.float32) for kind in kinds}
    path = tmp_path / "t.gguf"
    writer = Writer(path, metadata)
    encoded = {name: quantize(values, kind) for (name, values), kind in zip(tensors.items(), kinds)}
    for (name, values), kind in zip(tensors.items(), kinds):
        writer.declare(name, tuple(reversed(values.shape)), kind)
    writer.begin()
    for name, data in encoded.items():
        writer.write(name, data)
    writer.close()
    package = Package.local(path)
    assert package.metadata == metadata
    for (name, values), kind in zip(tensors.items(), kinds):
        decoded = package.tensor(name)
        expected = dequantize(TensorInfo(name, (64, 3), kind, 0), encoded[name].reshape(-1))
        assert np.array_equal(decoded, expected), name
        tolerance = {"F32": 0, "F16": 1e-3, "BF16": 1e-2, "Q8_0": 2e-2, "Q4_0": 0.3}[kind.name]
        assert np.abs(decoded - values).max() <= tolerance * np.abs(values).max(), name


def test_mxfp4_tables_equal_the_gguf_decoder_bit_for_bit() -> None:
    """The table decode of MXFP4 gives gguf-py's values bit for bit, over every scale byte and every
    nibble pair, as a stacked expert matrix decodes."""
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, size=(4096, 17), dtype=np.uint8)
    blocks[:256, 0] = np.arange(256, dtype=np.uint8)
    blocks[:256, 1:] = np.arange(256, dtype=np.uint8)[:, None]
    blocks[256:512, 1:] = np.arange(256, dtype=np.uint8).reshape(16, 16).repeat(16, axis=0)
    info = TensorInfo("t.MXFP4", (64, 4096 * 32 // 64), GGMLQuantizationType.MXFP4, 0)
    ours = dequantize(info, blocks.reshape(-1))
    theirs = ggml_dequantize(blocks.reshape(info.array_shape[0], -1), GGMLQuantizationType.MXFP4)
    assert np.array_equal(ours.view(np.uint32), np.asarray(theirs, dtype=np.float32).reshape(ours.shape).view(np.uint32))
