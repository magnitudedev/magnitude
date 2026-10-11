from pathlib import Path

import numpy as np
from gguf.constants import GGMLQuantizationType, GGUFValueType

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
