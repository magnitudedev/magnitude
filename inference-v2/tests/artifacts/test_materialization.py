import json
import struct

import mlx.core as mx
import pytest

from magnitude_engine.artifacts.layouts import ExpertSchema, logical_tensors
from magnitude_engine.artifacts.materialization import (
    ResidentMaterializer,
    TensorPartition,
    materialize,
)
from magnitude_engine.artifacts.tensors import TensorCatalog
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader


def expert_artifact(path, *, split):
    base = "language_model.model.layers.0.mlp.switch_mlp"
    header, payload = {}, bytearray()
    components = ("gate_proj", "up_proj", "down_proj")
    records = []
    if split:
        for expert in range(3):
            for index, projection in enumerate(components):
                records.append(
                    (
                        f"{base}.experts.{expert}.{projection}.weight",
                        [2, 2],
                        [float(10 * expert + index)] * 4,
                    )
                )
    else:
        for index, projection in enumerate(components):
            records.append(
                (
                    f"{base}.{projection}.weight",
                    [3, 2, 2],
                    [float(10 * expert + index) for expert in range(3) for _ in range(4)],
                )
            )
    for name, shape, values in records:
        start = len(payload)
        payload.extend(struct.pack(f"<{len(values)}f", *values))
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header).encode()
    path.mkdir()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return TensorCatalog.inspect(path)


def declaration():
    return {
        "format": "expert_major_safetensors",
        "version": 1,
        "expert_axis": 0,
        "key_template": "{base}.experts.{expert}.{projection}.{component}",
        "physical_order": "layer_expert_component",
        "group_must_not_cross_shards": True,
    }


def schema():
    return ExpertSchema(
        "language_model.model.layers",
        "mlp.switch_mlp",
        tuple((name, "weight") for name in ("gate_proj", "up_proj", "down_proj")),
        3,
    )


def test_stacked_and_expert_major_storage_materialize_identical_logical_tensors(tmp_path):
    conventional = logical_tensors(
        expert_artifact(tmp_path / "stacked", split=False), declaration=None
    )
    split = logical_tensors(
        expert_artifact(tmp_path / "split", split=True),
        declaration=declaration(),
        expert_schema=schema(),
    )
    assert conventional.keys() == split.keys()
    budget, reader = MemoryBudget(4096), PositionalReader(workers=3, batch_bytes=64)
    loader = ResidentMaterializer(budget, reader, owner="test.weights")
    first, second = loader.materialize(conventional), loader.materialize(split)
    for name in conventional:
        assert mx.array_equal(first.arrays[name], second.arrays[name]).item()
        assert sum(piece.size for piece in split[name].row(1)) == 16
    first.close()
    second.close()
    reader.close()
    assert budget.snapshot().reserved == 0


def test_expert_storage_cannot_be_inferred_or_misdeclared(tmp_path):
    catalog = expert_artifact(tmp_path / "split", split=True)
    with pytest.raises(ValueError, match="explicit"):
        logical_tensors(catalog, declaration=None, expert_schema=schema())
    for invalid in ({**declaration(), "version": True}, {**declaration(), "expert_axis": 1}):
        with pytest.raises(ValueError, match="unsupported"):
            logical_tensors(catalog, declaration=invalid, expert_schema=schema())


def test_materialization_partitions_are_preflighted_and_failure_closes_previous_owners(tmp_path):
    tensors = logical_tensors(expert_artifact(tmp_path / "stacked", split=False), declaration=None)
    names = tuple(tensors)
    events = []

    class Owner:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def materialize(self, subset):
            events.append((self.name, frozenset(subset)))
            if self.fail:
                raise MemoryError("allocation failed")
            return self

        def close(self):
            events.append(f"close {self.name}")

    first, failing = Owner("embedding"), Owner("expert", fail=True)
    with pytest.raises(ValueError, match="unique ownership"):
        materialize(
            tensors,
            (
                TensorPartition("embedding", frozenset(names), first),
                TensorPartition("expert", frozenset(names), failing),
            ),
        )
    assert events == []
    with pytest.raises(MemoryError):
        materialize(
            tensors,
            (
                TensorPartition("embedding", frozenset(names[:1]), first),
                TensorPartition("expert", frozenset(names[1:]), failing),
            ),
        )
    assert events == [
        ("embedding", frozenset(names[:1])),
        ("expert", frozenset(names[1:])),
        "close embedding",
    ]


@pytest.mark.parametrize("bits", [None, 4, 8])
def test_projection_packing_reads_final_layout_once_and_preserves_equation(tmp_path, bits):
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from magnitude_engine.artifacts.quantization import AffineEncoding
    from magnitude_engine.models.loading.packing import ProjectionPack
    from magnitude_engine.models.loading.parameters import materialize_parameters
    from magnitude_engine.models.projections import ParallelProjections
    from performance.benchmarks.references import separate_projections

    mx.random.seed(293)
    reference = nn.Module()
    reference.a, reference.b = nn.Linear(64, 32), nn.Linear(64, 96)
    if bits is not None:
        nn.quantize(reference, bits=bits, group_size=64)
    arrays = dict(tree_flatten(reference.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), arrays)
    tensors = logical_tensors(TensorCatalog.inspect(tmp_path), declaration=None)
    encoding = (
        {} if bits is None else {name + ".weight": AffineEncoding(bits, 64) for name in ("a", "b")}
    )
    model = nn.Module()
    model.a, model.b = nn.Linear(64, 32), nn.Linear(64, 96)
    if bits is not None:
        nn.quantize(model, bits=bits, group_size=64)
    # Exactly the original byte demand fits, so duplicate resident packing cannot pass.
    expected_bytes = sum(t.nbytes for t in tensors.values())
    budget, reader = MemoryBudget(expected_bytes), PositionalReader(workers=2)
    allocation = materialize_parameters(
        model,
        tensors,
        budget=budget,
        reader=reader,
        owner="weights",
        packs=(ProjectionPack(("a", "b")),),
        encodings=encoding,
    )
    try:
        grouped = ParallelProjections((model.a, model.b), allocation.projections[("a", "b")])
        control = separate_projections(grouped)
        for rows in (1, 17):
            x = mx.random.normal((rows, 64))
            outputs = grouped(x)
            for actual, original, borrowed in zip(
                outputs, (reference.a, reference.b), control, strict=True
            ):
                assert mx.allclose(actual, original(x), atol=1e-5, rtol=1e-5).item()
                assert mx.allclose(actual, borrowed(x), atol=1e-5, rtol=1e-5).item()
        assert budget.snapshot().reserved == expected_bytes
        for name, value in tree_flatten(model.parameters()):
            assert mx.array_equal(value, arrays[name]).item()
    finally:
        allocation.close()
        reader.close()
    assert budget.snapshot().reserved == 0


def test_logical_concatenation_interleaves_fragmented_expert_rows(tmp_path):
    from magnitude_engine.models.loading.packing import concatenate

    tensors = logical_tensors(
        expert_artifact(tmp_path / "split", split=True),
        declaration=declaration(),
        expert_schema=schema(),
    )
    base = "language_model.model.layers.0.mlp.switch_mlp"
    gate, up = tensors[base + ".gate_proj.weight"], tensors[base + ".up_proj.weight"]
    packed = concatenate("packed", (gate, up), 1)
    budget, reader = MemoryBudget(1024), PositionalReader(workers=2)
    weights = ResidentMaterializer(budget, reader, owner="packed").materialize({"packed": packed})
    try:
        expected = mx.array([[[float(10 * e + p)] * 2 for p in (0, 0, 1, 1)] for e in range(3)])
        assert mx.array_equal(weights.arrays["packed"], expected).item()
    finally:
        weights.close()
        reader.close()
    assert budget.snapshot().reserved == 0
