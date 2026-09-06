import mlx.core as mx
import numpy as np
import pytest

from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.tensors import TensorCatalog
from magnitude_engine.models.embeddings.resident import ResidentAffineEmbedding
from magnitude_engine.models.embeddings.streaming import StreamedEmbedding
from magnitude_engine.models.embeddings.table import AffineRowTable
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader


def table_fixture(tmp_path, dtype, bits):
    values = mx.random.normal((9, 64), key=mx.random.key(17)).astype(dtype)
    weight, scale, bias = mx.quantize(values, group_size=32, bits=bits)
    payload = {}
    for shard, (start, stop) in enumerate(((0, 4), (4, 9))):
        for name, tensor in zip(("weight", "scales", "biases"), (weight, scale, bias), strict=True):
            payload[f"table.{shard}.{name}"] = tensor[start:stop]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), payload)
    catalog = TensorCatalog.inspect(tmp_path)
    shards = tuple(
        tuple(
            catalog.tensors[f"table.{index}.{component}"]
            for component in ("weight", "scales", "biases")
        )
        for index in range(2)
    )
    encoding = AffineEncoding(bits, 32)
    return AffineRowTable(shards, encoding), ResidentAffineEmbedding(weight, scale, bias, encoding)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_streamed_rows_exactly_match_resident_decode_across_shards(tmp_path, dtype, bits):
    table, resident = table_fixture(tmp_path, dtype, bits)
    budget = MemoryBudget(1 << 24)
    reader = PositionalReader(workers=2)
    streamed = StreamedEmbedding(table, reader, budget, cache_bytes=4096)
    owner = ExecutionOwner()
    ids = mx.array([[8, 0, 8], [3, 4, 2]], dtype=mx.int32)
    with owner.scope() as scope:
        actual = streamed.lookup(ids, scope)
        expected = resident.lookup(ids, scope)
        pending = scope.seal(actual, expected)
    assert budget.snapshot().owners["target.embedding.staging"] > 0
    with pytest.raises(RuntimeError, match="retire"):
        streamed.close()
    pending.complete()
    assert mx.array_equal(actual, expected).item()
    assert budget.snapshot().owners.get("target.embedding.staging", 0) == 0
    assert streamed.metrics["unique_rows"] == 5
    assert streamed.metrics["bytes_read"] == 5 * table.row_bytes
    with owner.scope() as scope:
        again = streamed.lookup(ids, scope)
        pending = scope.seal(again)
    pending.complete()
    assert streamed.metrics["cache_hits"] == 5
    assert streamed.metrics["bytes_read"] == 5 * table.row_bytes
    streamed.close()
    reader.close()
    owner.close()
    assert budget.snapshot().reserved == 0


def test_bounded_lookahead_and_invalid_rows_release_reservations(tmp_path):
    table, _ = table_fixture(tmp_path, mx.float32, 4)
    budget = MemoryBudget(1 << 20)
    reader = PositionalReader()
    lookup = StreamedEmbedding(table, reader, budget, cache_bytes=0, max_pending=1)
    for ids in (np.array([-1]), np.array([9]), np.array([0.5])):
        with pytest.raises(ValueError):
            lookup.prepare(ids)
        assert budget.snapshot().reserved == 0
    lease = lookup.prepare(np.array([0, 1, 2]))
    with pytest.raises(MemoryError, match="queue"):
        lookup.prepare(np.array([1]))
    lease.close()
    assert budget.snapshot().reserved == 0
    lookup.close()
    reader.close()


def test_short_read_fails_without_publishing_partial_rows_or_leaking_staging(tmp_path):
    table, _ = table_fixture(tmp_path, mx.float32, 4)
    path = tmp_path / "model.safetensors"
    with path.open("r+b") as stream:
        stream.truncate(0)
    budget = MemoryBudget(1 << 20)
    reader = PositionalReader(workers=3)
    lookup = StreamedEmbedding(table, reader, budget, cache_bytes=4096)
    owner = ExecutionOwner()
    with pytest.raises(BaseExceptionGroup):
        with owner.scope() as scope:
            lookup.lookup(mx.array([1, 8]), scope)
    assert budget.snapshot().reserved == 0
    assert lookup.metrics["cache_hits"] == 0
    lookup.close()
    reader.close()
