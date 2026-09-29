import mlx.core as mx
import pytest

from magnitude_engine.artifacts.encoding import AffineMaterializer
from magnitude_engine.artifacts.layouts import logical_tensors
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.artifacts.tensors import TensorCatalog
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.io.reader import PositionalReader


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_projection_encoding_matches_mlx_with_one_source_projection_peak(tmp_path, bits, dtype):
    mx.random.seed(33)
    floating = {
        "up.weight": mx.random.normal((3, 64, 64)).astype(dtype),
        "down.weight": mx.random.normal((3, 64, 64)).astype(dtype),
        "norm.weight": mx.arange(64).astype(dtype),
    }
    mx.save_safetensors(str(tmp_path / "model.safetensors"), floating)
    tensors = logical_tensors(TensorCatalog.inspect(tmp_path), declaration=None)
    plan = {name: AffineEncoding(bits, 32) for name in floating if name != "norm.weight"}
    expected = {"norm.weight": floating["norm.weight"]}
    for name in plan:
        values = mx.quantize(floating[name], bits=bits, group_size=32)
        for suffix, value in zip((".weight", ".scales", ".biases"), values, strict=True):
            expected[name[:-7] + suffix] = value
    final = sum(a.nbytes for a in expected.values())
    peak = final + max(a.nbytes for a in floating.values())
    budget, reader = MemoryBudget(peak), PositionalReader(workers=2)
    try:
        owned = AffineMaterializer(budget, reader, plan, owner="head").materialize(tensors)
        assert owned.arrays.keys() == expected.keys()
        for name, value in expected.items():
            assert mx.array_equal(owned.arrays[name], value).item(), name
        assert budget.snapshot().reserved == final
        assert budget.snapshot().peak == peak
        owned.close()
        assert budget.snapshot().reserved == 0
        budget.limit = peak - 1
        with pytest.raises(MemoryError):
            AffineMaterializer(budget, reader, plan, owner="head").materialize(tensors)
        assert budget.snapshot().reserved == 0
    finally:
        reader.close()


def test_encoding_read_failure_releases_partial_final_storage(tmp_path):
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            "a.weight": mx.ones((64, 64)),
            "b.weight": mx.ones((64, 64)),
        },
    )
    tensors = logical_tensors(TensorCatalog.inspect(tmp_path), declaration=None)
    budget = MemoryBudget(1 << 20)

    class BrokenReader(PositionalReader):
        count = 0

        def submit(self, reads):
            self.count += 1
            if self.count == 2:
                raise OSError("injected second tensor failure")
            return super().submit(reads)

    reader = BrokenReader(workers=1)
    try:
        plan = {name: AffineEncoding(4, 32) for name in tensors}
        with pytest.raises(OSError, match="injected"):
            AffineMaterializer(budget, reader, plan, owner="head").materialize(tensors)
        assert budget.snapshot().reserved == 0
    finally:
        reader.close()
