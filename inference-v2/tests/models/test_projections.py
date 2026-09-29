import mlx.core as mx
import mlx.nn as nn
import pytest

from magnitude_engine.models.projections import bind_linear, bind_readout


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("batch", [2, 4, 9])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_quantized_decode_preserves_independent_rows(dtype, batch, bits, group_size):
    mx.random.seed(17)
    source = nn.QuantizedLinear(2048, 512, bias=True, group_size=group_size, bits=bits)
    source.set_dtype(dtype)
    inputs = mx.random.normal((batch, 1, 2048)).astype(dtype)
    mx.eval(source.parameters(), inputs)
    bound = bind_linear(source)
    assert bound.weight is source.weight
    assert bound.scales is source.scales
    expected = mx.concatenate([source(row[None]) for row in inputs])
    actual = mx.compile(bound)(inputs)
    assert mx.array_equal(actual, expected).item()
    # Matrix/prefill work retains the upstream path.
    wide = mx.broadcast_to(inputs[:1], (1, 16, 2048))
    assert mx.array_equal(bound(wide), source(wide)).item()


def test_tied_quantized_readout_borrows_vocabulary_and_preserves_rows():
    mx.random.seed(18)
    source = nn.QuantizedEmbedding(512, 2048, group_size=64, bits=4)
    source.set_dtype(mx.bfloat16)
    inputs = mx.random.normal((2, 1, 2048)).astype(mx.bfloat16)
    mx.eval(source.parameters(), inputs)
    bound = bind_readout(source)
    assert bound.weight is source.weight
    assert mx.array_equal(
        bound(inputs), mx.concatenate([source.as_linear(row[None]) for row in inputs])
    ).item()


@pytest.mark.parametrize("batch,query", [(1, 1), (3, 2), (9, 1), (2, 8)])
@pytest.mark.parametrize("width,outputs", [(2048, 257), (2048, 512), (128, 32), (512, 4096)])
@pytest.mark.parametrize("bits", [4, 8])
def test_short_queries_share_weights_without_changing_reduction(batch, query, width, outputs, bits):
    mx.random.seed(39)
    source = nn.QuantizedLinear(width, outputs, bias=True, group_size=64, bits=bits)
    source.set_dtype(mx.bfloat16)
    inputs = mx.random.normal((batch, query, width)).astype(mx.bfloat16)
    mx.eval(source.parameters(), inputs)
    bound = bind_linear(source)
    expected = mx.concatenate([source(row[None]) for row in inputs.reshape(-1, width)]).reshape(
        batch, query, outputs
    )
    actual = mx.compile(bound)(inputs)
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(bound(inputs[::-1]), actual[::-1]).item()
