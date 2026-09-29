"""Bounded physical gather plans; executed only at the replacement gate."""

import struct

import pytest

import ops
from ops.compiler.streaming import GatherLoop
from ops.runtime.imports import plan_import


def loop():
    content = struct.pack("=32f", *range(32))
    source = ops.MemorySource(content)
    binding = ops.Binding(
        ops.TensorSpec((8, 4), ops.DType.F32), "embedding", ops.Residency.STREAMED,
        (ops.SourcePlane(ops.SourceSpan(source, 0, len(content)), 8, 32),),
        ops.DenseImport(ops.DType.F32),
    )
    prototype = binding.region(0, 16, shape=(4, 4))
    return GatherLoop(1, 0, 2, 5, 2, 2, 4, binding, prototype, (), 4096)


def read(binding):
    span = binding.planes[0].span
    result = bytearray(span.length)
    assert span.source.read_into(span.offset, memoryview(result)) == span.length
    return struct.unpack("=16f", result)


def geometry(binding):
    plan = plan_import(binding)
    return tuple((item.source_spec, item.target_offset, item.elements, item.first_block)
                 for item in plan.steps)


def test_gather_preserves_original_token_order_and_encoding_groups():
    plan = loop()
    binding, locations = plan.gather((7, 2))
    assert locations == (3, 0)
    assert read(binding) == tuple((*range(8, 16), *range(24, 32)))
    assert geometry(binding) == geometry(plan.prototype)
    assert binding.source_bytes < plan.source.source_bytes


def test_duplicate_rows_read_one_group_and_pad_without_eager_table_storage():
    plan = loop()
    binding, locations = plan.gather((7, 7))
    assert locations == (1, 1)
    assert read(binding) == tuple((*range(24, 32), *([0] * 8)))
    assert geometry(binding) == geometry(plan.prototype)
    spans = binding.planes[0].span.source.spans
    assert len(spans) == 2
    assert isinstance(spans[1].source, ops.ZeroSource)


@pytest.mark.parametrize("rows", [(), (1, 2, 3), (-1,), (8,)])
def test_invalid_gather_fails_before_source_io(rows):
    with pytest.raises(ValueError):
        loop().gather(rows)
