import os
from contextlib import ExitStack

import numpy as np
import pytest

from magnitude_engine.numerics.random import words
from magnitude_engine.operations.sampling import (
    Draw,
    DrawDomain,
    SampledToken,
    SamplePosition,
    SampleSelector,
    SamplingSeed,
    SelectionFailure,
    SelectionKind,
    UnselectableDistribution,
)
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.machine import open_context


def philox(counter, key):
    """Independent integer reference using full Python products."""
    mask = 0xFFFFFFFF
    c0, c1, c2, c3 = map(int, counter)
    k0, k1 = map(int, key)
    for _ in range(10):
        a, b = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (b >> 32) ^ c1 ^ k0, b & mask, (a >> 32) ^ c3 ^ k1, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0, c1, c2, c3


@pytest.mark.device
def test_philox_known_answers_and_full_word_reference():
    # Published Random123 Philox4x32-10 known-answer vectors:
    # https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    addresses = np.array(
        [
            [0] * 6,
            [0xFFFFFFFF] * 6,
            [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0],
        ],
        np.uint32,
    )
    expected = np.array(
        [
            [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8],
            [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD],
            [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1],
        ],
        np.uint32,
    )
    extra = np.random.default_rng(392).integers(0, 2**32, (257, 6), dtype=np.uint32)
    expected = np.concatenate(
        (expected, np.array([philox(a[:4], a[4:]) for a in extra], np.uint32))
    )
    addresses = np.concatenate((addresses, extra))
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    with ExitStack() as cleanup:
        context = open_context(backend, 1024**2, 0)
        cleanup.callback(context.close)
        source = context.upload(TensorSpec(addresses.shape, DType.U32), addresses.tobytes())
        output = context.allocate(TensorSpec(expected.shape, DType.U32))
        cleanup.callback(source.close)
        cleanup.callback(output.close)
        plan = context.compile(words(len(addresses), cpu=backend == Backend.LLVM))
        ticket = context.submit([Prepared(context, plan, [source, output])])
        actual = np.frombuffer(context.read(output, after=ticket), np.uint32).reshape(
            expected.shape
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.device
def test_selection_reference_ties_invalid_rows_and_batch_order():
    rng = np.random.default_rng(124)
    logits = rng.normal(size=(7, 2059)).astype(np.float32)
    logits[0] = -np.inf
    logits[0, [2048, 37]] = 5
    logits[1] = -np.inf
    logits[2, 1050] = np.nan
    logits[3, 2049] = np.inf
    logits[5, ::2] = -np.inf
    draws = tuple(
        Draw(
            kind=SelectionKind.GREEDY if row < 4 else SelectionKind.CATEGORICAL,
            seed=SamplingSeed(2**63 + row),
            position=SamplePosition(2**40 + row),
            domain=DrawDomain.TARGET,
        )
        for row in range(7)
    )
    expected = [
        SampledToken(token=37),
        UnselectableDistribution(reason=SelectionFailure.EMPTY),
        UnselectableDistribution(reason=SelectionFailure.NONFINITE),
        UnselectableDistribution(reason=SelectionFailure.NONFINITE),
    ]
    for row in range(4, 7):
        draw = draws[row]
        w = draw.words()
        uniforms = np.array(
            [
                ((philox((token, w[3], w[4], w[5]), w[1:3])[0] >> 9) + 0.5) * 2**-23
                for token in range(logits.shape[1])
            ]
        )
        scores = logits[row] - np.log(-np.log(uniforms))
        expected.append(SampledToken(token=int(scores.argmax())))
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    with ExitStack() as cleanup:
        context = open_context(backend, 2 * 1024**2, 0)
        cleanup.callback(context.close)
        selector = SampleSelector(context)
        cleanup.callback(selector.close)
        for indices in (list(range(7)), [6, 4, 1, 0, 2, 5, 3], [4], [5], [6]):
            source = context.upload(
                TensorSpec((len(indices), logits.shape[1]), DType.F32), logits[indices].tobytes()
            )
            output = context.allocate(TensorSpec((len(indices), 2), DType.I32))
            try:
                prepared = selector.prepare(source, tuple(draws[i] for i in indices), output)
                ticket = context.submit(prepared)
                source.close()
                assert selector.read(output, after=ticket) == tuple(expected[i] for i in indices)
            finally:
                source.close()
                output.close()
        assert context.allocated_bytes == 0
