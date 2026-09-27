"""Packed history and dense current rows share one reference across schedules."""

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.kernels.schedules import ProbabilityTransfer
from ops.kv_codecs import decode_kv_reference, encode_kv_reference


class ProbabilitySchedule:
    def __init__(self, transfer):
        self.transfer = transfer

    def select(self, request, default):
        if hasattr(default, "probability_transfer"):
            return next(s for s in request.candidates if s.probability_transfer == self.transfer)
        return default


@pytest.mark.device
@pytest.mark.parametrize("backend", ["metal", "cuda"])
@pytest.mark.parametrize("codec", [ops.affine_k8_uniform_v4, ops.rotated_k4_uniform_v4])
@pytest.mark.parametrize("has_history", [True, False])
@pytest.mark.parametrize(
    "mode,transfer",
    [
        ("prefill", ProbabilityTransfer.SHARED),
        ("prefill", ProbabilityTransfer.INFERRED),
        ("decode", ProbabilityTransfer.SHARED),
    ],
)
def test_persistent_attention_tail_empty_and_mixed_sources(
    backend, codec, mode, transfer, has_history
):
    if backend == "metal" and not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(712)
    rows, heads, width, capacity, kvheads = 33, 16, 256, 97, 2
    tensors = [
        torch.from_numpy(rng.normal(0, 0.3, shape).astype(np.float32)).bfloat16()
        for shape in ((rows, heads, width), (rows, kvheads, width), (rows, kvheads, width))
    ]
    state_spec = ops.kv_state_spec(capacity, kvheads, ops.DType.BF16, codec(width, width))
    physical = encode_kv_reference(
        rng.normal(0, 0.3, state_spec.shape).astype(np.float32), state_spec.representation
    )
    logical = decode_kv_reference(physical, state_spec).astype(np.float64)
    visible = np.array(
        [
            [(row % 3) * 3, 75 - row % 5, row % 3, min(row + 1, rows - row % 3)]
            for row in range(rows)
        ],
        np.int32,
    )
    if not has_history:
        visible[:, :2] = 0
    # Empty and differently based intervals remain valid inside one sequence.
    # A shared tile must stage their union and preserve each row's own mask.
    visible[0] = [0, 0, 0, 0]
    visible[1] = [3, 75, 0, 0] if has_history else [0, 0, 0, 0]
    visible[2] = [0, 0, 1, 2]
    q, k, v = (t.float().numpy().astype(np.float64) for t in tensors)
    expected = np.zeros(q.shape, np.float64)
    for row, (start, count, current, current_count) in enumerate(visible):
        if count + current_count == 0:
            continue
        for head in range(heads):
            kv = head // 8
            keys = np.concatenate(
                (
                    logical[start : start + count, kv, :width],
                    k[current : current + current_count, kv],
                )
            )
            values = np.concatenate(
                (
                    logical[start : start + count, kv, width:],
                    v[current : current + current_count, kv],
                )
            )
            score = keys @ q[row, head] / 16
            probability = np.exp(score - score.max())
            expected[row, head] = (probability / probability.sum()) @ values
    expected = torch.from_numpy(expected.astype(np.float32)).bfloat16().float().numpy()
    specs = (
        ops.TensorSpec(tuple(tensors[0].shape), ops.DType.BF16),
        state_spec,
        *(ops.TensorSpec(tuple(t.shape), ops.DType.BF16) for t in tensors[1:]),
        ops.TensorSpec(visible.shape, ops.DType.I32),
    )
    contents = (
        tensors[0].view(torch.uint8).numpy().tobytes(),
        physical,
        *(t.view(torch.uint8).numpy().tobytes() for t in tensors[1:]),
        visible.tobytes(),
    )
    signature = ops.Signature(
        tuple(
            ops.Argument(
                s, name, ops.ValueKind.RESOURCE if name == "history" else ops.ValueKind.INPUT
            )
            for s, name in zip(
                specs, ("query", "history", "keys", "values", "visible"), strict=True
            )
        )
    )
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend=backend, maximum_bytes=64 << 20)
    ) as device:
        # Packed planes must honor a borrowed allocation's nonzero byte offset.
        backings = tuple(
            device.upload(ops.TensorSpec((len(b) + 256,), ops.DType.U8), bytes([0x55]) * 256 + b)
            for b in contents
        )
        resources = tuple(backing.view(s, 256) for s, backing in zip(specs, backings, strict=True))
        program = None
        try:
            try:
                program = ops.compile(
                    lambda q, h, k, v, r: ops.persistent_attention(q, h, k, v, r, sequence_count=1),
                    signature=signature,
                    device=device,
                    constants={},
                    options=ops.CompileOptions(mode=mode, schedules=ProbabilitySchedule(transfer)),
                )
            except Exception as error:
                if (
                    transfer == ProbabilityTransfer.INFERRED
                    and "Layout infer conflict between scores and probability" in str(error)
                ):
                    pytest.xfail(
                        "TileLang rejects incompatible score/probability fragment ownership"
                    )
                raise
            execution = program.submit(
                resources[0], *resources[2:], resources={"history": resources[1]}
            )
            execution.completion.wait()
            try:
                actual = execution.outputs[0].native.cpu().float().numpy()
                np.testing.assert_allclose(actual, expected, rtol=0.03, atol=0.003)
                assert device.read(resources[1]) == physical
            finally:
                for output in execution.outputs:
                    output.close()
        finally:
            if program is not None:
                program.close()
            for resource in resources:
                resource.close()
            for backing in backings:
                backing.close()
