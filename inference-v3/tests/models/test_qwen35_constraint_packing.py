"""Selection masks follow both ordinary and packed-control model invocations."""

import pytest

import ops
from engine.models.qwen35.description import MixerKind
from engine.models.qwen35.tensor_program import InvocationSpecs, TensorProgram
from tests.models.test_qwen35_tensor_program import Residency, _description
from tests.ops.test_compiler import Runtime


class DeferredRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.submissions = 0
        self.transfers = []

    def compile(self, program, signature):
        executable = super().compile(program, signature)
        bind = executable.bind

        def recorded_bind(*args):
            bound = bind(*args)
            submit = bound.submit

            def recorded_submit(*args):
                self.submissions += 1
                return submit(*args)

            bound.submit = recorded_submit
            return bound

        executable.bind = recorded_bind
        return executable

    def upload_async(self, spec, content):
        self.transfers.append(bytes(content))
        return super().upload_async(spec, content)


@pytest.mark.parametrize("failure", [None, "mask_error", "wrong_size"])
def test_deferred_masks_follow_forward_and_abort_without_advancing(failure):
    import struct

    from engine.data import TokenId
    from engine.models.qwen35.inputs import InputPlan
    from engine.models.qwen35.runtime import DenseRuntime
    from engine.models.sequence import LogitsSelection, ModelRequest
    from tests.models.test_qwen35_runtime import ModelResidency, _description

    native = DeferredRuntime()
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(_description(), device, residency, max_sequences=3)
    sequences = [model.create(InputPlan.text((TokenId(1),))) for _ in range(3)]

    class Provider:
        def __init__(self, mask):
            self.content, self.calls = mask, 0

        def mask(self):
            assert native.submissions == 1  # Numerical work is already submitted.
            self.calls += 1
            if failure == "mask_error":
                raise ValueError("mask failed")
            return b"" if failure == "wrong_size" else self.content

    providers = [Provider(b"\x01\x00\x00\x80"), None, Provider(b"\x02\x00\x00\x40")]
    requests = tuple(
        ModelRequest(sequence, (TokenId(1),), LogitsSelection.LAST, (0,) * 6, provider)
        for sequence, provider in zip(sequences, providers, strict=True)
    )
    batch = None
    try:
        if failure:
            with pytest.raises(ValueError, match="mask"):
                model.prepare(requests)
            assert native.submissions == 1 and not native.transfers
        else:
            batch = model.prepare(requests)
            assert native.submissions == 2
            assert native.transfers == [
                bytes(3 * 6 * 4) + providers[0].content + providers[2].content
                + struct.pack("<3i", 0, -1, 1)
            ]
            assert providers[0].calls == providers[2].calls == 1
            specs = next(iter(model.program._compiled))[1]
            assert specs.output_rows is not None and specs.draws is None and specs.masks is None
            batch.close()
            batch = None
        assert all(sequence.position == 0 and sequence.pending is None for sequence in sequences)
        assert not model._forwards
        device.drain()  # Abandoning logical work does not discharge physical work.
    finally:
        if batch is not None:
            batch.close()
        for sequence in sequences:
            sequence.close()
        model.close()
        for resource in residency.resources:
            resource.close()
        device.close()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("sampled", [False, True])
def test_selected_logits_allow_independent_or_fused_sampling(packed, sampled):
    original = _description()
    description = original.model_copy(
        update={
            "geometry": original.geometry.model_copy(update={"layers": (MixerKind.RECURRENT,)}),
            "blocks": (original.blocks[1],),
        }
    )
    device = ops.DeviceRuntime(Runtime(), budget_bytes=1 << 24)
    residency = Residency(device)
    program = TensorProgram(description, device, residency)
    specs = InvocationSpecs(
        batch=1,
        tokens=ops.TensorSpec((1,), ops.DType.I32),
        coordinates=ops.TensorSpec((1, 3), ops.DType.I32),
        recurrent_offsets=ops.TensorSpec((2,), ops.DType.I32),
        output_rows=ops.TensorSpec((1,), ops.DType.I32),
        draws=ops.TensorSpec((1, 6), ops.DType.U32) if sampled else None,
        masks=ops.TensorSpec((1, 1), ops.DType.U32) if sampled else None,
        mask_rows=ops.TensorSpec((1,), ops.DType.I32) if sampled else None,
        destinations=(),
        visible=(),
        attention_state=(),
        convolution_state=(ops.TensorSpec((1, 16, 2), ops.DType.F16),),
        delta_state=(ops.TensorSpec((1, 2, 4, 4), ops.DType.F32),),
        packed_controls=packed,
    )
    resources = []
    execution = None
    try:
        compiled = program.specialize("decode", specs, precision="reference")
        nodes = [node for node in compiled.graph.nodes if node.operation == "sample_constrained"]
        assert len(nodes) == int(sampled)
        if sampled:
            node = nodes[0]
            assert compiled.graph.value(node.inputs[2]).spec == specs.masks
            assert compiled.graph.value(node.inputs[3]).spec == specs.mask_rows
        else:
            assert not any(node.operation == "sample" for node in compiled.graph.nodes)
        dynamic_specs = (specs.control_record,) if packed else specs.control_specs
        dynamic = [device.allocate(spec) for spec in dynamic_specs]
        resources.extend(dynamic)
        bound = {
            "recurrent.0.0.convolution": device.allocate(specs.convolution_state[0]),
            "recurrent.0.0.delta": device.allocate(specs.delta_state[0]),
        }
        resources.extend(bound.values())
        execution = compiled.submit(*dynamic, resources=bound)
        execution.completion.wait()
        assert len(execution.outputs) == 3 + int(sampled)
        assert execution.outputs[0].spec.shape == (1, 32)
        if sampled:
            assert execution.outputs[1].spec.shape == (1, 2)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        for resource in resources:
            resource.close()
        program.close()
        for resource in residency.resources:
            resource.close()
        device.close()


@pytest.mark.parametrize("packed", [False, True])
def test_exact_mask_bytes_and_unconstrained_row_survive_control_packing(packed):
    import struct

    from engine.data import TokenId
    from engine.models.qwen35.inputs import InputPlan
    from engine.models.qwen35.runtime import DenseRuntime
    from engine.models.sequence import LogitsSelection, ModelRequest
    from tests.models.test_qwen35_runtime import ModelResidency, _description

    class RecordingRuntime(Runtime):
        def __init__(self):
            super().__init__()
            self.uploads = []

        def upload(self, spec, content):
            self.uploads.append((spec, bytes(content)))
            return super().upload(spec, content)

    native = RecordingRuntime()
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(_description(), device, residency, max_sequences=3)
    tokens = (TokenId(1),) if packed else (TokenId(1), TokenId(2))
    sequences = [model.create(InputPlan.text(tokens)) for _ in range(3)]
    masks = (b"\x01\x00\x00\x80", None, b"\x02\x00\x00\x40")
    batch = None
    try:
        batch = model.prepare(
            tuple(
                ModelRequest(sequence, tokens, LogitsSelection.LAST, (0,) * 6, mask)
                for sequence, mask in zip(sequences, masks, strict=True)
            )
        )
        _, specs, _ = next(iter(model.program._compiled))
        assert specs.packed_controls is packed
        expected_masks = masks[0] + masks[2]
        expected_rows = struct.pack("<3i", 0, -1, 1)
        if packed:
            payload = next(
                content for spec, content in native.uploads if spec == specs.control_record
            )
            mask_index = specs.control_specs.index(specs.masks)
            offset = sum(
                spec.elements * spec.dtype.itemsize for spec in specs.control_specs[:mask_index]
            )
            assert payload[offset : offset + 8] == expected_masks
            assert payload[offset + 8 : offset + 20] == expected_rows
        else:
            assert (specs.masks, expected_masks) in native.uploads
            assert (specs.mask_rows, expected_rows) in native.uploads
    finally:
        if batch is not None:
            batch.completion.wait()
            batch.close()
        for sequence in sequences:
            sequence.close()
        model.close()
        for resource in residency.resources:
            resource.close()
        device.close()
