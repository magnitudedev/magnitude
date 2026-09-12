import magnitensor as mt
from magnitude_engine.data import TokenId
from magnitude_engine.models.qwen35.inputs import InputPlan
from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.models.sequence import LogitsSelection, ModelRequest
from tests.magnitensor.test_compiler import Runtime
from tests.models.test_qwen35_tensor_program import Residency, _description


class ModelResidency(Residency):
    identity = _description().artifact_identity


def test_model_state_checkpoint_and_decode_use_magnitensor_submissions():
    native = Runtime()
    device = mt.Device(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(_description(), device, residency, max_sequences=3)
    sequence = model.create(InputPlan.text((TokenId(1), TokenId(2))))

    prefill = model.prepare(
        (
            ModelRequest(
                sequence,
                (TokenId(1), TokenId(2)),
                LogitsSelection.LAST,
                (0, 0, 0, 0, 0, 0),
            ),
        )
    )
    assert len(native.executables[-1].bound.calls) == 1
    prefill.completion.wait()
    prefill.advances[0].commit()
    prefill.close()
    assert sequence.position == 2

    checkpoint = sequence.checkpoint()
    fork = checkpoint.fork()
    assert fork.position == sequence.position
    assert len(native.executables[-1].bound.calls) == 1

    decode = model.prepare(
        (
            ModelRequest(
                fork,
                (TokenId(3),),
                LogitsSelection.LAST,
                (0, 0, 0, 1, 0, 0),
            ),
        )
    )
    assert len(native.executables[-1].bound.calls) == 1
    decode.completion.wait()
    decode.advances[0].commit()
    decode.close()
    assert fork.position == 3
    assert sequence.position == 2

    fork.close()
    checkpoint.close()
    sequence.close()
    model.close()
    for resource in residency.resources:
        resource.close()
    device.close()


def test_mixed_length_prefill_batch_uses_explicit_recurrent_row_offsets():
    native = Runtime()
    device = mt.Device(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(_description(), device, residency, max_sequences=3)
    first = model.create(InputPlan.text((TokenId(1),)))
    second = model.create(InputPlan.text((TokenId(2), TokenId(3), TokenId(4))))
    batch = model.prepare(
        (
            ModelRequest(first, (TokenId(1),), LogitsSelection.NONE),
            ModelRequest(
                second,
                (TokenId(2), TokenId(3), TokenId(4)),
                LogitsSelection.LAST,
                (0, 0, 0, 0, 0, 0),
            ),
        )
    )
    assert len(native.executables[-1].bound.calls) == 1
    assert any(
        parameter.name == "v2" and parameter.spec == mt.TensorSpec((3,), mt.DType.I32)
        for parameter in native.programs[-1].parameters
    )
    batch.completion.wait()
    batch.close()
    second.close()
    first.close()
    model.close()
    for resource in residency.resources:
        resource.close()
    device.close()


def test_prefill_uses_reusable_physical_row_capacity():
    native = Runtime()
    device = mt.Device(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(
        _description(), device, residency, max_sequences=2, prefill_rows=4
    )

    first = model.create(InputPlan.text((TokenId(1), TokenId(2))))
    first_batch = model.prepare(
        (
            ModelRequest(
                first,
                (TokenId(1), TokenId(2)),
                LogitsSelection.LAST,
                (0, 0, 0, 0, 0, 0),
            ),
        )
    )
    programs = len(native.programs)
    assert any(
        parameter.spec == mt.TensorSpec((4,), mt.DType.I32)
        for parameter in native.programs[-1].parameters
    )
    first_batch.completion.wait()
    first_batch.close()
    first.close()

    second = model.create(InputPlan.text((TokenId(3), TokenId(4), TokenId(5))))
    second_batch = model.prepare(
        (
            ModelRequest(
                second,
                (TokenId(3), TokenId(4), TokenId(5)),
                LogitsSelection.LAST,
                (0, 0, 0, 0, 0, 0),
            ),
        )
    )
    assert len(native.programs) == programs
    second_batch.completion.wait()
    second_batch.close()
    second.close()

    model.close()
    for resource in residency.resources:
        resource.close()
    device.close()


def test_runtime_allocates_only_configured_context_capacity():
    native = Runtime()
    device = mt.Device(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(
        _description(),
        device,
        residency,
        max_sequences=3,
        context_capacity=8,
    )

    assert model.context_capacity == 8
    assert model.states.attention[0].spec.shape == (2, 24, 1, 4)

    model.close()
    for resource in residency.resources:
        resource.close()
    device.close()


def test_prime_materializes_state_prefill_logits_prefill_and_decode():
    native = Runtime()
    device = mt.Device(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(
        _description(),
        device,
        residency,
        max_sequences=1,
        prefill_rows=4,
        context_capacity=8,
    )

    model.prime(4, 8)

    assert len(native.programs) == 3
    model.close()
    for resource in residency.resources:
        resource.close()
    device.close()
