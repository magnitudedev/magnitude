import ops
from engine.data import TokenId
from engine.models.qwen35.inputs import InputPlan
from engine.models.qwen35.runtime import DenseRuntime
from engine.models.sequence import LogitsSelection, ModelRequest
from tests.models.test_qwen35_tensor_program import Residency, _description
from tests.ops.test_compiler import Runtime


class ModelResidency(Residency):
    identity = _description().artifact_identity


def test_model_state_checkpoint_and_decode_use_ops_submissions():
    native = Runtime()
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
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
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
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
    assert all(specs.recurrent_sequence_length is None for _, specs, _ in model.program._compiled)
    assert any(
        parameter.name == "v2" and parameter.spec == ops.TensorSpec((3,), ops.DType.I32)
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
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(_description(), device, residency, max_sequences=2, prefill_rows=4)

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
        parameter.spec == ops.TensorSpec((4,), ops.DType.I32)
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

    assert all(specs.recurrent_sequence_length is None for _, specs, _ in model.program._compiled)
    for repeat in range(2):
        tokens = (TokenId(1), TokenId(2), TokenId(3), TokenId(4))
        full = model.create(InputPlan.text(tokens))
        full_batch = model.prepare((
            ModelRequest(full, tokens, LogitsSelection.LAST, (0, 0, 0, 0, 0, 0)),
        ))
        full_programs = [compiled for (_, specs, _), compiled in model.program._compiled.items()
                         if specs.recurrent_sequence_length == 4]
        assert len(full_programs) == 1
        recurrence = [node for node in full_programs[0].graph.nodes
                      if node.operation == "gated_delta_recurrence"]
        assert recurrence and all(node.attributes["sequence_length"] == 4 for node in recurrence)
        if repeat == 0:
            specialized_count = len(native.programs)
            assert specialized_count > programs
        else:
            assert len(native.programs) == specialized_count
        full_batch.completion.wait()
        full_batch.close()
        full.close()

    model.close()
    for resource in residency.resources:
        resource.close()
    device.close()


def test_runtime_allocates_only_configured_context_capacity():
    native = Runtime()
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
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
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
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

    assert len(native.programs) == 5
    assert {(mode, specs.recurrent_sequence_length)
            for mode, specs, _ in model.program._compiled} == {
        ("prefill", 4), ("prefill", None), ("decode", None),
    }
    model.close()
    for resource in residency.resources:
        resource.close()
    device.close()


def test_explicit_forward_observation_and_capture_share_production_execution(tmp_path):
    from engine.models.qwen35.inspection import inspect_forwards, publish_forward
    from ops.lab.store import ObservationStore
    from tests.ops.test_model_evidence import run

    native = Runtime()
    device = ops.DeviceRuntime(native, budget_bytes=1 << 24)
    residency = ModelResidency(device)
    model = DenseRuntime(_description(), device, residency, max_sequences=2)
    sequence = model.create(InputPlan.text((TokenId(1), TokenId(2))))
    try:
        with ObservationStore(tmp_path / 'model.sqlite') as store:
            collected = []
            def observed(invocation, observation):
                collected.append(publish_forward(store, run().context, invocation, observation))
            with inspect_forwards(model, observed=observed, kernel_limit=None):
                batch = model.prepare((ModelRequest(sequence, (TokenId(1), TokenId(2)),
                                      LogitsSelection.LAST, (0, 0, 0, 0, 0, 0)),))
                batch.completion.wait()
                batch.advances[0].commit()
                batch.close()
            assert len(collected) == 1
            assert collected[0].context.conditions['positions'] == [0]
            assert collected[0].observations[0].status.value == 'complete'
            assert store.models()[0].identity == 'qwen-4b'
            assert store.configurations()[0].formulas
        captures = []
        with inspect_forwards(model, captured=captures.append):
            batch = model.prepare((ModelRequest(sequence, (TokenId(3),),
                                  LogitsSelection.LAST, (0, 0, 0, 0, 0, 0)),))
            batch.completion.wait()
            batch.close()
        assert len(captures) == 1
        assert captures[0].positions == (2,)
        assert sequence.position == 2  # Uncommitted measured advance was aborted.
    finally:
        sequence.close()
        model.close()
        for resource in residency.resources:
            resource.close()
        device.close()
