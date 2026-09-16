"""Evidence survives failed diagnostics or independent checks."""

from types import SimpleNamespace

import pytest
from test_evidence import measurement

from roofline.contracts import Protocol
from roofline.integrations.magnitude import Magnitude
from roofline.store import Store


@pytest.mark.parametrize(
    "failure,correctness,status",
    [
        (ArithmeticError("wrong output"), "failed", "failed"),
        (RuntimeError("reference unavailable"), "unchecked", "incomplete"),
        (None, "passed", "complete"),
    ],
)
def test_workload_keeps_samples_when_check_or_diagnostics_fail(
    tmp_path, monkeypatch, failure, correctness, status
):
    from analytical import publication
    from ops.performance import publication as adapter

    numerical = publication().manifests[0]
    monkeypatch.setattr(adapter, "manifest", lambda *a, **kw: numerical)

    class Workload(Magnitude):
        def workload_sample(self, observed=None, captured=None):
            if captured:
                captured(
                    SimpleNamespace(
                        compiled=SimpleNamespace(
                            formulas=SimpleNamespace(graph=SimpleNamespace(fingerprint="test"))
                        )
                    )
                )
            if observed is not None:
                raise RuntimeError("timestamp capacity")
            return 0.1

        def check_workload(self):
            if failure:
                raise failure
            return [{"position": 0, "passed": True}]

        def base(self, **kwargs):
            return measurement(workload={"tokens": "fixture"}).model_dump(
                exclude={
                    "samples_seconds",
                    "status",
                    "correctness",
                    "scopes",
                    "artifacts",
                    "details",
                    "unavailable",
                    "error",
                    "costs",
                    "pair_ids",
                }
            )

    owner = object.__new__(Workload)
    owner.experiment = SimpleNamespace(
        scope="decode", protocol=Protocol(), context=1, steps=1, source="test"
    )
    owner.request = SimpleNamespace(model=SimpleNamespace(sha256="a" * 64))
    owner.device = SimpleNamespace(
        characterization=None, evidence_identity="device", compiler_identity="compiler"
    )
    owner.attempt = SimpleNamespace(target=SimpleNamespace(capacities=()), target_name="local")
    with Store(tmp_path) as owner.store:
        (result,) = owner.measure()
        assert result.samples_seconds == (0.1, 0.1, 0.1)
        assert result.correctness == correctness
        assert result.status == status
        assert any("timestamp capacity" in message for message in result.unavailable)
        assert result.artifacts["production-observations"]
        from formula_performance.records import Publication

        analytical = Publication.model_validate_json(
            owner.store.blob(result.artifacts["formula-performance"])
        )
        assert analytical.observations[0].boundary == "ordinary-sequential-workload"
        assert analytical.observations[0].samples == result.samples_seconds
        assert analytical.observations[0].correctness == correctness


def test_shared_boundary_preparation_respects_bound_prompt():
    events = []

    class Component(Magnitude):
        def sequence(self):
            return SimpleNamespace(close=lambda: events.append("close"))

        def prefill(self, sequence):
            events.append("prefill")

        def tokens(self):
            return SimpleNamespace(continuation=(10, 11))

        def component(self, sequence, tokens, position):
            assert events == ["prefill"]
            assert tokens == (10,) and position == 0
            events.append("restore-and-measure")
            return "measurement"

        def forward(self, *args, **kwargs):
            events.append("advance")

    owner = object.__new__(Component)
    owner.experiment = SimpleNamespace(scope="decode/component[0]", step=0, steps=2)
    owner.request = SimpleNamespace(inputs={"component": "verified-boundary"})
    owner.runners = {}
    assert owner.measure() == ["measurement"]
    assert events == ["prefill", "restore-and-measure", "advance", "close"]


def test_packed_weight_reference_preserves_decoded_coefficients(monkeypatch):
    import gguf
    import numpy as np
    import ops
    from engine.models.qwen35 import tensor_program
    from engine.weights.descriptor import WeightTransform
    from engine.weights.formats.gguf import Encoding
    from ops.tensor.primitive import round_reference

    source = np.linspace(-0.83, 0.91, 32, dtype=np.float32).reshape(1, 32)
    packed = gguf.quantize(source, gguf.GGMLQuantizationType.Q8_0)
    decoded = gguf.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
    assert np.any(decoded != round_reference(decoded, ops.DType.BF16))
    name = "projection.weight"
    role = SimpleNamespace(name=name, transform=WeightTransform.IDENTITY)
    monkeypatch.setattr(tensor_program, "weight_roles", lambda _: ((role, ops.DType.BF16),))
    owner = object.__new__(Magnitude)
    owner.description = None
    owner.format = SimpleNamespace(
        directory=SimpleNamespace(
            data_offset=0,
            tensor=lambda _: SimpleNamespace(
                offset=0, nbytes=packed.nbytes, shape=source.shape, encoding=Encoding.Q8_0
            ),
        ),
        source=SimpleNamespace(read=lambda offset, size: packed.tobytes()),
    )
    representation = ops.Affine(ops.Code(8), 32, ops.DirectCoefficients(ops.DType.F32))
    represented = SimpleNamespace(
        name=name,
        spec=ops.TensorSpec(source.shape, ops.DType.BF16, representation=representation),
    )
    np.testing.assert_array_equal(owner.decode_weight(represented), decoded)
    dense = SimpleNamespace(name=name, spec=ops.TensorSpec(source.shape, ops.DType.BF16))
    np.testing.assert_array_equal(
        owner.decode_weight(dense), round_reference(decoded, ops.DType.BF16)
    )
