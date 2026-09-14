"""The connection: stable semantic work, scoped traffic, evidence and tree display."""

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

import ops
from ops.formula import Unit, units
from ops.lab.characterization import Characterization, ProbeProtocol, Rate
from ops.lab.fixtures import Fixture
from ops.lab.records import ResourceDemand, ResourceLimit, Roofline
from ops.lab.roofline import RooflineCoverageError, demands, model
from ops.lab.tui import roofline_label
from ops.performance.resources import Resource
from ops.performance.semantics import sampling
from ops.performance.traffic import boundary_traffic


@ops.formula(id="qualification.roofline.composed")
def composed(x, weight):
    return ops.silu(ops.linear(x, weight))


def boundary(function=composed):
    signature = ops.Signature((ops.Argument(ops.TensorSpec((2, 4), ops.DType.F32), "x"),
                               ops.Argument(ops.TensorSpec((8, 4), ops.DType.F32), "weight", ops.ValueKind.CONSTANT)))
    graph = ops.trace(function, signature)
    fixture = Fixture.from_inputs(graph, {graph.inputs[0]: np.ones((2, 4), np.float32),
                                          graph.constants[0]: np.ones((8, 4), np.float32)})
    root, = ops.FormulaTree(graph).roots
    return fixture, root


def profile_for(boundary):
    rates = tuple(Rate(resource=demand.resource, dtype=demand.dtype, value=1e9,
                       unit=Unit(f"{demand.unit.name}/s", f"{demand.unit.dimension}/time"),
                       measurement=f"probe:{demand.resource.value}", metric="fixture", working_set_bytes=1024,
                       conditions=("synthetic unit-test evidence, never a real device measurement",))
                  for demand in demands(boundary))
    profile = Characterization(identity="test-profile", key="test-key", created=datetime.now(UTC),
                               device="test-device", compiler="test-compiler", capabilities="test-caps",
                               protocol=ProbeProtocol(), rates=rates)
    return SimpleNamespace(characterization=profile, evidence_identity=profile.device,
                           compiler_identity=profile.compiler, capabilities=SimpleNamespace(fingerprint=profile.capabilities))


def test_composed_demands_count_matrix_and_vector_work_once():
    fixture, root = boundary()
    work = {item.resource: item for item in demands(fixture.boundary(root))}
    assert work[Resource.MATRIX_ARITHMETIC].lower == 128
    assert work[Resource.VECTOR_ARITHMETIC].lower == 48
    assert work[Resource.SPECIAL_FUNCTIONS].lower == 16
    # x + weight + final result; not the internal projection result.
    assert work[Resource.EXECUTION_COPY].lower == (8 + 32 + 16) * 4
    linear, = root.occurrences(ops.linear)
    child = {item.resource: item for item in demands(fixture.boundary(linear))}
    assert child[Resource.MATRIX_ARITHMETIC].lower == 128
    assert Resource.SPECIAL_FUNCTIONS not in child


def test_storage_reference_is_bandwidth_not_operand_arithmetic_precision():
    fixture, root = boundary()
    selected = fixture.boundary(root)
    device = profile_for(selected)
    device.characterization = device.characterization.model_copy(update={
        "rates": tuple(rate.model_copy(update={"dtype": ops.DType.F32})
                       if rate.resource == Resource.EXECUTION_COPY else rate
                       for rate in device.characterization.rates),
    })
    result = model(selected, device)
    memory, = (limit for limit in result.limits if limit.demand.resource == Resource.EXECUTION_COPY)
    assert memory.demand.dtype == ops.DType.U8
    assert memory.rate_dtype == ops.DType.F32


def test_matrix_resource_probe_reuses_operands_without_duplicate_work_equations():
    from ops.lab.characterization import matrix
    from ops.performance.semantics import formula_work

    fixture, root = boundary(matrix)
    graph = fixture.root
    assert formula_work(graph, root.call).work.matrix.lower == 64 * 128
    assert boundary_traffic(graph, {}) == (8 + 32 + 16) * 4


def test_resource_model_joins_mixed_formula_and_rejects_missing_evidence():
    fixture, root = boundary()
    selected = fixture.boundary(root)
    device = profile_for(selected)
    result = model(selected, device)
    assert result.seconds == pytest.approx(224 / 1e9)
    assert result.bottleneck == Resource.EXECUTION_COPY.value
    device.characterization = device.characterization.model_copy(update={
        "rates": tuple(rate for rate in device.characterization.rates if rate.resource != Resource.SPECIAL_FUNCTIONS),
    })
    with pytest.raises(RooflineCoverageError, match="special-functions"):
        model(selected, device)


def test_device_identity_is_required_but_component_check_tolerance_is_not_a_rate_key():
    fixture, root = boundary()
    selected = fixture.boundary(root)
    device = profile_for(selected)
    assert model(selected, device).limits
    device.compiler_identity = "different-native-build"
    with pytest.raises(RooflineCoverageError, match="device and compiler"):
        model(selected, device)


def test_same_resource_precisions_add_and_independent_constraints_take_max():
    def limit(dtype, amount):
        return ResourceLimit(demand=ResourceDemand(resource=Resource.MATRIX_ARITHMETIC, dtype=dtype,
                                                  lower=amount, upper=amount, unit=units.flop, basis="fixture"),
                             rate=1000, rate_dtype=dtype, measurement="fixture", assumptions=())
    result = Roofline(revision="fixture", characterization="fixture", assumptions=(),
                      limits=(limit(ops.DType.F16, 30), limit(ops.DType.F32, 70)))
    assert result.seconds == pytest.approx(.1)
    assert "200%" in roofline_label(result, .05)
    assert "exceeds reference" in roofline_label(result, .05)


def test_embedding_traffic_uses_unique_selected_rows_not_table_capacity():
    table = ops.TensorSpec((100, 8), ops.DType.F32)
    index = ops.TensorSpec((3,), ops.DType.I32)
    graph = ops.trace(ops.embedding, ops.Signature((ops.Argument(index), ops.Argument(table))))
    values = {graph.inputs[0]: np.array([2, 2, 7], np.int32)}
    # Two table rows, three indices, three output rows.
    assert boundary_traffic(graph, values) == 2 * 8 * 4 + 3 * 4 + 3 * 8 * 4


def test_attention_traffic_uses_union_of_visible_ranges_not_capacity():
    query = ops.TensorSpec((2, 2, 4), ops.DType.F16)
    history = ops.TensorSpec((2, 100, 1, 4), ops.DType.F16)
    visible = ops.TensorSpec((2, 2), ops.DType.I32)
    graph = ops.trace(lambda q, h, v: ops.causal_attention(q, h, v),
                      ops.Signature((ops.Argument(query), ops.Argument(history, "history", ops.ValueKind.RESOURCE),
                                     ops.Argument(visible))))
    control = next(value.id for value in graph.values if value.producer is None and value.spec == visible)
    assert boundary_traffic(graph, {control: np.array([[1, 3], [2, 3]], np.int32)}) == (
        query.storage_nbytes * 2 + visible.storage_nbytes + 2 * 4 * 1 * 4 * 2)


def test_view_only_boundary_does_not_invent_a_read_or_write():
    spec = ops.TensorSpec((4,), ops.DType.F32)
    graph = ops.trace(lambda x: ops.reshape(x, (2, 2)), ops.Signature((ops.Argument(spec),)))
    assert boundary_traffic(graph, {}) == 0


def test_state_written_inside_parent_is_not_counted_as_an_external_reread():
    spec = ops.TensorSpec((8,), ops.DType.U8)
    extent = ops.TensorSpec((2,), ops.DType.I64)
    graph = ops.trace(lambda x, state, region: ops.byte_copy(x, state, region) + x,
                      ops.Signature((ops.Argument(spec), ops.Argument(spec, "state", ops.ValueKind.RESOURCE),
                                     ops.Argument(extent))))
    control = next(value.id for value in graph.values if value.producer is None and value.spec == extent)
    # Eight source bytes, control words, state publication, final output. No
    # external state reread after the copy produces it inside this boundary.
    assert boundary_traffic(graph, {control: np.array([0, 8], np.int64)}) == 8 + 16 + 8 + 8


def test_sampling_counts_actual_randomized_tokens_not_row_capacity():
    logits = np.array([[1, 2, -np.inf], [1, 2, 3]], np.float32)
    draws = np.zeros((2, 6), np.uint32)
    draws[0, 0] = 1
    specs = (ops.TensorSpec(logits.shape, ops.DType.F32), ops.TensorSpec(draws.shape, ops.DType.U32))
    counted = sampling(specs, {}, (), values=(logits, draws))
    assert counted.integer.lower == 202
    assert counted.special.lower == 4
    assert counted.floating.lower == 10
    assert counted.comparisons.lower == 3
    with pytest.raises(ValueError, match="concrete"):
        sampling(specs, {}, ())


def test_probe_geometry_is_explicit_and_fits_small_device_plans():
    for capacity in (1 << 20, 64 << 20, 1 << 30):
        protocol = ProbeProtocol.for_capacity(capacity)
        assert max(protocol.copy_sizes) * 6 + 32 < capacity
        assert protocol.matrix_width >= 32
        assert 48 * protocol.matrix_width ** 2 < capacity


def test_comparison_probe_has_explicit_semantics_and_one_fused_operation():
    from ops.lab.characterization import comparisons
    from ops.compiler.compilation import analyze_graph
    from tests.ops.test_compiler import Runtime

    spec = ops.TensorSpec((256,), ops.DType.F32)
    graph = ops.trace(comparisons, ops.Signature((ops.Argument(spec), ops.Argument(spec))))
    reference = ops.evaluate_reference(graph, {graph.inputs[0]: np.zeros(256, np.float32),
                                               graph.inputs[1]: np.ones(256, np.float32)})
    np.testing.assert_array_equal(reference.outputs[0], np.zeros(256, np.float32))
    fixture = Fixture.from_reference(graph, reference)
    root, = ops.FormulaTree(graph).roots
    count, = (item for item in demands(fixture.boundary(root)) if item.resource == Resource.COMPARISONS)
    assert count.lower == 32 * 256
    runtime = Runtime()
    plan = analyze_graph(graph, capabilities=runtime.capabilities, compiler_identity="fixture",
                         available_bytes=1 << 20, options=ops.CompileOptions(mode="prefill"))
    assert len(plan.operations) == 1


def test_source_demand_unions_real_encoding_groups_by_snapshot():
    spec = ops.TensorSpec((16, 4), ops.DType.F32)
    index = ops.TensorSpec((3,), ops.DType.I32)
    graph = ops.trace(ops.embedding, ops.Signature((ops.Argument(index),
                                                   ops.Argument(spec, "table", ops.ValueKind.CONSTANT))))
    data = np.arange(64, dtype=np.float32).reshape(spec.shape)
    source = ops.MemorySource(data.tobytes())
    binding = ops.Binding(spec, "source-table", ops.Residency.STREAMED,
                          (ops.SourcePlane(ops.SourceSpan(source, 0, data.nbytes), 1, 4),),
                          ops.DenseImport(ops.DType.F32))
    fixture = Fixture(graph, {graph.inputs[0]: np.array([2, 2, 7], np.int32), graph.constants[0]: data},
                      bindings={graph.constants[0]: binding})
    root, = ops.FormulaTree(graph).roots
    term, = (item for item in demands(fixture.boundary(root)) if item.resource == Resource.SOURCE_IMPORT)
    assert term.source == source.info
    assert term.lower == term.upper == 2 * 4 * 4


def test_bfloat_matrix_reference_cannot_silently_use_float16():
    fixture, root = boundary()
    selected = fixture.boundary(root)
    device = profile_for(selected)
    # Exact widening policy is covered through a real BF16 semantic trace.
    spec = ops.TensorSpec((2, 4), ops.DType.BF16)
    weight = ops.TensorSpec((8, 4), ops.DType.BF16)
    graph = ops.trace(composed, ops.Signature((ops.Argument(spec), ops.Argument(weight))))
    reference = Fixture.from_inputs(graph, {graph.inputs[0]: np.ones(spec.shape, np.float32),
                                            graph.inputs[1]: np.ones(weight.shape, np.float32)})
    target, = ops.FormulaTree(graph).roots
    result = model(reference.boundary(target), device)
    matrix, = (term for term in result.limits if term.demand.resource == Resource.MATRIX_ARITHMETIC)
    assert matrix.demand.dtype == ops.DType.BF16
    assert matrix.rate_dtype == ops.DType.F32
    device.characterization = device.characterization.model_copy(update={
        "rates": tuple(rate.model_copy(update={"dtype": ops.DType.F16})
                       if rate.resource == Resource.MATRIX_ARITHMETIC else rate
                       for rate in device.characterization.rates),
    })
    with pytest.raises(RooflineCoverageError, match="matrix-arithmetic/bfloat16"):
        model(reference.boundary(target), device)
