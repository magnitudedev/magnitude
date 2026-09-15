
from tests.ops.target_fixture import matrix_query
"""Parent fusion preserves independent child formulas and their storage rounding."""

from dataclasses import replace

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.compiler.compilation import analyze_graph
from ops.kernels.publication import residual_epilogue
from ops.lab import Fixture
from ops.lab.records import Outcome
from ops.lab.runner import MeasurementRunner
from ops.lab.store import ObservationStore
from tests.ops import composed_formulas as formulas
from tests.ops.test_compiler import Runtime


def identity_binding(shape):
    width = shape[-1]
    codes = np.broadcast_to(np.eye(width, dtype=np.uint8), shape).copy()
    packed = codes.ravel()[::2] | codes.ravel()[1::2] << 4
    groups = codes.size // 64
    contents = (packed.tobytes(), np.full(groups, 0x3F80, np.uint16).tobytes(),
                np.zeros(groups, np.uint16).tobytes())
    planes = tuple(ops.SourcePlane(ops.SourceSpan(ops.MemorySource(content), 0, len(content)), group, size)
                   for content, group, size in zip(contents, (2, 64, 64), (1, 2, 2), strict=True))
    spec = ops.TensorSpec(shape, ops.DType.BF16).with_representation(
        ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16)))
    return ops.Binding(spec, f"identity:{shape}", ops.Residency.RESIDENT, planes, ops.CanonicalImport()), codes.astype(np.float32)


def case(kind, rows):
    width, experts = 512, 3
    hidden = np.full((rows, width), np.float32(0.400390625), np.float32)
    skip = np.linspace(1, 2, rows * width, dtype=np.float32).reshape(rows, width)
    arguments = [("x", ops.TensorSpec(hidden.shape, ops.DType.BF16), hidden),
                 ("skip", ops.TensorSpec(skip.shape, ops.DType.F32), skip)]
    weights = {}
    if kind == "routed":
        routes = np.tile(np.array([2, 0], np.int32), (rows, 1))
        scores = np.tile(np.array([0.25, 0.75], np.float32), (rows, 1))
        arguments += [("ids", ops.TensorSpec(routes.shape, ops.DType.I32), routes),
                      ("probabilities", ops.TensorSpec(scores.shape, ops.DType.F32), scores)]
        for name in ("eg", "eu", "ed"):
            weights[name] = identity_binding((experts, width, width))
        names = ("sg", "su", "sd")
        function = formulas.routed_residual
    else:
        names = ("gate", "up", "down")
        function = formulas.dense_residual
    for name in names:
        weights[name] = identity_binding((width, width))
    signature_args = [ops.Argument(spec, name) for name, spec, _ in arguments]
    signature_args.extend(ops.Argument(binding.spec, name, ops.ValueKind.CONSTANT)
                          for name, (binding, _) in weights.items())
    if kind == "routed":
        signature_args.append(ops.Argument(ops.TensorSpec((width,), ops.DType.F32), "sr"))
        arguments.append(("sr", ops.TensorSpec((width,), ops.DType.F32), np.zeros(width, np.float32)))
    graph = ops.trace(function, ops.Signature(tuple(signature_args)))
    by_name = {value.name: value.id for value in graph.values if value.producer is None}
    values = {by_name[name]: value for name, _, value in arguments}
    values.update({by_name[name]: value for name, (_, value) in weights.items()})
    bindings = {by_name[name]: binding for name, (binding, _) in weights.items()}
    return function, graph, values, bindings


@pytest.mark.parametrize("kind,mode,expected", [("dense", "decode", 2), ("dense", "prefill", 2),
                                               ("routed", "decode", 2), ("routed", "prefill", 7)])
def test_parent_absorbs_residual_but_exposed_child_keeps_its_own_publication(kind, mode, expected):
    _, graph, _, bindings = case(kind, 9)
    native = Runtime()
    compiler_target = replace(native.compiler_target,
        matrix_query=matrix_query((ops.MatrixTile(8, 8, 8, ops.DType.F32, ops.DType.F32),)),

        )
    options = ops.CompileOptions(mode=mode)
    plan = analyze_graph(graph, compiler_target=compiler_target, compiler_identity="fixture", available_bytes=64 << 20,
                         options=options, constants=bindings)
    assert plan.diagnostics.dispatches == expected
    parent, = ops.FormulaTree(graph).roots
    child, = parent.occurrences(formulas.dense_feedforward if kind == "dense" else formulas.routed_feedforward)
    child_output = child.call.outputs[0].value
    assert residual_epilogue(graph, child_output) is not None
    observed = replace(graph, outputs=(*graph.outputs, child_output))
    assert residual_epilogue(observed, child_output) is None
    visible = analyze_graph(observed, compiler_target=compiler_target, compiler_identity="fixture", available_bytes=64 << 20,
                            options=options, constants=bindings)
    assert visible.diagnostics.dispatches == expected + 2
    assert len(parent.children) == len(ops.FormulaTree(observed).roots[0].children)


@pytest.mark.device
@pytest.mark.parametrize("kind", ("dense", "routed"))
@pytest.mark.parametrize("mode,rows", (("decode", 1), ("prefill", 9), ("prefill", 129)))
def test_parent_residual_publication_through_checked_formula_measurement(kind, mode, rows, tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("requires the designated Metal foundation device")
    _, graph, values, bindings = case(kind, rows)
    fixture = Fixture.from_inputs(graph, values, bindings=bindings)
    target, = ops.FormulaTree(graph).roots
    with ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=64 << 20)) as device:
        with ObservationStore(tmp_path / "residual.sqlite") as store:
            runner = MeasurementRunner(fixture, device, store, ops.CompileOptions(mode=mode))
            try:
                result = runner.measure(target).measurement
                assert result.outcome == Outcome.COMPLETE, result.error
                assert result.checked
                assert all(sample.memory.peak_bytes >= sample.memory.baseline_bytes for sample in result.samples)
                assert store.history(result.series).latest == result
            finally:
                runner.close()
