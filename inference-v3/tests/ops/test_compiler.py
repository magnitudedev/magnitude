from __future__ import annotations

import ops
from tests.ops.composed_formulas import activation_residual


def test_affine_high_plane_has_independent_byte_alignment():
    spec = ops.TensorSpec((1,), ops.DType.F16).with_representation(
        ops.Affine(ops.Code(4, 1), 1, ops.DirectCoefficients(ops.DType.F16))
    )
    assert spec.storage_nbytes == 4


class Allocation:
    def __init__(self, size):
        self.allocated_bytes = size
        self.identity = object()

    def view(self, spec, offset=0):
        return self.identity, spec, offset

    def close(self):
        pass


class Completion:
    def ready(self):
        return True

    def wait(self):
        pass


class Bound:
    def __init__(self, static, dynamic):
        self.static = dict(static)
        self.dynamic = dynamic
        self.calls = []

    def submit(self, dynamic):
        self.calls.append(dynamic)
        return Completion()

    def close(self):
        pass


class Executable:
    def __init__(self):
        self.bound = None

    def bind(self, static, dynamic_indices):
        self.bound = Bound(static, dynamic_indices)
        return self.bound

    def close(self):
        pass


class Runtime:
    def capture_kernels(self, limit):
        return None

    compiler_target = ops.CompilerTarget(
        32,
        256,
        32 * 1024,

        reference_schedules=True,


        identity="test",
    )
    compiler_identity = "test-compiler"
    runtime_identity = "test-runtime"

    def __init__(self):
        self.programs = []
        self.executables = []

    def allocate(self, size, alignment):
        return Allocation(size)

    def upload(self, spec, content):
        return Allocation(len(content))

    def compile(self, program, signature):
        self.programs.append(program)
        executable = Executable()
        self.executables.append(executable)
        return executable

    def join(self, completions):
        return Completion()

    def close(self):
        pass


def test_compile_uses_one_unit_and_prebinds_constants_and_temporary_slots():
    runtime = Runtime()
    device = ops.DeviceRuntime(runtime, budget_bytes=1 << 20)
    spec = ops.TensorSpec((4, 8), ops.DType.F32)
    weight_spec = ops.TensorSpec((8, 8), ops.DType.F32)
    signature = ops.Signature(
        (
            ops.Argument(spec, "hidden"),
            ops.Argument(weight_spec, "weight", ops.ValueKind.CONSTANT),
        )
    )
    weight = device.allocate(weight_spec)

    def model(hidden, weight):
        projected = ops.linear(hidden, weight)
        return activation_residual(projected)

    compiled = ops.compile(
        model,
        signature=signature,
        device=device,
        constants={"weight": weight},
        options=ops.CompileOptions(mode="decode"),
    )
    assert len(runtime.programs) == 1
    assert len(runtime.executables[0].bound.static) >= 1
    assert compiled.diagnostics.submissions == (
        (
            "linear.dense-vector@0",
            "pointwise.fused@1:2",
        ),
    )

    hidden = device.allocate(spec)
    execution = compiled.submit(hidden)
    assert len(runtime.executables[0].bound.calls) == 1
    execution.completion.wait()
    for output in execution.outputs:
        output.close()
    compiled.close()
    hidden.close()
    weight.close()
    device.close()


def test_analyze_plans_without_allocating_or_compiling():
    runtime = Runtime()
    spec = ops.TensorSpec((4, 8), ops.DType.F32)
    plan = ops.analyze(
        activation_residual,
        signature=ops.Signature((ops.Argument(spec, "value"),)),
        compiler_target=runtime.compiler_target,
        compiler_identity=runtime.compiler_identity,
        options=ops.CompileOptions(mode="prefill"),
    )
    assert plan.graph.outputs
    assert plan.diagnostics.dispatches == 1
    assert not runtime.programs
    assert not runtime.executables


def test_compile_can_prebind_a_stable_mutable_resource():
    runtime = Runtime()
    device = ops.DeviceRuntime(runtime, budget_bytes=1 << 20)
    spec = ops.TensorSpec((4, 8), ops.DType.F32)
    signature = ops.Signature(
        (
            ops.Argument(spec, "input"),
            ops.Argument(spec, "state", ops.ValueKind.RESOURCE),
        )
    )
    state = device.allocate(spec)
    compiled = ops.compile(
        lambda input, state: input + state,
        signature=signature,
        device=device,
        constants={},
        static_resources={"state": state},
        options=ops.CompileOptions(mode="decode"),
    )

    bound = runtime.executables[0].bound
    assert len(bound.static) == 1
    # The invocation still binds its input and compiler-allocated output; the
    # stable mutable state slot is absent from the dynamic ABI.
    assert len(bound.dynamic) == 2
    input_resource = device.allocate(spec)
    execution = compiled.submit(input_resource)
    assert len(bound.calls[0]) == 2
    execution.completion.wait()
    for output in execution.outputs:
        output.close()
    input_resource.close()
    compiled.close()
    state.close()
    device.close()


def test_symbolic_signature_is_specialized_before_tracing():
    runtime = Runtime()
    device = ops.DeviceRuntime(runtime, budget_bytes=1 << 20)
    tokens = ops.Dim("tokens", maximum=16)
    signature = ops.Signature((ops.Argument(ops.TensorSpec((tokens, 4), ops.DType.F32)),))

    compiled = ops.compile(
        lambda value: ops.silu(value),
        signature=signature,
        device=device,
        constants={},
        options=ops.CompileOptions(mode="prefill", dimensions={"tokens": 7}),
    )
    assert compiled.graph.values[0].spec.shape == (7, 4)
    compiled.close()
    device.close()
