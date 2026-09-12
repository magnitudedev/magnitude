from __future__ import annotations

import magnitensor as mt


def test_affine_high_plane_has_independent_byte_alignment():
    spec = mt.TensorSpec((1,), mt.DType.F16).with_representation(
        mt.Affine(mt.Code(4, 1), 1, mt.DirectCoefficients(mt.DType.F16))
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
    capabilities = mt.Capabilities(
        32,
        256,
        32 * 1024,
        native_multi_launch=True,
        partial_binding=True,
        fingerprint="test",
    )
    compiler_identity = "test-compiler"

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
    device = mt.Device(runtime, budget_bytes=1 << 20)
    spec = mt.TensorSpec((4, 8), mt.DType.F32)
    weight_spec = mt.TensorSpec((8, 8), mt.DType.F32)
    signature = mt.Signature(
        (
            mt.Argument(spec, "hidden"),
            mt.Argument(weight_spec, "weight", mt.ValueKind.CONSTANT),
        )
    )
    weight = device.allocate(weight_spec)

    def model(hidden, weight):
        projected = mt.linear(hidden, weight)
        return projected + mt.silu(projected)

    compiled = mt.compile(
        model,
        signature=signature,
        device=device,
        constants={"weight": weight},
        options=mt.CompileOptions(mode="decode"),
    )
    assert len(runtime.programs) == 1
    assert len(runtime.executables[0].bound.static) >= 1
    assert compiled.diagnostics.submissions == (
        (
            "linear.portable@0",
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
    spec = mt.TensorSpec((4, 8), mt.DType.F32)
    plan = mt.analyze(
        lambda value: mt.silu(value) + value,
        signature=mt.Signature((mt.Argument(spec, "value"),)),
        capabilities=runtime.capabilities,
        compiler_identity=runtime.compiler_identity,
        options=mt.CompileOptions(mode="prefill"),
    )
    assert plan.graph.outputs
    assert plan.diagnostics.dispatches == 1
    assert not runtime.programs
    assert not runtime.executables


def test_symbolic_signature_is_specialized_before_tracing():
    runtime = Runtime()
    device = mt.Device(runtime, budget_bytes=1 << 20)
    tokens = mt.Dim("tokens", maximum=16)
    signature = mt.Signature((mt.Argument(mt.TensorSpec((tokens, 4), mt.DType.F32)),))

    compiled = mt.compile(
        lambda value: mt.silu(value),
        signature=signature,
        device=device,
        constants={},
        options=mt.CompileOptions(mode="prefill", dimensions={"tokens": 7}),
    )
    assert compiled.graph.values[0].spec.shape == (7, 4)
    compiled.close()
    device.close()
