"""Numerical coverage of the complete packet and attention producer/consumer paths."""

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from tests.ops import composed_formulas


def _affine_weight(device, rng, shape, dtype=ops.DType.F16):
    codes = rng.integers(0, 16, shape, dtype=np.uint8)
    packed = codes.ravel()[::2] | (codes.ravel()[1::2] << 4)
    groups = codes.size // 64
    # Exact BF16 coefficients: scale 1/64, bias -1/8.
    payload = (
        packed.tobytes()
        + np.full(groups, 0x3C80, dtype=np.uint16).tobytes()
        + np.full(groups, 0xBE00, dtype=np.uint16).tobytes()
    )
    spec = ops.TensorSpec(shape, dtype).with_representation(
        ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16))
    )
    return device.upload(spec, payload), codes.astype(np.float32) / 64 - 0.125


@pytest.mark.device
@pytest.mark.parametrize("branches", [3, 4])
@pytest.mark.parametrize("packets", ["affine", "signed", "mixed"])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_parallel_packet_regions_cover_branch_boundaries_and_partial_groups(branches, packets, mode):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")

    rng = np.random.default_rng(413)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=4 << 20))
    resources = []
    compiled = execution = None
    try:
        hidden = rng.normal(0, 0.05, (2 if mode == "decode" else 9, 512)).astype(np.float16)
        source = device.upload(ops.TensorSpec(hidden.shape, ops.DType.F16), hidden.tobytes())
        resources.append(source)
        decoded = []
        for index, outputs in enumerate((3, 5, 7, 2)[:branches]):
            if packets == "signed" or packets == "mixed" and index == 1:
                shape = (outputs, 512)
                codes = rng.integers(-16, 16, shape, dtype=np.int8)
                scales = np.full(codes.size // 32, 1 / 64, np.float16)
                spec = ops.TensorSpec(shape, ops.DType.F16).with_representation(
                    ops.Affine(ops.Code(8, interpretation=ops.CodeInterpretation.TWOS_COMPLEMENT),
                               32, ops.DirectCoefficients(ops.DType.F16)))
                weight = device.upload(spec, codes.tobytes() + scales.tobytes())
                values = codes.astype(np.float32) / 64
            else:
                weight, values = _affine_weight(device, rng, (outputs, 512))
            resources.append(weight)
            decoded.append(values)
        names = ("x", *(f"weight{index}" for index in range(branches)))
        compiled = ops.compile(
            composed_formulas.parallel_projections,
            signature=ops.Signature(tuple(
                ops.Argument(resource.spec, name, ops.ValueKind.INPUT if index == 0 else ops.ValueKind.CONSTANT)
                for index, (name, resource) in enumerate(zip(names, resources, strict=True))
            )),
            device=device, constants=dict(zip(names[1:], resources[1:], strict=True)),
            options=ops.CompileOptions(mode=mode),
        )
        assert compiled.diagnostics.dispatches == 1
        execution = compiled.submit(source)
        execution.completion.wait()
        for output, weight in zip(execution.outputs, decoded, strict=True):
            expected = (hidden.astype(np.float32) @ weight.T).astype(np.float16)
            actual = np.frombuffer(device.read(output), np.float16).reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-4)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


@pytest.mark.device
def test_packet_prefill_preserves_decode_coefficient_precision():
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(641)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=8 << 20))
    resources, compiled_functions = [], []
    try:
        codes = rng.integers(0, 16, (32, 512), dtype=np.uint8)
        packed = codes.ravel()[::2] | (codes.ravel()[1::2] << 4)
        coefficients = torch.tensor([0.01, -0.1], dtype=torch.bfloat16)
        scale, bias = coefficients.float().numpy()
        bits = coefficients.view(torch.uint16).numpy()
        groups = codes.size // 64
        spec = ops.TensorSpec(codes.shape, ops.DType.BF16).with_representation(
            ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16))
        )
        weight = device.upload(
            spec,
            packed.tobytes()
            + np.full(groups, bits[0], np.uint16).tobytes()
            + np.full(groups, bits[1], np.uint16).tobytes(),
        )
        resources.append(weight)
        row = torch.from_numpy(rng.normal(0, 0.2, (1, 512)).astype(np.float32)).to(torch.bfloat16)
        expected = row.float().numpy() @ (codes.astype(np.float32) * scale + bias).T
        for rows, mode in ((1, "decode"), (9, "prefill")):
            source = device.upload(
                ops.TensorSpec((rows, 512), ops.DType.BF16),
                row.repeat(rows, 1).view(torch.uint16).numpy().tobytes(),
            )
            resources.append(source)
            compiled = ops.compile(
                lambda x, w: ops.linear(x, w, output_dtype=ops.DType.F32),
                signature=ops.Signature(
                    (
                        ops.Argument(source.spec, "x"),
                        ops.Argument(weight.spec, "w", ops.ValueKind.CONSTANT),
                    )
                ),
                device=device,
                constants={"w": weight},
                options=ops.CompileOptions(mode=mode),
            )
            compiled_functions.append(compiled)
            execution = compiled.submit(source)
            try:
                execution.completion.wait()
                np.testing.assert_allclose(
                    execution.outputs[0].native.cpu().numpy(),
                    np.repeat(expected, rows, axis=0),
                    atol=3e-5,
                    rtol=3e-5,
                )
            finally:
                for output in execution.outputs:
                    output.close()
    finally:
        for compiled in reversed(compiled_functions):
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


@pytest.mark.device
@pytest.mark.parametrize("mode,rows", (("decode", 1), ("prefill", 8)))
def test_packed_embedding_to_projection_preserves_every_packet(mode, rows):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(604)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=8 << 20))
    resources = []
    compiled = execution = None
    try:
        table, table_values = _affine_weight(device, rng, (16, 512))
        resources.append(table)
        weight, weight_values = _affine_weight(device, rng, (32, 512))
        resources.append(weight)
        indices = np.arange(rows, dtype=np.int32)
        tokens = device.upload(ops.TensorSpec((rows,), ops.DType.I32), indices.tobytes())
        resources.append(tokens)

        def function(ids, embedding, projection):
            value = ops.embedding(ids, embedding)
            return value, ops.linear(value, projection)

        compiled = ops.compile(
            function,
            signature=ops.Signature(
                (
                    ops.Argument(tokens.spec, "tokens"),
                    ops.Argument(table.spec, "embedding", ops.ValueKind.CONSTANT),
                    ops.Argument(weight.spec, "projection", ops.ValueKind.CONSTANT),
                )
            ),
            device=device,
            constants={"embedding": table, "projection": weight},
            options=ops.CompileOptions(mode=mode),
        )
        execution = compiled.submit(tokens)
        execution.completion.wait()
        np.testing.assert_array_equal(
            execution.outputs[0].native.cpu().numpy(), table_values[indices]
        )
        np.testing.assert_allclose(
            execution.outputs[1].native.cpu().numpy(),
            table_values[indices] @ weight_values.T,
            rtol=3e-3,
            atol=3e-3,
        )
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


@pytest.mark.device
@pytest.mark.parametrize("rows", [1, 129, 256])
def test_production_gate_up_and_projection_match_reference(rows):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(907)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=32 << 20))
    resources = []
    compiled = execution = None
    try:
        hidden = rng.normal(0, 0.05, (rows, 512)).astype(np.float16)
        source = device.upload(ops.TensorSpec(hidden.shape, ops.DType.F16), hidden.tobytes())
        resources.append(source)
        weights, values = [], []
        for _ in range(3):
            resource, value = _affine_weight(device, rng, (512, 512))
            resources.append(resource)
            weights.append(resource)
            values.append(value)

        def function(x, gate, up, down):
            projection = ops.linear(x, down)
            return projection, composed_formulas.dense_feedforward(x, gate, up, down)

        names = ("x", "gate", "up", "down")
        compiled = ops.compile(
            function,
            signature=ops.Signature(
                tuple(
                    ops.Argument(
                        resource.spec,
                        name,
                        ops.ValueKind.INPUT if index == 0 else ops.ValueKind.CONSTANT,
                    )
                    for index, (name, resource) in enumerate(zip(names, resources, strict=True))
                )
            ),
            device=device,
            constants=dict(zip(names[1:], weights, strict=True)),
            options=ops.CompileOptions(mode="decode" if rows == 1 else "prefill"),
        )
        execution = compiled.submit(source)
        execution.completion.wait()
        gate_value = (hidden.astype(np.float32) @ values[0].T).astype(np.float16).astype(np.float32)
        up_value = (hidden.astype(np.float32) @ values[1].T).astype(np.float16).astype(np.float32)
        activated_gate = (
            (gate_value / (1 + np.exp(-gate_value))).astype(np.float16).astype(np.float32)
        )
        activation = (activated_gate * up_value).astype(np.float16)
        expected = (
            (hidden.astype(np.float32) @ values[2].T).astype(np.float16),
            (activation.astype(np.float32) @ values[2].T).astype(np.float16),
        )
        for output, reference in zip(execution.outputs, expected, strict=True):
            np.testing.assert_allclose(output.native.cpu().numpy(), reference, rtol=3e-2, atol=3e-3)
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


@pytest.mark.device
@pytest.mark.parametrize("rows", [129, 256])
@pytest.mark.parametrize("floating", [ops.DType.F16, ops.DType.BF16])
def test_specialized_routed_shared_pipeline_covers_sparse_routes_and_partial_tiles(rows, floating):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(607)
    width, experts, selected = 512, 4, 2
    hidden = rng.normal(0, 0.1, (rows, width)).astype(np.float16)
    routes = np.tile(np.asarray([0, 1], np.int32), (rows, 1))
    scores = np.tile(np.asarray([0.25, 0.75], np.float32), (rows, 1))
    router = rng.normal(0, 0.1, (width,)).astype(np.float16)
    from ops.tensor.primitive import round_reference

    def publish(value):
        return round_reference(value, floating).astype(np.float32)

    def contents(value):
        dtype = torch.bfloat16 if floating == ops.DType.BF16 else torch.float16
        return torch.from_numpy(value).to(dtype).view(torch.uint16).numpy().tobytes()

    hidden, router = publish(hidden), publish(router)
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=32 << 20))
    resources = []
    compiled = execution = None
    try:
        for array, dtype in (
            (hidden, floating),
            (routes, ops.DType.I32),
            (scores, ops.DType.F32),
        ):
            resources.append(device.upload(ops.TensorSpec(array.shape, dtype),
                                           contents(array) if dtype == floating else array.tobytes()))
        decoded = []
        for shape in [(experts, width, width)] * 3 + [(width, width)] * 3:
            resource, values = _affine_weight(device, rng, shape, floating)
            resources.append(resource)
            decoded.append(values)
        resources.append(device.upload(ops.TensorSpec(router.shape, floating), contents(router)))

        names = ("x", "ids", "probabilities", "eg", "eu", "ed", "sg", "su", "sd", "sr")
        signature = ops.Signature(
            tuple(
                ops.Argument(
                    resource.spec, name, ops.ValueKind.INPUT if i < 3 else ops.ValueKind.CONSTANT
                )
                for i, (name, resource) in enumerate(zip(names, resources, strict=True))
            )
        )
        compiled = ops.compile(
            composed_formulas.routed_feedforward,
            signature=signature,
            device=device,
            constants=dict(zip(names[3:], resources[3:], strict=True)),
            options=ops.CompileOptions(mode="prefill"),
        )
        continuous = (floating == ops.DType.BF16 and "gemm.shared_instruction_tiles" in device.capabilities.features
                      and any(item.input_dtype == ops.DType.BF16 and (item.m, item.n, item.k) == (8, 8, 8)
                              for item in device.capabilities.matrix_instructions))
        assert compiled.diagnostics.dispatches == (7 if continuous else 8)
        execution = compiled.submit(*resources[:3])
        execution.completion.wait()

        def feedforward(gate, up, down, *, explicit_silu=False):
            g = publish(hidden @ gate.T)
            u = publish(hidden @ up.T)
            activated_gate = g / (1 + np.exp(-g))
            if explicit_silu:
                activated_gate = publish(activated_gate)
            activated = publish(activated_gate * u)
            return publish(activated @ down.T)

        eg, eu, ed, sg, su, sd = decoded
        expected = np.stack(
            [scores[:, i, None] * feedforward(eg[i], eu[i], ed[i]) for i in range(selected)]
        ).sum(axis=0)
        coefficient = 1 / (1 + np.exp(-(hidden.astype(np.float32) @ router.astype(np.float32))))
        shared = publish(publish(coefficient)[:, None] * feedforward(sg, su, sd, explicit_silu=True))
        expected = publish(publish(expected) + shared)
        np.testing.assert_allclose(
            execution.outputs[0].native.float().cpu().numpy(), expected, rtol=3e-2, atol=3e-3
        )
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()


@pytest.mark.device
@pytest.mark.parametrize(
    "rows,capacity,prefix",
    [(9, 64, 0), (9, 8192, 0), (9, 8192, 8000), (129, 256, 0), (129, 8192, 0), (129, 8192, 8000)],
)
@pytest.mark.parametrize("floating", [ops.DType.F16, ops.DType.BF16])
def test_attention_gate_and_packed_output_are_one_complete_region(rows, capacity, prefix, floating):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(605)
    heads, width = 4, 256
    arrays = (
        rng.normal(0, 0.2, (rows, heads, width)).astype(np.float16),
        rng.normal(0, 0.2, (2, capacity, 1, width)).astype(np.float16),
        np.asarray([[3, prefix + i + 1] for i in range(rows)], dtype=np.int32),
        rng.normal(0, 0.2, (rows, heads, width)).astype(np.float16),
    )
    if rows == 129 and prefix == 0:
        arrays[2][0, 1] = 0
    device = ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=32 << 20))
    resources = []
    compiled = execution = None
    try:
        rounded = []
        for value in arrays:
            dtype = ops.DType.I32 if value.dtype == np.int32 else floating
            if dtype == ops.DType.BF16:
                tensor = torch.from_numpy(value).to(torch.bfloat16)
                payload = tensor.view(torch.uint16).numpy().tobytes()
                rounded.append(tensor.float().numpy())
            else:
                payload = value.tobytes()
                rounded.append(value)
            resources.append(device.upload(ops.TensorSpec(value.shape, dtype), payload))
        weight, decoded = _affine_weight(device, rng, (32, heads * width), floating)
        resources.append(weight)

        signature = ops.Signature(
            tuple(
                ops.Argument(
                    resource.spec,
                    name,
                    ops.ValueKind.CONSTANT
                    if name == "projection"
                    else (ops.ValueKind.RESOURCE if name == "history" else ops.ValueKind.INPUT),
                )
                for resource, name in zip(
                    resources, ("query", "history", "visible", "gate", "projection"), strict=True
                )
            )
        )
        compiled = ops.compile(
            composed_formulas.attention_output,
            signature=signature,
            device=device,
            constants={"projection": weight},
            options=ops.CompileOptions(mode="prefill"),
        )
        assert compiled.diagnostics.dispatches == (3 if capacity > 4096 else 2)
        expected_family = "attention.matrix-streaming-gated-output"
        assert any(
            c.name.startswith(expected_family)
            for c in compiled.diagnostics.operations
        )
        execution = compiled.submit(
            resources[0], resources[2], resources[3], resources={"history": resources[1]}
        )
        execution.completion.wait()
        query, history, visible, gate = rounded
        attended = np.empty(query.shape, dtype=np.float32)
        for row in range(rows):
            start, count = visible[row]
            if count == 0:
                attended[row] = 0
                continue
            score = (
                query[row].astype(np.float32)
                @ history[0, start : start + count, 0].astype(np.float32).T
                / 16
            )
            probability = np.exp(score - score.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            attended[row] = probability @ history[1, start : start + count, 0].astype(np.float32)

        def publish(value):
            dtype = torch.bfloat16 if floating == ops.DType.BF16 else torch.float16
            return torch.from_numpy(value).to(dtype).float().numpy()

        coefficient = publish(1 / (1 + np.exp(-gate.astype(np.float32))))
        activation = publish(publish(attended) * coefficient)
        expected = publish(activation.reshape(rows, -1) @ decoded.T)
        np.testing.assert_allclose(
            execution.outputs[0].native.float().cpu().numpy(), expected, rtol=3e-2, atol=3e-3
        )
    finally:
        if execution is not None:
            for output in execution.outputs:
                output.close()
        if compiled is not None:
            compiled.close()
        for resource in reversed(resources):
            resource.close()
        device.close()
