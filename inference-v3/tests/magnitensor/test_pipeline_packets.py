"""Numerical coverage of the complete packet and attention producer/consumer paths."""

import numpy as np
import pytest
import torch

import magnitensor as mt


def _affine_weight(device, rng, shape, dtype=mt.DType.F16):
    codes = rng.integers(0, 16, shape, dtype=np.uint8)
    packed = codes.ravel()[::2] | (codes.ravel()[1::2] << 4)
    groups = codes.size // 64
    # Exact BF16 coefficients: scale 1/64, bias -1/8.
    payload = (
        packed.tobytes()
        + np.full(groups, 0x3C80, dtype=np.uint16).tobytes()
        + np.full(groups, 0xBE00, dtype=np.uint16).tobytes()
    )
    spec = mt.TensorSpec(shape, dtype).with_representation(
        mt.Affine(mt.Code(4), 64, mt.DirectCoefficients(mt.DType.BF16, mt.DType.BF16))
    )
    return device.upload(spec, payload), codes.astype(np.float32) / 64 - 0.125


@pytest.mark.device
def test_packet_prefill_preserves_decode_coefficient_precision():
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(641)
    device = mt.device("metal", budget_bytes=8 << 20)
    resources, compiled_functions = [], []
    try:
        codes = rng.integers(0, 16, (32, 512), dtype=np.uint8)
        packed = codes.ravel()[::2] | (codes.ravel()[1::2] << 4)
        coefficients = torch.tensor([0.01, -0.1], dtype=torch.bfloat16)
        scale, bias = coefficients.float().numpy()
        bits = coefficients.view(torch.uint16).numpy()
        groups = codes.size // 64
        spec = mt.TensorSpec(codes.shape, mt.DType.BF16).with_representation(
            mt.Affine(mt.Code(4), 64, mt.DirectCoefficients(mt.DType.BF16, mt.DType.BF16))
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
                mt.TensorSpec((rows, 512), mt.DType.BF16),
                row.repeat(rows, 1).view(torch.uint16).numpy().tobytes(),
            )
            resources.append(source)
            compiled = mt.compile(
                lambda x, w: mt.linear(x, w, output_dtype=mt.DType.F32),
                signature=mt.Signature(
                    (
                        mt.Argument(source.spec, "x"),
                        mt.Argument(weight.spec, "w", mt.ValueKind.CONSTANT),
                    )
                ),
                device=device,
                constants={"w": weight},
                options=mt.CompileOptions(mode=mode),
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
    device = mt.device("metal", budget_bytes=8 << 20)
    resources = []
    compiled = execution = None
    try:
        table, table_values = _affine_weight(device, rng, (16, 512))
        resources.append(table)
        weight, weight_values = _affine_weight(device, rng, (32, 512))
        resources.append(weight)
        indices = np.arange(rows, dtype=np.int32)
        tokens = device.upload(mt.TensorSpec((rows,), mt.DType.I32), indices.tobytes())
        resources.append(tokens)

        def function(ids, embedding, projection):
            value = mt.embedding(ids, embedding)
            return value, mt.linear(value, projection)

        compiled = mt.compile(
            function,
            signature=mt.Signature(
                (
                    mt.Argument(tokens.spec, "tokens"),
                    mt.Argument(table.spec, "embedding", mt.ValueKind.CONSTANT),
                    mt.Argument(weight.spec, "projection", mt.ValueKind.CONSTANT),
                )
            ),
            device=device,
            constants={"embedding": table, "projection": weight},
            options=mt.CompileOptions(mode=mode),
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
@pytest.mark.parametrize("rows", [129, 256])
def test_production_gate_up_and_projection_match_reference(rows):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(907)
    device = mt.device("metal", budget_bytes=32 << 20)
    resources = []
    compiled = execution = None
    try:
        hidden = rng.normal(0, 0.05, (rows, 512)).astype(np.float16)
        source = device.upload(mt.TensorSpec(hidden.shape, mt.DType.F16), hidden.tobytes())
        resources.append(source)
        weights, values = [], []
        for _ in range(3):
            resource, value = _affine_weight(device, rng, (512, 512))
            resources.append(resource)
            weights.append(resource)
            values.append(value)

        def function(x, gate, up, down):
            projection = mt.linear(x, down)
            activated = mt.silu(mt.linear(x, gate)) * mt.linear(x, up)
            return projection, mt.linear(activated, down)

        names = ("x", "gate", "up", "down")
        compiled = mt.compile(
            function,
            signature=mt.Signature(
                tuple(
                    mt.Argument(
                        resource.spec,
                        name,
                        mt.ValueKind.INPUT if index == 0 else mt.ValueKind.CONSTANT,
                    )
                    for index, (name, resource) in enumerate(zip(names, resources, strict=True))
                )
            ),
            device=device,
            constants=dict(zip(names[1:], weights, strict=True)),
            options=mt.CompileOptions(mode="prefill"),
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
def test_persistent_routed_shared_pipeline_covers_sparse_routes_and_worker_reuse(rows):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    rng = np.random.default_rng(607)
    width, experts, selected = 512, 4, 2
    hidden = rng.normal(0, 0.1, (rows, width)).astype(np.float16)
    routes = np.tile(np.asarray([0, 1], np.int32), (rows, 1))
    scores = np.tile(np.asarray([0.25, 0.75], np.float32), (rows, 1))
    router = rng.normal(0, 0.1, (width,)).astype(np.float16)
    device = mt.device("metal", budget_bytes=32 << 20)
    resources = []
    compiled = execution = None
    try:
        for array, dtype in (
            (hidden, mt.DType.F16),
            (routes, mt.DType.I32),
            (scores, mt.DType.F32),
        ):
            resources.append(device.upload(mt.TensorSpec(array.shape, dtype), array.tobytes()))
        decoded = []
        for shape in [(experts, width, width)] * 3 + [(width, width)] * 3:
            resource, values = _affine_weight(device, rng, shape)
            resources.append(resource)
            decoded.append(values)
        resources.append(device.upload(mt.TensorSpec(router.shape, mt.DType.F16), router.tobytes()))

        def function(x, ids, probabilities, eg, eu, ed, sg, su, sd, sr):
            routed = mt.routed_experts(x, ids, probabilities, eg, eu, ed)
            shared = mt.linear(mt.silu(mt.linear(x, sg)) * mt.linear(x, su), sd)
            coefficient = mt.cast(mt.sigmoid(mt.row_dot(x, sr, output_dtype=mt.DType.F32)), x.dtype)
            return routed + shared * coefficient

        names = ("x", "ids", "probabilities", "eg", "eu", "ed", "sg", "su", "sd", "sr")
        signature = mt.Signature(
            tuple(
                mt.Argument(
                    resource.spec, name, mt.ValueKind.INPUT if i < 3 else mt.ValueKind.CONSTANT
                )
                for i, (name, resource) in enumerate(zip(names, resources, strict=True))
            )
        )
        compiled = mt.compile(
            function,
            signature=signature,
            device=device,
            constants=dict(zip(names[3:], resources[3:], strict=True)),
            options=mt.CompileOptions(mode="prefill"),
        )
        assert compiled.diagnostics.dispatches == 4
        execution = compiled.submit(*resources[:3])
        execution.completion.wait()

        def feedforward(gate, up, down, *, explicit_silu=False):
            g = (hidden.astype(np.float32) @ gate.T).astype(np.float16).astype(np.float32)
            u = (hidden.astype(np.float32) @ up.T).astype(np.float16).astype(np.float32)
            activated_gate = g / (1 + np.exp(-g))
            if explicit_silu:
                activated_gate = activated_gate.astype(np.float16).astype(np.float32)
            activated = (activated_gate * u).astype(np.float16).astype(np.float32)
            return (activated @ down.T).astype(np.float16).astype(np.float32)

        eg, eu, ed, sg, su, sd = decoded
        expected = np.stack(
            [scores[:, i, None] * feedforward(eg[i], eu[i], ed[i]) for i in range(selected)]
        ).sum(axis=0)
        coefficient = 1 / (1 + np.exp(-(hidden.astype(np.float32) @ router.astype(np.float32))))
        shared = (
            coefficient.astype(np.float16).astype(np.float32)[:, None]
            * feedforward(sg, su, sd, explicit_silu=True)
        ).astype(np.float16)
        expected = (
            expected.astype(np.float16).astype(np.float32) + shared.astype(np.float32)
        ).astype(np.float16)
        np.testing.assert_allclose(
            execution.outputs[0].native.cpu().numpy(), expected, rtol=3e-2, atol=3e-3
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
@pytest.mark.parametrize("floating", [mt.DType.F16, mt.DType.BF16])
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
    device = mt.device("metal", budget_bytes=32 << 20)
    resources = []
    compiled = execution = None
    try:
        rounded = []
        for value in arrays:
            dtype = mt.DType.I32 if value.dtype == np.int32 else floating
            if dtype == mt.DType.BF16:
                tensor = torch.from_numpy(value).to(torch.bfloat16)
                payload = tensor.view(torch.uint16).numpy().tobytes()
                rounded.append(tensor.float().numpy())
            else:
                payload = value.tobytes()
                rounded.append(value)
            resources.append(device.upload(mt.TensorSpec(value.shape, dtype), payload))
        weight, decoded = _affine_weight(device, rng, (32, heads * width), floating)
        resources.append(weight)

        def function(query, history, visible, gate, projection):
            attended = mt.causal_attention(query, history, visible, sequence_count=1)
            return mt.linear(
                mt.reshape(attended * mt.sigmoid(gate), (rows, heads * width)), projection
            )

        signature = mt.Signature(
            tuple(
                mt.Argument(
                    resource.spec,
                    name,
                    mt.ValueKind.CONSTANT
                    if name == "projection"
                    else (mt.ValueKind.RESOURCE if name == "history" else mt.ValueKind.INPUT),
                )
                for resource, name in zip(
                    resources, ("query", "history", "visible", "gate", "projection"), strict=True
                )
            )
        )
        compiled = mt.compile(
            function,
            signature=signature,
            device=device,
            constants={"projection": weight},
            options=mt.CompileOptions(mode="prefill"),
        )
        assert compiled.diagnostics.dispatches == (3 if capacity > 4096 else 2)
        expected_family = "attention.matrix-streaming-gated-output"
        assert any(
            c.selected and c.name.startswith(expected_family)
            for c in compiled.diagnostics.candidates
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
            dtype = torch.bfloat16 if floating == mt.DType.BF16 else torch.float16
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
