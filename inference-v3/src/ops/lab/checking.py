"""Independent operation checks over a diagnostic replay of production kernels.

Replay preserves each selected physical operation and its fusion. Only host
submission boundaries change, so scratch can be inspected before it is reused.
No diagnostic duration is a performance sample.
"""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np

from ..compiler.compilation import materialize
from ..compiler.execution import execution_graph
from ..compiler.lowering import SubmissionUnit
from ..kv import KVRepresentation
from ..kv_codecs import encode_kv_reference
from ..tensor.primitive import primitives
from .fixtures import decode_dense
from .preparation import NumericalMismatch, check_value


def assert_close(actual, expected, spec, protocol):
    try:
        if spec.dtype.floating:
            np.testing.assert_allclose(
                actual,
                expected,
                atol=protocol.absolute_tolerance,
                rtol=protocol.relative_tolerance,
                equal_nan=True,
            )
        else:
            np.testing.assert_array_equal(actual, expected)
    except AssertionError as error:
        raise NumericalMismatch(str(error)) from error


def appended_bytes(initial, spec, keys, values, destinations):
    """Encode only the written rows; retain every untouched byte and padding bit."""
    representation = spec.representation
    logical = np.concatenate((keys, values), axis=-1)
    encoded = encode_kv_reference(logical, representation)
    result = bytearray(initial)
    capacity, heads, _ = spec.shape
    rows = len(destinations)
    source_planes = representation.planes(rows * heads)
    destination_planes = representation.planes(capacity * heads)
    selected = [(i, int(d)) for i, d in enumerate(destinations) if d >= 0]
    if any(d >= capacity for _, d in selected) or len({d for _, d in selected}) != len(selected):
        raise NumericalMismatch("KV write destinations are not distinct in-capacity rows")
    for source, destination in zip(source_planes, destination_planes, strict=True):
        dtype = np.dtype(f"V{source.dtype.itemsize}")
        src = np.frombuffer(
            encoded, dtype=dtype, offset=source.offset, count=source.nbytes // dtype.itemsize
        )
        dst = np.frombuffer(
            result,
            dtype=dtype,
            offset=destination.offset,
            count=destination.nbytes // dtype.itemsize,
        )
        if source.name.endswith(".codes") and representation.packing_version == 2:
            src = src.reshape(heads, -1, rows, 4)
            dst = dst.reshape(heads, -1, capacity, 4)
            for i, d in selected:
                dst[:, :, d, :] = src[:, :, i, :]
        else:
            src = src.reshape(rows, heads, -1)
            dst = dst.reshape(capacity, heads, -1)
            for i, d in selected:
                dst[d] = src[i]
    return bytes(result)


def checked_append(initial, spec, expected, actual, input_specs, protocol):
    """Prove producer accuracy, then encode its actual values independently."""
    for reference, captured, port in zip(expected, actual, input_specs, strict=True):
        assert_close(captured, reference, port, protocol)
    return appended_bytes(initial, spec, *actual)


class CheckedExecution:
    """A checked result, including complete mutated and read-only resources."""

    def __init__(self, outputs, resources, operations, encoded_writes, readonly=()):
        self.outputs, self.resources = outputs, resources
        self.operations, self.encoded_writes = operations, encoded_writes
        self.readonly = frozenset(readonly)

    def compare(self, device, outputs, resources, protocol):
        for actual, expected in zip(outputs, self.outputs, strict=True):
            if isinstance(actual.spec.representation, KVRepresentation):
                if device.read(actual) != expected:
                    raise NumericalMismatch(
                        "production encoded output differs from checked execution"
                    )
            else:
                check_value(device, actual, decode_dense(expected, actual.spec), protocol)
        for name, expected in self.resources.items():
            actual = resources[name]
            if name in self.readonly or isinstance(actual.spec.representation, KVRepresentation):
                content = device.read(actual)
                if content != expected:
                    differences = sum(a != b for a, b in zip(content, expected, strict=True))
                    raise NumericalMismatch(
                        f"production encoded state {name} differs from checked execution: "
                        f"{differences} bytes"
                    )
            else:
                check_value(device, actual, decode_dense(expected, actual.spec), protocol)


class OperationChecker:
    def __init__(self, compiled, decode_weight, protocol):
        if compiled.plan is None:
            raise ValueError("numerical inspection requires the production compilation plan")
        plan = compiled.plan
        units = tuple(
            SubmissionUnit(i, (operation,), "numerical inspection")
            for i, operation in enumerate(plan.operations)
            if operation.kernel_count
        )
        # Submission indices must be dense even when metadata-only operations exist.
        units = tuple(replace(unit, index=i) for i, unit in enumerate(units))
        plan = replace(
            plan, submissions=units, execution=execution_graph(plan.graph, units, plan.bindings)
        )
        self.compiled = materialize(
            plan,
            device=compiled.device,
            constants={**compiled._constants, **compiled._source_bindings},
            # Production may prebind mutable KV slabs. Inspection must bind its
            # private copies at invocation, never advance those production slabs.
            static_resources={},
        )
        self.decode_weight, self.protocol = decode_weight, protocol

    def close(self):
        self.compiled.close()

    def check(self, inputs, resources):
        compiled, protocol = self.compiled, self.protocol
        graph, device = compiled.graph, compiled.device
        state = {}
        count = encoded_writes = 0

        def inspect(phase, unit, physical, workspace):
            nonlocal count, encoded_writes
            operation = unit.calls[0].operation
            if phase == "before":
                state.clear()
                state["reference"] = {
                    identity: self.decode_weight(graph.value(identity))
                    if identity in graph.constants
                    else decode_dense(device.read(physical[identity]), graph.value(identity).spec)
                    for identity in operation.inputs
                }
                state["bytes"] = {
                    identity: device.read(physical[identity])
                    for identity in operation.inputs
                    if graph.value(identity).resource_id is not None
                }
                return
            reference = state["reference"]
            encoded = dict(state["bytes"])
            exposed = set(operation.inputs) | set(operation.outputs)

            def captured(identity):
                root = graph.alias_root(identity)
                if identity in operation.workspace_values:
                    resource = workspace[operation.name, operation.workspace_values[identity]]
                elif identity in exposed:
                    resource = physical[identity]
                elif root in exposed:
                    resource = physical[root]
                else:
                    raise ValueError(
                        f"operation {operation.name} does not expose codec input {identity}"
                    )
                return decode_dense(device.read(resource), graph.value(identity).spec)

            for node_id in sorted(operation.nodes):
                node = graph.node(node_id)
                if node.operation == "kv_append" and isinstance(
                    graph.value(node.inputs[0]).spec.representation, KVRepresentation
                ):
                    # Validate producer accuracy before crossing the discontinuity.
                    # Actual producer values never serve as their own oracle.
                    actual = tuple(captured(i) for i in node.inputs[1:])
                    encoded[node.outputs[0]] = checked_append(
                        encoded[node.inputs[0]],
                        graph.value(node.inputs[0]).spec,
                        tuple(reference[i] for i in node.inputs[1:]),
                        actual,
                        tuple(graph.value(i).spec for i in node.inputs[1:]),
                        protocol,
                    )
                    reference.update(zip(node.inputs[1:], actual, strict=True))
                    encoded_writes += 1
                results = primitives.get(node.operation).evaluate(
                    tuple(reference[i] for i in node.inputs),
                    node.attributes,
                    tuple(graph.value(i).spec for i in node.outputs),
                )
                reference.update(zip(node.outputs, results, strict=True))
            for identity in operation.outputs:
                if identity in encoded:
                    if device.read(physical[identity]) != encoded[identity]:
                        raise NumericalMismatch(
                            f"{operation.name}: encoded state bytes disagree with codec"
                        )
                else:
                    try:
                        check_value(device, physical[identity], reference[identity], protocol)
                    except NumericalMismatch as failure:
                        raise NumericalMismatch(
                            f"{operation.name}, output {identity}: {failure}"
                        ) from failure
            written = {r for n in operation.nodes for r, _, _ in graph.node(n).effects.writes}
            for identity, content in state["bytes"].items():
                if (
                    graph.value(identity).resource_id not in written
                    and device.read(physical[identity]) != content
                ):
                    raise NumericalMismatch(f"{operation.name} modified read-only state")
            count += 1
            state.clear()

        with ExitStack() as owned:
            copied = {}
            by_resource = {}
            for identity in graph.resources:
                value = graph.value(identity)
                original = resources[value.name]
                if value.resource_id not in by_resource:
                    copy = device.upload(original.spec, device.read(original))
                    owned.callback(copy.close)
                    by_resource[value.resource_id] = copy
                copied[value.name] = by_resource[value.resource_id]
            execution = compiled.submit(*inputs, resources=copied, inspect=inspect)
            try:
                execution.completion.wait()
                return CheckedExecution(
                    tuple(device.read(output) for output in execution.outputs),
                    {name: device.read(resource) for name, resource in copied.items()},
                    count,
                    encoded_writes,
                    readonly={
                        graph.value(i).name
                        for i in graph.resources
                        if graph.value(i).resource_id
                        not in {r for node in graph.nodes for r, _, _ in node.effects.writes}
                    },
                )
            finally:
                for output in execution.outputs:
                    output.close()
