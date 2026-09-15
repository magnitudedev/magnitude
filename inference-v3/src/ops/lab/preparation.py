"""Prepare and observe an isolated formula through the production execution path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import numpy as np

from ..binding import Binding, Residency
from ..compiler.compilation import CompiledFunction, CompileOptions, analyze_graph, materialize
from ..compiler.dependencies import CodeDependency
from ..runtime.observation import RuntimeObservation
from ..runtime.resources import DeviceRuntime, Execution, Resource
from ..tensor.graph import ValueKind
from .fixtures import FormulaFixture, decode_dense, encode_dense
from .records import Implementation, MeasurementProtocol


@dataclass(frozen=True, slots=True)
class InvocationInputs:
    dynamic: tuple[Resource, ...]
    state: dict[int, Resource]


class NumericalMismatch(ArithmeticError):
    """A value disagrees with the oracle; distinct from illegal state mutation."""


class PreparedFormula:
    """Resident fixtures and one compiled production specialization.

    The worker owns refresh/cache policy. This object never claims a cached
    executable incorporates edited Python definitions; it identifies the exact
    program it actually compiled and can be replaced independently of fixtures.
    """

    def __init__(self, fixture: FormulaFixture, device: DeviceRuntime, options: CompileOptions):
        self.fixture, self.device, self.options = fixture, device, options
        self._closed = False
        self._owned = ExitStack()
        self._dynamic: dict[int, Resource] = {}
        self._state_content: dict[int, bytes] = {}
        self._readonly_state: dict[int, Resource] = {}
        self._readonly_content: dict[int, bytes] = {}
        graph = fixture.isolated.graph
        written = {identity for node in graph.nodes for identity, _, _ in node.effects.writes}
        readonly_backing: dict[int, tuple[Resource, bytes]] = {}
        constants: dict[int, Resource | Binding] = {}
        try:
            for identity, tensor in fixture.inputs.items():
                value = graph.value(identity)
                if value.kind == ValueKind.RESOURCE:
                    content = (tensor.physical if isinstance(tensor.physical, bytes)
                               else encode_dense(tensor.reference, tensor.spec))
                    if value.resource_id in written:
                        self._state_content[identity] = content
                    else:
                        # Read-only history is resident input, not mutable state
                        # that must be uploaded again before every sample.
                        assert value.resource_id is not None
                        existing = readonly_backing.get(value.resource_id)
                        if existing is None:
                            resource = device.upload(tensor.spec, content)
                            self._owned.callback(resource.close)
                            readonly_backing[value.resource_id] = resource, content
                        else:
                            resource, previous_content = existing
                            if resource.spec != tensor.spec or content != previous_content:
                                raise ValueError("aliased fixture state views need a shared backing/offset description")
                        self._readonly_state[identity] = resource
                        self._readonly_content[value.resource_id] = content
                    continue
                physical = tensor.physical
                if isinstance(physical, Binding):
                    if value.kind == ValueKind.CONSTANT:
                        constants[identity] = physical
                        # Initial resident import belongs to preparation, not to
                        # any subsequent invocation's complete-operation interval.
                        if physical.residency == Residency.RESIDENT:
                            resident = device.resolve(physical)
                            self._owned.callback(resident.close)
                        continue
                    if physical.residency == Residency.STREAMED:
                        raise ValueError("streamed fixture inputs must be declared immutable bindings")
                    resource = device.resolve(physical)
                elif isinstance(physical, Resource):
                    if physical.device is not device:
                        raise ValueError("fixture resource belongs to another device owner")
                    resource = physical.fork()
                else:
                    resource = device.upload(tensor.spec, physical)
                self._owned.callback(resource.close)
                if value.kind == ValueKind.CONSTANT:
                    constants[identity] = resource
                else:
                    self._dynamic[identity] = resource
            self.plan = analyze_graph(
                graph, capabilities=device.capabilities, compiler_identity=device.compiler_identity,
                available_bytes=device.available_bytes, options=options, constants=constants,
            )
            self.compiled: CompiledFunction = materialize(self.plan, device=device, constants=constants)
            self._owned.callback(self.compiled.close)
            definitions = []
            authored: dict[tuple[str, str], CodeDependency] = {
                (item.module, item.symbol): item for item in self.compiled.code_dependencies
            }
            for value in constants.values():
                if isinstance(value, Binding):
                    importer = device._imports.get(value.fingerprint)
                    if importer is not None:
                        for program in importer._programs.values():
                            for item in program.code_dependencies:
                                authored[item.module, item.symbol] = item
            for operation in self.plan.operations:
                for dependency in operation.dependencies:
                    authored[dependency.module, dependency.symbol] = dependency
                if operation.definition is not None:
                    definitions.append(operation.definition.identity)
                    for dependency in operation.definition.dependencies:
                        authored[dependency.module, dependency.symbol] = dependency
                if operation.source_loop is not None:
                    definitions.extend(template.definition.identity for template in operation.source_loop.templates)
                    for template in operation.source_loop.templates:
                        for dependency in template.definition.dependencies:
                            authored[dependency.module, dependency.symbol] = dependency
            self.code_dependencies = tuple(authored[key] for key in sorted(authored))
            dependencies = tuple(dict.fromkeys(definitions))
            payload = (dependencies,
                       tuple((item.module, item.symbol, item.fingerprint) for item in self.code_dependencies),
                       self.plan.graph.fingerprint, options.mode, options.precision,
                       device.compiler_identity, device.capabilities.fingerprint)
            fingerprint = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
            self.implementation = Implementation(
                fingerprint=fingerprint, compiler=device.compiler_identity, dependencies=dependencies,
                authored=self.code_dependencies,
            )
        except BaseException:
            self._owned.close()
            raise

    @contextmanager
    def inputs(self) -> Iterator[InvocationInputs]:
        """Reset written state; reuse resident read-only inputs between samples."""
        self.device.check()
        if self._closed:
            raise RuntimeError("prepared formula is closed")
        graph = self.compiled.graph
        original = self.fixture.isolated.graph
        # Pruning retains all declared inputs/constants/resources in their order;
        # translate by the declaration list rather than relying on incidental IDs.
        source_inputs = (*original.inputs, *original.constants, *original.resources)
        compiled_inputs = (*graph.inputs, *graph.constants, *graph.resources)
        identities = dict(zip(compiled_inputs, source_inputs, strict=True))
        state = {}
        backing: dict[int, tuple[Resource, bytes]] = {}
        with ExitStack() as retained:
            for identity in graph.resources:
                source_id = identities[identity]
                if source_id in self._readonly_state:
                    state[identity] = self._readonly_state[source_id]
                    continue
                value = graph.value(identity)
                content = self._state_content[source_id]
                assert value.resource_id is not None
                existing = backing.get(value.resource_id)
                if existing is None:
                    resource = self.device.upload(value.spec, content)
                    retained.callback(resource.close)
                    backing[value.resource_id] = resource, content
                else:
                    resource, previous_content = existing
                    if resource.spec != value.spec or content != previous_content:
                        raise ValueError("aliased fixture state views need a shared backing/offset description")
                state[identity] = resource
            yield InvocationInputs(tuple(self._dynamic[identities[identity]] for identity in graph.inputs), state)

    def execute(self, inputs: InvocationInputs) -> Execution:
        return self.compiled.submit(*inputs.dynamic, resources=inputs.state)

    @staticmethod
    def retire(execution: Execution) -> None:
        try:
            execution.completion.wait()
        finally:
            # Completion keeps its own claims if a native wait fails.
            for output in execution.outputs:
                output.close()

    def check(self, inputs: InvocationInputs, protocol: MeasurementProtocol) -> None:
        execution = self.execute(inputs)
        mismatches = []

        def check_value(resource, expected):
            try:
                self._check_value(resource, expected, protocol)
            except NumericalMismatch as failure:
                mismatches.append(str(failure))

        try:
            execution.completion.wait()
            # The isolated reference retains the original output order, including
            # duplicate ports. Pruning does not reorder observable outputs.
            for resource, expected in zip(execution.outputs, self.fixture.reference.outputs, strict=True):
                check_value(resource, expected)
            # Check complete mutated resources too, including untouched regions
            # and writes that are observable without being a formula return port.
            latest = {}
            for value in self.fixture.isolated.graph.values:
                if value.resource_id is not None and (
                    value.resource_id not in latest or
                    (value.resource_version or 0) > (latest[value.resource_id].resource_version or 0)
                ):
                    latest[value.resource_id] = value
            for identity, resource in inputs.state.items():
                value = self.compiled.graph.value(identity)
                immutable = self._readonly_content.get(value.resource_id)
                if immutable is not None:
                    # No numerical write is permitted here. Byte equality is
                    # stronger and avoids decoding a large untouched KV cache
                    # into FP32 just to compare it with itself on each request.
                    if self.device.read(resource) != immutable:
                        raise AssertionError("operation modified a read-only formula input")
                    continue
                final = latest[value.resource_id]
                expected = self.fixture.reference.values[final.id]
                check_value(resource, np.asarray(expected).reshape(resource.spec.shape))
            by_resource = {self.compiled.graph.value(identity).resource_id: resource
                           for identity, resource in inputs.state.items()}
            for identity, resource in zip(self.compiled.graph.outputs, execution.outputs, strict=True):
                value = self.compiled.graph.value(identity)
                if value.resource_id is not None:
                    backing = by_resource[value.resource_id]
                    if resource._lease.allocation is not backing._lease.allocation:
                        raise AssertionError("formula state alias was materialized as unrelated backing")
            if mismatches:
                raise NumericalMismatch("\n".join(mismatches))
        finally:
            for output in execution.outputs:
                output.close()

    def _check_value(self, resource: Resource, expected, protocol: MeasurementProtocol) -> None:
        actual = decode_dense(self.device.read(resource), resource.spec)
        if tuple(actual.shape) != tuple(np.shape(expected)):
            raise AssertionError("operation output shape differs from formula reference")
        try:
            if resource.spec.dtype.floating:
                np.testing.assert_allclose(actual, expected, atol=protocol.absolute_tolerance,
                                           rtol=protocol.relative_tolerance, equal_nan=True)
            else:
                np.testing.assert_array_equal(actual, expected)
        except AssertionError as error:
            raise NumericalMismatch(str(error)) from error

    def sample(self, inputs: InvocationInputs, *, kernel_limit: int | None = None) -> RuntimeObservation:
        with self.device.observe(kernel_limit=kernel_limit) as capture:
            execution = self.execute(inputs)
            self.retire(execution)
        return capture.result

    def close(self) -> None:
        if not self._closed:
            # Don't discard registered cleanup on a failed drain. A later owner
            # retry can still finish the work and release every prepared claim.
            self.device.drain()
            self._owned.close()
            self._closed = True
