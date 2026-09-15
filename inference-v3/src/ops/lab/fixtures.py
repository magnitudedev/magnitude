"""Reuse traced reference values and production bindings at typed formula boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from weakref import WeakValueDictionary

import numpy as np
from numpy.typing import NDArray

from ..binding import Binding
from ..formula import FormulaHandle, FormulaIndex
from ..isolation import IsolatedFormula
from ..representations import Dense
from ..runtime.resources import Resource
from ..tensor.graph import Graph, Value, _stable
from ..tensor.primitive import ReferenceResult, evaluate_reference, primitives
from ..tensor.types import DENSE, DType, TensorSpec


def _storage_dtype(spec: TensorSpec) -> DType:
    if spec.representation is None:
        return spec.dtype
    if isinstance(spec.representation, Dense):
        return spec.representation.dtype
    raise ValueError("encoded fixture storage requires the production Binding")


def encode_dense(value: NDArray, spec: TensorSpec) -> bytes:
    """Encode reference fixture data, never an operation's numerical execution."""
    if spec.layout != DENSE:
        raise ValueError("non-dense fixture storage requires explicit physical bindings")
    dtype = _storage_dtype(spec)
    if tuple(value.shape) != spec.shape:
        raise ValueError("fixture shape differs from its traced tensor")
    if dtype == DType.BF16:
        floating = np.asarray(value, dtype=np.float32)
        bits = floating.view(np.uint32)
        # Fixture inputs must already denote representable values. Do not
        # silently round data while retaining a reference for the unrounded data.
        if np.any((bits & 0xFFFF != 0) & ~np.isnan(floating)):
            raise ValueError("BF16 reference fixture contains unrepresentable input values")
        return np.ascontiguousarray(bits >> 16, dtype=np.uint16).tobytes()
    encoded = np.asarray(value, dtype=dtype.value)
    if not np.array_equal(value, encoded, equal_nan=True):
        raise ValueError("fixture values are not representable in their declared storage dtype")
    return np.ascontiguousarray(encoded).tobytes()


def decode_dense(content: bytes, spec: TensorSpec) -> NDArray:
    from ..kv import KVRepresentation
    if isinstance(spec.representation, KVRepresentation):
        from ..kv_codecs import decode_kv_reference
        return decode_kv_reference(content, spec)
    if spec.layout != DENSE:
        raise ValueError("non-dense observations require their declared physical decoder")
    dtype = _storage_dtype(spec)
    if len(content) != spec.storage_nbytes:
        raise ValueError("observed storage byte count differs from the tensor specification")
    if dtype == DType.BF16:
        result = (np.frombuffer(content, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        result = np.frombuffer(content, dtype=dtype.value)
    return result.reshape(spec.shape)


def tensor_identity(spec, reference, physical):
    digest = hashlib.sha256()
    digest.update(json.dumps(_stable(spec), sort_keys=True).encode())
    digest.update(reference.dtype.str.encode())
    digest.update(memoryview(np.ascontiguousarray(reference)).cast("B"))
    if isinstance(physical, Binding):
        digest.update(physical.value_identity.encode())
    elif isinstance(physical, bytes):
        digest.update(physical)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class FixtureTensor:
    spec: TensorSpec
    reference: NDArray
    physical: Binding | Resource | bytes
    identity: str


@dataclass(frozen=True, slots=True, weakref_slot=True)
class FormulaFixture:
    isolated: IsolatedFormula
    inputs: Mapping[int, FixtureTensor]
    identity: str
    _reference: ReferenceResult | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))

    @property
    def reference(self) -> ReferenceResult:
        if self._reference is None:
            reference = evaluate_reference(self.isolated.graph, {
                identity: tensor.reference for identity, tensor in self.inputs.items()
            })
            for value in reference.values.values():
                if isinstance(value, np.ndarray):
                    value.flags.writeable = False
            object.__setattr__(self, "_reference", reference)
        return self._reference

    def storage(self) -> dict[int, int]:
        """Unique retained host backing, including encoded fixtures and oracle values."""
        result = {}

        def include(value):
            if isinstance(value, np.ndarray):
                while isinstance(value.base, np.ndarray):
                    value = value.base
                result[id(value)] = value.nbytes
            elif isinstance(value, bytes):
                result[id(value)] = len(value)

        for tensor in self.inputs.values():
            include(tensor.reference)
            include(tensor.physical)
        if self._reference is not None:
            for value in self._reference.values.values():
                include(value)
        return result


class Fixture:
    """One prepared model value table, shared by all its formula measurements.

    Values can come from a previously evaluated independent reference or captured
    model boundaries. A leaf is checked by evaluating its own formula, not by
    trusting captured implementation outputs. No benchmark equations are authored.
    """

    def __init__(
        self, root: Graph, values: Mapping[int, NDArray], *,
        bindings: Mapping[int, Binding | Resource | bytes] | None = None,
        capture: Callable[[Value], NDArray] | None = None,
    ):
        self.root = root
        self._capture = capture
        self._bindings = dict(bindings or {})
        self._values: dict[int, NDArray] = {}
        # These are shared indexes, not independent retaining caches. The worker
        # owns the single bounded preparation/reference cache.
        self._tensors: WeakValueDictionary[int, FixtureTensor] = WeakValueDictionary()
        self._boundaries: WeakValueDictionary[FormulaHandle, FormulaFixture] = WeakValueDictionary()
        for identity, value in values.items():
            if type(identity) is not int or not 0 <= identity < len(root.values):
                raise ValueError("fixture values must refer to this trace's tensor values")
            array = np.asarray(value)
            if array.dtype.hasobject or tuple(array.shape) != root.value(identity).spec.shape:
                raise ValueError("fixture value must have concrete numeric storage and the traced shape")
            self._values[identity] = array
        for identity, binding in self._bindings.items():
            spec = root.value(identity).spec
            compatible = len(binding) == spec.storage_nbytes if isinstance(binding, bytes) else binding.spec == spec
            if (identity not in self._values and capture is None) or not compatible:
                raise ValueError("physical fixture binding requires matching reference values and specification")

    @classmethod
    def from_inputs(
        cls, root: Graph, inputs: Mapping[int, NDArray], *,
        bindings: Mapping[int, Binding | Resource | bytes] | None = None,
        capture: Callable[[Value], NDArray] | None = None,
    ) -> Fixture:
        """Derive selected boundary inputs from this production trace on demand.

        The optional capture supplies immutable root inputs (for example lazily
        decoded artifact values). Intermediate values come from the existing
        primitive references; there is no separately authored benchmark equation.
        Upstream preparation is paid once per selected boundary, not per sample.
        """
        declared = set((*root.inputs, *root.constants, *root.resources))
        if not set(inputs) <= declared:
            raise ValueError("from_inputs accepts only declared production inputs")
        known = {identity: np.array(value, copy=True) for identity, value in inputs.items()}
        for value in known.values():
            value.flags.writeable = False

        def derive(target: Value):
            if target.id in known:
                return known[target.id]
            available = {identity: tensor.reference for identity, tensor in fixture._tensors.items()}
            available.update(known)
            needed = set()
            pending = [target.id]
            while pending:
                identity = pending.pop()
                if identity in available or identity in needed:
                    continue
                needed.add(identity)
                producer = root.value(identity).producer
                if producer is not None:
                    pending.extend(root.node(producer).inputs)
            values = dict(available)
            for identity in sorted(needed & declared):
                if capture is None:
                    raise KeyError(f"fixture has no value for production input {identity}")
                values[identity] = np.array(capture(root.value(identity)), copy=True)
                value = values[identity]
                spec = root.value(identity).spec
                if value.dtype.hasobject or tuple(value.shape) != spec.shape:
                    raise ValueError("captured production input has incompatible numeric storage or shape")
                value.flags.writeable = False
            node_ids = {root.value(identity).producer for identity in needed}
            node_ids.discard(None)
            consumers = {}
            for node_id in node_ids:
                for identity in root.node(node_id).inputs:
                    consumers[identity] = consumers.get(identity, 0) + 1
            for node_id in sorted(node_ids):
                node = root.node(node_id)
                primitive = primitives.get(node.operation)
                if primitive.reference is None:
                    raise NotImplementedError(f"{node.operation} has no independent reference")
                # The shared evaluator supplies immutable views, not repeated
                # whole-weight copies for each mathematical use.
                result = primitive.evaluate(tuple(values[identity] for identity in node.inputs),
                                            node.attributes, tuple(root.value(identity).spec for identity in node.outputs))
                values.update(zip(node.outputs, result, strict=True))
                for identity in node.inputs:
                    consumers[identity] -= 1
                    if not consumers[identity] and identity != target.id:
                        values.pop(identity, None)
                for identity in node.outputs:
                    if identity != target.id and not consumers.get(identity):
                        values.pop(identity, None)
            return values[target.id]

        fixture = cls(root, known, bindings=bindings, capture=derive)
        return fixture

    @classmethod
    def from_reference(
        cls, root: Graph, reference: ReferenceResult, *,
        bindings: Mapping[int, Binding | Resource] | None = None,
    ) -> Fixture:
        return cls(root, reference.values, bindings=bindings)

    def _tensor(self, identity: int) -> FixtureTensor:
        cached = self._tensors.get(identity)
        if cached is not None:
            return cached
        if identity not in self._values:
            if self._capture is None:
                raise KeyError(f"fixture has no captured value for formula input {identity}")
            value = np.asarray(self._capture(self.root.value(identity)))
            if value.dtype.hasobject or tuple(value.shape) != self.root.value(identity).spec.shape:
                raise ValueError("captured fixture input has incompatible numeric storage or shape")
        else:
            value = self._values[identity]
        # Snapshot only the selected boundary, not every intermediate in a large
        # model table. Its content identity remains fixed for all later samples.
        value = np.array(value, copy=True)
        value.flags.writeable = False
        spec = self.root.value(identity).spec
        physical = self._bindings.get(identity)
        from ..kv import KVRepresentation
        if isinstance(spec.representation, KVRepresentation):
            if isinstance(physical, Resource):
                physical = physical.device.read(physical)
            if physical is None:
                raise ValueError("represented KV fixtures require original packed bytes or a resource binding")
        if physical is None:
            physical = encode_dense(value, spec)
        cached = FixtureTensor(spec, value, physical, tensor_identity(spec, value, physical))
        self._tensors[identity] = cached
        return cached

    @property
    def source_spans(self):
        """Actual streamed paths available for an explicit characterization request."""
        from ..binding import Residency
        return tuple(plane.span for binding in self._bindings.values()
                     if isinstance(binding, Binding) and binding.residency == Residency.STREAMED
                     for plane in binding.planes)

    def boundary(self, target: FormulaHandle) -> FormulaFixture:
        if not isinstance(target, FormulaHandle) or target.graph is not self.root:
            raise ValueError("measurement target must belong to the fixture trace")
        cached = self._boundaries.get(target)
        if cached is not None:
            return cached
        isolated = target.isolate()
        inputs = {port.local: self._tensor(port.original) for port in isolated.inputs}
        payload = tuple((port.local, inputs[port.local].identity) for port in isolated.inputs)
        identity = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
        cached = FormulaFixture(isolated, inputs, identity)
        self._boundaries[target] = cached
        return cached

    def release_boundary(self, target: FormulaHandle) -> None:
        """Allow bounded worker caches to discard derived references without losing inputs."""
        self._boundaries.pop(target, None)

    def capture_boundary(self, target: FormulaHandle, device, options):
        """Execute upstream production operations once, stopping before target.

        This is an explicit preparation build: exposing boundary values can change
        fusion. Its execution is never reported as production parent timing.
        Returns an immutable boundary snapshot and its compiled dependencies.
        """
        from ..compiler.compilation import analyze_graph, materialize
        from ..tensor.graph import ValueKind

        if target.graph is not self.root or not target.call.complete:
            raise ValueError("capture requires a complete scope in this production trace")
        isolated = target.isolate()
        cutoff = min(target.call.nodes) if target.call.nodes else 0
        nodes = self.root.nodes[:cutoff]
        declared = (*self.root.inputs, *self.root.constants, *self.root.resources)
        available = set(declared) | {v for node in nodes for v in node.outputs}
        outputs = tuple(port.original for port in isolated.inputs
                        if self.root.value(port.original).kind != ValueKind.CONSTANT)
        if not set(outputs) <= available:
            raise ValueError("selected boundary requires computation outside its upstream prefix")
        prefix = replace(self.root, nodes=nodes, outputs=outputs, formulas=FormulaIndex(tuple(
            call.remap({v: v for v in available}, {node.id: node.id for node in nodes})
            for call in self.root.formulas)))
        captured = {}
        captured_bytes = {}
        with ExitStack() as retained:
            constants, resources = {}, {}
            backing = {}
            for identity in declared:
                value = self.root.value(identity)
                physical = self._bindings.get(identity)
                if value.kind == ValueKind.CONSTANT and isinstance(physical, Binding):
                    constants[identity] = physical
                    continue
                # Mutable starting state is copied; preparation must never advance
                # the caller's retained history. Immutable resources may be leased.
                tensor = self._tensor(identity)
                if isinstance(physical, Resource) and value.kind != ValueKind.RESOURCE:
                    resource = physical.fork()
                else:
                    content = (tensor.physical if isinstance(tensor.physical, bytes)
                               else encode_dense(tensor.reference, tensor.spec))
                    shared = backing.get(value.resource_id) if value.resource_id is not None else None
                    if shared is not None:
                        resource, prior = shared
                        if prior != content or resource.spec != tensor.spec:
                            raise ValueError("capture requires compatible aliased state views")
                        resources[identity] = resource
                        continue
                    resource = device.upload(tensor.spec, content)
                    if value.resource_id is not None:
                        backing[value.resource_id] = resource, content
                retained.callback(resource.close)
                resources[identity] = resource
                if value.kind == ValueKind.CONSTANT:
                    constants[identity] = resource
            plan = analyze_graph(prefix, compiler_target=device.compiler_target,
                                 compiler_identity=device.compiler_identity, options=options,
                                 available_bytes=device.available_bytes, constants=constants)
            compiled = materialize(plan, device=device, constants=constants)
            retained.callback(compiled.close)
            original = (*prefix.inputs, *prefix.constants, *prefix.resources)
            local = (*compiled.graph.inputs, *compiled.graph.constants, *compiled.graph.resources)
            mapping = dict(zip(local, original, strict=True))
            execution = compiled.submit(*(resources[mapping[i]] for i in compiled.graph.inputs),
                                        resources={i: resources[mapping[i]] for i in compiled.graph.resources})
            try:
                execution.completion.wait()
                for identity, resource in zip(outputs, execution.outputs, strict=True):
                    content = device.read(resource)
                    captured[identity] = decode_dense(content, resource.spec).copy()
                    captured_bytes[identity] = content
            finally:
                for resource in execution.outputs:
                    resource.close()
            dependencies = compiled.code_dependencies
        # Preserve production immutable weight bindings. Captured state owns bytes;
        # no live alias can be mutated behind a replay fixture's identity.
        bindings = {port.original: self._bindings[port.original] for port in isolated.inputs
                    if port.original in self._bindings and port.original not in captured}
        bindings.update(captured_bytes)
        snapshot = Fixture(self.root, captured, bindings=bindings,
                           capture=lambda value: self._tensor(value.id).reference)
        boundary = snapshot.boundary(target)
        return boundary, dependencies, prefix.fingerprint
