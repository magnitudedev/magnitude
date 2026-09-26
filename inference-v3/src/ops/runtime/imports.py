"""Planned bounded source I/O and TileLang conversion, owned by the runtime."""

from __future__ import annotations

import math
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum

from ..binding import Binding, CanonicalImport, DenseImport, EncodedImport, SourceSpan, Transform
from ..tensor.types import DType, TensorSpec


class ImportActionKind(StrEnum):
    READ = "source-read"
    UPLOAD = "upload"
    CONVERT = "convert"
    PUBLISH = "publish"


@dataclass(frozen=True, slots=True)
class ImportAction:
    index: int
    kind: ImportActionKind
    predecessors: tuple[int, ...]
    bytes: int
    step: int


@dataclass(frozen=True, slots=True)
class ImportStep:
    source: SourceSpan
    source_spec: TensorSpec
    target_offset: int
    elements: int
    first_block: int = 0


@dataclass(frozen=True, slots=True)
class ImportPlan:
    binding: Binding
    steps: tuple[ImportStep, ...]
    actions: tuple[ImportAction, ...]
    host_peak_bytes: int
    execution_peak_bytes: int


def plan_import(binding: Binding, *, chunk_bytes: int = 8 << 20) -> ImportPlan:
    """Metadata only. Source handles are never read during analysis."""
    if chunk_bytes < 4:
        raise ValueError("source import chunks must contain at least one storage word")
    if binding.resource is not None:
        return ImportPlan(binding, (), (), 0, 0)
    recipe = binding.recipe
    steps = []
    destination = 0
    for plane in binding.planes:
        groups = max(1, chunk_bytes // plane.group_bytes)
        capacity = groups * plane.group_bytes
        for first in range(0, plane.span.length, capacity):
            length = min(capacity, plane.span.length - first)
            elements = length // plane.group_bytes * plane.group_elements
            span = SourceSpan(plane.span.source, plane.span.offset + first, length)
            if isinstance(recipe, DenseImport):
                source_dtype = DType.U16 if recipe.source_dtype == DType.BF16 else recipe.source_dtype
                spec = TensorSpec((elements,), source_dtype)
                steps.append(ImportStep(span, spec, destination, elements))
                destination += elements * binding.spec.dtype.itemsize
            else:
                spec = TensorSpec((length,), DType.U8)
                steps.append(ImportStep(span, spec, destination, elements,
                                        first // recipe.block_bytes if isinstance(recipe, EncodedImport) else 0))
                destination += length
    actions = []
    host_peak = execution_stage = 0
    previous = ()
    for index, step in enumerate(steps):
        host_peak = max(host_peak, 2 * step.source.length)
        converted = step.elements * binding.spec.dtype.itemsize if isinstance(recipe, DenseImport) else 0
        published = (step.elements * binding.spec.storage_nbytes // binding.spec.elements
                     if isinstance(recipe, EncodedImport) else converted or step.source.length)
        execution_stage = max(execution_stage, step.source_spec.storage_nbytes + converted + 16)
        for kind, size in ((ImportActionKind.READ, step.source.length),
                           (ImportActionKind.UPLOAD, step.source_spec.storage_nbytes),
                           (ImportActionKind.CONVERT, published),
                           (ImportActionKind.PUBLISH, published)):
            if kind == ImportActionKind.CONVERT and isinstance(recipe, CanonicalImport):
                continue
            action = ImportAction(len(actions), kind, previous, size, index)
            actions.append(action)
            previous = (action.index,)
    return ImportPlan(binding, tuple(steps), tuple(actions), host_peak,
                      binding.spec.storage_nbytes + execution_stage)


class PreparedImport:
    """Selected conversion programs plus the exact source/transfer action plan."""
    def __init__(self, device, plan: ImportPlan):
        self.device, self.plan = device, plan
        self._programs = {}
        self._closed = False
        self._prepared = False

    def _program(self, key, build):
        from ..tensor.graph import _stable
        import json

        shared_key = (json.dumps(_stable((self.plan.binding.spec, self.plan.binding.recipe)), sort_keys=True), key)
        existing = self.device._import_programs.get(shared_key)
        if existing is None:
            existing = build()
            self.device._import_programs[shared_key] = existing
        self._programs[key] = existing
        return existing

    def prepare(self):
        """Compile selected import programs before source reads or invocation."""
        from ..compiler.compilation import CompileOptions, compile
        from ..tensor.graph import ValueKind
        from ..tensor.tracing import Argument, Signature
        from ..tensor import ops

        if self._closed:
            raise RuntimeError("import executable is closed")
        if self._prepared:
            return self
        binding, recipe = self.plan.binding, self.plan.binding.recipe
        extent_spec = TensorSpec((2,), DType.I32)
        for step in self.plan.steps:
            if isinstance(recipe, EncodedImport):
                valid = step.source.length // recipe.block_bytes
                key = ("encoded", step.source_spec, valid)
                def build_encoded(step=step, valid=valid):
                    def convert(source, extent, *, target):
                        return ops.quantized_import(source, target, extent,
                                                    codec=recipe.codec, staged_tiles=valid)
                    return compile(convert, signature=Signature((Argument(step.source_spec), Argument(extent_spec)),
                                   {"target": Argument(binding.spec, "target", ValueKind.RESOURCE)}),
                                   device=self.device, constants={}, options=CompileOptions(mode="import"))
                self._program(key, build_encoded)
                continue
            converted = isinstance(recipe, DenseImport) and (
                recipe.source_dtype != binding.spec.dtype or recipe.transform != Transform.IDENTITY
            )
            value_spec = step.source_spec
            if converted:
                key = ("dense", step.source_spec, binding.spec.dtype, recipe)
                def build_dense(step=step):
                    def convert(value):
                        decoded = (ops.decode_bfloat16(value, DType.F32)
                                   if recipe.source_dtype == DType.BF16 else ops.cast(value, DType.F32))
                        if recipe.transform == Transform.NEGATIVE_EXP:
                            decoded = -1.0 * ops.exp(decoded)
                        return ops.cast(decoded, binding.spec.dtype)
                    return compile(convert, signature=Signature((Argument(step.source_spec),)),
                                   device=self.device, constants={}, options=CompileOptions(mode="import"))
                self._program(key, build_dense)
                value_spec = TensorSpec(step.source_spec.shape, binding.spec.dtype)
            source_spec = TensorSpec((value_spec.storage_nbytes,), DType.U8)
            target_spec = TensorSpec((binding.spec.storage_nbytes,), DType.U8)
            key = ("publish", source_spec, target_spec)
            def build_publish(source_spec=source_spec, target_spec=target_spec):
                return compile(lambda source, extent, *, target: ops.byte_copy(source, target, extent),
                               signature=Signature((Argument(source_spec), Argument(TensorSpec((2,), DType.I64))),
                               {"target": Argument(target_spec, "target", ValueKind.RESOURCE)}),
                               device=self.device, constants={}, options=CompileOptions(mode="import"))
            self._program(key, build_publish)
        self._prepared = True
        return self

    def load(self, *, plan: ImportPlan | None = None):
        self.device.check()
        if self._closed:
            raise RuntimeError("import executable is closed")
        if not self._prepared:
            raise RuntimeError("imports must be prepared before source execution")
        selected = self.plan if plan is None else plan
        if plan is not None:
            def geometry(value):
                return (value.binding.spec, value.binding.recipe,
                        tuple((step.source_spec, step.source.length, step.target_offset,
                               step.elements, step.first_block) for step in value.steps))
            if geometry(selected) != geometry(self.plan):
                raise ValueError("rebound source plan changes prepared import geometry")
        binding, recipe = selected.binding, selected.binding.recipe
        if binding.resource is not None:
            if binding.resource.device is not self.device:
                raise ValueError("binding resource belongs to another runtime")
            return binding.resource.fork()
        target = self.device.allocate(binding.spec)
        try:
            for step in selected.steps:
                with ExitStack() as retained:
                    source_reservation = self.device.memory.reserve(step.source.length, self.device.host_domains)
                    retained.callback(source_reservation.close)
                    content = self.device.read_source(step.source, value_identity=binding.value_identity)
                    source = self.device.upload(step.source_spec, content)
                    retained.callback(source.close)
                    del content
                    source_reservation.close()
                    extent_spec = TensorSpec((2,), DType.I32)
                    if isinstance(recipe, EncodedImport):
                        valid = step.source.length // recipe.block_bytes
                        key = ("encoded", step.source_spec, valid)
                        program = self._programs[key]
                        extent = self.device.upload(extent_spec, struct.pack("=ii", valid, step.first_block))
                        retained.callback(extent.close)
                        execution = program.submit(source, extent, resources={"target": target})
                    else:
                        value = source
                        if isinstance(recipe, DenseImport) and (
                            recipe.source_dtype != binding.spec.dtype or recipe.transform != Transform.IDENTITY
                        ):
                            key = ("dense", step.source_spec, binding.spec.dtype, recipe)
                            program = self._programs[key]
                            converted = program.submit(source)
                            for converted_output in converted.outputs:
                                retained.callback(converted_output.close)
                            converted.completion.wait()
                            value = converted.outputs[0]
                        source_bytes = value.view(TensorSpec((value.spec.storage_nbytes,), DType.U8))
                        target_bytes = target.view(TensorSpec((binding.spec.storage_nbytes,), DType.U8))
                        retained.callback(source_bytes.close)
                        retained.callback(target_bytes.close)
                        key = ("publish", source_bytes.spec, target_bytes.spec)
                        program = self._programs[key]
                        extent = self.device.upload(TensorSpec((2,), DType.I64),
                                                    struct.pack("=qq", step.target_offset, source_bytes.spec.elements))
                        retained.callback(extent.close)
                        execution = program.submit(source_bytes, extent, resources={"target": target_bytes})
                    for value in execution.outputs:
                        retained.callback(value.close)
                    execution.completion.wait()
            return target
        except BaseException:
            target.close()
            raise

    def close(self):
        if not self._closed:
            # Shared conversion executables belong to DeviceRuntime, not to a
            # particular source binding or provider lifetime.
            self._programs.clear()
            self._closed = True
