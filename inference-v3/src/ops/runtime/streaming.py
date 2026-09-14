"""Execute selected source loops with bounded, completion-retained tile storage."""

from __future__ import annotations

import struct

from ..compiler.lowering import KernelBinding, BoundOperation
from ..compiler.unit import KernelCall, ParameterKind, TileCompilationUnit, UnitParameter
from ..tensor.types import DType, TensorSpec
from .resources import Completion, NativeSubmissionError


def _submit_stage(device, entrypoint, resources):
    try:
        return device.submit_native(entrypoint, tuple(value.native for value in resources))
    except NativeSubmissionError as error:
        # Local tiles can leave scope before the enclosing program's failure
        # handler runs. Pin them here, even if draining itself fails.
        Completion(device, error.completion, tuple(value.fork() for value in resources)).wait()
        raise


def _prepare_kernel(device, definition, output):
    shared = device._source_kernels.get(definition.identity)
    if shared is None:
        parameters = tuple(UnitParameter(index, f"p{index}", ("value", index, 0), port.spec,
                                         ParameterKind.DYNAMIC)
                           for index, port in enumerate(definition.ports))
        operation = BoundOperation(definition.name, frozenset({0}),
                                   tuple(index for index in range(len(parameters)) if index != output),
                                   (output,), lambda operands: None, definition=definition, kernel_count=1)
        bindings = tuple(KernelBinding(index, parameter.name, "write" if index == output else "read")
                         for index, parameter in enumerate(parameters))
        unit = TileCompilationUnit(definition.name, definition.identity, parameters,
                                   (KernelCall(operation, bindings, ()),), (output,))
        executable = device.compile_native(unit, unit.signature)
        try:
            entrypoint = executable.bind({}, tuple(range(len(parameters))))
        except BaseException:
            executable.close()
            raise
        shared = executable, entrypoint
        device._source_kernels[definition.identity] = shared
    return shared[1]


class CompiledGatherLoop:
    programs = ()
    def __init__(self, device, loop):
        self.device, self.loop = device, loop
        self._closed = False
        self._entrypoint = _prepare_kernel(device, loop.templates[0].definition, 2)
        self._imports = [(0, loop.prototype, device.prepare_binding(loop.prototype))]

    def submit(self, values, *, sources=None):
        from contextlib import ExitStack
        from .imports import plan_import

        self.device.check()
        if self._closed:
            raise RuntimeError("source gather is closed")
        indices, output = values[self.loop.hidden], values[self.loop.output]
        formats = {DType.I8: "b", DType.U8: "B", DType.I16: "h", DType.U16: "H",
                   DType.I32: "i", DType.U32: "I", DType.I64: "q"}
        reader = struct.Struct("=" + formats[indices.spec.dtype])
        map_spec = TensorSpec((self.loop.step, 2), DType.I64)
        completions = []
        with ExitStack() as host:
            readback = self.device.memory.reserve(2 * indices.spec.storage_nbytes, self.device.host_domains)
            host.callback(readback.close)
            content = self.device.read(indices)
            mapping_storage = self.device.memory.reserve(map_spec.storage_nbytes, self.device.host_domains)
            host.callback(mapping_storage.close)
            destinations = bytearray(map_spec.storage_nbytes)
            for first in range(0, self.loop.tokens, self.loop.step):
                count = min(self.loop.step, self.loop.tokens - first)
                rows = tuple(reader.unpack_from(content, (first + index) * reader.size)[0]
                             for index in range(count))
                source = (sources or {}).get(self.loop.weight, self.loop.source)
                binding, locations = self.loop.gather(rows, source=source)
                for index in range(self.loop.step):
                    struct.pack_into("=qq", destinations, index * 16,
                                     locations[index] if index < count else 0,
                                     first + index if index < count else -1)
                weight = self._imports[0][2].load(plan=plan_import(binding))
                mapping = completion = None
                try:
                    mapping = self.device.upload(map_spec, destinations)
                    native = _submit_stage(self.device, self._entrypoint, (weight, mapping, output))
                    completions.append(native)
                    completion = Completion(self.device, native, (weight, mapping))
                    completion.wait()
                finally:
                    if completion is None:
                        if mapping is not None:
                            mapping.close()
                        weight.close()
        return self.device.join(tuple(completions))

    def close(self):
        self._imports.clear()
        self._entrypoint = None
        self._closed = True


class CompiledExpertLoop:
    def __init__(self, device, loop):
        from ..compiler.compilation import materialize

        self.device, self.loop = device, loop
        self._closed = False
        self._imports = []
        self._stages = tuple(_prepare_kernel(device, template.definition, 2) for template in loop.templates)
        self._inner = materialize(loop.inner_plan, device=device)
        try:
            self._ports = tuple(next(identity for identity in self._inner.graph.constants
                                     if self._inner.graph.value(identity).name == name)
                                for name in ("gate", "up", "down"))
            self.programs = (self._inner,)
        except BaseException:
            self._inner.close()
            raise

    def _stage(self, index, resources):
        native = _submit_stage(self.device, self._stages[index], resources)
        completion = Completion(self.device, native, tuple(value.fork() for value in resources))
        completion.wait()
        return native

    def submit(self, values, *, sources=None):
        from contextlib import ExitStack

        self.device.check()
        if self._closed:
            raise RuntimeError("streamed expert operation is closed")
        hidden, routes, scores, output = (values[identity] for identity in
                                          (self.loop.hidden, self.loop.routes, self.loop.scores, self.loop.output))
        formats = {DType.I8: "b", DType.U8: "B", DType.I16: "h", DType.U16: "H",
                   DType.I32: "i", DType.U32: "I", DType.I64: "q"}
        reader = struct.Struct("=" + formats[routes.spec.dtype])
        with ExitStack() as retained:
            reservation = self.device.memory.reserve(2 * routes.spec.storage_nbytes, self.device.host_domains)
            retained.callback(reservation.close)
            content = self.device.read(routes)
            groups = {}
            for index in range(routes.spec.elements):
                expert = reader.unpack_from(content, index * reader.size)[0]
                if not 0 <= expert < self.loop.experts:
                    raise ValueError("expert route is outside the source bank")
                groups.setdefault(expert, []).append(index)
            del content
            reservation.close()
            map_storage = self.device.memory.reserve(self.loop.route_map.storage_nbytes, self.device.host_domains)
            retained.callback(map_storage.close)
            mapping_bytes = bytearray(self.loop.route_map.storage_nbytes)
            gathered = self.device.allocate(self.loop.gathered)
            retained.callback(gathered.close)
            contributions = self.device.allocate(self.loop.contributions)
            retained.callback(contributions.close)
            banks = tuple((sources or {}).get(identity, binding)
                          for identity, binding in zip(self.loop.source_values, self.loop.sources, strict=True))
            for expert, assignments in sorted(groups.items()):
                regions = {port: self.loop.region(binding, expert)
                           for port, binding in zip(self._ports, banks, strict=True)}
                for first in range(0, len(assignments), self.loop.step):
                    chunk = assignments[first:first + self.loop.step]
                    for index in range(self.loop.step):
                        struct.pack_into("=i", mapping_bytes, index * 4, chunk[index] if index < len(chunk) else -1)
                    mapping = self.device.upload(self.loop.route_map, mapping_bytes)
                    execution = None
                    try:
                        self._stage(0, (hidden, mapping, gathered))
                        execution = self._inner.submit(gathered, sources=regions)
                        execution.completion.wait()
                        projected, = execution.outputs
                        self._stage(1, (projected, mapping, contributions))
                    finally:
                        if execution is not None:
                            for value in execution.outputs:
                                value.close()
                        mapping.close()
            native = self._stage(2, (contributions, scores, output))
        return self.device.join((native,))

    def close(self):
        if not self._closed:
            self._inner.close()
            self.programs = ()
            self._stages = ()
            self._closed = True


def compile_source_loop(device, loop):
    from ..compiler.streaming import ExpertLoop, GatherLoop, ProjectionLoop

    if isinstance(loop, GatherLoop):
        return CompiledGatherLoop(device, loop)
    if isinstance(loop, ProjectionLoop):
        return CompiledProjectionLoop(device, loop)
    if isinstance(loop, ExpertLoop):
        return CompiledExpertLoop(device, loop)
    raise TypeError(f"unsupported authored source loop: {type(loop).__name__}")


class CompiledProjectionLoop:
    programs = ()
    def __init__(self, device, loop):
        self.device, self.loop = device, loop
        self._closed = False
        self._programs = {}
        self._imports = []
        for template in loop.templates:
            self._programs[template.extent] = _prepare_kernel(device, template.definition, 3)
        # All source metadata and conversions are prepared before invocation.
        for first, region in loop.regions():
            self._imports.append((first, region, device.prepare_binding(region)))

    def submit(self, values, *, sources=None):
        from .imports import plan_import

        self.device.check()
        if self._closed:
            raise RuntimeError("source loop is closed")
        hidden, output = values[self.loop.hidden], values[self.loop.output]
        completions = []
        source = (sources or {}).get(self.loop.weight, self.loop.source)
        for first, binding, importer in self._imports:
            region = source.region(first * self.loop.width, binding.spec.elements, shape=binding.spec.shape)
            weight = importer.load(plan=plan_import(region))
            extent = None
            completion = None
            try:
                extent = self.device.upload(TensorSpec((1,), DType.I32), struct.pack("=i", first))
                arguments = (hidden, weight, extent, output)
                if self.loop.bias is not None:
                    arguments += (values[self.loop.bias],)
                entrypoint = self._programs[binding.spec.shape[0]]
                native = _submit_stage(self.device, entrypoint, arguments)
                completions.append(native)
                completion = Completion(self.device, native, (weight, extent))
                completion.wait()
            finally:
                # Once submitted, Completion owns both tile resources even when
                # a native wait fails. Drain must prove completion before release.
                if completion is None:
                    if extent is not None:
                        extent.close()
                    weight.close()
        return self.device.join(tuple(completions))

    def close(self):
        self._imports.clear()
        self._programs.clear()
        self._closed = True
