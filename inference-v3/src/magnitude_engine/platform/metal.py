"""Native Metal storage and batched encoding of TileLang-generated kernels."""

from __future__ import annotations

import math
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import (
    Executable,
    ExecutionOrder,
    NativeBuffer,
    NativeCommand,
    TensorSpec,
)
from magnitude_engine.platform.measurement import clock_ns

if TYPE_CHECKING:
    from tilelang.jit import PrimFunc


@cache
def runtime() -> ModuleType:
    from torch.utils.cpp_extension import load

    module = load(
        name="magnitude_metal_runtime",
        sources=[str(Path(__file__).with_name("metal_runtime.mm"))],
        extra_cflags=["-O2"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=False,
    )
    if not isinstance(module, ModuleType):
        raise RuntimeError("native Metal runtime did not load")
    return module


class _Buffer:
    def __init__(self, native):
        self.native = native

    @property
    def allocated_bytes(self) -> int:
        return int(self.native.allocated_bytes)

    def close(self) -> None:
        self.native.close()


class _Command:
    def __init__(self, native):
        self.native = native

    def close(self) -> None:
        self.native = None


class _Kernel:
    def __init__(
        self,
        native,
        permutation: tuple[int, ...],
        groups: tuple[int, ...],
        threads: tuple[int, ...],
    ):
        self.native, self.permutation = native, permutation
        self.groups, self.threads = groups, threads

    def bind(self, arguments: tuple[tuple[NativeBuffer, TensorSpec, int], ...]) -> NativeCommand:
        buffers, offsets = [], []
        for index in self.permutation:
            buffer, _, offset = arguments[index]
            if not isinstance(buffer, _Buffer):
                raise TypeError("operand belongs to another native runtime")
            buffers.append(buffer.native)
            offsets.append(offset)
        return _Command(runtime().Command(self.native, buffers, offsets, self.groups, self.threads))


class _SequenceKernel:
    def __init__(self, kernels, commands, references, order: ExecutionOrder):
        self.kernels = kernels
        bindings = tuple(
            tuple(refs[index] for index in kernel.permutation)
            if isinstance(kernel, _Kernel)
            else refs
            for kernel, refs in zip(kernels, references, strict=True)
        )
        self.native = runtime().Sequence(
            [command.native for command in commands],
            [[-1 if ref is None else ref.index for ref in refs] for refs in bindings],
            [[0 if ref is None else ref.offset for ref in refs] for refs in bindings],
            order == ExecutionOrder.INDEPENDENT,
        )

    def bind(self, arguments) -> NativeCommand:
        buffers, offsets = [], []
        for buffer, _, offset in arguments:
            if not isinstance(buffer, _Buffer):
                raise TypeError("operand belongs to another native runtime")
            buffers.append(buffer.native)
            offsets.append(offset)
        return _Command(self.native.bind(buffers, offsets))


class MetalDriver:
    backend = Backend.METAL

    def __init__(self, registry_id: int = 0):
        from magnitude_engine.platform.metal_compiler import MetalCompiler

        self.native = runtime().Device(registry_id)
        self.compiler = MetalCompiler()
        probe = self.native.compile("kernel void capabilities() {}", "capabilities")
        self.subgroup_width = int(probe.subgroup_width)

    def allocate(self, size: int) -> NativeBuffer:
        return _Buffer(self.native.allocate(size))

    def upload(self, content: bytes) -> NativeBuffer:
        buffer = self.native.allocate(len(content))
        try:
            buffer.write(content, 0)
            return _Buffer(buffer)
        except BaseException:
            buffer.close()
            raise

    def read(self, buffer: NativeBuffer, offset: int, size: int) -> bytes:
        if not isinstance(buffer, _Buffer):
            raise TypeError("readback belongs to another native runtime")
        return buffer.native.read(offset, size)

    def write(self, buffer: NativeBuffer, offset: int, content: bytes) -> None:
        if not isinstance(buffer, _Buffer):
            raise TypeError("upload belongs to another native runtime")
        buffer.native.write(content, offset)

    def dispatch(self, commands: tuple[NativeCommand, ...]) -> None:
        bindings = []
        for command in commands:
            if not isinstance(command, _Command) or command.native is None:
                raise TypeError("invalid native Metal command")
            bindings.append(command.native)
        self.native.dispatch(bindings)

    def bind_sequence(self, kernels, commands, references, order: ExecutionOrder):
        if any(not isinstance(command, _Command) or command.native is None for command in commands):
            raise TypeError("physical sequence contains a foreign or closed command")
        return _SequenceKernel(kernels, commands, references, order)

    def record(self, *, timing: bool = False):
        if timing:
            raise ValueError("use an interval for profiling")
        return self.native.record()

    def interval(self):
        return self.native.interval()

    def drain(self) -> None:
        self.native.drain()

    def compile(self, program: PrimFunc) -> Executable:
        artifact = self.compiler.compile(program)
        started = clock_ns()
        native = self.native.compile(artifact.source, artifact.entry)
        self.compiler.realized(clock_ns() - started)
        if native.subgroup_width != self.subgroup_width:
            raise ValueError("compiled Metal execution width differs from the selected plan")
        if math.prod(artifact.threads) > native.max_threads:
            raise ValueError("shader launch exceeds the realized device pipeline's thread limit")
        return Executable(
            artifact.signature,
            _Kernel(native, artifact.permutation, artifact.groups, artifact.threads),
            self,
        )
