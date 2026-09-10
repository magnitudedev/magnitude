"""Torch is used only for storage, TileLang launch, transfer, and completion.

No tensor arithmetic is implemented by this adapter. Backend selection is an
explicit endpoint choice; this module does not claim to discover the machine.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import (
    DType,
    Executable,
    ExecutionOrder,
    NativeBuffer,
    NativeCommand,
    TensorSpec,
)

if TYPE_CHECKING:
    import torch
    from tilelang.jit import PrimFunc


class _Buffer:
    def __init__(self, storage: torch.Tensor):
        self.storage: torch.Tensor | None = storage

    @property
    def allocated_bytes(self) -> int:
        return self.tensor().untyped_storage().nbytes()

    def close(self) -> None:
        self.storage = None

    def tensor(self) -> torch.Tensor:
        if self.storage is None:
            raise RuntimeError("native buffer has been released")
        return self.storage


def _buffer(buffer: NativeBuffer) -> _Buffer:
    if not isinstance(buffer, _Buffer):
        raise TypeError("buffer belongs to another runtime")
    return buffer


class _Immediate:
    def ready(self) -> bool:
        return True

    def wait(self) -> None:
        pass

    def finish(self) -> None:
        pass

    def seconds(self) -> None:
        return None


class _Event:
    def __init__(self, event):
        self.event = event

    def ready(self) -> bool:
        return self.event.query()

    def wait(self) -> None:
        self.event.synchronize()


class _CudaInterval:
    def __init__(self, torch_module, device):
        self.torch, self.device = torch_module, device
        with self.torch.cuda.device(device):
            self.start = self.torch.cuda.Event(enable_timing=True)
            self.end = self.torch.cuda.Event(enable_timing=True)
            self.start.record(self.torch.cuda.current_stream(device))

    def finish(self) -> None:
        self.end.record(self.torch.cuda.current_stream(self.device))

    def seconds(self) -> float:
        return self.start.elapsed_time(self.end) / 1000.0


class _Kernel:
    def __init__(self, kernel, torch_module, device):
        self.kernel, self.torch, self.device = kernel, torch_module, device

    def bind(self, arguments: tuple[tuple[NativeBuffer, TensorSpec, int], ...]) -> NativeCommand:
        bindings = []
        for buffer, spec, offset in arguments:
            storage = _buffer(buffer).tensor()
            dtype = getattr(self.torch, spec.dtype.value)
            bindings.append(storage[offset : offset + spec.nbytes].view(dtype).reshape(spec.shape))
        return _Command(self, tuple(bindings), arguments)


class _Command:
    def __init__(self, kernel: _Kernel, bindings, arguments):
        self.kernel, self.bindings, self.arguments = kernel, bindings, arguments

    def close(self) -> None:
        self.bindings = ()
        self.arguments = ()

    def launch(self) -> None:
        kernel = self.kernel
        with (
            kernel.torch.cuda.device(kernel.device)
            if kernel.device.type == "cuda"
            else nullcontext()
        ):
            kernel.kernel(*self.bindings)


class _SequenceKernel:
    def __init__(self, kernels, commands, references):
        self.kernels, self.references = kernels, references
        # Templates retain native buffer wrappers, never Torch views. A wrapper's
        # close releases storage even if a consumed Prepared still refers to the
        # template's code. Dynamic wrappers are not retained at all.
        self.arguments = tuple(
            tuple(
                (None if ref is not None else buffer, spec, offset)
                for (buffer, spec, offset), ref in zip(command.arguments, refs, strict=True)
            )
            for command, refs in zip(commands, references, strict=True)
        )

    def bind(self, arguments) -> NativeCommand:
        commands = []
        try:
            for kernel, operands, refs in zip(
                self.kernels, self.arguments, self.references, strict=True
            ):
                bound = []
                for (buffer, spec, offset), ref in zip(operands, refs, strict=True):
                    if ref is not None:
                        buffer, _, base = arguments[ref.index]
                        offset = base + ref.offset
                    bound.append((buffer, spec, offset))
                commands.append(kernel.bind(tuple(bound)))
        except BaseException:
            for command in commands:
                command.close()
            raise
        return _SequenceCommand(_LaunchSequence(tuple(commands)), arguments)


class _LaunchSequence:
    def __init__(self, commands):
        self.commands = commands

    def launch(self) -> None:
        for command in self.commands:
            command.launch()

    def close(self) -> None:
        for command in self.commands:
            command.close()
        self.commands = ()


class _SequenceCommand:
    def __init__(self, sequence, arguments=()):
        self.sequence, self.arguments = sequence, arguments

    def launch(self) -> None:
        if self.sequence is None:
            raise RuntimeError("physical sequence invocation is closed")
        self.sequence.launch()

    def close(self) -> None:
        if self.sequence is not None:
            self.sequence.close()
            self.sequence = None
            self.arguments = ()


class TorchDriver:
    def __init__(self, backend: Backend, ordinal: int = 0):
        import torch

        self.torch, self.backend = torch, backend
        if ordinal < 0:
            raise ValueError("negative endpoint ordinal")
        if backend == Backend.METAL:
            raise ValueError("Metal uses its native platform driver")
        elif backend in (Backend.CUDA, Backend.HIP):
            runtime_is_hip = torch.version.hip is not None
            if runtime_is_hip != (backend == Backend.HIP) or not torch.cuda.is_available():
                raise RuntimeError(f"{backend.value} runtime is unavailable")
            if ordinal >= torch.cuda.device_count():
                raise ValueError("endpoint ordinal exceeds runtime inventory")
            self.device = torch.device("cuda", ordinal)
        else:
            if ordinal != 0:
                raise ValueError("LLVM execution uses the process CPU domain")
            self.device = torch.device("cpu")

    @property
    def subgroup_width(self) -> int | None:
        return None

    def allocate(self, size: int) -> NativeBuffer:
        return _Buffer(self.torch.empty(size, dtype=self.torch.uint8, device=self.device))

    def dispatch(self, commands: tuple[NativeCommand, ...]) -> None:
        if any(not isinstance(command, (_Command, _SequenceCommand)) for command in commands):
            raise TypeError("commands belong to another runtime")
        for command in commands:
            assert isinstance(command, (_Command, _SequenceCommand))
            command.launch()

    def bind_sequence(self, kernels, commands, references, order: ExecutionOrder):
        if any(not isinstance(command, (_Command, _SequenceCommand)) for command in commands):
            raise TypeError("physical sequence contains a foreign command")
        # This adapter currently realizes both orders serially on its owning
        # stream. Independent regions retain their join/completion semantics.
        return _SequenceKernel(kernels, commands, references)

    def upload(self, content: bytes) -> NativeBuffer:
        # Blocking copy establishes upload completion before the host staging
        # bytes are released. Asynchronous loading will use explicit tickets.
        host = self.torch.frombuffer(bytearray(content), dtype=self.torch.uint8)
        return _Buffer(host.to(self.device, copy=True, non_blocking=False))

    def read(self, buffer: NativeBuffer, offset: int, size: int) -> bytes:
        return _buffer(buffer).tensor()[offset : offset + size].cpu().numpy().tobytes()

    def write(self, buffer: NativeBuffer, offset: int, content: bytes) -> None:
        storage = _buffer(buffer).tensor()
        if offset < 0 or offset + len(content) > storage.numel():
            raise ValueError("upload exceeds native allocation")
        host = self.torch.frombuffer(bytearray(content), dtype=self.torch.uint8)
        storage[offset : offset + len(content)].copy_(host, non_blocking=False)

    def record(self, *, timing: bool = False):
        if timing:
            raise ValueError("use a profiling interval; completion events are not timing samples")
        if self.backend == Backend.LLVM:
            return _Immediate()
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(self.device))
        return _Event(event)

    def interval(self):
        if self.backend == Backend.LLVM:
            return _Immediate()
        return _CudaInterval(self.torch, self.device)

    def drain(self) -> None:
        if self.backend != Backend.LLVM:
            self.torch.cuda.synchronize(self.device)

    def compile(self, program: PrimFunc) -> Executable:
        import tilelang

        signature = tuple(
            TensorSpec(
                tuple(int(n) for n in program.buffer_map[p].shape),
                DType(str(program.buffer_map[p].dtype)),
            )
            for p in program.params
        )
        if self.backend == Backend.LLVM:
            # TileLang's host codegen legalizes BF16 storage to uint16. Lower
            # arithmetic conversions first, while the BF16 type is still known;
            # otherwise LLVM treats the stored bit pattern as an integer value.
            module = tilelang.tvm.IRModule({"main": program})
            # This TVM revision does not recursively legalize Bind values.
            # Inline pure local bindings before visiting their arithmetic.
            module = tilelang.transform.LetInline()(module)
            program = tilelang.tvm.tirx.transform.BF16ComputeLegalize()(module)["main"]
        with self.torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext():
            kernel = tilelang.compile(
                program,
                target=self.backend.value,
                execution_backend="tvm_ffi",
            )
        return Executable(signature, _Kernel(kernel, self.torch, self.device), self)
