"""Torch is used only for storage, TileLang launch, transfer, and completion.

No tensor arithmetic is implemented by this adapter. Backend selection is an
explicit endpoint choice; this module does not claim to discover the machine.
Kernels reach the device through TileLang's public compilation entry and its
supported execution adapter for the selected target.

Operand binding is the per-invocation hot path: a decode step launches several
hundred commands. Static operands of a captured sequence are viewed once when
the sequence is bound; only changing operands are viewed per invocation, and a
view is one storage-level operation. Small host uploads stage through pinned
memory and copy asynchronously on the owning stream; the staging tensor is
retained until a recorded completion proves the copy finished.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import (
    DType,
    Executable,
    ExecutionOrder,
    InputReference,
    NativeBuffer,
    NativeCommand,
    TensorSpec,
)

if TYPE_CHECKING:
    import torch
    from tilelang.jit import PrimFunc

# Uploads up to this size stage through pinned memory and copy asynchronously;
# larger ones (weights) copy synchronously, where the stream round trip is
# negligible against the copy itself and no pinned duplicate is retained.
STAGED_UPLOAD_BYTES = 1 << 20


@dataclass(frozen=True)
class MetalSelection:
    """Discovered Metal device that the Torch MPS endpoint must correspond to.

    Torch exposes one MPS device and no public identity accessor. The discovered
    device's recommended working set is the public property both sides report,
    so it is the correspondence check an endpoint selection can truthfully make.
    """

    name: str
    working_set_bytes: int


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
    """Stream completion marker that also releases the uploads it proves complete."""

    def __init__(self, event, staging: tuple[torch.Tensor, ...]):
        self.event, self.staging = event, staging

    def ready(self) -> bool:
        done = self.event.query()
        if done:
            self.staging = ()
        return done

    def wait(self) -> None:
        self.event.synchronize()
        self.staging = ()


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


class _MpsInterval:
    def __init__(self, torch_module):
        self.start = torch_module.mps.Event(enable_timing=True)
        self.end = torch_module.mps.Event(enable_timing=True)
        self.start.record()
        self._seconds: float | None = None

    def finish(self) -> None:
        self.end.record()

    def seconds(self) -> float:
        # Torch's MPS elapsed_time consumes the events' completion notification;
        # a second query of the same pair blocks forever. A finished interval's
        # duration is immutable, so it is read from the events exactly once.
        seconds = self._seconds
        if seconds is None:
            seconds = self.start.elapsed_time(self.end) / 1000.0
            self._seconds = seconds
        return seconds


class _Views:
    """One-operation operand views: a typed prototype per dtype, then ``set_``."""

    def __init__(self, torch_module, device):
        self.prototypes = {
            dtype: torch_module.empty(0, dtype=getattr(torch_module, dtype.value), device=device)
            for dtype in DType
        }

    def view(self, buffer: NativeBuffer, spec: TensorSpec, offset: int) -> torch.Tensor:
        storage = _buffer(buffer).tensor().untyped_storage()
        return (
            self.prototypes[spec.dtype]
            .new_empty(0)
            .set_(storage, offset // spec.dtype.itemsize, spec.shape)
        )


class _Kernel:
    def __init__(self, kernel, views: _Views, launch_context):
        self.kernel, self.views, self.launch_context = kernel, views, launch_context
        self.launch = kernel.torch_function

    def bind(self, arguments: tuple[tuple[NativeBuffer, TensorSpec, int], ...]) -> NativeCommand:
        view = self.views.view
        bindings = tuple(view(buffer, spec, offset) for buffer, spec, offset in arguments)
        return _Command(self, bindings, arguments)


class _Command:
    def __init__(self, kernel: _Kernel, bindings, arguments):
        self.kernel, self.bindings, self.arguments = kernel, bindings, arguments

    def close(self) -> None:
        self.bindings = ()
        self.arguments = ()

    def launch(self) -> None:
        with self.kernel.launch_context():
            self.kernel.launch(*self.bindings)


class _SequenceKernel:
    """Captured commands with static operands viewed once.

    Captured resources keep static allocations alive for the capture and every
    invocation, so a template never outlives the storage its views reference.
    Changing operands are viewed per invocation from the invocation's arguments.
    """

    def __init__(self, kernels, commands, references, views: _Views):
        self.views = views
        # Per command: (kernel, static operands, changing operands).
        # A kernel command keeps static operands as views; a nested sequence
        # keeps them as (buffer, spec, offset) because its own template views
        # them. Changing operands are (position, reference, spec).
        plan = []
        for kernel, command, refs in zip(kernels, commands, references, strict=True):
            nested = not isinstance(kernel, _Kernel)
            static: dict[int, object] = {}
            changing: list[tuple[int, InputReference, TensorSpec]] = []
            for position, ((buffer, spec, offset), ref) in enumerate(
                zip(command.arguments, refs, strict=True)
            ):
                if ref is not None:
                    changing.append((position, ref, spec))
                elif nested:
                    static[position] = (buffer, spec, offset)
                else:
                    static[position] = views.view(buffer, spec, offset)
            plan.append((kernel, nested, static, tuple(changing), len(command.arguments)))
        self.plan = tuple(plan)

    def bind(self, arguments) -> NativeCommand:
        view = self.views.view
        cache: dict[tuple[int, int, TensorSpec], torch.Tensor] = {}
        commands = []
        for kernel, nested, static, changing, width in self.plan:
            bound: list[object] = [None] * width
            for position, operand in static.items():
                bound[position] = operand
            for position, ref, spec in changing:
                buffer, _, base = arguments[ref.index]
                offset = base + ref.offset
                if nested:
                    bound[position] = (buffer, spec, offset)
                    continue
                key = (id(buffer), offset, spec)
                tensor = cache.get(key)
                if tensor is None:
                    tensor = cache[key] = view(buffer, spec, offset)
                bound[position] = tensor
            if nested:
                # The inner template rebinds its own commands and returns an
                # invocation with a launch of its own.
                commands.append(kernel.bind(tuple(bound)))
            else:
                commands.append(_Command(kernel, tuple(bound), ()))
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
    def __init__(self, backend: Backend, ordinal: int = 0, *, metal: MetalSelection | None = None):
        import torch

        self.torch, self.backend = torch, backend
        self._subgroup_width: int | None = None
        self._staging: list[torch.Tensor] = []
        if ordinal < 0:
            raise ValueError("negative endpoint ordinal")
        if backend == Backend.METAL:
            if metal is None:
                raise ValueError("Metal execution requires the discovered device selection")
            if not torch.backends.mps.is_available():
                raise RuntimeError("metal runtime is unavailable")
            if ordinal != 0 or torch.mps.device_count() != 1:
                raise ValueError("Torch exposes exactly one MPS device, ordinal 0")
            if torch.mps.recommended_max_memory() != metal.working_set_bytes:
                raise RuntimeError(
                    f"Torch MPS device does not correspond to Metal device {metal.name!r}"
                )
            self.device = torch.device("mps")
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
        self.views = _Views(torch, self.device)
        if self.device.type == "cuda":
            device = self.device
            self.launch_context = lambda: torch.cuda.device(device)
        else:
            self.launch_context = nullcontext
        if backend == Backend.METAL:
            from magnitude_engine.numerics.capabilities import capability_probe

            self._subgroup_width = self._pipeline_width(self.compile(capability_probe()))

    @property
    def subgroup_width(self) -> int | None:
        return self._subgroup_width

    @staticmethod
    def _pipeline_width(executable: Executable) -> int:
        kernel = executable.native
        assert isinstance(kernel, _Kernel)
        return int(kernel.kernel.adapter.thread_execution_width)

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
        return _SequenceKernel(kernels, commands, references, self.views)

    def _copy_in(self, destination: torch.Tensor, content: bytes) -> None:
        """Copy host bytes into device storage on the owning stream.

        Small transfers stage through pinned memory and complete asynchronously
        in stream order; the staging tensor stays referenced until a recorded
        completion (or a drain) proves the copy finished. Large transfers and
        the host device copy synchronously.
        """
        host = self.torch.frombuffer(bytearray(content), dtype=self.torch.uint8)
        if self.device.type == "cpu" or len(content) > STAGED_UPLOAD_BYTES:
            destination.copy_(host, non_blocking=False)
            return
        staging = self.torch.empty(len(content), dtype=self.torch.uint8, pin_memory=True)
        staging.copy_(host)
        destination.copy_(staging, non_blocking=True)
        self._staging.append(staging)

    def upload(self, content: bytes) -> NativeBuffer:
        storage = self.torch.empty(len(content), dtype=self.torch.uint8, device=self.device)
        self._copy_in(storage, content)
        return _Buffer(storage)

    def read(self, buffer: NativeBuffer, offset: int, size: int) -> bytes:
        return _buffer(buffer).tensor()[offset : offset + size].cpu().numpy().tobytes()

    def write(self, buffer: NativeBuffer, offset: int, content: bytes) -> None:
        storage = _buffer(buffer).tensor()
        if offset < 0 or offset + len(content) > storage.numel():
            raise ValueError("upload exceeds native allocation")
        self._copy_in(storage[offset : offset + len(content)], content)

    def _take_staging(self) -> tuple[torch.Tensor, ...]:
        staging = tuple(self._staging)
        self._staging.clear()
        return staging

    def record(self, *, timing: bool = False):
        if timing:
            raise ValueError("use a profiling interval; completion events are not timing samples")
        if self.backend == Backend.LLVM:
            return _Immediate()
        if self.backend == Backend.METAL:
            event = self.torch.mps.Event()
            event.record()
            return _Event(event, self._take_staging())
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream(self.device))
        return _Event(event, self._take_staging())

    def interval(self):
        if self.backend == Backend.LLVM:
            return _Immediate()
        if self.backend == Backend.METAL:
            return _MpsInterval(self.torch)
        return _CudaInterval(self.torch, self.device)

    def drain(self) -> None:
        if self.backend == Backend.METAL:
            self.torch.mps.synchronize()
        elif self.backend != Backend.LLVM:
            self.torch.cuda.synchronize(self.device)
        self._staging.clear()

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
        with self.launch_context():
            kernel = tilelang.compile(
                program,
                target=self.backend.value,
                execution_backend="torch" if self.backend == Backend.METAL else "tvm_ffi",
            )
        executable = Executable(signature, _Kernel(kernel, self.views, self.launch_context), self)
        if self._subgroup_width is not None:
            if self._pipeline_width(executable) != self._subgroup_width:
                raise ValueError("compiled Metal execution width differs from the selected plan")
        return executable
