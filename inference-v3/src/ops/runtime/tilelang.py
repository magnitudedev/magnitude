"""TileLang realization of Ops's physical runtime contract."""

from __future__ import annotations

import sys
import hashlib
import json
import platform
from dataclasses import asdict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from threading import Lock
from typing import Any, cast

import torch

from ..compiler.lowering import Capabilities, MatrixInstruction
from ..compiler.unit import TileCompilationUnit
from ..representations import Dense
from ..tensor.types import DType, TensorSpec
from .resources import NativeAllocation, NativeBoundEntrypoint, NativeCompletion, NativeExecutable, NativeSubmissionError
from .observation import KernelActivity

_COMPILER_RECURSION_LOCK = Lock()


class _KernelCapture:
    def __init__(self, capture):
        self._capture = capture
        self.clock = capture.clock

    def start(self) -> None:
        self._capture.start()

    def finish(self) -> tuple[KernelActivity, ...]:
        return tuple(KernelActivity(item.name, item.elapsed_ns) for item in self._capture.finish())

    def close(self) -> None:
        self._capture.close()


class _Allocation(NativeAllocation):
    def __init__(self, tensor: torch.Tensor):
        self._tensor: torch.Tensor | None = tensor

    def _require_tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError("allocation is closed")
        return self._tensor

    @property
    def allocated_bytes(self) -> int:
        return self._require_tensor().untyped_storage().nbytes()

    def view(self, spec: TensorSpec, offset: int = 0) -> torch.Tensor:
        tensor = self._require_tensor()
        representation = spec.representation
        if representation is not None and not isinstance(representation, Dense):
            # Encoded compute kernels consume aligned packets, not individual
            # bytes.  Keep the canonical layout byte-addressed at the tensor
            # boundary while exposing its physical ABI as packed words.
            shape, dtype = ((spec.storage_nbytes + 3) // 4,), DType.U32
        else:
            shape = cast(tuple[int, ...], spec.shape)
            dtype = spec.dtype if not isinstance(representation, Dense) else representation.dtype
        torch_dtype = getattr(torch, dtype.value)
        width = torch.empty((), dtype=torch_dtype).element_size()
        if offset % width:
            raise ValueError("view offset is not aligned to its element type")
        count = 1
        for extent in shape:
            count *= extent
        return tensor[offset : offset + count * width].view(torch_dtype).view(shape)

    def close(self) -> None:
        self._tensor = None


class _Completion(NativeCompletion):
    def __init__(self, event=None, synchronize: Callable[[], None] | None = None):
        self._event = event
        self._synchronize = synchronize
        self._done = event is None and synchronize is None

    def ready(self) -> bool:
        if self._done:
            return True
        if self._event is None:
            return False
        self._done = bool(self._event.query())
        return self._done

    def wait(self) -> None:
        if self._done:
            return
        if self._event is not None:
            self._event.synchronize()
        elif self._synchronize is not None:
            self._synchronize()
        self._done = True


class _JoinedCompletion(NativeCompletion):
    def __init__(self, completions: tuple[NativeCompletion, ...]):
        self._completions = completions

    def ready(self) -> bool:
        return all(completion.ready() for completion in self._completions)

    def wait(self) -> None:
        for completion in self._completions:
            completion.wait()


class _BoundEntrypoint(NativeBoundEntrypoint):
    def __init__(self, bound, completion: Callable[[], NativeCompletion]):
        self._bound = bound
        self._completion = completion

    def submit(self, dynamic: tuple[Any, ...]) -> NativeCompletion:
        if self._bound is None:
            raise RuntimeError("bound entrypoint is closed")
        try:
            self._bound(*dynamic)
        except BaseException as error:
            # A composite entrypoint may already have encoded earlier kernels.
            # Its failure carries a real completion, not permission to free inputs.
            raise NativeSubmissionError(error, self._completion()) from error
        return self._completion()

    def close(self) -> None:
        self._bound = None


class _Executable(NativeExecutable):
    def __init__(self, kernel, completion: Callable[[], NativeCompletion]):
        self._kernel = kernel
        self._completion = completion

    def bind(
        self, static: Mapping[int, Any], dynamic_indices: tuple[int, ...]
    ) -> NativeBoundEntrypoint:
        if self._kernel is None:
            raise RuntimeError("executable is closed")
        return _BoundEntrypoint(self._kernel.bind(dict(static), dynamic_indices), self._completion)

    def close(self) -> None:
        self._kernel = None


class TileLangRuntime:
    """Ops-owned storage and submission over a resolved TileLang backend."""

    def __init__(self, target: Any = "auto", *, ordinal: int = 0):
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("device ordinal must be a nonnegative integer")
        from tilelang.backend.module import create_backend_context

        self._context = create_backend_context(target, execution_backend="tvm_ffi")
        self._ordinal = ordinal
        kind = self._context.target.kind.name
        if kind == "metal":
            if ordinal:
                raise ValueError("Metal exposes only process device ordinal zero")
            self._device = torch.device("mps")
            self._completion = lambda: _Completion(synchronize=torch.mps.synchronize)
        elif kind in ("cuda", "hip"):
            self._device = torch.device("cuda", ordinal)

            def completion() -> NativeCompletion:
                event = torch.cuda.Event()
                event.record()
                return _Completion(event=event)

            self._completion = completion
        else:
            if ordinal:
                raise ValueError("CPU execution exposes only process device ordinal zero")
            self._device = torch.device("cpu")
            self._completion = _Completion
        self._capabilities = _capabilities(self._context.capabilities)
        from tilelang.cache import compiler_identity

        self._compiler_identity = (f"{compiler_identity()}:{self._context.target}:"
                                   f"{self._context.execution_backend.name}")
        from tilelang.backend import runtime_info

        self._runtime_identity = hashlib.sha256(json.dumps({
            "native": asdict(runtime_info(self._context.target, ordinal=ordinal)),
            "os": (platform.system(), platform.release(), platform.version(), platform.machine()),
            "python": (platform.python_implementation(), platform.python_version()),
            "transfer_runtime": (torch.__version__, torch.version.git_version),
        }, sort_keys=True).encode()).hexdigest()

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def compiler_identity(self) -> str:
        return self._compiler_identity

    @property
    def runtime_identity(self) -> str:
        return self._runtime_identity

    def allocate(self, size: int, alignment: int) -> NativeAllocation:
        if size <= 0 or alignment <= 0:
            raise ValueError("allocation size and alignment must be positive")
        allocated = (size + alignment - 1) // alignment * alignment
        return _Allocation(torch.empty(allocated, dtype=torch.uint8, device=self._device))

    def upload(self, spec: TensorSpec, content: bytes) -> NativeAllocation:
        representation = spec.representation
        size = len(content)
        if representation is not None and not isinstance(representation, Dense):
            size = (size + 3) // 4 * 4
        staging = bytearray(size)
        staging[:len(content)] = content
        host = torch.frombuffer(staging, dtype=torch.uint8)
        # Keep the staged allocation distinct even for a CPU endpoint: the owner
        # has separate transient-host and retained-execution reservations.
        return _Allocation(host.to(self._device, copy=True))

    def download(self, value: torch.Tensor) -> bytes:
        # Host transfer is an ABI operation. Numerical work remains in the
        # compiled TileLang program.
        return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()

    def compile(self, program: object, signature: tuple[TensorSpec, ...]) -> NativeExecutable:
        import tilelang

        del signature
        unit = cast(TileCompilationUnit, program)
        context = self._context
        with _compiler_recursion_budget(unit):
            kernel = tilelang.compile(
                _build_reusable_module(unit),
                out_idx=[],
                execution_backend="tvm_ffi",
                target=context.target,
                target_host=context.target_host,
            )
        return _Executable(kernel, self._completion)

    def join(self, completions: tuple[NativeCompletion, ...]) -> NativeCompletion:
        return _JoinedCompletion(completions)

    def capture_kernels(self, limit: int):
        from tilelang.backend import kernel_capture

        capture = kernel_capture(self._context.target, ordinal=self._ordinal, max_kernels=limit)
        return None if capture is None else _KernelCapture(capture)

    def close(self) -> None:
        self._device = None


def _annotation(T, spec: TensorSpec):
    from ..compiler.program import annotation

    return annotation(T, spec)


@contextmanager
def _compiler_recursion_budget(unit: TileCompilationUnit):
    """Give recursive TIR visitors enough stack for a maximal multi-kernel function."""
    required = 2_000 + 8 * sum(call.operation.kernel_count for call in unit.calls)
    with _COMPILER_RECURSION_LOCK:
        previous = sys.getrecursionlimit()
        sys.setrecursionlimit(max(previous, required))
        try:
            yield
        finally:
            sys.setrecursionlimit(previous)


def _build_reusable_module(unit: TileCompilationUnit):
    """Factor repeated layer schedules into reusable functions in one native module."""
    import tilelang.language as T

    parameter_by_name = {parameter.name: parameter for parameter in unit.parameters}
    templates = {}
    calls = []
    definitions = []
    for call in unit.calls:
        operand_parameters = tuple(
            parameter_by_name[binding.parameter] for binding in call.bindings
        ) + tuple(unit.parameters[index] for index in call.workspace)
        operand_specs = tuple(parameter.spec for parameter in operand_parameters)
        definition = call.operation.definition
        if definition is None:
            raise ValueError("materialization requires the operation's executable definition")
        if len(operand_specs) != len(definition.ports) or any(
            actual.storage_nbytes < port.spec.storage_nbytes
            for actual, port in zip(operand_specs, definition.ports, strict=True)
        ):
            raise ValueError("materialization backing does not contain the declared operation ports")
        key = definition.identity
        template = templates.get(key)
        if template is None:
            definitions.append(definition.private_definition())
            template = definition.name
            templates[key] = template
        calls.append((template, operand_parameters, definition.ports))

    entry_parameters = tuple(
        (parameter.name, _annotation(T, parameter.spec)) for parameter in unit.parameters
    )

    def entry_body(private, *bound) -> None:
        by_name = {
            parameter.name: value for parameter, value in zip(unit.parameters, bound, strict=True)
        }
        for schedule, operands, ports in calls:
            arguments = []
            for parameter, port in zip(operands, ports, strict=True):
                value = by_name[parameter.name]
                if parameter.spec != port.spec:
                    representation = port.spec.representation
                    if representation is not None and not isinstance(representation, Dense):
                        shape, dtype = ((port.spec.storage_nbytes + 3) // 4,), "uint32"
                    else:
                        shape = port.spec.shape
                        dtype = (representation.dtype if isinstance(representation, Dense) else port.spec.dtype).value
                    value = T.view(value, shape=shape, dtype=dtype)
                arguments.append(value)
            private[schedule](*arguments)

    return T.build_prim_module("main", entry_parameters, entry_body, definitions)


def _capabilities(value) -> Capabilities:
    shared = value.shared_memory_bytes > 0
    atomics = frozenset(dtype for dtype in DType if value.supports(f"atomic.add.{dtype.value}"))
    return Capabilities(
        subgroup_width=value.subgroup_width,
        threads_per_group=value.max_threads_per_group,
        shared_memory_bytes=value.shared_memory_bytes,
        matrix_instructions=tuple(
            MatrixInstruction(
                item.m, item.n, item.k, DType(item.input_dtype), DType(item.accumulation_dtype)
            )
            for item in value.matrix_instructions
        ),
        supported_dtypes=frozenset(DType(item) for item in value.supported_dtypes),
        memory_scopes=(
            frozenset({"global", "shared", "local"}) if shared else frozenset({"global", "local"})
        ),
        barrier_scopes=frozenset({"workgroup"}) if shared else frozenset(),
        asynchronous_copy=value.supports("async_copy"),
        subgroup_exchange=value.supports("subgroup_exchange"),
        vector_bytes=(1, 2, 4, 8, 16),
        atomics=atomics,
        features=frozenset(value.features),
        alignments={dtype: dtype.itemsize for dtype in DType},
        native_multi_launch=value.native_multi_launch,
        partial_binding=value.native_argument_binding,
        max_kernels_per_program=value.max_kernels_per_program,
        fingerprint=value.fingerprint,
    )


def describe_configuration(configuration):
    """Inspect the selected endpoint's compiler contract without opening a live owner."""
    from tilelang.backend.module import create_backend_context
    from tilelang.cache import compiler_identity

    if len(configuration.selected_endpoints) != 1:
        raise ValueError("distributed operation planning is not yet implemented")
    endpoint = configuration.selected_endpoints[0]
    context = create_backend_context(str(endpoint.backend), execution_backend="tvm_ffi")
    return (_capabilities(context.capabilities),
            f"{compiler_identity()}:{context.target}:{context.execution_backend.name}")
