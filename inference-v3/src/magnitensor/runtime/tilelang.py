"""TileLang realization of Magnitensor's physical runtime contract."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from threading import Lock
from typing import Any, cast

import torch

from ..compiler.lowering import Capabilities, MatrixInstruction
from ..compiler.unit import TileCompilationUnit
from ..representations import Dense
from ..tensor.types import DType, TensorSpec
from .resources import NativeAllocation, NativeBoundEntrypoint, NativeCompletion, NativeExecutable

_COMPILER_RECURSION_LOCK = Lock()


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
            shape, dtype = (spec.storage_nbytes,), DType.U8
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
        self._bound(*dynamic)
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
    """Magnitensor-owned storage and submission over a resolved TileLang backend."""

    def __init__(self, target: Any = "auto", *, ordinal: int = 0):
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("device ordinal must be a nonnegative integer")
        from tilelang.backend.module import create_backend_context

        self._context = create_backend_context(target, execution_backend="tvm_ffi")
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

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def compiler_identity(self) -> str:
        import tilelang

        context = self._context
        return f"tilelang-{tilelang.__version__}:{context.target}:{context.execution_backend.name}"

    def allocate(self, size: int, alignment: int) -> NativeAllocation:
        if size <= 0 or alignment <= 0:
            raise ValueError("allocation size and alignment must be positive")
        return _Allocation(torch.empty(size, dtype=torch.uint8, device=self._device))

    def upload(self, spec: TensorSpec, content: bytes) -> NativeAllocation:
        del spec
        host = torch.frombuffer(bytearray(content), dtype=torch.uint8)
        return _Allocation(host.to(self._device))

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
                _build_prim_func(unit),
                out_idx=[],
                execution_backend="tvm_ffi",
                target=context.target,
                target_host=context.target_host,
            )
        return _Executable(kernel, self._completion)

    def join(self, completions: tuple[NativeCompletion, ...]) -> NativeCompletion:
        return _JoinedCompletion(completions)

    def close(self) -> None:
        self._device = None


def _annotation(T, spec: TensorSpec):
    representation = spec.representation
    if representation is not None and not isinstance(representation, Dense):
        return T.Tensor((spec.storage_nbytes,), T.uint8)
    dtype = spec.dtype if not isinstance(representation, Dense) else representation.dtype
    return T.Tensor(spec.shape, dtype.value)


@contextmanager
def _compiler_recursion_budget(unit: TileCompilationUnit):
    """Give recursive TIR visitors enough stack for a maximal multi-kernel function."""
    required = 2_000 + 8 * sum(call.candidate.kernel_count for call in unit.calls)
    with _COMPILER_RECURSION_LOCK:
        previous = sys.getrecursionlimit()
        sys.setrecursionlimit(max(previous, required))
        try:
            yield
        finally:
            sys.setrecursionlimit(previous)


def _build_prim_func(unit: TileCompilationUnit):
    import tilelang.language as T

    parameters = tuple(
        (parameter.name, _annotation(T, parameter.spec)) for parameter in unit.parameters
    )

    def body(*bound) -> None:
        by_name = {
            parameter.name: value for parameter, value in zip(unit.parameters, bound, strict=True)
        }
        specs_by_name = {parameter.name: parameter.spec for parameter in unit.parameters}
        for call in unit.calls:
            operands = tuple(by_name[binding.parameter] for binding in call.bindings)
            operands += tuple(bound[index] for index in call.workspace)
            try:
                call.candidate.emitter(operands)
            except BaseException as error:
                bindings = ", ".join(
                    f"{binding.parameter}:{specs_by_name[binding.parameter]!r}"
                    for binding in call.bindings
                )
                error.add_note(f"while emitting {call.candidate.name} with {bindings}")
                raise

    return T.build_prim_func(unit.name, parameters, body)


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
