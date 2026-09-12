"""TileLang-backed realization of Magnitensor's runtime contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ..compiler.lowering import Capabilities, MatrixInstruction
from ..compiler.unit import TileCompilationUnit
from ..representations import Dense
from ..tensor.types import DType, TensorSpec
from .resources import NativeAllocation, NativeBoundEntrypoint, NativeCompletion, NativeExecutable


class _Allocation(NativeAllocation):
    def __init__(self, native):
        self._native = native

    @property
    def allocated_bytes(self) -> int:
        return self._native.allocated_bytes

    def view(self, spec: TensorSpec, offset: int = 0) -> Any:
        representation = spec.representation
        if representation is not None and not isinstance(representation, Dense):
            return self._native.view((spec.storage_nbytes,), DType.U8.value, offset)
        dtype = spec.dtype if not isinstance(representation, Dense) else representation.dtype
        return self._native.view(spec.shape, dtype.value, offset)

    def close(self) -> None:
        self._native.close()


class _Completion(NativeCompletion):
    def __init__(self, completions: tuple[Any, ...]):
        self._completions = completions

    def ready(self) -> bool:
        return all(completion.ready() for completion in self._completions)

    def wait(self) -> None:
        for completion in self._completions:
            completion.wait()


class _BoundEntrypoint(NativeBoundEntrypoint):
    def __init__(self, native):
        self._native = native

    def submit(self, dynamic: tuple[Any, ...]) -> NativeCompletion:
        return _Completion((self._native.submit(dynamic),))

    def close(self) -> None:
        self._native.close()


class _Executable(NativeExecutable):
    def __init__(self, native):
        self._native = native

    def bind(
        self, static: Mapping[int, Any], dynamic_indices: tuple[int, ...]
    ) -> NativeBoundEntrypoint:
        return _BoundEntrypoint(self._native.bind(static, dynamic_indices))

    def close(self) -> None:
        self._native.close()


class TileLangRuntime:
    """Opaque TileLang endpoint implementing Magnitensor's physical contract."""

    def __init__(self, target: Any = "auto"):
        from tilelang.runtime import open_device

        self._device: Any = open_device(target)
        self._capabilities = _capabilities(self._device.capabilities)

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    @property
    def compiler_identity(self) -> str:
        return self._device.compiler_identity

    def allocate(self, size: int, alignment: int) -> NativeAllocation:
        return _Allocation(self._device.allocate(size, alignment))

    def upload(self, spec: TensorSpec, content: bytes) -> NativeAllocation:
        del spec
        return _Allocation(self._device.upload(content))

    def compile(self, program: object, signature: tuple[TensorSpec, ...]) -> NativeExecutable:
        del signature
        unit = cast(TileCompilationUnit, program)
        return _Executable(self._device.compile(_build_prim_func(unit)))

    def join(self, completions: tuple[NativeCompletion, ...]) -> NativeCompletion:
        flattened = []
        for completion in completions:
            if isinstance(completion, _Completion):
                flattened.extend(completion._completions)
            else:
                flattened.append(completion)
        return _Completion(tuple(flattened))

    def close(self) -> None:
        self._device.close()


def _annotation(T, spec: TensorSpec):
    representation = spec.representation
    if representation is not None and not isinstance(representation, Dense):
        return T.Tensor((spec.storage_nbytes,), T.uint8)
    dtype = spec.dtype if not isinstance(representation, Dense) else representation.dtype
    return T.Tensor(spec.shape, dtype.value)


def _build_prim_func(unit: TileCompilationUnit):
    import tilelang.language as T

    parameters = tuple(
        (parameter.name, _annotation(T, parameter.spec)) for parameter in unit.parameters
    )

    def body(*bound) -> None:
        by_name = {
            parameter.name: value for parameter, value in zip(unit.parameters, bound, strict=True)
        }
        for call in unit.calls:
            operands = tuple(by_name[binding.parameter] for binding in call.bindings)
            operands += tuple(bound[index] for index in call.workspace)
            call.candidate.emitter(operands)

    return T.build_prim_func(unit.name, parameters, body)


def _capabilities(value) -> Capabilities:
    return Capabilities(
        subgroup_width=value.subgroup_width,
        threads_per_group=value.threads_per_group,
        shared_memory_bytes=value.shared_memory_bytes,
        matrix_instructions=tuple(
            MatrixInstruction(
                item.m, item.n, item.k, DType(item.input_dtype), DType(item.accumulation_dtype)
            )
            for item in value.matrix_instructions
        ),
        supported_dtypes=frozenset(DType(item) for item in value.supported_dtypes),
        memory_scopes=value.memory_scopes,
        barrier_scopes=value.barrier_scopes,
        asynchronous_copy=value.asynchronous_copy,
        subgroup_exchange=value.subgroup_exchange,
        vector_bytes=value.vector_bytes,
        atomics=frozenset(DType(item) for item in value.atomics),
        alignments={DType(dtype): alignment for dtype, alignment in value.alignments},
        native_multi_launch=value.native_multi_launch,
        partial_binding=value.partial_binding,
        max_kernels_per_program=value.max_kernels_per_program,
        fingerprint=value.fingerprint,
    )
