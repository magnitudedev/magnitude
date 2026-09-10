"""Unwindable construction of a composite invocation before any submission."""

from contextlib import ExitStack
from typing import Protocol

from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.platform.execution import DeviceContext, Prepared, Tensor, TensorSpec


class Closeable(Protocol):
    def close(self) -> None: ...


class Preparation:
    def __init__(self, context: DeviceContext):
        self.context = context
        self._cleanup = ExitStack()
        self._commands: list[Prepared] = []
        self._finished = False

    def own[T: Closeable](self, tensor: T) -> T:
        self._cleanup.callback(tensor.close)
        return tensor

    def allocate(self, spec: TensorSpec) -> Tensor:
        result = self.context.allocate(spec)
        self._cleanup.callback(result.close)
        return result

    def upload(self, spec: TensorSpec, content: bytes) -> Tensor:
        result = self.context.upload(spec, content)
        self._cleanup.callback(result.close)
        return result

    def indices(self, values: tuple[int, ...], shape: tuple[int, ...] | None = None) -> Tensor:
        result = self.context.indices(values, shape)
        self._cleanup.callback(result.close)
        return result

    def view(self, tensor: Tensor, spec: TensorSpec, offset: int = 0) -> Tensor:
        result = tensor.view(spec, offset)
        self._cleanup.callback(result.close)
        return result

    def parameter(self, parameter: Parameter) -> Tensor:
        result = parameter.acquire()
        self._cleanup.callback(result.close)
        return result

    def add(self, *commands: Prepared) -> None:
        if self._finished:
            raise RuntimeError("preparation is finished")
        self._commands.extend(commands)

    def finish(self) -> tuple[Prepared, ...]:
        if self._finished:
            raise RuntimeError("preparation is finished")
        self._finished = True
        return tuple(self._commands)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if kind is not None or not self._finished:
            for command in reversed(self._commands):
                command.close()
        self._cleanup.close()
