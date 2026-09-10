"""Reusable commands with typed changing operands and invocation-owned resources.

Capture consumes prepared commands. Static weights/scratch are retained once;
changing inputs (including state and KV extent pins) are owned only by invocations.
The native adapter binds the entire sequence in one call.
"""

from contextlib import ExitStack
from dataclasses import dataclass

from magnitude_engine.platform.execution import (
    DeviceContext,
    Executable,
    ExecutionOrder,
    InputReference,
    Lease,
    Prepared,
    Tensor,
    TensorSpec,
)


@dataclass(frozen=True)
class InputLayout:
    specifications: tuple[TensorSpec, ...]
    # Equal allocation groups and relative byte offsets preserve alias meaning.
    aliases: tuple[tuple[int, int], ...]

    @classmethod
    def of(cls, inputs: tuple[Tensor, ...]) -> "InputLayout":
        allocations = {}
        aliases = []
        for tensor in inputs:
            tensor._lease.check()
            allocation = tensor._lease._allocation
            group, origin = allocations.setdefault(allocation, (len(allocations), tensor.offset))
            aliases.append((group, tensor.offset - origin))
        return cls(tuple(tensor.spec for tensor in inputs), tuple(aliases))


class _Capture:
    """Static resources have the lifetime of the capture and its invocations."""

    def __init__(self, context: DeviceContext, ownership: ExitStack):
        self.context, self.ownership, self.claims = context, ownership, 0


class _CaptureLease:
    def __init__(self, capture: _Capture):
        self.capture, self.closed = capture, False
        capture.claims += 1

    @property
    def context(self) -> DeviceContext:
        return self.capture.context

    def fork(self) -> "_CaptureLease":
        self.context.check()
        if self.closed:
            raise RuntimeError("captured resources are closed")
        return _CaptureLease(self.capture)

    def close(self) -> None:
        self.context.check_thread()
        if not self.closed:
            self.closed = True
            self.capture.claims -= 1
            if self.capture.claims == 0:
                self.capture.ownership.close()


class BoundSequence:
    """Capture a dependency region.

    INDEPENDENT declares that children have no read/write or write/write conflict;
    their reads must already be available and outputs must not overlap. Backends
    may overlap those children. Completion of the region joins every child, so an
    enclosing ordered sequence can consume their outputs normally. Serial execution
    is a valid realization when a backend does not expose concurrent dispatch.
    """

    def __init__(
        self,
        context: DeviceContext,
        commands: tuple[Prepared, ...],
        inputs: tuple[Tensor, ...] = (),
        *,
        order: ExecutionOrder = ExecutionOrder.ORDERED,
    ):
        context.check()
        if not isinstance(order, ExecutionOrder):
            raise TypeError("execution order must be explicit")
        if not commands or len({id(command) for command in commands}) != len(commands):
            raise ValueError("binding requires distinct nonempty prepared commands")
        if any(command.context is not context or command._consumed for command in commands):
            raise ValueError("binding requires unsubmitted work from its execution owner")
        if any(tensor.context is not context for tensor in inputs):
            raise ValueError("binding inputs belong to another execution owner")
        self.layout = InputLayout.of(inputs)
        with ExitStack() as consumed, ExitStack() as ownership:
            for command in commands:
                consumed.callback(command.close)
            allocations = set()

            def retain(claims):
                for lease in claims:
                    if isinstance(lease, Lease):
                        if lease._allocation in allocations:
                            continue
                        allocations.add(lease._allocation)
                    retained = lease.fork()
                    ownership.callback(retained.close)

            references = []
            for command in commands:
                bindings = []
                for (allocation, spec, offset), claims in zip(
                    command._arguments, command._operand_claims, strict=True
                ):
                    candidates = [
                        (tensor.spec.nbytes, index, offset - tensor.offset)
                        for index, tensor in enumerate(inputs)
                        if allocation is tensor._lease._allocation
                        and tensor.offset <= offset
                        and offset + spec.nbytes <= tensor.offset + tensor.spec.nbytes
                    ]
                    if candidates:
                        _, index, relative = min(candidates)
                        bindings.append(InputReference(index, relative))
                    else:
                        if any(allocation is tensor._lease._allocation for tensor in inputs):
                            raise ValueError("changing operand does not cover a captured view")
                        bindings.append(None)
                        retain(claims)
                retain(command._extra_claims)
                references.append(tuple(bindings))
            native = context.driver.bind_sequence(
                tuple(command.kernel for command in commands),
                tuple(command._command for command in commands),
                tuple(references),
                order,
            )
            self._executable: Executable | None = Executable(
                self.layout.specifications,
                native,
                context.driver,
                sum(command.dispatches for command in commands),
            )
            self.context = context
            self._resources = _CaptureLease(_Capture(context, ownership.pop_all()))

    def prepare(self, inputs: tuple[Tensor, ...] = ()) -> Prepared:
        self.context.check()
        if self._executable is None:
            raise RuntimeError("physical binding is closed")
        if InputLayout.of(inputs) != self.layout:
            raise ValueError("changing operand layout differs from the captured sequence")
        return Prepared(self.context, self._executable, inputs, resources=(self._resources,))

    def close(self) -> None:
        self.context.check_thread()
        if self._executable is not None:
            self._executable = None
            self._resources.close()
