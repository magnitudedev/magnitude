"""One bound launch for generated regions and explicitly opaque handwritten kernels."""

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .assembly import source_files
from .graph import Tensor, Value
from .plan import Launch, Parameter, Scalar, Source, identifier
from .primitive import Primitive
from .runtime import generated_kernel


@dataclass(frozen=True)
class Kernel(Primitive):
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    source: Source
    launch: Launch
    template: tuple[tuple[str, Parameter], ...] = ()
    constants: tuple[Scalar, ...] = ()

    def __post_init__(self):
        names = [v.name for v in (*self.inputs, *self.outputs)]
        names.extend(k for k, _ in self.template)
        names.extend(v.name for v in self.constants)
        if len(names) != len(set(names)):
            raise ValueError("kernel symbols must be unique across all bindings")

    def bind(self):
        sources = source_files(self.source)
        header = "".join(f"#define {v.name} ({v.literal})\n" for v in self.constants)
        header += "\n".join(text for _, text in sources[:-1])
        return BoundKernel(
            self.inputs, self.outputs, sources[-1][1], header, sources, self.launch, self.template
        )

    def infer(self, inputs):
        if inputs != tuple(v.tensor for v in self.inputs):
            raise ValueError("opaque kernel operands differ from the declared binding")
        return tuple(v.tensor for v in self.outputs)

    def lower(self, inputs):
        self.infer(inputs)
        return self.bind()


@dataclass(frozen=True)
class BoundKernel:
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    source: str
    header: str
    sources: tuple[tuple[str, str], ...]
    launch: Launch
    template: tuple[tuple[str, Parameter], ...] = ()
    description: str = "handwritten Metal dispatch; opaque internal algorithm"
    _kernel: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self):

        names = [v.name for v in (*self.inputs, *self.outputs)]
        names.extend(k for k, _ in self.template)
        for name in names:
            identifier(name)
        if not self.outputs or len(names) != len(set(names)):
            raise ValueError("kernel bindings require unique names and output ownership")
        object.__setattr__(
            self,
            "_kernel",
            generated_kernel(
                self.source,
                tuple(v.name for v in self.inputs),
                tuple(v.name for v in self.outputs),
                self.header,
            ),
        )

    def __call__(self, *arrays):
        if len(arrays) != len(self.inputs):
            raise ValueError("kernel operand count differs from its binding")
        if any(
            Tensor(array.shape, array.dtype) != value.tensor
            for array, value in zip(arrays, self.inputs, strict=True)
        ):
            raise ValueError("kernel operand shape or dtype differs from its binding")
        if all(v.tensor.size == 0 for v in self.outputs):
            return tuple(mx.zeros(v.tensor.shape, v.tensor.dtype) for v in self.outputs)
        # Layout normalization is an explicit MLX node, visible to enclosing compile.
        # Already contiguous operands do not copy; no hidden custom-kernel layout fixup.
        return tuple(
            self._kernel(
                inputs=[mx.contiguous(a) for a in arrays],
                output_shapes=[v.tensor.shape for v in self.outputs],
                output_dtypes=[v.tensor.dtype for v in self.outputs],
                grid=self.launch.grid,
                threadgroup=self.launch.threadgroup,
                template=list(self.template),
            )
        )


def dispatch(source, *, inputs, outputs, launch, template=(), constants=()):
    """Bind array operands to the same immutable launch used by compiled regions."""

    operands = tuple(Value(name, Tensor(array.shape, array.dtype)) for name, array in inputs)
    results = tuple(Value(name, tensor) for name, tensor in outputs)
    kernel = Kernel(operands, results, source, launch, template, constants)
    return kernel(*(array for _, array in inputs))


@dataclass(frozen=True)
class ConstantInputs:
    kernel: BoundKernel
    constants: tuple[mx.array, ...]

    def __call__(self, *arrays):
        return self.kernel(*arrays, *self.constants)
