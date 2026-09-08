"""Lower ordinary MLX scalar dataflow into a fused, typed device program."""

from dataclasses import dataclass
from functools import partial

import mlx.core as mx

from ._emitter import (
    Assign,
    Binary,
    Call,
    Cast,
    For,
    If,
    Index,
    Let,
    Literal,
    Member,
    Negative,
    Select,
    Symbol,
    emit,
)
from .graph import Graph, Node, Value, signature
from .kernel import BoundKernel, ConstantInputs
from .plan import Launch
from .proposal import Proposal

# The scalar numerical operation is distinct from its execution schedule.
BINARY = {
    "Add": "+",
    "Subtract": "-",
    "Multiply": "*",
    "Divide": "/",
    "Equal": "==",
    "NotEqual": "!=",
    "Greater": ">",
    "GreaterEqual": ">=",
    "Less": "<",
    "LessEqual": "<=",
}
UNARY = {
    "Abs": "metal::abs",
    "Exp": "metal::precise::exp",
    "Log": "metal::precise::log",
    "Sin": "metal::precise::sin",
    "Cos": "metal::precise::cos",
    "Tanh": "metal::precise::tanh",
    "Sqrt": "metal::precise::sqrt",
    "Rsqrt": "metal::precise::rsqrt",
}


def supported(node: Node) -> bool:
    if (
        not isinstance(node.operation, str)
        or any(
            v.tensor.dtype
            not in (
                mx.float32,
                mx.float16,
                mx.bfloat16,
                mx.int32,
                mx.uint32,
                mx.int64,
                mx.uint64,
                mx.int16,
                mx.uint16,
                mx.int8,
                mx.uint8,
                mx.bool_,
            )
            for v in (*node.inputs, *node.outputs)
        )
        or len(node.outputs) != 1
    ):
        return False
    args = tuple(Symbol(f"arg{i}") for i in range(len(node.inputs)))
    return scalar_expression(node, args, "Native") is not None


def address(value: Value, shape: tuple[int, ...], linear: Symbol):
    source = value.tensor.shape
    if source == shape:
        return linear
    if len(source) > len(shape):
        raise ValueError("cannot broadcast to a smaller rank")
    stride = 1
    terms = []
    for size, target in zip(
        reversed((1,) * (len(shape) - len(source)) + source), reversed(shape), strict=True
    ):
        if size not in (1, target):
            raise ValueError("invalid elementwise broadcast")
        if size > 1:
            # The logical output coordinate is independent of physical array strides;
            # the MLX invocation explicitly requests contiguous inputs.
            terms.append((size, target, stride))
        stride *= target
    source_stride = 1
    result = Literal(0)
    for size, target, output_stride in terms:
        coordinate = Binary("%", Binary("/", linear, Literal(output_stride)), Literal(target))
        result = Binary("+", result, Binary("*", coordinate, Literal(source_stride)))
        source_stride *= size
    return result


def scalar_expression(node, args, dtype):
    """The single definition of scalar eligibility and typed expression emission."""
    op, out = node.operation, node.outputs[0]
    if op == "Sqrt":
        if len(node.attributes) != 1 or type(node.attributes[0]) is not bool:
            return None
    elif op == "AsType":
        if node.attributes != (out.tensor.dtype,):
            return None
    elif op in ("Broadcast", "Full"):
        if node.attributes != (out.tensor.shape,):
            return None
    elif node.attributes:
        return None
    body = []
    if op in BINARY:
        value = Binary(BINARY[op], *args)
    elif op in UNARY:
        value = Call(
            "metal::precise::rsqrt" if op == "Sqrt" and node.attributes[0] else UNARY[op], args
        )
    elif op == "Sigmoid":
        # Preserve MLX's precise unary exp boundary and stable sigmoid expression.
        e = Symbol(f"exp_{out.name}")
        body.append(
            Let(e, dtype, Cast(dtype, Call("metal::precise::exp", (Call("metal::abs", args),))))
        )
        y = Binary("/", Literal(1), Binary("+", Literal(1), e))
        value = Select(Binary("<", args[0], Literal(0)), y, Binary("-", Literal(1), y))
    elif op == "Negative":
        value = Negative(args[0])
    elif op == "Square":
        value = Binary("*", args[0], args[0])
    elif op in ("AsType", "Broadcast", "Full", "StopGradient"):
        value = args[0]
    elif op == "Select":
        value = Select(*args)
    elif op in ("Maximum", "Minimum"):
        value = Call("metal::max" if op == "Maximum" else "metal::min", args)
        if out.tensor.dtype in (mx.float32, mx.float16, mx.bfloat16):
            # MLX maximum/minimum propagate NaNs; Metal max/min alone do not.
            value = Select(Binary(">" if op == "Maximum" else "<", *args), *args)
            value = Select(Call("metal::isnan", (args[0],)), args[0], value)
    else:
        return None
    return body, value


def scalar_program(nodes, values, types):
    body = []
    for node in nodes:
        args = tuple(values[v.name] for v in node.inputs)
        out = node.outputs[0]
        dtype = types[out.tensor.dtype]
        expression = scalar_expression(node, args, dtype)
        if expression is None:
            raise ValueError("scalar node has no qualified emission")
        prefix, value = expression
        body.extend(prefix)
        symbol = Symbol(f"value_{out.name}")
        body.append(Let(symbol, dtype, Cast(dtype, value)))
        values[out.name] = symbol
    return body, values


@dataclass(frozen=True)
class Elementwise:
    """An output-linear tile; scheduling changes do not alter scalar dtype boundaries."""

    items_per_thread: int = 4
    threads: int = 128

    def lower(self, graph: Graph):
        if not graph.outputs or any(not supported(n) for n in graph.nodes):
            raise ValueError("elementwise schedule requires supported scalar operations")
        shape = graph.outputs[0].tensor.shape
        if any(v.tensor.shape != shape for v in graph.outputs):
            raise ValueError("one elementwise region requires aligned output shapes")
        if self.items_per_thread < 1 or not 1 <= self.threads <= 1024:
            raise ValueError("invalid elementwise tile")
        inputs = (*graph.inputs, *(v for v, _ in graph.constants))
        constants = tuple(a for _, a in graph.constants)
        types = {
            v.tensor.dtype: f"D{i}"
            for i, v in enumerate((*inputs, *(v for n in graph.nodes for v in n.outputs)))
        }
        index = Symbol("index")
        values = {
            v.name: (
                Symbol(v.name)
                if v.tensor.shape == ()
                else Index(Symbol(v.name), address(v, shape, index))
            )
            for v in inputs
        }
        body, values = scalar_program(graph.nodes, values, types)
        body.extend(
            Assign(Index(Symbol(f"out{i}"), index), values[v.name])
            for i, v in enumerate(graph.outputs)
        )
        count = graph.outputs[0].tensor.size
        body = [If(Binary("<", index, Literal(count)), tuple(body))]
        start = Binary(
            "*", Member(Symbol("thread_position_in_grid"), "x"), Literal(self.items_per_thread)
        )
        source = emit(
            (
                For(
                    index,
                    start,
                    Binary("+", start, Literal(self.items_per_thread)),
                    tuple(body),
                    unroll=True,
                ),
            )
        )

        kernel = BoundKernel(
            inputs,
            signature(
                tuple(f"out{i}" for i in range(len(graph.outputs))),
                tuple(v.tensor for v in graph.outputs),
            ),
            source,
            "",
            (),
            Launch(
                (max(1, (count + self.items_per_thread - 1) // self.items_per_thread), 1, 1),
                (self.threads, 1, 1),
            ),
            tuple((name, dtype) for dtype, name in types.items()),
            description=f"{len(graph.nodes)} captured scalar operations; native dtype boundaries",
        )
        return ConstantInputs(kernel, constants) if constants else kernel


def plan_scalar(graph, bindings):
    if bindings or not all(supported(n) for n in graph.nodes):
        return None
    if len({v.tensor.shape for n in graph.nodes for v in n.outputs}) != 1:
        return None
    return Proposal(graph, partial(Elementwise().lower, graph))
