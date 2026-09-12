"""Magnitensor: composable inference tensors compiled through TileLang."""

from .compiler.compilation import CompiledFunction, CompileOptions, compile
from .compiler.lowering import (
    Candidate,
    Capabilities,
    LoweringRegistry,
    MatrixInstruction,
    lowerings,
)
from .compiler.tuning import TuningDatabase, TuningKey, TuningRecord
from .kernels import register_builtin_lowerings
from .representations import (
    Affine,
    Code,
    Codebook,
    CodeInterpretation,
    Dense,
    DirectCoefficients,
    HierarchicalCoefficients,
    TensorRepresentation,
)
from .runtime.resources import CapacityError, Completion, Device, Execution, Resource
from .runtime.tilelang import TileLangRuntime
from .tensor.graph import Effects, Graph, Node, Value, ValueKind
from .tensor.operation import (
    NumericalContract,
    Operation,
    evaluate_reference,
    operation,
    operations,
)
from .tensor.ops import (
    add,
    cast,
    causal_attention,
    concatenate,
    delta_recurrence,
    divide,
    embedding,
    exp,
    kv_append,
    linear,
    matmul,
    multiply,
    reshape,
    rms_norm,
    rotary,
    route_topk,
    routed_experts,
    scalar,
    sigmoid,
    silu,
    softmax,
    subtract,
    tanh,
    transpose,
)
from .tensor.tracing import Argument, Signature, Tensor, trace
from .tensor.types import DENSE, Dim, DType, Layout, TensorSpec

register_builtin_lowerings(lowerings)


def device(target="auto", *, budget_bytes: int) -> Device:
    return Device(TileLangRuntime(target), budget_bytes=budget_bytes)


__all__ = [
    "Affine",
    "Argument",
    "Candidate",
    "Capabilities",
    "CapacityError",
    "Code",
    "CodeInterpretation",
    "Codebook",
    "CompileOptions",
    "CompiledFunction",
    "Completion",
    "DENSE",
    "DType",
    "Dense",
    "Device",
    "Dim",
    "DirectCoefficients",
    "Effects",
    "Execution",
    "Graph",
    "HierarchicalCoefficients",
    "Layout",
    "LoweringRegistry",
    "MatrixInstruction",
    "Node",
    "NumericalContract",
    "Operation",
    "Resource",
    "Signature",
    "Tensor",
    "TensorRepresentation",
    "TensorSpec",
    "TileLangRuntime",
    "TuningDatabase",
    "TuningKey",
    "TuningRecord",
    "Value",
    "ValueKind",
    "add",
    "cast",
    "causal_attention",
    "compile",
    "concatenate",
    "delta_recurrence",
    "device",
    "divide",
    "embedding",
    "evaluate_reference",
    "exp",
    "kv_append",
    "linear",
    "lowerings",
    "matmul",
    "multiply",
    "operation",
    "operations",
    "reshape",
    "rms_norm",
    "rotary",
    "route_topk",
    "routed_experts",
    "scalar",
    "sigmoid",
    "silu",
    "softmax",
    "subtract",
    "tanh",
    "trace",
    "transpose",
]
