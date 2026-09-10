"""Typed component admission into the common benchmark runner."""

from typing import overload

from magnitude_engine.models.qwen35.runtime import DenseRuntime
from magnitude_engine.operations.attention import CausalAttention
from magnitude_engine.operations.linear import EncodedLinear
from magnitude_engine.operations.recurrent import DeltaRecurrence
from performance.attention import AttentionMetrics, AttentionProcedure, AttentionWorkload
from performance.linear import LinearProcedure, LinearWorkload
from performance.metrics import LinearMetrics
from performance.model import ModelMetrics, ModelProcedure, ModelWorkload
from performance.recurrent import RecurrentMetrics, RecurrentProcedure, RecurrentWorkload
from performance.runner import Policy, Result, collect

_DEFAULT_POLICY = Policy()


@overload
def run(
    component: CausalAttention, workload: AttentionWorkload, *, policy: Policy = _DEFAULT_POLICY
) -> Result[AttentionMetrics]: ...


@overload
def run(
    component: DeltaRecurrence, workload: RecurrentWorkload, *, policy: Policy = _DEFAULT_POLICY
) -> Result[RecurrentMetrics]: ...


@overload
def run(
    component: EncodedLinear, workload: LinearWorkload, *, policy: Policy = _DEFAULT_POLICY
) -> Result[LinearMetrics]: ...


@overload
def run(
    component: DenseRuntime, workload: ModelWorkload, *, policy: Policy = _DEFAULT_POLICY
) -> Result[ModelMetrics]: ...


def run(
    component: EncodedLinear | DenseRuntime | DeltaRecurrence | CausalAttention,
    workload: LinearWorkload | ModelWorkload | RecurrentWorkload | AttentionWorkload,
    *,
    policy: Policy = _DEFAULT_POLICY,
) -> (
    Result[LinearMetrics]
    | Result[ModelMetrics]
    | Result[RecurrentMetrics]
    | Result[AttentionMetrics]
):
    if isinstance(component, CausalAttention) and isinstance(workload, AttentionWorkload):
        return collect(AttentionProcedure(), component, workload, policy)
    if isinstance(component, DeltaRecurrence) and isinstance(workload, RecurrentWorkload):
        return collect(RecurrentProcedure(), component, workload, policy)
    if isinstance(component, EncodedLinear) and isinstance(workload, LinearWorkload):
        return collect(LinearProcedure(), component, workload, policy)
    if isinstance(component, DenseRuntime) and isinstance(workload, ModelWorkload):
        return collect(ModelProcedure(), component, workload, policy)
    raise TypeError("benchmark workload does not match its bound component")
