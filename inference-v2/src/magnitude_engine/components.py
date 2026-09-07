"""Device-free component facts. Execution, observation and theory have separate owners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from magnitude_engine.models.definition import ModelDefinition

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Facts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Configuration(Facts):
    settings: dict[str, int | float | bool | str | None] = Field(default_factory=dict)


class AttentionGeometry(Facts):
    query_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    key_width: int = Field(gt=0)
    value_width: int = Field(gt=0)
    element_bytes: int = Field(gt=0)
    window: int | None = Field(default=None, gt=0)
    kv_source: int | None = None


class RecurrentGeometry(Facts):
    key_heads: int = Field(gt=0)
    value_heads: int = Field(gt=0)
    key_width: int = Field(gt=0)
    value_width: int = Field(gt=0)
    element_bytes: int = Field(gt=0)


class TensorFacts(Facts):
    identity: str
    shape: tuple[int, ...]
    bytes: int = Field(ge=0)
    dtype: str


class MatrixFacts(Facts):
    identity: str
    input_width: int = Field(gt=0)
    output_width: int = Field(gt=0)
    experts: int | None = Field(default=None, gt=0)


class WeightUse(StrEnum):
    FULL = "full"
    EMBEDDING = "embedding"
    EXPERTS = "experts"


class NeuralParameters(Facts):
    arrays: dict[str, TensorFacts] = Field(default_factory=dict)
    matrices: tuple[MatrixFacts, ...] = ()
    weight_use: WeightUse = WeightUse.FULL
    top_k: int | None = Field(default=None, gt=0)
    settings: dict[str, int | float | bool | str | None] = Field(default_factory=dict)


class KVGeometry(Facts):
    heads: int = Field(gt=0)
    key_width: int = Field(gt=0)
    value_width: int = Field(gt=0)


class KVStorage(Facts):
    layers: tuple[KVGeometry, ...]
    element_bytes: int = Field(gt=0)
    page_size: int = Field(gt=0)
    slab_pages: int = Field(gt=0)
    max_pages: int = Field(gt=0)


class RecurrentStorage(Facts):
    layouts: tuple[tuple[TensorFacts, ...], ...]


class NativeStorage(Facts):
    cache_types: tuple[str, ...]


class OpaqueParameters(Facts):
    opaque: bool = True
    arrays: dict[str, TensorFacts] = Field(default_factory=dict)


@dataclass(frozen=True)
class Contract[P: Facts]:
    identity: str
    parameters: type[P]

    def read(self, value: dict) -> P:
        # JSON reconstruction validates the typed shape, including unknown fields.
        import json

        return TypeAdapter(self.parameters).validate_json(json.dumps(value), strict=True)


class Source(StrEnum):
    MLX = "MLX"
    LM = "LM"
    VLM = "VLM"
    MAG = "MAG"


@dataclass(frozen=True)
class Implementation[P: Facts]:
    contract: Contract[P]
    source: Source
    variant: str

    @property
    def identity(self) -> str:
        return f"{self.contract.identity}:{self.source.value}:{self.variant}"


# Contract identity and parameter shape are production facts; dimensions are not.
ATTENTION = Contract("MODEL:ATTENTION", AttentionGeometry)
RECURRENCE = Contract("MODEL:GATED_DELTA", RecurrentGeometry)
EMBEDDING = Contract("MODEL:EMBEDDING", NeuralParameters)
EXPERTS = Contract("MODEL:EXPERTS", NeuralParameters)
QWEN35 = Contract("MODEL:QWEN35", NeuralParameters)
QWEN_ATTENTION = Contract("MODEL:QWEN35.ATTENTION", NeuralParameters)
QWEN_RECURRENCE = Contract("MODEL:QWEN35.RECURRENCE", NeuralParameters)
QWEN_FEEDFORWARD = Contract("MODEL:QWEN35.FEEDFORWARD", NeuralParameters)
QWEN_READOUT = Contract("MODEL:QWEN35.READOUT", NeuralParameters)
QWEN_MTP = Contract("MODEL:QWEN35.MTP", NeuralParameters)
GEMMA4 = Contract("MODEL:GEMMA4", NeuralParameters)
GEMMA_INPUTS = Contract("MODEL:GEMMA4.INPUTS", NeuralParameters)
GEMMA_ATTENTION = Contract("MODEL:GEMMA4.ATTENTION", NeuralParameters)
GEMMA_KV = Contract("MODEL:GEMMA4.KV", NeuralParameters)
GEMMA_FEEDFORWARD = Contract("MODEL:GEMMA4.FEEDFORWARD", NeuralParameters)
GEMMA_MLP = Contract("MODEL:GEMMA4.MLP", NeuralParameters)
GEMMA_EXPERT_BRANCH = Contract("MODEL:GEMMA4.EXPERT_BRANCH", NeuralParameters)
GEMMA_READOUT = Contract("MODEL:GEMMA4.READOUT", NeuralParameters)
FORWARD = Contract("MODEL:FORWARD", OpaqueParameters)
EXECUTOR = Contract("MODEL:EXECUTOR", NeuralParameters)
LOADING = Contract("MODEL:LOADING", Configuration)
KV_STORE = Contract("KV:STORE", KVStorage)
KV_APPEND = Contract("KV:APPEND", KVStorage)
KV_BRANCH = Contract("KV:BRANCH", Configuration)
RECURRENT_STATE = Contract("STATE:RECURRENT", RecurrentStorage)
NATIVE_STATE = Contract("STATE:CHECKPOINTS", NativeStorage)
HYBRID_STATE = Contract("STATE:QWEN35", Configuration)
ENGINE = Contract("ENGINE:INFERENCE", Configuration)
ADMISSION = Contract("SCHEDULING:ADMISSION", Configuration)
SCHEDULING = Contract("SCHEDULING:SERVICE", Configuration)
PREFILL = Contract("SCHEDULING:PREFILL", Configuration)
BATCHING = Contract("BATCHING:ASSEMBLY", Configuration)
DEVICE = Contract("EXECUTION:DEVICE", Configuration)
MEMORY = Contract("MEMORY:ACCOUNTING", Configuration)
PREFIX = Contract("CACHE:PREFIX", Configuration)
GENERATION = Contract("GENERATION:PLAIN", Configuration)
SPECULATION = Contract("GENERATION:SPECULATION", Configuration)
SAMPLING = Contract("GENERATION:SAMPLING", Configuration)
ACCEPTANCE = Contract("GENERATION:ACCEPTANCE", Configuration)

CONTRACTS: tuple[Contract[Any], ...] = (
    ATTENTION,
    RECURRENCE,
    EMBEDDING,
    EXPERTS,
    QWEN35,
    QWEN_ATTENTION,
    QWEN_RECURRENCE,
    QWEN_FEEDFORWARD,
    QWEN_READOUT,
    QWEN_MTP,
    GEMMA4,
    GEMMA_INPUTS,
    GEMMA_ATTENTION,
    GEMMA_KV,
    GEMMA_FEEDFORWARD,
    GEMMA_MLP,
    GEMMA_EXPERT_BRANCH,
    GEMMA_READOUT,
    FORWARD,
    EXECUTOR,
    LOADING,
    KV_STORE,
    KV_APPEND,
    KV_BRANCH,
    RECURRENT_STATE,
    NATIVE_STATE,
    HYBRID_STATE,
    ENGINE,
    ADMISSION,
    SCHEDULING,
    PREFILL,
    BATCHING,
    DEVICE,
    MEMORY,
    PREFIX,
    GENERATION,
    SPECULATION,
    SAMPLING,
    ACCEPTANCE,
)
_BY_ID = {c.identity: c for c in CONTRACTS}


def implementation(identity: str) -> Implementation[Any]:
    address, source, variant = identity.rsplit(":", 2)
    if not variant or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in variant):
        raise ValueError(f"invalid implementation variant: {variant}")
    try:
        return Implementation(_BY_ID[address], Source(source), variant)
    except (KeyError, ValueError) as error:
        raise ValueError(f"unknown component implementation: {identity}") from error


@dataclass(frozen=True)
class Component:
    contract: Contract[Any]
    source: Source
    variant: str | Callable[[Any], str]
    model: ModelDefinition | None = None

    def identity(self, value: object) -> Implementation[Any]:
        variant = self.variant(value) if callable(self.variant) else self.variant
        return Implementation(self.contract, self.source, variant)


def component[T](
    contract: Contract[Any],
    *,
    source: Source,
    variant: str | Callable[[Any], str],
    model: ModelDefinition | None = None,
) -> Callable[[T], T]:
    """Declare an execution component. Leaves construction and calls unchanged."""

    def declare(owner: T) -> T:
        if "__component__" in vars(owner):
            raise TypeError("component already declared")
        cast(Any, owner).__component__ = Component(contract, source, variant, model)
        return owner

    return declare


def component_of(value: object) -> Component:
    # Exact declarations: an undeclared subclass must not inherit an identity silently.
    import inspect

    owner = value.__func__ if inspect.ismethod(value) else value
    namespace = (
        vars(owner) if inspect.isfunction(owner) or isinstance(owner, type) else vars(type(owner))
    )
    declaration = namespace.get("__component__")
    if not isinstance(declaration, Component):
        raise TypeError(f"undeclared execution component: {type(value).__qualname__}")
    return declaration
