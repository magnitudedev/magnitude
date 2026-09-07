"""Device-free parameter records for capture, formulas and persisted observations."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
