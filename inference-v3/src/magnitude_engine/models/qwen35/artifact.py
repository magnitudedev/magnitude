"""Validate GGUF's Qwen representation into explicit architecture-owned roles."""

import math
from enum import StrEnum

from pydantic import Field, PositiveFloat, PositiveInt, TypeAdapter, model_validator

from magnitude_engine.artifacts.gguf import Directory
from magnitude_engine.artifacts.identity import ArtifactIdentity
from magnitude_engine.artifacts.weights import WeightDescriptor
from magnitude_engine.data import Record
from magnitude_engine.numerics.semantics import HeadMapping


class MixerKind(StrEnum):
    ATTENTION = "attention"
    RECURRENT = "recurrent"


class Geometry(Record):
    hidden: PositiveInt
    intermediate: PositiveInt
    vocabulary: PositiveInt
    context_limit: PositiveInt
    layers: tuple[MixerKind, ...]
    attention_heads: PositiveInt
    kv_heads: PositiveInt
    attention_width: PositiveInt
    rotary_width: PositiveInt
    rotary_base: PositiveFloat
    rotary_sections: tuple[int, int, int, int]
    epsilon: PositiveFloat
    convolution_width: PositiveInt
    recurrent_key_heads: PositiveInt
    recurrent_value_heads: PositiveInt
    recurrent_width: PositiveInt
    recurrent_head_mapping: HeadMapping = HeadMapping.TILED

    @model_validator(mode="after")
    def validate_geometry(self):
        if not math.isfinite(self.epsilon) or not math.isfinite(self.rotary_base):
            raise ValueError("Qwen numerical parameters must be finite")
        if not self.layers or self.attention_heads % self.kv_heads:
            raise ValueError("invalid Qwen layer/head geometry")
        if self.recurrent_value_heads % self.recurrent_key_heads:
            raise ValueError("recurrent value heads must be a multiple of key heads")
        if self.rotary_width % 2 or self.rotary_width > self.attention_width:
            raise ValueError("invalid partial rotary width")
        if (
            any(n < 0 for n in self.rotary_sections)
            or sum(self.rotary_sections) * 2 != self.rotary_width
        ):
            raise ValueError("rotary sections do not cover the rotary width")
        if self.rotary_sections[3] != 0:
            raise ValueError("Qwen's current input contract has three rotary axes")
        if self.convolution_width < 2:
            raise ValueError("Qwen recurrent convolution requires retained history")
        return self

    @property
    def recurrent_channels(self) -> int:
        return (2 * self.recurrent_key_heads + self.recurrent_value_heads) * self.recurrent_width


class AttentionWeights(Record):
    query_gate: WeightDescriptor
    key: WeightDescriptor
    value: WeightDescriptor
    query_norm: WeightDescriptor
    key_norm: WeightDescriptor
    output: WeightDescriptor


class RecurrentWeights(Record):
    query_key_value: WeightDescriptor
    gate: WeightDescriptor
    alpha: WeightDescriptor
    beta: WeightDescriptor
    convolution: WeightDescriptor
    decay: WeightDescriptor
    time_bias: WeightDescriptor
    norm: WeightDescriptor
    output: WeightDescriptor


class BlockWeights(Record):
    input_norm: WeightDescriptor
    mixer: AttentionWeights | RecurrentWeights
    feedforward_norm: WeightDescriptor
    feedforward_gate: WeightDescriptor
    feedforward_up: WeightDescriptor
    feedforward_down: WeightDescriptor


class DenseArtifact(Record):
    artifact_identity: ArtifactIdentity = Field(pattern=r"^[0-9a-f]{64}$")
    geometry: Geometry
    embedding: WeightDescriptor
    output_norm: WeightDescriptor
    output: WeightDescriptor
    blocks: tuple[BlockWeights, ...]


def inspect_dense(directory: Directory, identity: ArtifactIdentity) -> DenseArtifact:
    if directory.value("general.architecture") != "qwen35":
        raise ValueError("dense Qwen binding requires the qwen35 GGUF architecture")
    metadata = {item.name: item.value for item in directory.metadata}
    if metadata.get("qwen35.rope.scaling.type", "none") != "none":
        raise ValueError("scaled Qwen rotary definitions are not yet qualified")

    def integer(key: str) -> int:
        value = directory.value("qwen35." + key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"qwen35.{key} must be a positive integer")
        return value

    def number(key: str) -> float:
        value = directory.value("qwen35." + key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"qwen35.{key} must be numeric")
        return float(value)

    layer_count = integer("block_count")
    flags = metadata.get("qwen35.attention.recurrent_layers")
    if flags is None:
        interval = integer("full_attention_interval")
        kinds = tuple(
            MixerKind.ATTENTION if (i + 1) % interval == 0 else MixerKind.RECURRENT
            for i in range(layer_count)
        )
    else:
        if (
            not isinstance(flags, tuple)
            or len(flags) != layer_count
            or any(type(v) is not bool for v in flags)
        ):
            raise ValueError("invalid per-layer recurrent flags")
        kinds = tuple(MixerKind.RECURRENT if flag else MixerKind.ATTENTION for flag in flags)
    embedding = directory.tensor("token_embd.weight")
    if len(embedding.shape) != 2:
        raise ValueError("Qwen embedding must be a matrix")
    vocabulary = metadata.get("tokenizer.ggml.tokens")
    if vocabulary is not None and (
        not isinstance(vocabulary, tuple) or len(vocabulary) != embedding.shape[0]
    ):
        raise ValueError("tokenizer vocabulary differs from the embedding table")
    sections = directory.value("qwen35.rope.dimension_sections")
    if (
        not isinstance(sections, tuple)
        or len(sections) != 4
        or any(type(n) is not int for n in sections)
    ):
        raise ValueError("Qwen rotary sections must contain four integers")
    geometry = Geometry(
        hidden=integer("embedding_length"),
        intermediate=integer("feed_forward_length"),
        vocabulary=embedding.shape[0],
        context_limit=integer("context_length"),
        layers=kinds,
        attention_heads=integer("attention.head_count"),
        kv_heads=integer("attention.head_count_kv"),
        attention_width=integer("attention.key_length"),
        rotary_width=integer("rope.dimension_count"),
        rotary_base=number("rope.freq_base"),
        rotary_sections=TypeAdapter(tuple[int, int, int, int]).validate_python(
            sections, strict=True
        ),
        epsilon=number("attention.layer_norm_rms_epsilon"),
        convolution_width=integer("ssm.conv_kernel"),
        recurrent_key_heads=integer("ssm.group_count"),
        recurrent_value_heads=integer("ssm.time_step_rank"),
        recurrent_width=integer("ssm.state_size"),
    )
    if integer("attention.value_length") != geometry.attention_width:
        raise ValueError("Qwen attention requires equal key/value head widths")
    if integer("ssm.inner_size") != geometry.recurrent_value_heads * geometry.recurrent_width:
        raise ValueError("Qwen recurrent inner size differs from head geometry")
    consumed: set[str] = set()

    def tensor(name: str, shape: tuple[int, ...]) -> WeightDescriptor:
        value = directory.tensor(name)
        if value.shape != shape:
            raise ValueError(f"Qwen weight {name}: expected {shape}, received {value.shape}")
        consumed.add(name)
        return WeightDescriptor(name=name, shape=shape)

    g = geometry
    embedding = tensor(embedding.name, (g.vocabulary, g.hidden))
    output_norm = tensor("output_norm.weight", (g.hidden,))
    output_name = (
        "output.weight"
        if any(t.name == "output.weight" for t in directory.tensors)
        else embedding.name
    )
    output = tensor(output_name, (g.vocabulary, g.hidden))
    blocks = []
    for i, kind in enumerate(kinds):
        prefix = f"blk.{i}."
        if kind == MixerKind.ATTENTION:
            mixer = AttentionWeights(
                query_gate=tensor(
                    prefix + "attn_q.weight", (2 * g.attention_heads * g.attention_width, g.hidden)
                ),
                key=tensor(prefix + "attn_k.weight", (g.kv_heads * g.attention_width, g.hidden)),
                value=tensor(prefix + "attn_v.weight", (g.kv_heads * g.attention_width, g.hidden)),
                query_norm=tensor(prefix + "attn_q_norm.weight", (g.attention_width,)),
                key_norm=tensor(prefix + "attn_k_norm.weight", (g.attention_width,)),
                output=tensor(
                    prefix + "attn_output.weight", (g.hidden, g.attention_heads * g.attention_width)
                ),
            )
        else:
            mixer = RecurrentWeights(
                query_key_value=tensor(
                    prefix + "attn_qkv.weight", (g.recurrent_channels, g.hidden)
                ),
                gate=tensor(
                    prefix + "attn_gate.weight",
                    (g.recurrent_value_heads * g.recurrent_width, g.hidden),
                ),
                alpha=tensor(prefix + "ssm_alpha.weight", (g.recurrent_value_heads, g.hidden)),
                beta=tensor(prefix + "ssm_beta.weight", (g.recurrent_value_heads, g.hidden)),
                convolution=tensor(
                    prefix + "ssm_conv1d.weight", (g.recurrent_channels, g.convolution_width)
                ),
                decay=tensor(prefix + "ssm_a", (g.recurrent_value_heads,)),
                time_bias=tensor(prefix + "ssm_dt.bias", (g.recurrent_value_heads,)),
                norm=tensor(prefix + "ssm_norm.weight", (g.recurrent_width,)),
                output=tensor(
                    prefix + "ssm_out.weight",
                    (g.hidden, g.recurrent_value_heads * g.recurrent_width),
                ),
            )
        blocks.append(
            BlockWeights(
                input_norm=tensor(prefix + "attn_norm.weight", (g.hidden,)),
                mixer=mixer,
                feedforward_norm=tensor(prefix + "post_attention_norm.weight", (g.hidden,)),
                feedforward_gate=tensor(prefix + "ffn_gate.weight", (g.intermediate, g.hidden)),
                feedforward_up=tensor(prefix + "ffn_up.weight", (g.intermediate, g.hidden)),
                feedforward_down=tensor(prefix + "ffn_down.weight", (g.hidden, g.intermediate)),
            )
        )
    unused = {t.name for t in directory.tensors} - consumed
    if unused:
        raise ValueError(f"dense Qwen artifact contains unbound weight roles: {sorted(unused)}")
    return DenseArtifact(
        artifact_identity=identity,
        geometry=g,
        embedding=embedding,
        output_norm=output_norm,
        output=output,
        blocks=tuple(blocks),
    )
