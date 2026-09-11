"""Read an MLX Safetensors directory into Qwen's weight roles."""

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

from magnitude_engine.kernels.semantics import HeadMapping
from magnitude_engine.models.qwen35.description import (
    AttentionWeights,
    BlockWeights,
    DenseDescription,
    Geometry,
    MixerKind,
    RecurrentWeights,
)
from magnitude_engine.weights.descriptor import WeightTransform
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat


class RotaryConfig(BaseModel):
    rope_theta: PositiveFloat
    partial_rotary_factor: PositiveFloat
    rope_type: str
    mrope_section: tuple[int, int, int]
    mrope_interleaved: bool


class TextConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    model_type: str
    hidden_size: PositiveInt
    intermediate_size: PositiveInt
    vocab_size: PositiveInt
    max_position_embeddings: PositiveInt
    num_hidden_layers: PositiveInt
    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    head_dim: PositiveInt
    linear_num_key_heads: PositiveInt
    linear_num_value_heads: PositiveInt
    linear_key_head_dim: PositiveInt
    linear_value_head_dim: PositiveInt
    linear_conv_kernel_dim: PositiveInt
    rms_norm_eps: PositiveFloat
    layer_types: list[str]
    rope_parameters: RotaryConfig
    tie_word_embeddings: bool
    hidden_act: str
    attn_output_gate: bool
    attention_bias: bool


def describe(artifact: MLXFormat) -> DenseDescription:
    text = TextConfig.model_validate(artifact.config["text_config"])
    rope = text.rope_parameters
    if (
        text.model_type != "qwen3_5_text"
        or text.hidden_act != "silu"
        or not text.attn_output_gate
        or text.attention_bias
        or text.linear_key_head_dim != text.linear_value_head_dim
        or rope.rope_type != "default"
        or not rope.mrope_interleaved
    ):
        raise ValueError("unsupported dense Qwen definition")
    kind = {"full_attention": MixerKind.ATTENTION, "linear_attention": MixerKind.RECURRENT}
    if len(text.layer_types) != text.num_hidden_layers or any(
        x not in kind for x in text.layer_types
    ):
        raise ValueError("invalid Qwen layer order")
    g = Geometry(
        hidden=text.hidden_size,
        intermediate=text.intermediate_size,
        vocabulary=text.vocab_size,
        context_limit=text.max_position_embeddings,
        layers=tuple(kind[x] for x in text.layer_types),
        attention_heads=text.num_attention_heads,
        kv_heads=text.num_key_value_heads,
        attention_width=text.head_dim,
        rotary_width=int(text.head_dim * rope.partial_rotary_factor),
        rotary_base=rope.rope_theta,
        rotary_sections=(*rope.mrope_section, 0),
        epsilon=text.rms_norm_eps,
        convolution_width=text.linear_conv_kernel_dim,
        recurrent_key_heads=text.linear_num_key_heads,
        recurrent_value_heads=text.linear_num_value_heads,
        recurrent_width=text.linear_key_head_dim,
        recurrent_head_mapping=HeadMapping.GROUPED,
    )
    prefix = "language_model.model."

    def weight(name, shape):
        return artifact.descriptor(prefix + name, shape)

    embedding = weight("embed_tokens.weight", (g.vocabulary, g.hidden))
    blocks = []
    for i, mixer_kind in enumerate(g.layers):
        p = f"layers.{i}."
        if mixer_kind == MixerKind.ATTENTION:
            a = p + "self_attn."
            mixer = AttentionWeights(
                query_gate=weight(
                    a + "q_proj.weight", (2 * g.attention_heads * g.attention_width, g.hidden)
                ),
                key=weight(a + "k_proj.weight", (g.kv_heads * g.attention_width, g.hidden)),
                value=weight(a + "v_proj.weight", (g.kv_heads * g.attention_width, g.hidden)),
                query_norm=weight(a + "q_norm.weight", (g.attention_width,)),
                key_norm=weight(a + "k_norm.weight", (g.attention_width,)),
                output=weight(
                    a + "o_proj.weight", (g.hidden, g.attention_heads * g.attention_width)
                ),
            )
        else:
            a = p + "linear_attn."
            mixer = RecurrentWeights(
                query_key_value=weight(a + "in_proj_qkv.weight", (g.recurrent_channels, g.hidden)),
                gate=weight(
                    a + "in_proj_z.weight", (g.recurrent_value_heads * g.recurrent_width, g.hidden)
                ),
                alpha=weight(a + "in_proj_a.weight", (g.recurrent_value_heads, g.hidden)),
                beta=weight(a + "in_proj_b.weight", (g.recurrent_value_heads, g.hidden)),
                convolution=weight(
                    a + "conv1d.weight", (g.recurrent_channels, g.convolution_width)
                ),
                decay=weight(a + "A_log", (g.recurrent_value_heads,)).model_copy(
                    update={"transform": WeightTransform.NEGATIVE_EXP}
                ),
                time_bias=weight(a + "dt_bias", (g.recurrent_value_heads,)),
                norm=weight(a + "norm.weight", (g.recurrent_width,)),
                output=weight(
                    a + "out_proj.weight", (g.hidden, g.recurrent_value_heads * g.recurrent_width)
                ),
            )
        blocks.append(
            BlockWeights(
                input_norm=weight(p + "input_layernorm.weight", (g.hidden,)),
                mixer=mixer,
                feedforward_norm=weight(p + "post_attention_layernorm.weight", (g.hidden,)),
                feedforward_gate=weight(p + "mlp.gate_proj.weight", (g.intermediate, g.hidden)),
                feedforward_up=weight(p + "mlp.up_proj.weight", (g.intermediate, g.hidden)),
                feedforward_down=weight(p + "mlp.down_proj.weight", (g.hidden, g.intermediate)),
            )
        )
    return DenseDescription(
        artifact_identity=artifact.identity,
        geometry=g,
        embedding=embedding,
        output_norm=weight("norm.weight", (g.hidden,)),
        output=embedding
        if text.tie_word_embeddings
        else artifact.descriptor("language_model.lm_head.weight", (g.vocabulary, g.hidden)),
        blocks=tuple(blocks),
    )
