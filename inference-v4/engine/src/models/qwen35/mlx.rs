//! MLX dense Qwen roles, preserving V3's grouped recurrent heads and A_log import.
use super::*;
use crate::weights::{descriptor::Transform, mlx::MlxArtifact};
use serde::Deserialize;
#[derive(Deserialize)]
struct RotaryConfig {
    rope_theta: f64,
    partial_rotary_factor: f64,
    rope_type: String,
    mrope_section: [u64; 3],
    mrope_interleaved: bool,
}
#[derive(Deserialize)]
struct TextConfig {
    model_type: String,
    hidden_size: u64,
    intermediate_size: u64,
    vocab_size: u64,
    max_position_embeddings: u64,
    num_hidden_layers: usize,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    head_dim: u64,
    linear_num_key_heads: u64,
    linear_num_value_heads: u64,
    linear_key_head_dim: u64,
    linear_value_head_dim: u64,
    linear_conv_kernel_dim: u64,
    rms_norm_eps: f64,
    layer_types: Vec<String>,
    rope_parameters: RotaryConfig,
    tie_word_embeddings: bool,
    hidden_act: String,
    attn_output_gate: bool,
    attention_bias: bool,
}
pub fn describe(artifact: &MlxArtifact) -> Result<Description, Error> {
    let text: TextConfig = serde_json::from_value(
        artifact
            .config()
            .get("text_config")
            .ok_or_else(|| invalid("missing Qwen text_config"))?
            .clone(),
    )
    .map_err(|e| invalid(format!("invalid Qwen text configuration: {e}")))?;
    let rope = &text.rope_parameters;
    if text.model_type != "qwen3_5_text"
        || text.hidden_act != "silu"
        || !text.attn_output_gate
        || text.attention_bias
        || text.linear_key_head_dim != text.linear_value_head_dim
        || rope.rope_type != "default"
        || !rope.mrope_interleaved
    {
        return Err(invalid("unsupported dense Qwen definition"));
    }
    if text.layer_types.len() != text.num_hidden_layers {
        return Err(invalid("invalid Qwen layer order"));
    }
    let layers = text
        .layer_types
        .iter()
        .map(|s| match s.as_str() {
            "full_attention" => Ok(MixerKind::Attention),
            "linear_attention" => Ok(MixerKind::Recurrent),
            _ => Err(invalid("invalid Qwen layer order")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rotary_width = text.head_dim as f64 * rope.partial_rotary_factor;
    if !rope.partial_rotary_factor.is_finite()
        || rope.partial_rotary_factor <= 0.0
        || !rotary_width.is_finite()
        || rotary_width >= u64::MAX as f64
    {
        return Err(invalid("invalid Qwen partial rotary factor"));
    }
    let g = Geometry {
        activation_dtype: DType::BF16,
        hidden: text.hidden_size,
        intermediate: text.intermediate_size,
        vocabulary: text.vocab_size,
        context_limit: text.max_position_embeddings,
        layers,
        attention_heads: text.num_attention_heads,
        kv_heads: text.num_key_value_heads,
        attention_width: text.head_dim,
        rotary_width: rotary_width as u64,
        rotary_base: rope.rope_theta,
        rotary_sections: [
            rope.mrope_section[0],
            rope.mrope_section[1],
            rope.mrope_section[2],
            0,
        ],
        epsilon: text.rms_norm_eps,
        convolution_width: text.linear_conv_kernel_dim,
        recurrent_key_heads: text.linear_num_key_heads,
        recurrent_value_heads: text.linear_num_value_heads,
        recurrent_width: text.linear_key_head_dim,
        recurrent_head_mapping: HeadMapping::Grouped,
        experts: None,
    };
    g.validate()?;
    let recurrent_inner = product(&[g.recurrent_value_heads, g.recurrent_width])?;
    let weight = |name: &str, shape: &[u64]| {
        artifact.descriptor(&format!("language_model.model.{name}"), shape)
    };
    let embedding = weight("embed_tokens.weight", &[g.vocabulary, g.hidden])?;
    let mut blocks = Vec::new();
    for (i, kind) in g.layers.iter().enumerate() {
        let p = format!("layers.{i}.");
        let mixer = if *kind == MixerKind::Attention {
            let a = format!("{p}self_attn.");
            MixerWeights::Attention(Box::new(AttentionWeights {
                query_gate: weight(
                    &format!("{a}q_proj.weight"),
                    &[
                        product(&[2, g.attention_heads, g.attention_width])?,
                        g.hidden,
                    ],
                )?,
                key: weight(
                    &format!("{a}k_proj.weight"),
                    &[product(&[g.kv_heads, g.attention_width])?, g.hidden],
                )?,
                value: weight(
                    &format!("{a}v_proj.weight"),
                    &[product(&[g.kv_heads, g.attention_width])?, g.hidden],
                )?,
                query_norm: weight(&format!("{a}q_norm.weight"), &[g.attention_width])?,
                key_norm: weight(&format!("{a}k_norm.weight"), &[g.attention_width])?,
                output: weight(
                    &format!("{a}o_proj.weight"),
                    &[g.hidden, product(&[g.attention_heads, g.attention_width])?],
                )?,
            }))
        } else {
            let a = format!("{p}linear_attn.");
            let mut decay = weight(&format!("{a}A_log"), &[g.recurrent_value_heads])?;
            decay.transform = Transform::NegativeExp;
            MixerWeights::Recurrent(Box::new(RecurrentWeights {
                query_key_value: weight(
                    &format!("{a}in_proj_qkv.weight"),
                    &[g.recurrent_channels()?, g.hidden],
                )?,
                gate: weight(
                    &format!("{a}in_proj_z.weight"),
                    &[recurrent_inner, g.hidden],
                )?,
                alpha: weight(
                    &format!("{a}in_proj_a.weight"),
                    &[g.recurrent_value_heads, g.hidden],
                )?,
                beta: weight(
                    &format!("{a}in_proj_b.weight"),
                    &[g.recurrent_value_heads, g.hidden],
                )?,
                convolution: weight(
                    &format!("{a}conv1d.weight"),
                    &[g.recurrent_channels()?, g.convolution_width],
                )?,
                decay,
                time_bias: weight(&format!("{a}dt_bias"), &[g.recurrent_value_heads])?,
                norm: weight(&format!("{a}norm.weight"), &[g.recurrent_width])?,
                output: weight(&format!("{a}out_proj.weight"), &[g.hidden, recurrent_inner])?,
            }))
        };
        blocks.push(BlockWeights {
            input_norm: weight(&format!("{p}input_layernorm.weight"), &[g.hidden])?,
            mixer,
            feedforward_norm: weight(&format!("{p}post_attention_layernorm.weight"), &[g.hidden])?,
            feedforward: FeedForwardWeights::Dense(Box::new(DenseFeedForwardWeights {
                gate: weight(
                    &format!("{p}mlp.gate_proj.weight"),
                    &[g.intermediate, g.hidden],
                )?,
                up: weight(
                    &format!("{p}mlp.up_proj.weight"),
                    &[g.intermediate, g.hidden],
                )?,
                down: weight(
                    &format!("{p}mlp.down_proj.weight"),
                    &[g.hidden, g.intermediate],
                )?,
            })),
        });
    }
    Ok(Description {
        artifact_identity: artifact.identity(),
        output_norm: weight("norm.weight", &[g.hidden])?,
        output: if text.tie_word_embeddings {
            embedding.clone()
        } else {
            artifact.descriptor("language_model.lm_head.weight", &[g.vocabulary, g.hidden])?
        },
        geometry: g,
        embedding,
        blocks,
    })
}
