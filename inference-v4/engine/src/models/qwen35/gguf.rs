//! The boundary between Qwen architecture roles and the GGUF container.
use super::*;
use crate::weights::{
    descriptor::Transform,
    gguf::{Directory, Scalar, Value},
};
use std::collections::HashSet;
struct Metadata<'a> {
    directory: &'a Directory,
    prefix: String,
}
impl Metadata<'_> {
    fn value(&self, key: &str) -> Result<&Value, Error> {
        self.directory
            .value(&format!("{}{key}", self.prefix))
            .ok_or_else(|| invalid(format!("missing Qwen metadata {}{key}", self.prefix)))
    }
    fn optional(&self, key: &str) -> Option<&Value> {
        self.directory.value(&format!("{}{key}", self.prefix))
    }
    fn integer(&self, key: &str) -> Result<u64, Error> {
        self.value(key)?
            .unsigned()
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid(format!("{}{key} must be a positive integer", self.prefix)))
    }
    fn number(&self, key: &str) -> Result<f64, Error> {
        match self.value(key)? {
            Value::Scalar(Scalar::Float(n)) => Ok(*n),
            Value::Scalar(Scalar::Unsigned(n)) => Ok(*n as f64),
            Value::Scalar(Scalar::Signed(n)) => Ok(*n as f64),
            _ => Err(invalid(format!("{}{key} must be numeric", self.prefix))),
        }
    }
}
pub fn inspect(directory: &Directory, identity: ArtifactIdentity) -> Result<Description, Error> {
    let architecture = directory
        .value("general.architecture")
        .and_then(Value::string)
        .ok_or_else(|| invalid("missing GGUF architecture"))?;
    let routed = match architecture {
        "qwen35" => false,
        "qwen35moe" => true,
        _ => {
            return Err(invalid(format!(
                "unsupported Qwen GGUF architecture {architecture:?}"
            )))
        }
    };
    let m = Metadata {
        directory,
        prefix: format!("{architecture}."),
    };
    if m.optional("rope.scaling.type")
        .is_some_and(|v| v.string() != Some("none"))
    {
        return Err(invalid(
            "scaled Qwen rotary definitions are not yet qualified",
        ));
    }
    let block_count = m.integer("block_count")?;
    let speculative = match m.optional("nextn_predict_layers") {
        None => 0,
        Some(v) => v
            .unsigned()
            .filter(|n| *n < block_count)
            .ok_or_else(|| invalid("nextn_predict_layers must be below block count"))?,
    };
    let layer_count = block_count - speculative;
    // Each real layer requires distinct stored weights. Reject impossible counts
    // before allocating architecture vectors from untrusted metadata.
    if layer_count > directory.tensors.len() as u64 {
        return Err(invalid("Qwen layer count exceeds available weight roles"));
    }
    let layers = match m.optional("attention.recurrent_layers") {
        None => {
            let interval = m.integer("full_attention_interval")?;
            (0..layer_count)
                .map(|i| {
                    if (i + 1) % interval == 0 {
                        MixerKind::Attention
                    } else {
                        MixerKind::Recurrent
                    }
                })
                .collect::<Vec<_>>()
        }
        Some(Value::Array(flags)) if flags.len() as u64 == layer_count => flags
            .iter()
            .map(|v| match v {
                Scalar::Bool(true) => Ok(MixerKind::Recurrent),
                Scalar::Bool(false) => Ok(MixerKind::Attention),
                _ => Err(invalid("invalid per-layer recurrent flags")),
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(invalid("invalid per-layer recurrent flags")),
    };
    let embedding = directory
        .tensor("token_embd.weight")
        .ok_or_else(|| invalid("missing Qwen embedding"))?;
    if embedding.shape.len() != 2 {
        return Err(invalid("Qwen embedding must be a matrix"));
    }
    if let Some(v) = directory.value("tokenizer.ggml.tokens") {
        if !matches!(v,Value::Array(a) if a.len() as u64==embedding.shape[0]) {
            return Err(invalid("tokenizer vocabulary differs from embedding table"));
        }
    }
    let Value::Array(sections) = m.value("rope.dimension_sections")? else {
        return Err(invalid("Qwen rotary sections must contain four integers"));
    };
    let sections = sections
        .iter()
        .map(|v| match v {
            Scalar::Unsigned(n) => Some(*n),
            Scalar::Signed(n) => u64::try_from(*n).ok(),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .and_then(|v| <[u64; 4]>::try_from(v).ok())
        .ok_or_else(|| invalid("Qwen rotary sections must contain four nonnegative integers"))?;
    let experts = if routed {
        Some(ExpertGeometry {
            count: m.integer("expert_count")?,
            selected: m.integer("expert_used_count")?,
            intermediate: m.integer("expert_feed_forward_length")?,
            shared_intermediate: m.integer("expert_shared_feed_forward_length")?,
            normalize_selected: true,
        })
    } else {
        None
    };
    let g = Geometry {
        activation_dtype: DType::BF16,
        hidden: m.integer("embedding_length")?,
        intermediate: if let Some(e) = &experts {
            e.intermediate
        } else {
            m.integer("feed_forward_length")?
        },
        vocabulary: embedding.shape[0],
        context_limit: m.integer("context_length")?,
        layers,
        attention_heads: m.integer("attention.head_count")?,
        kv_heads: m.integer("attention.head_count_kv")?,
        attention_width: m.integer("attention.key_length")?,
        rotary_width: m.integer("rope.dimension_count")?,
        rotary_base: m.number("rope.freq_base")?,
        rotary_sections: sections,
        epsilon: m.number("attention.layer_norm_rms_epsilon")?,
        convolution_width: m.integer("ssm.conv_kernel")?,
        recurrent_key_heads: m.integer("ssm.group_count")?,
        recurrent_value_heads: m.integer("ssm.time_step_rank")?,
        recurrent_width: m.integer("ssm.state_size")?,
        recurrent_head_mapping: HeadMapping::Tiled,
        experts,
    };
    g.validate()?;
    let recurrent_inner = product(&[g.recurrent_value_heads, g.recurrent_width])?;
    if m.integer("attention.value_length")? != g.attention_width
        || m.integer("ssm.inner_size")? != recurrent_inner
    {
        return Err(invalid(
            "Qwen attention/recurrent inner size differs from head geometry",
        ));
    }
    let mut consumed = HashSet::new();
    let mut tensor = |name: &str, shape: &[u64]| -> Result<WeightDescriptor, Error> {
        let value = directory
            .tensor(name)
            .ok_or_else(|| invalid(format!("missing Qwen weight {name:?}")))?;
        if value.shape != shape {
            return Err(invalid(format!(
                "Qwen weight {name:?}: expected {shape:?}, received {:?}",
                value.shape
            )));
        }
        product(shape)?;
        consumed.insert(name.to_string());
        Ok(WeightDescriptor {
            name: name.into(),
            shape: shape.into(),
            transform: Transform::Identity,
        })
    };
    let embedding = tensor(&embedding.name, &[g.vocabulary, g.hidden])?;
    let output_norm = tensor("output_norm.weight", &[g.hidden])?;
    let output = tensor(
        if directory.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            &embedding.name
        },
        &[g.vocabulary, g.hidden],
    )?;
    let mut blocks = Vec::new();
    for (i, kind) in g.layers.iter().enumerate() {
        let p = format!("blk.{i}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        let mixer = if *kind == MixerKind::Attention {
            MixerWeights::Attention(Box::new(AttentionWeights {
                query_gate: weight(
                    "attn_q.weight",
                    &[
                        product(&[2, g.attention_heads, g.attention_width])?,
                        g.hidden,
                    ],
                )?,
                key: weight(
                    "attn_k.weight",
                    &[product(&[g.kv_heads, g.attention_width])?, g.hidden],
                )?,
                value: weight(
                    "attn_v.weight",
                    &[product(&[g.kv_heads, g.attention_width])?, g.hidden],
                )?,
                query_norm: weight("attn_q_norm.weight", &[g.attention_width])?,
                key_norm: weight("attn_k_norm.weight", &[g.attention_width])?,
                output: weight(
                    "attn_output.weight",
                    &[g.hidden, product(&[g.attention_heads, g.attention_width])?],
                )?,
            }))
        } else {
            MixerWeights::Recurrent(Box::new(RecurrentWeights {
                query_key_value: weight("attn_qkv.weight", &[g.recurrent_channels()?, g.hidden])?,
                gate: weight("attn_gate.weight", &[recurrent_inner, g.hidden])?,
                alpha: weight("ssm_alpha.weight", &[g.recurrent_value_heads, g.hidden])?,
                beta: weight("ssm_beta.weight", &[g.recurrent_value_heads, g.hidden])?,
                convolution: weight(
                    "ssm_conv1d.weight",
                    &[g.recurrent_channels()?, g.convolution_width],
                )?,
                decay: weight("ssm_a", &[g.recurrent_value_heads])?,
                time_bias: weight("ssm_dt.bias", &[g.recurrent_value_heads])?,
                norm: weight("ssm_norm.weight", &[g.recurrent_width])?,
                output: weight("ssm_out.weight", &[g.hidden, recurrent_inner])?,
            }))
        };
        let feedforward = if let Some(e) = &g.experts {
            FeedForwardWeights::Routed(Box::new(RoutedFeedForwardWeights {
                router: weight("ffn_gate_inp.weight", &[e.count, g.hidden])?,
                shared_router: weight("ffn_gate_inp_shexp.weight", &[g.hidden])?,
                expert_gate: weight("ffn_gate_exps.weight", &[e.count, e.intermediate, g.hidden])?,
                expert_up: weight("ffn_up_exps.weight", &[e.count, e.intermediate, g.hidden])?,
                expert_down: weight("ffn_down_exps.weight", &[e.count, g.hidden, e.intermediate])?,
                shared_gate: weight("ffn_gate_shexp.weight", &[e.shared_intermediate, g.hidden])?,
                shared_up: weight("ffn_up_shexp.weight", &[e.shared_intermediate, g.hidden])?,
                shared_down: weight("ffn_down_shexp.weight", &[g.hidden, e.shared_intermediate])?,
            }))
        } else {
            FeedForwardWeights::Dense(Box::new(DenseFeedForwardWeights {
                gate: weight("ffn_gate.weight", &[g.intermediate, g.hidden])?,
                up: weight("ffn_up.weight", &[g.intermediate, g.hidden])?,
                down: weight("ffn_down.weight", &[g.hidden, g.intermediate])?,
            }))
        };
        blocks.push(BlockWeights {
            input_norm: weight("attn_norm.weight", &[g.hidden])?,
            mixer,
            feedforward_norm: weight("post_attention_norm.weight", &[g.hidden])?,
            feedforward,
        });
    }
    for t in &directory.tensors {
        if consumed.contains(&t.name) {
            continue;
        }
        let speculative_role = t
            .name
            .strip_prefix("blk.")
            .and_then(|s| s.split_once('.'))
            .and_then(|(i, _)| i.parse::<u64>().ok().filter(|n| i == n.to_string()))
            .is_some_and(|i| i >= layer_count && i < block_count);
        if !speculative_role {
            return Err(invalid(format!(
                "Qwen artifact contains unbound weight role {:?}",
                t.name
            )));
        }
    }
    Ok(Description {
        artifact_identity: identity,
        geometry: g,
        embedding,
        output_norm,
        output,
        blocks,
    })
}
