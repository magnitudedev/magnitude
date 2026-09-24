//! Qwen 3.5 artifact recognition and family-specific numerical interpretation.
//!
//! This adapter contains only facts that can vary across model families. It
//! recognizes Qwen 3.5 GGUF metadata and binds stored tensor names to the
//! family-neutral numerical contracts consumed by the engine.

pub mod inputs;

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    Package, PackageIdentity,
};
use magnitude_model_contracts::{
    checked_product, ActivationDType, AttentionGeometry, AttentionWeights, BlockGeometry,
    BlockWeights, DecoderGeometry, DenseFeedForwardWeights, ExpertGeometry, FamilyId,
    FeedForwardGeometry, FeedForwardWeights, FusedQkvWeights, HeadBlock, HeadWeights,
    InputSemantics, LayerNormWeights, MixerGeometry, MixerWeights, ModelDefinition,
    RecurrentGeometry, RecurrentHeadMapping, RecurrentWeights, RotarySemantics,
    RoutedFeedForwardWeights, RowRange, TextCoordinateSemantics, VisionActivation,
    VisionAttentionWeights, VisionBlockWeights, VisionDescription, VisionFeedForwardWeights,
    VisionGeometry, VisionMergerWeights, VisionPreprocessing, WeightDescriptor,
};
use std::{collections::HashSet, error, fmt};

/// Qwen 3.5 architecture recognized from literal artifact metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Dense,
    Routed,
}

impl Architecture {
    pub const fn family_id(self) -> &'static str {
        match self {
            Self::Dense => "qwen35",
            Self::Routed => "qwen35moe",
        }
    }

    const fn is_routed(self) -> bool {
        matches!(self, Self::Routed)
    }
}

/// Failure to recognize or interpret a Qwen 3.5 artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for Error {}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

fn product(dimensions: &[u64]) -> Result<u64, Error> {
    checked_product(dimensions).map_err(|error| invalid(error.to_string()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalMixerKind {
    Attention,
    Recurrent,
}
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

fn positive(directory: &Directory, key: &str) -> Result<u64, Error> {
    directory
        .value(key)
        .and_then(Value::unsigned)
        .filter(|value| *value > 0)
        .ok_or_else(|| invalid(format!("{key} must be a positive integer")))
}

fn number(directory: &Directory, key: &str) -> Result<f64, Error> {
    let value = match directory.value(key) {
        Some(Value::Scalar(Scalar::Float(value))) => *value,
        Some(Value::Scalar(Scalar::Unsigned(value))) => *value as f64,
        Some(Value::Scalar(Scalar::Signed(value))) => *value as f64,
        _ => return Err(invalid(format!("{key} must be numeric"))),
    };
    if !value.is_finite() {
        return Err(invalid(format!("{key} must be finite")));
    }
    Ok(value)
}

fn triple(directory: &Directory, key: &str) -> Result<[f64; 3], Error> {
    let Some(Value::Array(values)) = directory.value(key) else {
        return Err(invalid(format!("{key} must contain three numbers")));
    };
    let values = values
        .iter()
        .map(|value| match value {
            Scalar::Float(value) if value.is_finite() => Some(*value),
            Scalar::Unsigned(value) => Some(*value as f64),
            Scalar::Signed(value) => Some(*value as f64),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid(format!("{key} must contain three finite numbers")))?;
    <[f64; 3]>::try_from(values).map_err(|_| invalid(format!("{key} must contain three numbers")))
}

/// Interpret a validated Qwen3-VL projector inventory without importing it.
pub fn describe_projector(directory: &Directory) -> Result<VisionDescription, Error> {
    if directory
        .value("general.architecture")
        .and_then(Value::string)
        != Some("clip")
        || directory.value("general.type").and_then(Value::string) != Some("mmproj")
        || directory
            .value("clip.has_vision_encoder")
            .and_then(|value| match value {
                Value::Scalar(Scalar::Bool(value)) => Some(*value),
                _ => None,
            })
            != Some(true)
        || directory
            .value("clip.projector_type")
            .and_then(Value::string)
            != Some("qwen3vl_merger")
    {
        return Err(invalid("unsupported Qwen3-VL projector identity"));
    }
    let depth = positive(directory, "clip.vision.block_count")?;
    let hidden = positive(directory, "clip.vision.embedding_length")?;
    let intermediate = positive(directory, "clip.vision.feed_forward_length")?;
    let heads = positive(directory, "clip.vision.attention.head_count")?;
    let patch = positive(directory, "clip.vision.patch_size")?;
    let merge = positive(directory, "clip.vision.spatial_merge_size")?;
    let image_size = positive(directory, "clip.vision.image_size")?;
    let output_hidden = positive(directory, "clip.vision.projection_dim")?;
    let epsilon = number(directory, "clip.vision.attention.layer_norm_epsilon")?;
    let mean = triple(directory, "clip.vision.image_mean")?;
    let std = triple(directory, "clip.vision.image_std")?;
    let pixel_bound = |key: &str, default: u64| -> Result<u64, Error> {
        match directory.value(key) {
            None => Ok(default),
            Some(value) => value
                .unsigned()
                .filter(|bound| *bound > 0)
                .ok_or_else(|| invalid(format!("{key} must be a positive integer"))),
        }
    };
    let min_pixels = pixel_bound("clip.vision.image_min_pixels", 65_536)?;
    let max_pixels = pixel_bound("clip.vision.image_max_pixels", 16_777_216)?;
    if let Some(value) = directory.value("clip.use_gelu") {
        if !matches!(value, Value::Scalar(Scalar::Bool(_))) {
            return Err(invalid("clip.use_gelu must be boolean"));
        }
    }
    if let Some(deepstack) = directory.value("clip.vision.is_deepstack_layers") {
        if !matches!(deepstack, Value::Array(values) if values.len() as u64 == depth && values.iter().all(|value| matches!(value, Scalar::Bool(false))))
        {
            return Err(invalid("deepstack projector layers are unsupported"));
        }
    }

    let mut consumed = HashSet::new();
    let mut tensor = |name: &str, shape: &[u64]| -> Result<WeightDescriptor, Error> {
        let value = directory
            .tensor(name)
            .ok_or_else(|| invalid(format!("missing projector weight {name:?}")))?;
        if value.shape != shape {
            return Err(invalid(format!(
                "projector weight {name:?}: expected {shape:?}, received {:?}",
                value.shape
            )));
        }
        product(shape)?;
        consumed.insert(name.to_string());
        Ok(WeightDescriptor {
            name: name.into(),
            shape: shape.into(),
        })
    };
    let patch_embeddings = ["v.patch_embd.weight", "v.patch_embd.weight.1"]
        .into_iter()
        .map(|name| tensor(name, &[hidden, 3, patch, patch]))
        .collect::<Result<Vec<_>, _>>()?;
    let patch_bias = tensor("v.patch_embd.bias", &[hidden])?;
    let position = directory
        .tensor("v.position_embd.weight")
        .ok_or_else(|| invalid("missing projector position embedding"))?;
    if position.shape.len() != 2 || position.shape[1] != hidden {
        return Err(invalid("invalid projector position embedding shape"));
    }
    let table_rows = position.shape[0];
    let table_side = (table_rows as f64).sqrt() as u64;
    if table_side.checked_mul(table_side) != Some(table_rows) {
        return Err(invalid("projector position table is not square"));
    }
    let position_embedding = tensor("v.position_embd.weight", &[table_rows, hidden])?;
    let qkv_rows = product(&[3, hidden])?;
    let norm = |weight: WeightDescriptor, bias: WeightDescriptor| LayerNormWeights { weight, bias };
    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        blocks.push(VisionBlockWeights {
            input_norm: norm(
                weight("ln1.weight", &[hidden])?,
                weight("ln1.bias", &[hidden])?,
            ),
            attention: VisionAttentionWeights {
                qkv: FusedQkvWeights {
                    weight: weight("attn_qkv.weight", &[qkv_rows, hidden])?,
                    bias: weight("attn_qkv.bias", &[qkv_rows])?,
                    query: RowRange {
                        start: 0,
                        rows: hidden,
                    },
                    key: RowRange {
                        start: hidden,
                        rows: hidden,
                    },
                    value: RowRange {
                        start: hidden * 2,
                        rows: hidden,
                    },
                },
                output: weight("attn_out.weight", &[hidden, hidden])?,
                output_bias: weight("attn_out.bias", &[hidden])?,
            },
            feedforward_norm: norm(
                weight("ln2.weight", &[hidden])?,
                weight("ln2.bias", &[hidden])?,
            ),
            feedforward: VisionFeedForwardWeights {
                up: weight("ffn_up.weight", &[intermediate, hidden])?,
                up_bias: weight("ffn_up.bias", &[intermediate])?,
                down: weight("ffn_down.weight", &[hidden, intermediate])?,
                down_bias: weight("ffn_down.bias", &[hidden])?,
            },
        });
    }
    let output_norm = norm(
        tensor("v.post_ln.weight", &[hidden])?,
        tensor("v.post_ln.bias", &[hidden])?,
    );
    let merger_hidden = product(&[hidden, merge, merge])?;
    let merger = VisionMergerWeights {
        hidden: tensor("mm.0.weight", &[merger_hidden, merger_hidden])?,
        hidden_bias: tensor("mm.0.bias", &[merger_hidden])?,
        output: tensor("mm.2.weight", &[output_hidden, merger_hidden])?,
        output_bias: tensor("mm.2.bias", &[output_hidden])?,
    };
    for value in &directory.tensors {
        if !consumed.contains(&value.name) {
            return Err(invalid(format!(
                "projector contains unbound weight role {:?}",
                value.name
            )));
        }
    }
    let description = VisionDescription {
        geometry: VisionGeometry {
            // f16 intermediates around an F32 residual stream: 1.9% relative
            // RMS from an F64 forward of the 4B projector, where bf16 puts
            // the features 12.7% away (see the vision kernel family).
            activation_dtype: ActivationDType::F16,
            depth,
            hidden,
            intermediate,
            heads,
            patch,
            merge,
            image_size,
            output_hidden,
            epsilon,
            table_side,
            temporal_patch: patch_embeddings.len() as u64,
            channels: 3,
            block_activation: VisionActivation::GeluTanh,
            merger_activation: VisionActivation::GeluErf,
        },
        preprocessing: VisionPreprocessing {
            processor: "qwen3vl_merger".into(),
            mean,
            std,
            min_pixels,
            max_pixels,
        },
        patch_embeddings,
        patch_bias,
        position_embedding,
        blocks,
        output_norm,
        merger,
    };
    description
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(description)
}

/// Recognize the supported Qwen 3.5 architecture without interpreting its
/// geometry or tensor roles.
pub fn recognize(directory: &Directory) -> Result<Architecture, Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
        .ok_or_else(|| invalid("missing GGUF architecture"))?
    {
        "qwen35" => Ok(Architecture::Dense),
        "qwen35moe" => Ok(Architecture::Routed),
        architecture => Err(invalid(format!(
            "unsupported Qwen GGUF architecture {architecture:?}"
        ))),
    }
}

pub fn inspect_package(package: &Package) -> Result<ModelDefinition, Error> {
    inspect_components(
        package.target().directory(),
        package.projector().map(|projector| projector.directory()),
        package.identity(),
    )
}

/// Inspect already-opened package components. This exists for composition roots
/// and header-only fixtures; component ownership remains with `Package`.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, Error> {
    let architecture = recognize(directory)?;
    let family_id = architecture.family_id();
    let m = Metadata {
        directory,
        prefix: format!("{family_id}."),
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
    if speculative > 1 {
        return Err(invalid(
            "Qwen artifacts with more than one nextn prediction layer are not admitted until their GGUF roles and execution semantics are qualified",
        ));
    }
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
                        LocalMixerKind::Attention
                    } else {
                        LocalMixerKind::Recurrent
                    }
                })
                .collect::<Vec<_>>()
        }
        Some(Value::Array(flags)) if flags.len() as u64 == layer_count => flags
            .iter()
            .map(|v| match v {
                Scalar::Bool(true) => Ok(LocalMixerKind::Recurrent),
                Scalar::Bool(false) => Ok(LocalMixerKind::Attention),
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
    if sections[3] != 0 {
        return Err(invalid("Qwen rotary fourth section must be zero"));
    }
    let experts = if architecture.is_routed() {
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
    let hidden = m.integer("embedding_length")?;
    let intermediate = if let Some(experts) = &experts {
        experts.intermediate
    } else {
        m.integer("feed_forward_length")?
    };
    let attention_heads = m.integer("attention.head_count")?;
    let kv_heads = m.integer("attention.head_count_kv")?;
    let attention_width = m.integer("attention.key_length")?;
    let recurrent_key_heads = m.integer("ssm.group_count")?;
    let recurrent_value_heads = m.integer("ssm.time_step_rank")?;
    let recurrent_width = m.integer("ssm.state_size")?;
    let recurrent_inner = product(&[recurrent_value_heads, recurrent_width])?;
    if m.integer("attention.value_length")? != attention_width
        || m.integer("ssm.inner_size")? != recurrent_inner
    {
        return Err(invalid(
            "Qwen attention/recurrent inner size differs from head geometry",
        ));
    }
    let attention = AttentionGeometry {
        heads: attention_heads,
        kv_heads,
        width: attention_width,
        rotary: RotarySemantics::Interleaved {
            width: m.integer("rope.dimension_count")?,
            base: m.number("rope.freq_base")?,
            sections: sections.into(),
            axis_pattern: vec![0, 1, 2],
        },
    };
    let recurrent = RecurrentGeometry {
        convolution_width: m.integer("ssm.conv_kernel")?,
        key_heads: recurrent_key_heads,
        value_heads: recurrent_value_heads,
        width: recurrent_width,
        head_mapping: RecurrentHeadMapping::Tiled,
    };
    let feedforward = match &experts {
        Some(experts) => FeedForwardGeometry::Routed(experts.clone()),
        None => FeedForwardGeometry::Dense { intermediate },
    };
    let block_geometry = layers
        .iter()
        .map(|kind| BlockGeometry {
            mixer: match kind {
                LocalMixerKind::Attention => MixerGeometry::Attention(attention.clone()),
                LocalMixerKind::Recurrent => MixerGeometry::Recurrent(recurrent.clone()),
            },
            feedforward: feedforward.clone(),
        })
        .collect();
    let g = DecoderGeometry {
        activation_dtype: ActivationDType::BF16,
        hidden,
        vocabulary: embedding.shape[0],
        context_limit: m.integer("context_length")?,
        epsilon: m.number("attention.layer_norm_rms_epsilon")?,
        blocks: block_geometry,
    };
    g.validate().map_err(|error| invalid(error.to_string()))?;
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
    for (i, kind) in layers.iter().enumerate() {
        let p = format!("blk.{i}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        let mixer = if *kind == LocalMixerKind::Attention {
            MixerWeights::Attention(Box::new(AttentionWeights {
                query_gate: weight(
                    "attn_q.weight",
                    &[product(&[2, attention_heads, attention_width])?, g.hidden],
                )?,
                key: weight(
                    "attn_k.weight",
                    &[product(&[kv_heads, attention_width])?, g.hidden],
                )?,
                value: weight(
                    "attn_v.weight",
                    &[product(&[kv_heads, attention_width])?, g.hidden],
                )?,
                query_norm: weight("attn_q_norm.weight", &[attention_width])?,
                key_norm: weight("attn_k_norm.weight", &[attention_width])?,
                output: weight(
                    "attn_output.weight",
                    &[g.hidden, product(&[attention_heads, attention_width])?],
                )?,
            }))
        } else {
            MixerWeights::Recurrent(Box::new(RecurrentWeights {
                query_key_value: weight(
                    "attn_qkv.weight",
                    &[
                        recurrent
                            .channels()
                            .map_err(|error| invalid(error.to_string()))?,
                        g.hidden,
                    ],
                )?,
                gate: weight("attn_gate.weight", &[recurrent_inner, g.hidden])?,
                alpha: weight("ssm_alpha.weight", &[recurrent_value_heads, g.hidden])?,
                beta: weight("ssm_beta.weight", &[recurrent_value_heads, g.hidden])?,
                convolution: weight(
                    "ssm_conv1d.weight",
                    &[
                        recurrent
                            .channels()
                            .map_err(|error| invalid(error.to_string()))?,
                        recurrent.convolution_width,
                    ],
                )?,
                decay: weight("ssm_a", &[recurrent_value_heads])?,
                time_bias: weight("ssm_dt.bias", &[recurrent_value_heads])?,
                norm: weight("ssm_norm.weight", &[recurrent_width])?,
                output: weight("ssm_out.weight", &[g.hidden, recurrent_inner])?,
            }))
        };
        let feedforward = if let Some(e) = &experts {
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
                gate: weight("ffn_gate.weight", &[intermediate, g.hidden])?,
                up: weight("ffn_up.weight", &[intermediate, g.hidden])?,
                down: weight("ffn_down.weight", &[g.hidden, intermediate])?,
            }))
        };
        blocks.push(BlockWeights {
            input_norm: weight("attn_norm.weight", &[g.hidden])?,
            mixer,
            feedforward_norm: weight("post_attention_norm.weight", &[g.hidden])?,
            feedforward,
        });
    }
    let head = if speculative == 0 {
        None
    } else {
        let head_intermediate = m.integer("feed_forward_length")?;
        let mut head_blocks = Vec::with_capacity(speculative as usize);
        for offset in 0..speculative {
            let index = layer_count + offset;
            let p = format!("blk.{index}.");
            let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
            head_blocks.push(HeadBlock {
                embedding_norm: weight("nextn.enorm.weight", &[g.hidden])?,
                hidden_norm: weight("nextn.hnorm.weight", &[g.hidden])?,
                combine: weight(
                    "nextn.eh_proj.weight",
                    &[g.hidden, product(&[2, g.hidden])?],
                )?,
                input_norm: weight("attn_norm.weight", &[g.hidden])?,
                attention: AttentionWeights {
                    query_gate: weight(
                        "attn_q.weight",
                        &[product(&[2, attention_heads, attention_width])?, g.hidden],
                    )?,
                    key: weight(
                        "attn_k.weight",
                        &[product(&[kv_heads, attention_width])?, g.hidden],
                    )?,
                    value: weight(
                        "attn_v.weight",
                        &[product(&[kv_heads, attention_width])?, g.hidden],
                    )?,
                    query_norm: weight("attn_q_norm.weight", &[attention_width])?,
                    key_norm: weight("attn_k_norm.weight", &[attention_width])?,
                    output: weight(
                        "attn_output.weight",
                        &[g.hidden, product(&[attention_heads, attention_width])?],
                    )?,
                },
                feedforward_norm: weight("post_attention_norm.weight", &[g.hidden])?,
                feedforward: DenseFeedForwardWeights {
                    gate: weight("ffn_gate.weight", &[head_intermediate, g.hidden])?,
                    up: weight("ffn_up.weight", &[head_intermediate, g.hidden])?,
                    down: weight("ffn_down.weight", &[g.hidden, head_intermediate])?,
                },
                output_norm: weight("nextn.shared_head_norm.weight", &[g.hidden])?,
            });
        }
        Some(HeadWeights {
            blocks: head_blocks,
        })
    };
    for t in &directory.tensors {
        if consumed.contains(&t.name) {
            continue;
        }
        return Err(invalid(format!(
            "Qwen artifact contains unbound weight role {:?}",
            t.name
        )));
    }
    if let Some(projector) = projector {
        // A derived target (a quantization, a fine-tune) names its base model
        // exactly as the projector does; an original model is its own base.
        let target_base = directory
            .value("general.base_model.0.name")
            .and_then(Value::string)
            .or_else(|| directory.value("general.name").and_then(Value::string));
        let projector_base = projector
            .value("general.base_model.0.name")
            .and_then(Value::string);
        if target_base
            .zip(projector_base)
            .is_some_and(|(target, projected)| target != projected)
        {
            return Err(invalid("projector base model differs from target"));
        }
    }
    let vision = projector.map(describe_projector).transpose()?;
    let definition = ModelDefinition {
        family: FamilyId(family_id.into()),
        artifact_identity: identity,
        inputs: InputSemantics {
            text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
            coordinate_axes: 3,
        },
        geometry: g,
        embedding,
        output_norm,
        output,
        blocks,
        head,
        vision,
    };
    definition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(definition)
}
