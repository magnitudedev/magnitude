//! Qwen 3.5 artifact recognition and family-specific numerical interpretation.
//!
//! This adapter contains only facts that can vary across model families. It
//! recognizes Qwen 3.5 GGUF metadata and binds stored tensor names to the
//! family-neutral numerical contracts consumed by the engine.

mod family;
pub mod inputs;

pub use family::Qwen35Family;

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    Package, PackageIdentity,
};
use magnitude_family_contracts::{
    checked_product, ActivationDType, ActivationFunction, Attention, AttentionGate, Block,
    CellReduction, Decoder, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm,
    ExpertSelection, FamilyId, FeedForwardUp, GateFunction, GatedDelta, Head, HeadBlock,
    HeadNorm, HistoryDomain, HistoryReads, ImportTransform, InputNorm, InputSemantics, KeyValue,
    MediaRowAttention, MergerStage, ModelDefinition, Operator, OutputForm, PositionSampling,
    RecurrentHeadMapping, ResidualForm, RmsNorm, Rotary, RouteNormalization, RoutedFfn, Router,
    RouterInput, RowRange, ScoreFunction, SharedExpert, SharedExpertGate, Sublayer,
    TextCoordinateSemantics, ValueNorm, ValueSource, VisionActivation, VisionAttention,
    VisionAttentionScale, VisionAttentionSpan, VisionBlock, VisionDescription, VisionFeedForward,
    VisionLinear, VisionMerger, VisionNorm, VisionPositions, VisionPreprocessing,
    VisionResampling, VisionResize, VisionStem, VisionUp, WeightDescriptor,
};
use magnitude_family_common::{HeaderError, Tensors};
use std::{error, fmt};

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

impl From<HeaderError> for Error {
    fn from(error: HeaderError) -> Self {
        invalid(error.to_string())
    }
}

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

/// Expert widths of a routed (`qwen35moe`) artifact.
struct Experts {
    count: u64,
    selected: u64,
    intermediate: u64,
    shared_intermediate: u64,
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

    let mut tensors = Tensors::new(directory);
    let mut tensor = |name: &str, shape: &[u64]| tensors.bind(name, shape);
    let patch_frames = ["v.patch_embd.weight", "v.patch_embd.weight.1"]
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
    if image_size.checked_rem(patch) != Some(0) || image_size / patch != table_side {
        return Err(invalid(
            "projector position table does not cover the nominal image",
        ));
    }
    let table = tensor("v.position_embd.weight", &[table_rows, hidden])?;
    if heads == 0 || hidden % heads != 0 || (hidden / heads) % 4 != 0 {
        return Err(invalid("projector heads do not split into 2-D rotary heads"));
    }
    let layer_norm = |weight: WeightDescriptor, bias: WeightDescriptor| VisionNorm::Layer {
        weight,
        bias,
        epsilon,
    };
    // One stored QKV projection, imported as three row ranges.
    let rows = |weight: &WeightDescriptor, part: u64| WeightDescriptor {
        name: weight.name.clone(),
        shape: {
            let mut shape = weight.shape.clone();
            shape[0] = hidden;
            shape
        },
        transforms: vec![ImportTransform::Rows(RowRange {
            start: part * hidden,
            rows: hidden,
        })],
    };
    let qkv_rows = product(&[3, hidden])?;
    let mut blocks = Vec::with_capacity(depth as usize);
    for index in 0..depth {
        let p = format!("v.blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
        let qkv = weight("attn_qkv.weight", &[qkv_rows, hidden])?;
        let qkv_bias = weight("attn_qkv.bias", &[qkv_rows])?;
        let split = |part| VisionLinear {
            weight: rows(&qkv, part),
            bias: Some(rows(&qkv_bias, part)),
            clamp: None,
        };
        blocks.push(VisionBlock {
            attention_norm: layer_norm(
                weight("ln1.weight", &[hidden])?,
                weight("ln1.bias", &[hidden])?,
            ),
            attention: VisionAttention {
                heads,
                width: hidden / heads,
                query: split(0),
                key: split(1),
                value: split(2),
                query_norm: None,
                key_norm: None,
                value_norm: None,
                rotary_base: 10_000.0,
                scale: VisionAttentionScale::InverseSqrtWidth,
                span: VisionAttentionSpan::Full,
                output: VisionLinear {
                    weight: weight("attn_out.weight", &[hidden, hidden])?,
                    bias: Some(weight("attn_out.bias", &[hidden])?),
                    clamp: None,
                },
            },
            attention_post_norm: None,
            feedforward_norm: layer_norm(
                weight("ln2.weight", &[hidden])?,
                weight("ln2.bias", &[hidden])?,
            ),
            feedforward: VisionFeedForward {
                up: VisionUp::Plain(VisionLinear {
                    weight: weight("ffn_up.weight", &[intermediate, hidden])?,
                    bias: Some(weight("ffn_up.bias", &[intermediate])?),
                    clamp: None,
                }),
                activation: VisionActivation::GeluTanh,
                down: VisionLinear {
                    weight: weight("ffn_down.weight", &[hidden, intermediate])?,
                    bias: Some(weight("ffn_down.bias", &[hidden])?),
                    clamp: None,
                },
            },
            feedforward_post_norm: None,
        });
    }
    let merger_norm = layer_norm(
        tensor("v.post_ln.weight", &[hidden])?,
        tensor("v.post_ln.bias", &[hidden])?,
    );
    let merger_hidden = product(&[hidden, merge, merge])?;
    let stages = vec![
        MergerStage {
            linear: VisionLinear {
                weight: tensor("mm.0.weight", &[merger_hidden, merger_hidden])?,
                bias: Some(tensor("mm.0.bias", &[merger_hidden])?),
                clamp: None,
            },
            activation: Some(VisionActivation::GeluErf),
        },
        MergerStage {
            linear: VisionLinear {
                weight: tensor("mm.2.weight", &[output_hidden, merger_hidden])?,
                bias: Some(tensor("mm.2.bias", &[output_hidden])?),
                clamp: None,
            },
            activation: None,
        },
    ];
    tensors.finish()?;
    let description = VisionDescription {
        // f16 intermediates around an F32 residual stream: 1.9% relative
        // RMS from an F64 forward of the 4B projector, where bf16 puts the
        // features 12.7% away (see the vision kernel family).
        activation_dtype: ActivationDType::F16,
        hidden,
        output_hidden,
        preprocessing: VisionPreprocessing {
            resize: VisionResize::PixelBounds {
                min_pixels,
                max_pixels,
            },
            resampling: VisionResampling::Bicubic,
            mean,
            std,
            channels: 3,
            patch,
            merge,
        },
        stem: VisionStem::Patch {
            frames: patch_frames,
            bias: Some(patch_bias),
            positions: VisionPositions {
                table,
                sampling: PositionSampling::AlignedCorners { side: table_side },
            },
            norm: None,
        },
        window: None,
        blocks,
        merger: VisionMerger {
            norm: Some(merger_norm),
            reduction: CellReduction::Concatenate,
            standardize: None,
            projection_norm: None,
            stages,
            output_norm: None,
        },
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
        Some(Experts {
            count: m.integer("expert_count")?,
            selected: m.integer("expert_used_count")?,
            intermediate: m.integer("expert_feed_forward_length")?,
            shared_intermediate: m.integer("expert_shared_feed_forward_length")?,
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
    let epsilon = m.number("attention.layer_norm_rms_epsilon")?;
    let rotary = Rotary::Interleaved {
        width: m.integer("rope.dimension_count")?,
        base: m.number("rope.freq_base")?,
        sections: sections.into(),
        axis_pattern: vec![0, 1, 2],
    };
    let convolution_width = m.integer("ssm.conv_kernel")?;
    let vocabulary = embedding.shape[0];
    let context_limit = m.integer("context_length")?;
    let recurrent_channels = product(&[
        recurrent_key_heads
            .checked_mul(2)
            .and_then(|heads| heads.checked_add(recurrent_value_heads))
            .ok_or_else(|| invalid("Qwen recurrent dimensions overflow"))?,
        recurrent_width,
    ])?;
    let mut tensors = Tensors::new(directory);
    let mut tensor = |name: &str, shape: &[u64]| -> Result<WeightDescriptor, Error> {
        Ok(tensors.bind(name, shape)?)
    };
    let embedding = tensor(&embedding.name, &[vocabulary, hidden])?;
    let output_norm = tensor("output_norm.weight", &[hidden])?;
    let output = tensor(
        if directory.tensor("output.weight").is_some() {
            "output.weight"
        } else {
            &embedding.name
        },
        &[vocabulary, hidden],
    )?;
    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };
    // One attention sublayer and one feed-forward sublayer, as every target
    // layer and every MTP block has them.
    let block = |tensor: &mut dyn FnMut(&str, &[u64]) -> Result<WeightDescriptor, Error>,
                 p: &str,
                 kind: LocalMixerKind,
                 intermediate: u64|
     -> Result<Block, Error> {
            let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
            let mixer = match kind {
                LocalMixerKind::Attention => Operator::Attention(Box::new(Attention {
                    heads: attention_heads,
                    kv_heads,
                    width: attention_width,
                    query: weight(
                        "attn_q.weight",
                        &[product(&[2, attention_heads, attention_width])?, hidden],
                    )?,
                    gate: AttentionGate::Interleaved {
                        function: GateFunction::Sigmoid,
                    },
                    query_norm: HeadNorm::Rms(rms(weight(
                        "attn_q_norm.weight",
                        &[attention_width],
                    )?)),
                    key_value: KeyValue::Owned {
                        key: weight(
                            "attn_k.weight",
                            &[product(&[kv_heads, attention_width])?, hidden],
                        )?,
                        value: ValueSource::Projected(weight(
                            "attn_v.weight",
                            &[product(&[kv_heads, attention_width])?, hidden],
                        )?),
                        key_norm: HeadNorm::Rms(rms(weight(
                            "attn_k_norm.weight",
                            &[attention_width],
                        )?)),
                        value_norm: ValueNorm::None,
                        domain: HistoryDomain::Token,
                    },
                    rotary: rotary.clone(),
                    scale: 1.0 / (attention_width as f64).sqrt(),
                    reads: HistoryReads::Visible,
                    media_rows: MediaRowAttention::Causal,
                    output: weight(
                        "attn_output.weight",
                        &[hidden, product(&[attention_heads, attention_width])?],
                    )?,
                })),
                LocalMixerKind::Recurrent => Operator::GatedDelta(Box::new(GatedDelta {
                    convolution_width,
                    key_heads: recurrent_key_heads,
                    value_heads: recurrent_value_heads,
                    width: recurrent_width,
                    head_mapping: RecurrentHeadMapping::Tiled,
                    query_key_value: weight("attn_qkv.weight", &[recurrent_channels, hidden])?,
                    gate: weight("attn_gate.weight", &[recurrent_inner, hidden])?,
                    alpha: weight("ssm_alpha.weight", &[recurrent_value_heads, hidden])?,
                    beta: weight("ssm_beta.weight", &[recurrent_value_heads, hidden])?,
                    convolution: weight(
                        "ssm_conv1d.weight",
                        &[recurrent_channels, convolution_width],
                    )?,
                    decay: weight("ssm_a", &[recurrent_value_heads])?,
                    time_bias: weight("ssm_dt.bias", &[recurrent_value_heads])?,
                    norm: rms(weight("ssm_norm.weight", &[recurrent_width])?),
                    output: weight("ssm_out.weight", &[hidden, recurrent_inner])?,
                })),
            };
            let feed_forward = if let Some(e) = &experts {
                Operator::RoutedFfn(Box::new(RoutedFfn {
                    experts: e.count,
                    selected: e.selected,
                    intermediate: e.intermediate,
                    router: Router {
                        weight: weight("ffn_gate_inp.weight", &[e.count, hidden])?,
                        input: RouterInput::Operator,
                        score: ScoreFunction::Softmax,
                        selection: ExpertSelection::TopK { bias: None },
                        normalization: RouteNormalization::Sum,
                        scale: 1.0,
                    },
                    expert_up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        gate: weight("ffn_gate_exps.weight", &[e.count, e.intermediate, hidden])?,
                        up: weight("ffn_up_exps.weight", &[e.count, e.intermediate, hidden])?,
                    },
                    expert_down: weight(
                        "ffn_down_exps.weight",
                        &[e.count, hidden, e.intermediate],
                    )?,
                    expert_scale: None,
                    latent: None,
                    shared: Some(SharedExpert {
                        intermediate: e.shared_intermediate,
                        up: FeedForwardUp::Gated {
                            activation: ActivationFunction::Silu,
                            gate: weight(
                                "ffn_gate_shexp.weight",
                                &[e.shared_intermediate, hidden],
                            )?,
                            up: weight("ffn_up_shexp.weight", &[e.shared_intermediate, hidden])?,
                        },
                        down: weight("ffn_down_shexp.weight", &[hidden, e.shared_intermediate])?,
                        gate: SharedExpertGate::Sigmoid(weight(
                            "ffn_gate_inp_shexp.weight",
                            &[hidden],
                        )?),
                    }),
                }))
            } else {
                Operator::DenseFfn(Box::new(DenseFfn {
                    intermediate,
                    up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        gate: weight("ffn_gate.weight", &[intermediate, hidden])?,
                        up: weight("ffn_up.weight", &[intermediate, hidden])?,
                    },
                    down: weight("ffn_down.weight", &[hidden, intermediate])?,
                }))
            };
            Ok(Block {
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::Rms(rms(weight("attn_norm.weight", &[hidden])?)),
                        op: mixer,
                        output: OutputForm::Residual,
                    },
                    Sublayer {
                        input: InputNorm::Rms(rms(weight(
                            "post_attention_norm.weight",
                            &[hidden],
                        )?)),
                        op: feed_forward,
                        output: OutputForm::Residual,
                    },
                ],
            })
        };
    let blocks = layers
        .iter()
        .enumerate()
        .map(|(i, kind)| block(&mut tensor, &format!("blk.{i}."), *kind, intermediate))
        .collect::<Result<Vec<_>, _>>()?;
    let head = if speculative == 0 {
        None
    } else {
        // A routed head's experts carry their own widths; only a dense head
        // reads the dense intermediate width.
        let head_intermediate = match &experts {
            Some(experts) => experts.intermediate,
            None => m.integer("feed_forward_length")?,
        };
        let mut head_blocks = Vec::with_capacity(speculative as usize);
        for offset in 0..speculative {
            let p = format!("blk.{}.", layer_count + offset);
            let block = block(&mut tensor, &p, LocalMixerKind::Attention, head_intermediate)?;
            let mut weight = |name: &str, shape: &[u64]| tensor(&format!("{p}{name}"), shape);
            head_blocks.push(HeadBlock {
                embedding_norm: rms(weight("nextn.enorm.weight", &[hidden])?),
                hidden_norm: rms(weight("nextn.hnorm.weight", &[hidden])?),
                combine: weight("nextn.eh_proj.weight", &[hidden, product(&[2, hidden])?])?,
                block,
                output_norm: ExitNorm::Rms(rms(weight(
                    "nextn.shared_head_norm.weight",
                    &[hidden],
                )?)),
            });
        }
        Some(Head {
            blocks: head_blocks,
        })
    };
    tensors.finish()?;
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
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary,
            context_limit,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding,
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
                hash_routing: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::Rms(rms(output_norm)),
                output,
                softcap: None,
            },
        },
        head,
        vision,
        draft: None,
    };
    definition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(definition)
}
