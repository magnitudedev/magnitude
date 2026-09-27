//! LFM2 and LFM2-MoE artifact recognition and numerical interpretation.
//!
//! Both architectures share one decoder. Every layer is an operator sublayer
//! (gated short convolution, or attention where the layer has key heads)
//! followed by a feed-forward sublayer (dense, or routed after the leading
//! dense layers of `lfm2moe`). Each sublayer is RMS-normalized on input and
//! adds its output to the residual stream.

mod family;

pub use family::Lfm2Family;

use magnitude_artifacts::{
    gguf::{Directory, Value},
    Package, PackageIdentity,
};
use magnitude_family_common::{rotary, HeaderError, Metadata, Tensors};
use magnitude_family_contracts::{
    checked_product, ActivationDType, ActivationFunction, Attention, AttentionGate, Block,
    Decoder, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, ExpertSelection, FamilyId,
    FeedForwardUp, HeadNorm, HistoryDomain, HistoryReads, ImportTransform, InputNorm,
    InputSemantics, KeyValue, MediaRowAttention, ModelDefinition, Operator, OutputForm,
    ResidualForm, RmsNorm, RouteNormalization, RoutedFfn, Router, RouterInput, RowRange,
    ScoreFunction, ShortConv, Sublayer, TextCoordinateSemantics, ValueNorm, ValueSource,
    WeightDescriptor,
};
use std::{error, fmt};

/// LFM2 architecture recognized from literal artifact metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    Dense,
    Routed,
}

impl Architecture {
    pub const fn family_id(self) -> &'static str {
        match self {
            Self::Dense => "lfm2",
            Self::Routed => "lfm2moe",
        }
    }

    /// Every metadata key under the architecture prefix this interpretation
    /// understands. Any other key may change the forward pass (sliding
    /// windows, rope scaling, expert weight scales), so it is rejected rather
    /// than ignored.
    const fn metadata_keys(self) -> &'static [&'static str] {
        const DENSE: &[&str] = &[
            "block_count",
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "vocab_size",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "shortconv.l_cache",
        ];
        const ROUTED: &[&str] = &[
            "block_count",
            "context_length",
            "embedding_length",
            "feed_forward_length",
            "vocab_size",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "attention.layer_norm_rms_epsilon",
            "rope.freq_base",
            "shortconv.l_cache",
            "expert_count",
            "expert_used_count",
            "expert_feed_forward_length",
            "expert_gating_func",
            "leading_dense_block_count",
        ];
        match self {
            Self::Dense => DENSE,
            Self::Routed => ROUTED,
        }
    }
}

/// Failure to recognize or interpret an LFM2 artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for Error {}

impl From<HeaderError> for Error {
    fn from(error: HeaderError) -> Self {
        match error {
            HeaderError::UnknownMetadata(key) => {
                invalid(format!("LFM2 metadata {key:?} is not understood"))
            }
            error => invalid(error.to_string()),
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error(message.into())
}

fn product(dimensions: &[u64]) -> Result<u64, Error> {
    checked_product(dimensions).map_err(|error| invalid(error.to_string()))
}

/// The operator of one layer, read from its key-head count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mixer {
    ShortConv,
    Attention { kv_heads: u64 },
}

/// Expert geometry of a routed (`lfm2moe`) artifact.
struct Experts {
    count: u64,
    selected: u64,
    intermediate: u64,
    score: ScoreFunction,
}

/// Recognize the supported LFM2 architectures without interpreting their
/// geometry or tensor roles.
pub fn recognize(directory: &Directory) -> Result<Architecture, Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
        .ok_or_else(|| invalid("missing GGUF architecture"))?
    {
        "lfm2" => Ok(Architecture::Dense),
        "lfm2moe" => Ok(Architecture::Routed),
        architecture => Err(invalid(format!(
            "unsupported LFM2 GGUF architecture {architecture:?}"
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
    if projector.is_some() {
        return Err(invalid("LFM2 packages have no projector"));
    }
    let family_id = architecture.family_id();
    let m = Metadata::new(directory, family_id, architecture.metadata_keys())?;

    let layer_count = m.integer("block_count")?;
    // Each layer requires distinct stored weights. Reject impossible counts
    // before allocating per-layer vectors from untrusted metadata.
    if layer_count > directory.tensors.len() as u64 {
        return Err(invalid("LFM2 layer count exceeds available weight roles"));
    }
    // A layer without key heads is a short-convolution layer; the per-layer
    // array is the only source of the pattern.
    let mixers = m
        .layer_counts("attention.head_count_kv", layer_count)?
        .into_iter()
        .map(|kv_heads| match kv_heads {
            0 => Mixer::ShortConv,
            kv_heads => Mixer::Attention { kv_heads },
        })
        .collect::<Vec<_>>();

    let hidden = m.integer("embedding_length")?;
    let heads = m.integer("attention.head_count")?;
    if !hidden.is_multiple_of(heads) {
        return Err(invalid("LFM2 hidden width is not a multiple of its heads"));
    }
    // Queries and outputs span the hidden width, so the head width follows.
    let width = hidden / heads;
    for key in ["attention.key_length", "attention.value_length"] {
        if m.optional_integer(key)?.is_some_and(|length| length != width) {
            return Err(invalid(format!(
                "{} differs from the hidden width per head",
                m.key(key)
            )));
        }
    }
    let epsilon = m.positive_number("attention.layer_norm_rms_epsilon")?;
    let base = m.positive_number("rope.freq_base")?;
    let convolution_width = m.integer("shortconv.l_cache")?;
    if convolution_width < 2 {
        return Err(invalid("LFM2 short convolution needs at least two taps"));
    }
    let context_limit = m.integer("context_length")?;
    let (experts, leading_dense) = match architecture {
        Architecture::Dense => (None, layer_count),
        Architecture::Routed => {
            let score = match m.count("expert_gating_func")? {
                1 => ScoreFunction::Softmax,
                2 => ScoreFunction::Sigmoid,
                _ => {
                    return Err(invalid(format!(
                        "{} must be 1 (softmax) or 2 (sigmoid)",
                        m.key("expert_gating_func")
                    )))
                }
            };
            let leading_dense = m
                .optional_count("leading_dense_block_count")?
                .unwrap_or(0);
            if leading_dense > layer_count {
                return Err(invalid(format!(
                    "{} must not exceed the layer count",
                    m.key("leading_dense_block_count")
                )));
            }
            let experts = Experts {
                count: m.integer("expert_count")?,
                selected: m.integer("expert_used_count")?,
                intermediate: m.integer("expert_feed_forward_length")?,
                score,
            };
            if experts.selected > experts.count {
                return Err(invalid("LFM2 selects more experts than it has"));
            }
            (Some(experts), leading_dense)
        }
    };
    // Only dense layers read the dense width; a fully routed artifact need not
    // declare it.
    let dense_intermediate = m.optional_integer("feed_forward_length")?;

    let mut tensors = Tensors::new(directory);
    let vocabulary = tensors.rows("token_embd.weight")?;
    if m.optional_integer("vocab_size")?
        .is_some_and(|size| size != vocabulary)
    {
        return Err(invalid("LFM2 vocab_size differs from the embedding rows"));
    }

    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };
    let embedding = tensors.bind("token_embd.weight", &[vocabulary, hidden])?;
    // `token_embd_norm` is the decoder's final norm, not an entry norm.
    let output_norm = tensors.bind("token_embd_norm.weight", &[hidden])?;
    let output = tensors.bind(
        if tensors.contains("output.weight") {
            "output.weight"
        } else {
            &embedding.name
        },
        &[vocabulary, hidden],
    )?;
    // NEOX rotation over the whole head: pair `p` rotates dimensions
    // `(p, p + W/2)` at frequency `base^(-2p/W)`.
    let rotary = rotary::table(width, width, base)?;
    let head_rows = product(&[heads, width])?;

    let mut blocks = Vec::with_capacity(mixers.len());
    for (index, mixer) in mixers.iter().enumerate() {
        let p = format!("blk.{index}.");
        let mut weight = |name: &str, shape: &[u64]| tensors.bind(&format!("{p}{name}"), shape);
        let op = match *mixer {
            Mixer::Attention { kv_heads } => {
                if !heads.is_multiple_of(kv_heads) {
                    return Err(invalid(format!(
                        "LFM2 layer {index}: {heads} heads do not group over {kv_heads} key heads"
                    )));
                }
                let kv_rows = product(&[kv_heads, width])?;
                Operator::Attention(Box::new(Attention {
                    heads,
                    kv_heads,
                    width,
                    query: weight("attn_q.weight", &[head_rows, hidden])?,
                    gate: AttentionGate::None,
                    query_norm: HeadNorm::Rms(rms(weight("attn_q_norm.weight", &[width])?)),
                    key_value: KeyValue::Owned {
                        key: weight("attn_k.weight", &[kv_rows, hidden])?,
                        value: ValueSource::Projected(weight("attn_v.weight", &[kv_rows, hidden])?),
                        key_norm: HeadNorm::Rms(rms(weight("attn_k_norm.weight", &[width])?)),
                        value_norm: ValueNorm::None,
                        domain: HistoryDomain::Token,
                    },
                    rotary: rotary.clone(),
                    scale: 1.0 / (width as f64).sqrt(),
                    reads: HistoryReads::Visible,
                    media_rows: MediaRowAttention::Causal,
                    output: weight("attn_output.weight", &[hidden, head_rows])?,
                }))
            }
            Mixer::ShortConv => {
                // `in_proj` stores the B, C and X projections as consecutive
                // row blocks of the hidden width.
                let fused = weight("shortconv.in_proj.weight", &[product(&[3, hidden])?, hidden])?;
                let chunk = |block: u64| WeightDescriptor {
                    name: fused.name.clone(),
                    shape: vec![hidden, hidden],
                    transforms: vec![ImportTransform::Rows(RowRange {
                        start: block * hidden,
                        rows: hidden,
                    })],
                };
                Operator::ShortConv(Box::new(ShortConv {
                    channels: hidden,
                    width: convolution_width,
                    input_gate: chunk(0),
                    output_gate: chunk(1),
                    value: chunk(2),
                    convolution: weight("shortconv.conv.weight", &[hidden, convolution_width])?,
                    output: weight("shortconv.out_proj.weight", &[hidden, hidden])?,
                }))
            }
        };
        let feed_forward = match &experts {
            Some(e) if index as u64 >= leading_dense => Operator::RoutedFfn(Box::new(RoutedFfn {
                experts: e.count,
                selected: e.selected,
                intermediate: e.intermediate,
                // Scores rank on `score + bias`; the combine weights are the
                // selected unbiased scores over their sum plus 1e-6 (the
                // released model definition; llama.cpp clamps the sum at
                // 6.1e-5 instead), with no scale.
                router: Router {
                    weight: weight("ffn_gate_inp.weight", &[e.count, hidden])?,
                    input: RouterInput::Operator,
                    score: e.score,
                    selection: ExpertSelection::TopK {
                        bias: Some(weight("exp_probs_b.bias", &[e.count])?),
                    },
                    normalization: RouteNormalization::SumPlusEpsilon(1e-6),
                    scale: 1.0,
                },
                expert_up: FeedForwardUp::Gated {
                    activation: ActivationFunction::Silu,
                    gate: weight("ffn_gate_exps.weight", &[e.count, e.intermediate, hidden])?,
                    up: weight("ffn_up_exps.weight", &[e.count, e.intermediate, hidden])?,
                },
                expert_down: weight("ffn_down_exps.weight", &[e.count, hidden, e.intermediate])?,
                expert_scale: None,
                latent: None,
                shared: None,
            })),
            _ => {
                let intermediate = dense_intermediate.ok_or_else(|| {
                    invalid(format!(
                        "missing LFM2 metadata {} for dense layer {index}",
                        m.key("feed_forward_length")
                    ))
                })?;
                Operator::DenseFfn(Box::new(DenseFfn {
                    intermediate,
                    up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        gate: weight("ffn_gate.weight", &[intermediate, hidden])?,
                        up: weight("ffn_up.weight", &[intermediate, hidden])?,
                    },
                    down: weight("ffn_down.weight", &[hidden, intermediate])?,
                }))
            }
        };
        blocks.push(Block {
            sublayers: vec![
                Sublayer {
                    input: InputNorm::Rms(rms(weight("attn_norm.weight", &[hidden])?)),
                    op,
                    output: OutputForm::Residual,
                },
                Sublayer {
                    input: InputNorm::Rms(rms(weight("ffn_norm.weight", &[hidden])?)),
                    op: feed_forward,
                    output: OutputForm::Residual,
                },
            ],
        });
    }
    tensors.finish()?;
    let definition = ModelDefinition {
        family: FamilyId(family_id.into()),
        artifact_identity: identity,
        inputs: InputSemantics {
            text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
            coordinate_axes: 1,
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
        head: None,
        vision: None,
        draft: None,
    };
    definition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(definition)
}
