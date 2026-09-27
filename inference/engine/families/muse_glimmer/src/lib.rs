//! Muse Glimmer artifact recognition and family-specific numerical
//! interpretation.
//!
//! Muse Glimmer is a dense decoder with sandwich norms: a weightless RMS of
//! the embedding enters the residual, every sublayer's output is normalized
//! before it is added, and attention carries an element-wise sigmoid gate
//! from its own projection. Windowed layers use rotary; full layers use none.
//! The readout multiplies logits by a scale and soft-caps them.
//!
//! Two import transforms make the stored tensors exact in the engine's
//! layouts: Muse pairs adjacent rotary dimensions `(2i, 2i + 1)`, so the
//! query and key rows of rotary layers (and their head norms, identically)
//! are permuted to the engine's `(i, i + P)` pairing; and the logit scale is
//! folded into the final norm's weight, since the readout is linear in it.

mod family;
pub mod inputs;
mod projector;

pub use family::MuseGlimmerFamily;
pub use projector::describe_projector;

use magnitude_artifacts::{
    gguf::{Directory, Value},
    PackageIdentity,
};
use magnitude_family_contracts::{
    checked_product, ActivationDType, ActivationFunction, Attention, AttentionGate, Block,
    Decoder, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, FamilyId, FeedForwardUp,
    GateFunction, GateGranularity, HeadNorm, HistoryDomain, HistoryReads, ImportTransform,
    InputNorm, InputSemantics, KeyValue, MediaRowAttention, ModelDefinition, Operator, OutputForm,
    ResidualForm, RmsNorm, Rotary, Sublayer, TextCoordinateSemantics, UnweightedRms, ValueNorm,
    ValueSource, WeightDescriptor,
};
use magnitude_family_common::{rotary, HeaderError, Metadata, Tensors};
use std::{error, fmt};

const ARCHITECTURE: &str = "muse-glimmer";

/// Every `muse-glimmer.*` key the adapter interprets; any other is rejected.
const KEYS: &[&str] = &[
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
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.freq_base_swa",
    "logit_scale",
    "final_logit_softcapping",
];

/// The epsilon of the post-sublayer norms. The architecture fixes it; headers
/// do not state it.
const POST_NORM_EPSILON: f64 = 1e-8;

/// Windowed layers repeat with this period when the header states no pattern;
/// the last layer of each period is full.
const DEFAULT_WINDOW_PERIOD: u64 = 4;

/// Failure to interpret a Muse Glimmer artifact.
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
        invalid(error.to_string())
    }
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error(message.into())
}

fn product(dimensions: &[u64]) -> Result<u64, Error> {
    checked_product(dimensions).map_err(|error| invalid(error.to_string()))
}

/// Whether the target declares the Muse Glimmer architecture. Recognition
/// interprets no geometry.
pub fn recognize(directory: &Directory) -> bool {
    directory
        .value("general.architecture")
        .and_then(Value::string)
        == Some(ARCHITECTURE)
}

/// Which layers keep only a window of history (and use rotary): a period
/// whose last layer is full, or one flag per layer.
fn window_layers(m: &Metadata, layers: u64) -> Result<Vec<bool>, Error> {
    let period = match m.value("attention.sliding_window_pattern") {
        None => DEFAULT_WINDOW_PERIOD,
        Some(Value::Array(_)) => return Ok(m.flags("attention.sliding_window_pattern", layers)?),
        Some(value) => value
            .unsigned()
            .ok_or_else(|| invalid("Muse Glimmer sliding-window pattern must be a period"))?,
    };
    // A zero period makes every layer windowed.
    Ok((0..layers)
        .map(|layer| period == 0 || layer % period < period - 1)
        .collect())
}

pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, Error> {
    if !recognize(directory) {
        return Err(invalid("not a Muse Glimmer artifact"));
    }
    let m = Metadata::new(directory, ARCHITECTURE, KEYS)?;
    let layers = m.integer("block_count")?;
    // Every layer binds distinct stored weights; reject impossible counts
    // before allocating per-layer vectors from untrusted metadata.
    if layers > directory.tensors.len() as u64 {
        return Err(invalid("Muse Glimmer layer count exceeds its stored weights"));
    }
    let hidden = m.integer("embedding_length")?;
    let intermediate = m.integer("feed_forward_length")?;
    let heads = m.integer("attention.head_count")?;
    let kv_heads = m.integer("attention.head_count_kv")?;
    let width = m.integer("attention.key_length")?;
    if m.integer("attention.value_length")? != width {
        return Err(invalid("Muse Glimmer key and value head widths differ"));
    }
    let epsilon = m.number("attention.layer_norm_rms_epsilon")?;
    let window = m.integer("attention.sliding_window")?;
    let windowed = window_layers(&m, layers)?;

    let rotated = m.optional_integer("rope.dimension_count")?.unwrap_or(width);
    let base = match m.optional_number("rope.freq_base_swa")? {
        Some(base) => base,
        None => m.number("rope.freq_base")?,
    };
    let rotary = rotary::table(rotated, width, base)?;
    let pairing = rotary::half_split_rows(rotated, width);

    let logit_scale = m.number("logit_scale")?;
    if logit_scale <= 0.0 {
        return Err(invalid("Muse Glimmer logit scale must be positive"));
    }
    let softcap = m
        .optional_number("final_logit_softcapping")?
        .filter(|cap| *cap != 0.0);

    let mut tensors = Tensors::new(directory);
    let vocabulary = tensors.rows("token_embd.weight")?;
    if m.optional_integer("vocab_size")?
        .is_some_and(|size| size != vocabulary)
    {
        return Err(invalid(
            "Muse Glimmer vocabulary size differs from its embedding",
        ));
    }
    let embedding = tensors.bind("token_embd.weight", &[vocabulary, hidden])?;
    // `W·(RMS(y)·w)·s = W·(RMS(y)·(s·w))`: the scale folds into the norm.
    let output_norm = tensors.bind_transformed(
        "output_norm.weight",
        &[hidden],
        vec![ImportTransform::Scale {
            factor: logit_scale,
        }],
    )?;
    let output = tensors.bind("output.weight", &[vocabulary, hidden])?;
    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };
    let post_norm = |weight: WeightDescriptor| {
        OutputForm::PostNorm(RmsNorm {
            weight,
            epsilon: POST_NORM_EPSILON,
        })
    };
    let query_rows = product(&[heads, width])?;
    let key_rows = product(&[kv_heads, width])?;

    let mut blocks = Vec::with_capacity(layers as usize);
    for (layer, windowed) in windowed.into_iter().enumerate() {
        let p = format!("blk.{layer}.");
        let mut weight = |name: &str, shape: &[u64]| tensors.bind(&format!("{p}{name}"), shape);
        let gate = weight("attn_gate.weight", &[query_rows, hidden])?;
        let value = weight("attn_v.weight", &[key_rows, hidden])?;
        let output = weight("attn_output.weight", &[hidden, query_rows])?;
        let input_norm = weight("attn_norm.weight", &[hidden])?;
        let attention_post_norm = weight("post_attention_norm.weight", &[hidden])?;
        let feed_forward_norm = weight("ffn_norm.weight", &[hidden])?;
        let feed_forward = DenseFfn {
            intermediate,
            up: FeedForwardUp::Gated {
                activation: ActivationFunction::Silu,
                gate: weight("ffn_gate.weight", &[intermediate, hidden])?,
                up: weight("ffn_up.weight", &[intermediate, hidden])?,
            },
            down: weight("ffn_down.weight", &[hidden, intermediate])?,
        };
        let feed_forward_post_norm = weight("post_ffw_norm.weight", &[hidden])?;
        // Rotary layers read queries and keys in the engine's pairing; the
        // permutation is invisible to the head norm and to `q·k` alike.
        let transforms = if windowed {
            vec![pairing.clone()]
        } else {
            Vec::new()
        };
        let mut paired = |name: &str, shape: &[u64]| {
            tensors.bind_transformed(&format!("{p}{name}"), shape, transforms.clone())
        };
        let attention = Attention {
            heads,
            kv_heads,
            width,
            query: paired("attn_q.weight", &[query_rows, hidden])?,
            gate: AttentionGate::Separate {
                weight: gate,
                function: GateFunction::Sigmoid,
                granularity: GateGranularity::Element,
            },
            query_norm: HeadNorm::Rms(rms(paired("attn_q_norm.weight", &[width])?)),
            key_value: KeyValue::Owned {
                key: paired("attn_k.weight", &[key_rows, hidden])?,
                value: ValueSource::Projected(value),
                key_norm: HeadNorm::Rms(rms(paired("attn_k_norm.weight", &[width])?)),
                value_norm: ValueNorm::None,
                domain: if windowed {
                    HistoryDomain::Window { tokens: window }
                } else {
                    HistoryDomain::Token
                },
            },
            rotary: if windowed {
                rotary.clone()
            } else {
                Rotary::None
            },
            scale: 1.0 / (width as f64).sqrt(),
            reads: HistoryReads::Visible,
            media_rows: MediaRowAttention::Causal,
            output,
        };
        blocks.push(Block {
            sublayers: vec![
                Sublayer {
                    input: InputNorm::Rms(rms(input_norm)),
                    op: Operator::Attention(Box::new(attention)),
                    output: post_norm(attention_post_norm),
                },
                Sublayer {
                    input: InputNorm::Rms(rms(feed_forward_norm)),
                    op: Operator::DenseFfn(Box::new(feed_forward)),
                    output: post_norm(feed_forward_post_norm),
                },
            ],
        });
    }
    tensors.finish()?;

    let definition = ModelDefinition {
        family: FamilyId(ARCHITECTURE.into()),
        artifact_identity: identity,
        inputs: InputSemantics {
            text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
            coordinate_axes: 1,
        },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary,
            context_limit: m.integer("context_length")?,
            residual: ResidualForm::Single,
            entry: EntryForm {
                embedding,
                scale: EmbeddingScale::Unit,
                norm: Some(UnweightedRms { epsilon }),
                per_layer: None,
                hash_routing: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::Rms(rms(output_norm)),
                output,
                softcap,
            },
        },
        head: None,
        vision: projector
            .map(|projector| describe_projector(projector, hidden, epsilon))
            .transpose()?,
        draft: None,
    };
    definition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(definition)
}
