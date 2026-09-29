//! Llama-architecture recognition and numerical interpretation.
//!
//! `llama` is a catch-all GGUF architecture string shared by many unrelated
//! checkpoints, so this family is strict: it admits exactly one feature set
//! (plain grouped-query attention with full rotary, a dense SwiGLU
//! feed-forward, pre-norm residuals and an untied head) and claims nothing
//! else. Rope scaling, biases, per-layer variation, q/k norms, a tied head and
//! any tensor role outside that set are not recognized.
//!
//! GGUF stores `attn_q`/`attn_k` rows in the adjacent-pair (NORM) rotary
//! layout; import permutes them into the engine's half-split (NEOX) pairs, so
//! the definition's rotary is a plain frequency table.

mod family;

pub use family::LlamaFamily;

use magnitude_artifacts::{
    gguf::{Directory, Value},
    PackageIdentity,
};
use magnitude_family_common::{rotary, HeaderError, Metadata, Tensors};
use magnitude_family_contracts::{
    checked_product, ActivationDType, ActivationFunction, Attention, AttentionGate, Block, Decoder,
    DefinitionError, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm, FamilyId,
    FeedForwardUp, HeadNorm, HistoryDomain, HistoryReads, InputNorm, InputSemantics, KeyValue,
    MediaRowAttention, ModelDefinition, Operator, OutputForm, ResidualForm, RmsNorm, Sublayer,
    ValueNorm, ValueSource, WeightDescriptor,
};
use std::{error, fmt};

/// The family identity every definition this crate builds carries.
pub const FAMILY_ID: &str = "llama";

const PREFIX: &str = "llama.";

/// Every `llama.*` key of the admitted feature set. Each is required and
/// scalar; any other key names a feature this family does not admit.
const ADMITTED_KEYS: [&str; 12] = [
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "rope.dimension_count",
    "rope.freq_base",
    "vocab_size",
];

/// Container and provenance namespaces that carry no numerical meaning for
/// the decoder (the tokenizer is interpreted by the chat layer).
const CONTAINER_NAMESPACES: [&str; 4] = ["general.", "tokenizer.", "quantize.", "split."];

const GLOBAL_ROLES: [&str; 3] = ["token_embd.weight", "output_norm.weight", "output.weight"];

const BLOCK_ROLES: [&str; 9] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// Failure to recognize or interpret a `llama` artifact.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// `general.architecture` is missing or not `llama`.
    Architecture(Option<String>),
    /// A metadata key outside the admitted feature set.
    UnknownMetadata(String),
    /// An admitted key given per layer.
    PerLayerMetadata(String),
    /// A required key is missing or not of its admitted type.
    Metadata {
        key: String,
        expected: &'static str,
    },
    /// Metadata that is well typed but outside the admitted geometry.
    Geometry(String),
    /// The head shares the embedding table.
    TiedOutput,
    /// A projector was supplied for a text-only family.
    Projector,
    MissingWeight(String),
    WeightShape {
        name: String,
        expected: Vec<u64>,
        received: Vec<u64>,
    },
    /// A stored tensor that no admitted role binds.
    UnboundWeight(String),
    Definition(DefinitionError),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Architecture(Some(architecture)) => {
                write!(
                    formatter,
                    "GGUF architecture {architecture:?} is not \"llama\""
                )
            }
            Self::Architecture(None) => formatter.write_str("missing GGUF architecture"),
            Self::UnknownMetadata(key) => write!(
                formatter,
                "llama metadata {key:?} names a feature outside the admitted set"
            ),
            Self::PerLayerMetadata(key) => {
                write!(formatter, "llama metadata {key:?} varies per layer")
            }
            Self::Metadata { key, expected } => {
                write!(formatter, "llama metadata {key:?} must be {expected}")
            }
            Self::Geometry(message) => write!(formatter, "llama geometry: {message}"),
            Self::TiedOutput => formatter.write_str("llama head is tied to the embedding"),
            Self::Projector => formatter.write_str("llama has no vision projector"),
            Self::MissingWeight(name) => write!(formatter, "missing llama weight {name:?}"),
            Self::WeightShape {
                name,
                expected,
                received,
            } => write!(
                formatter,
                "llama weight {name:?}: expected {expected:?}, received {received:?}"
            ),
            Self::UnboundWeight(name) => {
                write!(
                    formatter,
                    "llama artifact contains unbound weight role {name:?}"
                )
            }
            Self::Definition(error) => write!(formatter, "llama definition: {error}"),
        }
    }
}

impl error::Error for Error {}

impl From<HeaderError> for Error {
    fn from(error: HeaderError) -> Self {
        match error {
            HeaderError::UnknownMetadata(key) => Self::UnknownMetadata(key),
            HeaderError::MissingMetadata(key) => Self::Metadata {
                key,
                expected: "present",
            },
            HeaderError::MetadataType { key, expected } => Self::Metadata { key, expected },
            HeaderError::MissingWeight(name) => Self::MissingWeight(name),
            HeaderError::WeightShape {
                name,
                expected,
                received,
            } => Self::WeightShape {
                name,
                expected,
                received,
            },
            HeaderError::UnboundWeight(name) => Self::UnboundWeight(name),
            HeaderError::Definition(error) => Self::Definition(error),
            error @ (HeaderError::NotMatrix { .. }
            | HeaderError::CompanionScale { .. }
            | HeaderError::Rotary { .. }
            | HeaderError::Yarn) => Self::Geometry(error.to_string()),
        }
    }
}

/// Whether `name` is a tensor role of the admitted feature set.
fn admitted_role(name: &str) -> bool {
    if GLOBAL_ROLES.contains(&name) {
        return true;
    }
    name.strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(index, role)| {
            !index.is_empty()
                && index.bytes().all(|byte| byte.is_ascii_digit())
                && BLOCK_ROLES.contains(&role)
        })
}

/// Recognize the admitted `llama` feature set from metadata keys and tensor
/// role names, without interpreting geometry.
pub fn recognize(directory: &Directory) -> Result<(), Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
    {
        Some("llama") => {}
        architecture => return Err(Error::Architecture(architecture.map(str::to_owned))),
    }
    for metadata in &directory.metadata {
        match metadata.name.strip_prefix(PREFIX) {
            Some(key) if !ADMITTED_KEYS.contains(&key) => {
                return Err(Error::UnknownMetadata(metadata.name.clone()));
            }
            Some(_) if matches!(metadata.value, Value::Array(_)) => {
                return Err(Error::PerLayerMetadata(metadata.name.clone()));
            }
            Some(_) => {}
            None if CONTAINER_NAMESPACES
                .iter()
                .any(|namespace| metadata.name.starts_with(namespace)) => {}
            None => return Err(Error::UnknownMetadata(metadata.name.clone())),
        }
    }
    if let Some(tensor) = directory
        .tensors
        .iter()
        .find(|tensor| !admitted_role(&tensor.name))
    {
        return Err(Error::UnboundWeight(tensor.name.clone()));
    }
    if directory.tensor("output.weight").is_none() {
        return Err(Error::TiedOutput);
    }
    Ok(())
}

fn product(dimensions: &[u64]) -> Result<u64, Error> {
    checked_product(dimensions).map_err(Error::Definition)
}

/// Build the numerical definition of a recognized `llama` package from its
/// headers.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, Error> {
    recognize(directory)?;
    if projector.is_some() {
        return Err(Error::Projector);
    }
    let m = Metadata::new(directory, FAMILY_ID, &ADMITTED_KEYS)?;
    let block_count = m.integer("block_count")?;
    let context_limit = m.integer("context_length")?;
    let hidden = m.integer("embedding_length")?;
    let intermediate = m.integer("feed_forward_length")?;
    let heads = m.integer("attention.head_count")?;
    let kv_heads = m.integer("attention.head_count_kv")?;
    let width = m.integer("attention.key_length")?;
    let epsilon = m.positive_number("attention.layer_norm_rms_epsilon")?;
    let base = m.positive_number("rope.freq_base")?;
    let vocabulary = m.integer("vocab_size")?;
    if m.integer("attention.value_length")? != width {
        return Err(Error::Geometry(
            "value head width differs from key head width".into(),
        ));
    }
    if m.integer("rope.dimension_count")? != width || !width.is_multiple_of(2) {
        return Err(Error::Geometry(
            "rotary must cover the whole even head width".into(),
        ));
    }
    // Each block binds distinct stored weights; reject impossible counts
    // before allocating from untrusted metadata.
    if block_count > directory.tensors.len() as u64 {
        return Err(Error::Geometry(
            "block count exceeds the stored weight roles".into(),
        ));
    }
    let query_rows = product(&[heads, width])?;
    let key_rows = product(&[kv_heads, width])?;
    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };
    // Pair `p` rotates dimensions `(p, p + W/2)` at `base^(-2p/W)`.
    let rotary = rotary::table(width, width, base)?;
    // GGUF stores the query and key rows of each head in adjacent pairs.
    let half_split = rotary::half_split_rows(width, width);

    let mut tensors = Tensors::new(directory);
    let embedding = tensors.bind("token_embd.weight", &[vocabulary, hidden])?;
    let output_norm = tensors.bind("output_norm.weight", &[hidden])?;
    let output = tensors.bind("output.weight", &[vocabulary, hidden])?;
    let blocks = (0..block_count)
        .map(|index| {
            let role = |role: &str| format!("blk.{index}.{role}");
            let mut rotated = |name: &str, shape: &[u64]| {
                tensors.bind_transformed(&role(name), shape, vec![half_split.clone()])
            };
            let query = rotated("attn_q.weight", &[query_rows, hidden])?;
            let key = rotated("attn_k.weight", &[key_rows, hidden])?;
            let mut weight = |name: &str, shape: &[u64]| tensors.bind(&role(name), shape);
            let attention_norm = weight("attn_norm.weight", &[hidden])?;
            let attention = Attention {
                heads,
                kv_heads,
                width,
                query,
                gate: AttentionGate::None,
                query_norm: HeadNorm::None,
                key_value: KeyValue::Owned {
                    key,
                    value: ValueSource::Projected(weight("attn_v.weight", &[key_rows, hidden])?),
                    key_norm: HeadNorm::None,
                    value_norm: ValueNorm::None,
                    domain: HistoryDomain::Token,
                },
                rotary: rotary.clone(),
                scale: 1.0 / (width as f64).sqrt(),
                reads: HistoryReads::Visible,
                media_rows: MediaRowAttention::Causal,
                output: weight("attn_output.weight", &[hidden, query_rows])?,
            };
            let feed_forward = DenseFfn {
                intermediate,
                up: FeedForwardUp::Gated {
                    activation: ActivationFunction::Silu,
                    gate: weight("ffn_gate.weight", &[intermediate, hidden])?,
                    up: weight("ffn_up.weight", &[intermediate, hidden])?,
                },
                down: weight("ffn_down.weight", &[hidden, intermediate])?,
            };
            Ok(Block {
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::Rms(rms(attention_norm)),
                        op: Operator::Attention(Box::new(attention)),
                        output: OutputForm::Residual,
                    },
                    Sublayer {
                        input: InputNorm::Rms(rms(weight("ffn_norm.weight", &[hidden])?)),
                        op: Operator::DenseFfn(Box::new(feed_forward)),
                        output: OutputForm::Residual,
                    },
                ],
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    tensors.finish()?;

    let definition = ModelDefinition {
        family: FamilyId(FAMILY_ID.into()),
        artifact_identity: identity,
        inputs: InputSemantics {
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
    definition.validate().map_err(Error::Definition)?;
    Ok(definition)
}
