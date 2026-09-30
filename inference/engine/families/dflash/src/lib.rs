//! DFlash, DSpark and DFlash2 draft components (GGUF architecture `dflash`,
//! plan §3.8).
//!
//! A draft is a separate package component read against its target's
//! definition: its taps name target layers, it has no vocabulary of its own
//! (the target's embedding and vocabulary projection serve it, unless it
//! ships its own embedding table), and its width is the target's. DSpark is
//! the same architecture with a Markov bias and a confidence head, detected
//! by `markov_w1.weight`. DFlash2 wraps every draft sublayer in a grouped
//! dynamic causal convolution and selects an ordered candidate path, detected
//! by its `selector_*` roles; the shared architecture label names none of
//! them.
//!
//! GGUF stores the draft's `attn_q`/`attn_k` rows in the half-split (NEOX)
//! rotary layout, the engine's own, so rotary is a plain frequency table (YaRN
//! when the header declares it); the q/k norms apply per head.

use magnitude_artifacts::{
    gguf::{Directory, Value},
    TokenId,
};
use magnitude_family_common::{
    rotary::{self, Yarn},
    HeaderError, Metadata, Tensors,
};
use magnitude_family_contracts::{
    checked_product, ActivationFunction, Attention, AttentionGate, Block, BlockAttention,
    BlockLayout, CandidateSelector, ConfidenceHead, DefinitionError, DenseFfn, DraftDefinition,
    DraftEmbedding, DraftMethod, DynamicConvolution, FamilyError, FeedForwardUp, HeadNorm,
    HistoryDomain, HistoryReads, ImportTransform, InputNorm, KeyValue, LayerConvolutions,
    MarkovHead, MediaRowAttention, ModelDefinition, Operator, OutputForm, RmsNorm, Rotary,
    Sublayer, TapPoint, ValueNorm, ValueSource, WeightDescriptor,
};
use std::{error, fmt};

const ARCHITECTURE: &str = "dflash";
const PREFIX: &str = "dflash.";

/// Every admitted `dflash.*` key; any other key names a feature this family
/// does not admit.
const ADMITTED_KEYS: [&str; 28] = [
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.causal",
    "rope.freq_base",
    "rope.dimension_sections",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast",
    "rope.scaling.yarn_beta_slow",
    "rope.scaling.yarn_attn_factor",
    "block_size",
    "target_layers",
    "sample_from_anchor",
    "has_confidence_head",
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
];

/// Keys that hold one value per draft layer or a list.
const ARRAY_KEYS: [&str; 3] = [
    "target_layers",
    "attention.sliding_window_pattern",
    "rope.dimension_sections",
];

/// Container and provenance namespaces without numerical meaning for the
/// draft (its tokenizer is the target's; only the mask token is read).
const CONTAINER_NAMESPACES: [&str; 4] = ["general.", "tokenizer.", "quantize.", "split."];

const MASK_TOKEN: &str = "tokenizer.ggml.mask_token_id";

/// Draft-wide roles. A draft's own vocabulary projection and a reduced
/// vocabulary (`d2t`) are not admitted.
const GLOBAL_ROLES: [&str; 12] = [
    "fc.weight",
    "fc.scale",
    "enc.output_norm.weight",
    "output_norm.weight",
    "token_embd.weight",
    "markov_w1.weight",
    "markov_w2.weight",
    "conf_proj.weight",
    "conf_proj.bias",
    "selector_hidden.weight",
    "selector_predecessor.weight",
    "selector_successor.weight",
];

const BLOCK_ROLES: [&str; 13] = [
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_q_norm",
    "attn_k_norm",
    "attn_output",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "attn_conv_proj",
    "ffn_conv_proj",
];

/// DFlash2's per-layer convolution bases, stored without a `.weight` suffix.
const BLOCK_PARAMETERS: [&str; 2] = ["attn_conv_base", "ffn_conv_base"];

/// Failure to recognize or interpret a `dflash` draft.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// `general.architecture` is missing or not `dflash`.
    Architecture(Option<String>),
    /// A metadata key outside the admitted feature set.
    UnknownMetadata(String),
    /// A required key is missing or not of its admitted type.
    Metadata {
        key: String,
        expected: &'static str,
    },
    /// Metadata that is well typed but outside the admitted geometry.
    Geometry(String),
    /// The draft disagrees with its target's definition.
    Target(String),
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
            Self::Architecture(Some(architecture)) => write!(
                formatter,
                "GGUF architecture {architecture:?} is not \"{ARCHITECTURE}\""
            ),
            Self::Architecture(None) => formatter.write_str("missing GGUF architecture"),
            Self::UnknownMetadata(key) => write!(
                formatter,
                "draft metadata {key:?} names a feature outside the admitted set"
            ),
            Self::Metadata { key, expected } => {
                write!(formatter, "draft metadata {key:?} must be {expected}")
            }
            Self::Geometry(message) => write!(formatter, "draft geometry: {message}"),
            Self::Target(message) => write!(formatter, "draft and target disagree: {message}"),
            Self::MissingWeight(name) => write!(formatter, "missing draft weight {name:?}"),
            Self::WeightShape {
                name,
                expected,
                received,
            } => write!(
                formatter,
                "draft weight {name:?}: expected {expected:?}, received {received:?}"
            ),
            Self::UnboundWeight(name) => {
                write!(formatter, "draft contains unbound weight role {name:?}")
            }
            Self::Definition(error) => write!(formatter, "draft definition: {error}"),
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

/// Whether `name` is a tensor role of a `dflash` draft.
fn known_role(name: &str) -> bool {
    if GLOBAL_ROLES.contains(&name) {
        return true;
    }
    name.strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(index, role)| {
            !index.is_empty()
                && index.bytes().all(|byte| byte.is_ascii_digit())
                && (BLOCK_PARAMETERS.contains(&role)
                    || role.rsplit_once('.').is_some_and(|(role, suffix)| {
                        BLOCK_ROLES.contains(&role) && matches!(suffix, "weight" | "scale")
                    }))
        })
}

/// Recognize a `dflash` draft from metadata keys and tensor role names,
/// without interpreting geometry. No target family claims this
/// architecture, and this family claims nothing else.
pub fn recognize(directory: &Directory) -> Result<(), Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
    {
        Some(ARCHITECTURE) => {}
        architecture => return Err(Error::Architecture(architecture.map(str::to_owned))),
    }
    for metadata in &directory.metadata {
        match metadata.name.strip_prefix(PREFIX) {
            Some(key) if !ADMITTED_KEYS.contains(&key) => {
                return Err(Error::UnknownMetadata(metadata.name.clone()));
            }
            Some(key) if matches!(metadata.value, Value::Array(_)) != ARRAY_KEYS.contains(&key) => {
                return Err(Error::Metadata {
                    key: metadata.name.clone(),
                    expected: "of its admitted arity",
                });
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
        .find(|tensor| !known_role(&tensor.name))
    {
        return Err(Error::UnboundWeight(tensor.name.clone()));
    }
    Ok(())
}

/// Whether the family claims `directory` as a draft component.
pub fn recognizes(directory: &Directory) -> bool {
    recognize(directory).is_ok()
}

/// A parameter vector stored as `[n]` or as a `[1, n]` matrix.
fn bind_vector(
    tensors: &mut Tensors,
    name: &str,
    width: u64,
) -> Result<WeightDescriptor, HeaderError> {
    match tensors.shape(name)? {
        [1, columns] if *columns == width => {
            tensors.bind_transformed(name, &[1, width], vec![ImportTransform::Flatten])
        }
        _ => tensors.bind(name, &[width]),
    }
}

/// The draft's rotary table. Pairs are `(p, p + W/2)` over the whole head
/// (the stored rows are already in that order); YaRN, when declared, folds
/// into each pair's frequency and amplitude exactly.
fn rotary_table(m: &Metadata, width: u64) -> Result<Rotary, Error> {
    let base = m.positive_number("rope.freq_base")?;
    // Multi-axis sections whose every pair rotates by the first (text
    // position) axis are the plain table: the draft's rows are sequence
    // positions.
    if m.value("rope.dimension_sections").is_some() {
        let sections = m.counts("rope.dimension_sections")?;
        let pairs = sections.first().copied().unwrap_or(0);
        if sections.iter().skip(1).any(|pairs| *pairs != 0) || pairs.checked_mul(2) != Some(width) {
            return Err(Error::Geometry(format!(
                "rotary sections {sections:?} other than one text axis over the head"
            )));
        }
        if m.optional_string("rope.scaling.type")?.is_some() {
            return Err(Error::Geometry(
                "a draft's rotary is either sectioned or scaled".into(),
            ));
        }
    }
    match m.optional_string("rope.scaling.type")? {
        None => Ok(rotary::table(width, width, base)?),
        Some("yarn") => {
            let yarn = Yarn::read(m)?;
            // The header's multiplier on the YaRN amplitude; only the neutral
            // value is admitted.
            if yarn.attention_factor != 1.0 {
                return Err(Error::Geometry(
                    "YaRN attention factors other than 1.0 are not admitted".into(),
                ));
            }
            Ok(yarn.table(width, width, base)?)
        }
        Some(scaling) => Err(Error::Geometry(format!(
            "rope scaling {scaling:?} is not admitted"
        ))),
    }
}

fn product(dimensions: &[u64]) -> Result<u64, Error> {
    checked_product(dimensions).map_err(Error::Definition)
}

/// Build the numerical definition of a recognized draft against its
/// target's definition. `layer_entry` maps a target layer index, as the
/// draft names it, to the residual it taps (the target family's
/// [`magnitude_family_contracts::ModelFamily::layer_entry`]).
pub fn inspect(
    directory: &Directory,
    target: &ModelDefinition,
    layer_entry: &dyn Fn(u64) -> Result<TapPoint, FamilyError>,
) -> Result<DraftDefinition, Error> {
    recognize(directory)?;
    let m = Metadata::new(directory, ARCHITECTURE, &ADMITTED_KEYS)?;
    let decoder = &target.decoder;
    let block_count = m.integer("block_count")?;
    m.integer("context_length")?;
    let hidden = m.integer("embedding_length")?;
    let intermediate = m.integer("feed_forward_length")?;
    let heads = m.integer("attention.head_count")?;
    let kv_heads = m.integer("attention.head_count_kv")?;
    let width = m.integer("attention.key_length")?;
    let epsilon = m.positive_number("attention.layer_norm_rms_epsilon")?;
    let block_size = m.integer("block_size")?;
    if hidden != decoder.hidden {
        return Err(Error::Target(format!(
            "draft width {hidden} differs from the target's {}",
            decoder.hidden
        )));
    }
    if m.integer("attention.value_length")? != width {
        return Err(Error::Geometry(
            "value head width differs from key head width".into(),
        ));
    }
    if !width.is_multiple_of(2) || block_size < 2 {
        return Err(Error::Geometry(
            "draft heads need an even width and blocks two slots".into(),
        ));
    }
    // Each layer binds distinct stored weights; reject impossible counts
    // before allocating from untrusted metadata.
    if block_count > directory.tensors.len() as u64 {
        return Err(Error::Geometry(
            "block count exceeds the stored weight roles".into(),
        ));
    }
    let mask_token = directory
        .value(MASK_TOKEN)
        .and_then(Value::unsigned)
        .and_then(|token| u32::try_from(token).ok())
        .filter(|token| u64::from(*token) < decoder.vocabulary)
        .map(TokenId)
        .ok_or_else(|| Error::Metadata {
            key: MASK_TOKEN.into(),
            expected: "a token of the target's vocabulary",
        })?;

    let taps = m
        .counts("target_layers")?
        .into_iter()
        .map(|layer| layer_entry(layer).map_err(|error| Error::Target(error.to_string())))
        .collect::<Result<Vec<_>, _>>()?;
    if taps.is_empty() {
        return Err(Error::Geometry("draft taps no target layer".into()));
    }

    let (domains, block_attention): (Vec<_>, Vec<_>) =
        layer_attention(&m, block_count)?.into_iter().unzip();
    let rotary = rotary_table(&m, width)?;
    let query_rows = product(&[heads, width])?;
    let key_rows = product(&[kv_heads, width])?;
    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };

    // Projections and tables carry their NVFP4 second-level scale (`X.scale`
    // beside `X.weight`, `[1]` F32) when the file stores one.
    let mut binder = Tensors::new(directory);
    let fusion = binder.bind_scaled(
        "fc.weight",
        &[hidden, product(&[taps.len() as u64, hidden])?],
    )?;
    let fusion_norm = rms(binder.bind_scaled("enc.output_norm.weight", &[hidden])?);
    let output_norm = rms(binder.bind_scaled("output_norm.weight", &[hidden])?);
    let embedding = if binder.contains("token_embd.weight") {
        DraftEmbedding::Own(binder.bind_scaled("token_embd.weight", &[decoder.vocabulary, hidden])?)
    } else {
        DraftEmbedding::Target
    };
    let blocks = domains
        .into_iter()
        .enumerate()
        .map(|(index, domain)| {
            let role = |role: &str| format!("blk.{index}.{role}.weight");
            let mut weight = |name: &str, shape: &[u64]| binder.bind_scaled(&role(name), shape);
            let attention = Attention {
                heads,
                kv_heads,
                width,
                query: weight("attn_q", &[query_rows, hidden])?,
                gate: AttentionGate::None,
                query_norm: HeadNorm::Rms(rms(weight("attn_q_norm", &[width])?)),
                key_value: KeyValue::Owned {
                    key: weight("attn_k", &[key_rows, hidden])?,
                    value: ValueSource::Projected(weight("attn_v", &[key_rows, hidden])?),
                    key_norm: HeadNorm::Rms(rms(weight("attn_k_norm", &[width])?)),
                    value_norm: ValueNorm::None,
                    domain,
                },
                rotary: rotary.clone(),
                scale: 1.0 / (width as f64).sqrt(),
                reads: HistoryReads::Visible,
                media_rows: MediaRowAttention::Causal,
                output: weight("attn_output", &[hidden, query_rows])?,
            };
            let feed_forward = DenseFfn {
                intermediate,
                up: FeedForwardUp::Gated {
                    activation: ActivationFunction::Silu,
                    gate: weight("ffn_gate", &[intermediate, hidden])?,
                    up: weight("ffn_up", &[intermediate, hidden])?,
                },
                down: weight("ffn_down", &[hidden, intermediate])?,
            };
            Ok(Block {
                sublayers: vec![
                    Sublayer {
                        input: InputNorm::Rms(rms(weight("attn_norm", &[hidden])?)),
                        op: Operator::Attention(Box::new(attention)),
                        output: OutputForm::Residual,
                    },
                    Sublayer {
                        input: InputNorm::Rms(rms(weight("ffn_norm", &[hidden])?)),
                        op: Operator::DenseFfn(Box::new(feed_forward)),
                        output: OutputForm::Residual,
                    },
                ],
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;

    let dspark = binder.contains("markov_w1.weight");
    let dflash2 = binder.contains("selector_hidden.weight");
    if dspark && dflash2 {
        return Err(Error::Geometry(
            "a draft carries both DSpark and DFlash2 roles".into(),
        ));
    }
    let method = if dflash2 {
        if m.optional_flag("has_confidence_head")?.is_some()
            || m.optional_flag("sample_from_anchor")? == Some(true)
        {
            return Err(Error::Geometry(
                "DFlash2 drafts propose from their mask slots without a confidence head".into(),
            ));
        }
        dflash2_method(&m, &mut binder, hidden, decoder.vocabulary, block_count)?
    } else if dspark {
        let rank = directory
            .tensor("markov_w1.weight")
            .and_then(|tensor| tensor.shape.get(1).copied())
            .filter(|rank| *rank > 0)
            .ok_or_else(|| Error::Geometry("Markov embedding has no rank".into()))?;
        let table = [decoder.vocabulary, rank];
        let markov = MarkovHead {
            rank,
            embedding: binder.bind_scaled("markov_w1.weight", &table)?,
            projection: binder.bind_scaled("markov_w2.weight", &table)?,
        };
        if m.optional_flag("has_confidence_head")? == Some(false) {
            return Err(Error::Geometry(
                "a DSpark draft without its confidence head is not admitted".into(),
            ));
        }
        let confidence = ConfidenceHead {
            weight: bind_vector(&mut binder, "conf_proj.weight", hidden + rank)?,
            bias: bind_vector(&mut binder, "conf_proj.bias", 1)?,
        };
        DraftMethod::DSpark { markov, confidence }
    } else {
        if m.optional_flag("has_confidence_head")?.is_some() {
            return Err(Error::Geometry(
                "a confidence head requires the Markov head".into(),
            ));
        }
        DraftMethod::DFlash
    };
    if !dflash2
        && [
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ]
        .iter()
        .any(|key| m.value(key).is_some())
    {
        return Err(Error::Geometry(
            "convolution or selector metadata without the DFlash2 roles".into(),
        ));
    }
    // DSpark samples from the anchor unless the header says otherwise.
    let layout = if m.optional_flag("sample_from_anchor")?.unwrap_or(dspark) {
        BlockLayout::AnchorFirst
    } else {
        BlockLayout::MaskSlots
    };
    binder.finish()?;

    let draft = DraftDefinition {
        method,
        taps,
        fusion,
        fusion_norm,
        embedding,
        blocks,
        block_attention,
        output_norm,
        block_size,
        mask_token,
        layout,
    };
    draft
        .validate(decoder, target.inputs.coordinate_axes)
        .map_err(Error::Definition)?;
    Ok(draft)
}

/// DFlash2's convolutions around every draft sublayer and its candidate
/// selector.
fn dflash2_method(
    m: &Metadata<'_>,
    binder: &mut Tensors<'_>,
    hidden: u64,
    vocabulary: u64,
    block_count: u64,
) -> Result<DraftMethod, Error> {
    let kernel = m.integer("conv_kernel_size")?;
    let group = m.integer("conv_group_size")?;
    if kernel == 0 || group == 0 || !hidden.is_multiple_of(group) {
        return Err(Error::Geometry(
            "convolution groups must divide the hidden width".into(),
        ));
    }
    let coefficients = product(&[2, kernel, hidden / group])?;
    let convolutions = (0..block_count)
        .map(|index| {
            let mut convolution = |name: &str| -> Result<DynamicConvolution, Error> {
                Ok(DynamicConvolution {
                    base: binder.bind(
                        &format!("blk.{index}.{name}_conv_base"),
                        &[2, kernel, hidden],
                    )?,
                    projection: binder.bind_scaled(
                        &format!("blk.{index}.{name}_conv_proj.weight"),
                        &[coefficients, hidden],
                    )?,
                })
            };
            Ok(LayerConvolutions {
                attention: convolution("attn")?,
                feed_forward: convolution("ffn")?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let rank = m.integer("selector_rank")?;
    let top_k = m.integer("selector_top_k")?;
    if rank == 0 || top_k == 0 || top_k > vocabulary {
        return Err(Error::Geometry("candidate selector geometry".into()));
    }
    let codebook = [vocabulary, rank];
    let selector = CandidateSelector {
        rank,
        top_k,
        hidden: binder.bind_scaled("selector_hidden.weight", &[rank, hidden])?,
        predecessor: binder.bind_scaled("selector_predecessor.weight", &codebook)?,
        successor: binder.bind_scaled("selector_successor.weight", &codebook)?,
    };
    Ok(DraftMethod::DFlash2 {
        kernel,
        group,
        convolutions,
        selector,
    })
}

/// Each draft layer's history and block attention. A sliding layer (the
/// pattern's, or every layer when a window has no pattern) keeps the
/// `attention.sliding_window` positions with `q − k < W`, its own included,
/// and reads its block causally; a full layer keeps the whole context and
/// reads the whole block. `attention.causal`, when present, sets every
/// layer's block attention instead (the reference `is_causal` override).
fn layer_attention(
    m: &Metadata<'_>,
    block_count: u64,
) -> Result<Vec<(HistoryDomain, BlockAttention)>, Error> {
    let window = m.optional_integer("attention.sliding_window")?;
    let causal = m.optional_flag("attention.causal")?;
    let sliding = m
        .optional_flags("attention.sliding_window_pattern", block_count)?
        .unwrap_or_else(|| vec![window.is_some(); block_count as usize]);
    sliding
        .into_iter()
        .map(|sliding| {
            let domain = match (sliding, window) {
                (true, Some(tokens)) => HistoryDomain::Window { tokens },
                (true, None) => {
                    return Err(Error::Metadata {
                        key: m.key("attention.sliding_window"),
                        expected: "present for sliding layers",
                    })
                }
                (false, _) => HistoryDomain::Token,
            };
            let attention = match causal.unwrap_or(sliding) {
                true => BlockAttention::Causal,
                false => BlockAttention::Bidirectional,
            };
            Ok((domain, attention))
        })
        .collect()
}
