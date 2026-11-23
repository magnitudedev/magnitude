//! DFlash, DSpark, DFlash2 and LiLiCorr draft components (GGUF architecture
//! `dflash`, plan §3.8).
//!
//! A draft is a separate package component read against its target's
//! definition: its taps name target layers, it has no vocabulary of its own
//! (the target's embedding and vocabulary projection serve it, unless it
//! ships its own embedding table), and its width is the target's. DSpark is
//! the same architecture with a Markov bias and a confidence head, detected
//! by `markov_w1.weight`. A convolved backbone wraps every draft sublayer in
//! a grouped dynamic causal convolution, declared by `conv_kernel_size` and
//! `conv_group_size`. DFlash2 is a convolved backbone that selects an ordered
//! candidate path, detected by its `selector_*` roles. LiLiCorr scores the
//! block's candidate lattice with a corrector, declared by its `lilicorr.*`
//! metadata; its layers are stored as `blk.N` beyond `block_count` and its
//! feature projection as the block after them. The shared architecture label
//! names none of them. Any draft layer may carry per-head attention sinks
//! (`attn_sinks`).
//!
//! GGUF stores the draft's `attn_q`/`attn_k` rows in the half-split (NEOX)
//! rotary layout, the engine's own, so rotary is a plain frequency table (YaRN
//! when the header declares it); the q/k norms apply per head.

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    TokenId, UnsupportedReason,
};
use magnitude_family_common::{
    binders,
    rotary::{self, Yarn, YarnAmplitude},
    HeaderError, Metadata, Tensors,
};
use magnitude_family_contracts::{
    checked_product, ActivationFunction, Affine, Attention, AttentionGate, Block, BlockAttention,
    BlockConvolutions, BlockLayout, CandidateCorrector, CandidateSelector, CenteredNorm,
    ConfidenceHead, CorrectorLayer, DefinitionError, DenseFfn, DraftDefinition, DraftEmbedding,
    DraftMethod, DynamicConvolution, FamilyError, FeedForwardUp, GluLimit, HeadNorm, HistoryRows,
    HistorySource, HyperConnectionMixing, HyperConnections, InputNorm, LayerConvolutions,
    MarkovHead, MediaRowAttention, ModelDefinition, Operator, OutputForm, OutputProjection,
    OutputRotary, ProjectedRows, QueryKeys, QueryProjection, ResidualForm, RmsNorm, Rotary,
    RotaryLayout, RouteNormalization, RowProducer, RowRounding, ScoreFunction, SinkhornInput,
    Sublayer, TapPoint, TokenDomain, ValueNorm, ValueSource, WeightDescriptor, CORRECTOR_FEATURES,
};
use std::{error, fmt};

/// The GGUF architecture of a DFlash-family block drafter.
pub const ARCHITECTURE: &str = "dflash";

/// The LiLiCorr corrector's metadata namespace.
const LILICORR: &str = "lilicorr";

/// The corrector's feature norm epsilon: the reference implementation's
/// layer norm default (PyTorch `nn.LayerNorm`, llama.cpp `ggml_norm` at
/// 1e-5); the header does not carry it.
const CORRECTOR_FEATURE_EPSILON: f64 = 1e-5;

const MASK_TOKEN: &str = "tokenizer.ggml.mask_token_id";

/// The declaration of the draft's block attention, which is never inferred.
const CAUSALITY: &str = "dflash.attention.causal";

/// Failure to recognize or interpret a `dflash` draft.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// `general.architecture` is missing or not `dflash`.
    Architecture(Option<String>),
    /// No explicit checkpoint attention semantics were published.
    MissingCausality,
    /// Every metadata key and stored tensor outside the admitted feature set.
    Unadmitted {
        metadata: Vec<String>,
        tensors: Vec<String>,
    },
    /// A rope scaling the draft family does not execute.
    RopeScaling(String),
    /// A required key is missing or not of its admitted type.
    Metadata {
        key: String,
        expected: &'static str,
    },
    /// Metadata that is well typed but outside the admitted geometry.
    Geometry(String),
    /// A metadata array whose payload doesn't read or decode.
    Payload(String),
    /// The draft disagrees with its target's definition.
    Target(String),
    MissingWeight(String),
    WeightShape {
        name: String,
        expected: Vec<u64>,
        received: Vec<u64>,
    },
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
            Self::MissingCausality => formatter.write_str(
                "missing dflash.attention.causal; drafting requires explicit attention semantics",
            ),
            Self::Unadmitted { metadata, tensors } => write!(
                formatter,
                "draft features outside the admitted set: metadata {metadata:?}, tensors {tensors:?}"
            ),
            Self::RopeScaling(kind) => write!(formatter, "draft rope scaling {kind:?} is not admitted"),
            Self::Metadata { key, expected } => {
                write!(formatter, "draft metadata {key:?} must be {expected}")
            }
            Self::Geometry(message) => write!(formatter, "draft geometry: {message}"),
            Self::Payload(message) => formatter.write_str(message),
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
            Self::Definition(error) => write!(formatter, "draft definition: {error}"),
        }
    }
}

impl error::Error for Error {}

impl Error {
    /// The unsupported features this error names. A draft that disagrees
    /// with its target, or whose geometry or assembled definition are
    /// outside what drafting admits, does not fit the target it was bound
    /// to; one that omits a key it needs names that key (its attention
    /// causality is one a published file may predate); one missing a weight
    /// it needs, or whose metadata payload doesn't read, is `Unreadable`.
    pub fn reasons(&self) -> Vec<UnsupportedReason> {
        match self {
            Self::Architecture(architecture) => vec![UnsupportedReason::Architecture {
                architecture: architecture.clone().unwrap_or_default(),
            }],
            Self::Unadmitted { metadata, tensors } => metadata
                .iter()
                .map(|key| UnsupportedReason::UnrecognizedMetadata { key: key.clone() })
                .chain(
                    tensors
                        .iter()
                        .map(|name| UnsupportedReason::UnrecognizedTensor { name: name.clone() }),
                )
                .collect(),
            Self::RopeScaling(kind) => vec![UnsupportedReason::RopeScaling { kind: kind.clone() }],
            Self::MissingCausality => vec![UnsupportedReason::MissingMetadata {
                key: CAUSALITY.into(),
            }],
            Self::Geometry(_) | Self::Target(_) | Self::Definition(_) => {
                vec![UnsupportedReason::TargetMismatch {
                    detail: self.to_string(),
                }]
            }
            Self::WeightShape { name, .. } => {
                vec![UnsupportedReason::UnrecognizedTensor { name: name.clone() }]
            }
            Self::Metadata {
                key,
                expected: PRESENT,
            } => vec![UnsupportedReason::MissingMetadata { key: key.clone() }],
            Self::MissingWeight(_) | Self::Payload(_) => vec![UnsupportedReason::Unreadable],
            Self::Metadata { key, .. } => {
                vec![UnsupportedReason::UnrecognizedMetadata { key: key.clone() }]
            }
        }
    }
}

/// What an [`Error::Metadata`] for a missing key expects.
const PRESENT: &str = "present";

impl From<HeaderError> for Error {
    fn from(error: HeaderError) -> Self {
        match error {
            HeaderError::UnknownMetadata(metadata) => Self::Unadmitted {
                metadata,
                tensors: Vec::new(),
            },
            HeaderError::MissingMetadata(key) => Self::Metadata {
                key,
                expected: PRESENT,
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
            HeaderError::UnboundWeights(tensors) => Self::Unadmitted {
                metadata: Vec::new(),
                tensors,
            },
            HeaderError::Definition(error) => Self::Definition(error),
            HeaderError::RopeScaling(kind) => Self::RopeScaling(kind),
            error @ (HeaderError::NotMatrix { .. }
            | HeaderError::CompanionScale { .. }
            | HeaderError::Rotary { .. }
            | HeaderError::Yarn) => Self::Geometry(error.to_string()),
            error @ HeaderError::Payload { .. } => Self::Payload(error.to_string()),
        }
    }
}

/// Recognize a `dflash` draft from its declared architecture, without
/// interpreting geometry. No target family claims this architecture, and this
/// family claims nothing else.
pub fn recognize(directory: &Directory) -> Result<(), Error> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
    {
        Some(ARCHITECTURE) => Ok(()),
        architecture => Err(Error::Architecture(architecture.map(str::to_owned))),
    }
}

/// Whether the family claims `directory` as a draft component.
pub fn recognizes(directory: &Directory) -> bool {
    recognize(directory).is_ok()
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
        if m.optional_string("rope.scaling.type")?
            .is_some_and(|scaling| scaling != "none")
        {
            return Err(Error::Geometry(
                "a draft's rotary is either sectioned or scaled".into(),
            ));
        }
    }
    match m.optional_string("rope.scaling.type")? {
        None | Some("none") => Ok(rotary::table(width, width, base)?),
        Some("yarn") => {
            // The drafts' YaRN scales its amplitude by YaRN's mscale, under
            // the header's multiplier; only the neutral multiplier is
            // admitted.
            let amplitude = YarnAmplitude::mscale(m)?;
            if amplitude != (YarnAmplitude::Mscale { attention_factor: 1.0 }) {
                return Err(Error::Geometry(
                    "YaRN attention factors other than 1.0 are not admitted".into(),
                ));
            }
            Ok(Yarn::read(m, amplitude)?.table(
                width,
                width,
                base,
                RotaryLayout::HALF_SPLIT_LEADING,
                OutputRotary::None,
            )?)
        }
        Some(scaling) => Err(Error::RopeScaling(scaling.to_owned())),
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
    let m = Metadata::new(directory, ARCHITECTURE);
    let decoder = &target.decoder;
    let block_count = m.bounded_block_count("block_count")?;
    // The longest sequence the draft was trained on: no layer reads further
    // back than it (a full layer's history is a window of that length).
    let context_length = m.integer("context_length")?;
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
    let single_pass_mhc = m.optional_flag("hyper_connection.single_pass")? == Some(true);
    m.invariant_count(
        "attention.value_length",
        width,
        "equal to attention.key_length",
    )?;
    let _ = m.vocab_size(decoder.vocabulary)?;
    if !width.is_multiple_of(2) || block_size < 2 {
        return Err(Error::Geometry(
            "draft heads need an even width and blocks two slots".into(),
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

    let rms = |weight: WeightDescriptor| RmsNorm { weight, epsilon };

    // Projections and tables carry their NVFP4 second-level scale (`X.scale`
    // beside `X.weight`, `[1]` F32) when the file stores one.
    let mut binder = Tensors::new(directory);
    let fusion = binder.bind(
        "fc.weight",
        &[hidden, product(&[taps.len() as u64, hidden])?],
    )?;
    let fusion_norm = rms(binder.bind("enc.output_norm.weight", &[hidden])?);
    let output_norm = rms(binder.bind("output_norm.weight", &[hidden])?);
    let embedding = if binder.contains("token_embd.weight") {
        DraftEmbedding::Own(binder.bind("token_embd.weight", &[decoder.vocabulary, hidden])?)
    } else {
        DraftEmbedding::Target
    };
    let output = if binder.contains("output.weight") {
        Some(binder.bind("output.weight", &[decoder.vocabulary, hidden])?)
    } else {
        None
    };

    let (residual, blocks, block_attention) = if single_pass_mhc {
        deepseek41_dspark_blocks(
            &m,
            &mut binder,
            block_count,
            hidden,
            heads,
            kv_heads,
            width,
            epsilon,
        )?
    } else {
        let (domains, block_attention): (Vec<_>, Vec<_>) =
            layer_attention(&m, block_count, context_length, false)?
                .into_iter()
                .unzip();
        let rotary = rotary_table(&m, width)?;
        let query_rows = product(&[heads, width])?;
        let key_rows = product(&[kv_heads, width])?;
        let blocks = domains
            .into_iter()
            .enumerate()
            .map(|(index, domain)| {
                let role = |role: &str| format!("blk.{index}.{role}.weight");
                let sinks = role("attn_sinks");
                let sinks = if binder.contains(&sinks) {
                    Some(binder.bind(&sinks, &[heads])?)
                } else {
                    None
                };
                let mut weight = |name: &str, shape: &[u64]| binder.bind(&role(name), shape);
                let attention = Attention {
                    heads,
                    kv_heads,
                    width,
                    query: QueryProjection::Full(weight("attn_q", &[query_rows, hidden])?),
                    query_keys: QueryKeys::Direct,
                    gate: AttentionGate::None,
                    query_norm: HeadNorm::Rms(rms(weight("attn_q_norm", &[width])?)),
                    history: vec![HistorySource::visible(HistoryRows::Own(
                        RowProducer::Projected(ProjectedRows {
                            key: weight("attn_k", &[key_rows, hidden])?,
                            value: ValueSource::Projected(weight("attn_v", &[key_rows, hidden])?),
                            key_norm: HeadNorm::Rms(rms(weight("attn_k_norm", &[width])?)),
                            value_norm: ValueNorm::None,
                            domain,
                            rounding: RowRounding::None,
                        }),
                    ))],
                    rotary: rotary.clone(),
                    scale: 1.0 / (width as f64).sqrt(),
                    media_rows: MediaRowAttention::Causal,
                    sinks,
                    output: OutputProjection::Full(weight("attn_output", &[hidden, query_rows])?),
                };
                let feed_forward = DenseFfn {
                    intermediate,
                    up: FeedForwardUp::Gated {
                        activation: ActivationFunction::Silu,
                        limit: None,
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
        (ResidualForm::Single, blocks, block_attention)
    };

    let convolutions = block_convolutions(&m, &mut binder, hidden, block_count)?;
    let dspark = binder.contains("markov_w1.weight");
    let dflash2 = binder.contains("selector_hidden.weight");
    let lilicorr = directory
        .metadata
        .iter()
        .any(|entry| entry.name.starts_with("lilicorr."));
    if usize::from(dspark) + usize::from(dflash2) + usize::from(lilicorr) > 1 {
        return Err(Error::Geometry(
            "a draft carries the roles of more than one of DSpark, DFlash2 and LiLiCorr".into(),
        ));
    }
    // A selector-less header may still declare an empty selector.
    if !dflash2
        && ["selector_rank", "selector_top_k"]
            .into_iter()
            .map(|key| m.optional_count(key))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .any(|value| value.is_some_and(|value| value != 0))
    {
        return Err(Error::Geometry(
            "candidate selector metadata without the selector roles".into(),
        ));
    }
    if (dflash2 || lilicorr)
        && (m.optional_flag("has_confidence_head")?.is_some()
            || m.optional_flag("sample_from_anchor")? == Some(true))
    {
        return Err(Error::Geometry(
            "DFlash2 and LiLiCorr drafts propose from their mask slots without a confidence head"
                .into(),
        ));
    }
    let method = if dflash2 {
        if convolutions.is_none() {
            return Err(Error::Geometry(
                "a DFlash2 draft convolves its layers".into(),
            ));
        }
        DraftMethod::DFlash2 {
            selector: candidate_selector(&m, &mut binder, hidden, decoder.vocabulary)?,
        }
    } else if lilicorr {
        DraftMethod::LiLiCorr {
            corrector: Box::new(candidate_corrector(
                directory,
                &mut binder,
                hidden,
                block_count,
                block_size,
                epsilon,
            )?),
        }
    } else if dspark {
        let rank = directory
            .tensor("markov_w1.weight")
            .and_then(|tensor| tensor.shape.get(1).copied())
            .filter(|rank| *rank > 0)
            .ok_or_else(|| Error::Geometry("Markov embedding has no rank".into()))?;
        let table = [decoder.vocabulary, rank];
        let markov = MarkovHead {
            rank,
            embedding: binder.bind("markov_w1.weight", &table)?,
            projection: binder.bind("markov_w2.weight", &table)?,
        };
        if m.optional_flag("has_confidence_head")? == Some(false) {
            return Err(Error::Geometry(
                "a DSpark draft without its confidence head is not admitted".into(),
            ));
        }
        let conf_weight_shape = directory
            .tensor("conf_proj.weight")
            .map(|tensor| tensor.shape.as_slice());
        let confidence = if conf_weight_shape == Some(&[1, hidden + rank]) {
            ConfidenceHead {
                weight: binder.bind("conf_proj.weight", &[1, hidden + rank])?,
                bias: if binder.contains("conf_proj.bias") {
                    Some(binder.bind("conf_proj.bias", &[1])?)
                } else {
                    None
                },
            }
        } else {
            ConfidenceHead {
                weight: binder.bind("conf_proj.weight", &[hidden + rank])?,
                bias: if binder.contains("conf_proj.bias") {
                    Some(binder.bind("conf_proj.bias", &[1])?)
                } else {
                    None
                },
            }
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
        output,
        embedding_scale: m.optional_positive_number("embedding_scale")?.unwrap_or(1.0),
        residual,
        blocks,
        block_attention,
        convolutions,
        output_norm,
        logit_softcap: m.optional_number("final_logit_softcapping")?.filter(|cap| *cap != 0.0),
        block_size,
        mask_token,
        layout,
    };
    m.finish()?;
    draft
        .validate(decoder, target.inputs.coordinate_axes)
        .map_err(Error::Definition)?;
    Ok(draft)
}

/// A convolved backbone's convolutions around every draft sublayer, when the
/// header declares them.
fn block_convolutions(
    m: &Metadata<'_>,
    binder: &mut Tensors<'_>,
    hidden: u64,
    block_count: u64,
) -> Result<Option<BlockConvolutions>, Error> {
    let (Some(kernel), Some(group)) = (
        m.optional_integer("conv_kernel_size")?,
        m.optional_integer("conv_group_size")?,
    ) else {
        if m.value("conv_kernel_size").is_some() || m.value("conv_group_size").is_some() {
            return Err(Error::Geometry(
                "a convolved draft declares both its kernel and its group".into(),
            ));
        }
        return Ok(None);
    };
    if kernel == 0 || group == 0 || !hidden.is_multiple_of(group) {
        return Err(Error::Geometry(
            "convolution groups must divide the hidden width".into(),
        ));
    }
    let coefficients = product(&[2, kernel, hidden / group])?;
    let layers = (0..block_count)
        .map(|index| {
            let mut convolution = |name: &str| -> Result<DynamicConvolution, Error> {
                Ok(DynamicConvolution {
                    base: binder.bind(
                        &format!("blk.{index}.{name}_conv_base"),
                        &[2, kernel, hidden],
                    )?,
                    projection: binder.bind(
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
    Ok(Some(BlockConvolutions {
        kernel,
        group,
        layers,
    }))
}

/// DFlash2's candidate selector.
fn candidate_selector(
    m: &Metadata<'_>,
    binder: &mut Tensors<'_>,
    hidden: u64,
    vocabulary: u64,
) -> Result<CandidateSelector, Error> {
    let rank = m.integer("selector_rank")?;
    let top_k = m.integer("selector_top_k")?;
    if rank == 0 || top_k == 0 || top_k > vocabulary {
        return Err(Error::Geometry("candidate selector geometry".into()));
    }
    let codebook = [vocabulary, rank];
    Ok(CandidateSelector {
        rank,
        top_k,
        hidden: binder.bind("selector_hidden.weight", &[rank, hidden])?,
        predecessor: binder.bind("selector_predecessor.weight", &codebook)?,
        successor: binder.bind("selector_successor.weight", &codebook)?,
    })
}

/// LiLiCorr's candidate corrector. Its layers are stored as the blocks after
/// the draft's (`blk.{block_count + i}`, each layer's attention input
/// projection as `lilicorr.layers.{i}.attn.in_proj`), its feature
/// projection and norm as the block after them, and its other maps under
/// `lilicorr.`. Matrices carry their NVFP4 second-level scale when the file
/// stores one; biases, norms, embeddings and the attention bias are stored
/// plainly.
fn candidate_corrector(
    directory: &Directory,
    binder: &mut Tensors<'_>,
    hidden: u64,
    block_count: u64,
    block_size: u64,
    epsilon: f64,
) -> Result<CandidateCorrector, Error> {
    let m = Metadata::new(directory, LILICORR);
    let width = m.integer("hidden_size")?;
    let top_k = m.integer("candidate_topk")?;
    let heads = m.integer("num_heads")?;
    let layers = m.bounded_block_count("num_layers")?;
    let factor = m.integer("factor_dim")?;
    let mlp = m.integer("mlp_hidden_size")?;
    let vector_epsilon = m.positive_number("vector_eps")?;
    let logit_scale = m.positive_number("logit_scale")?;
    m.finish()?;
    if width == 0
        || top_k == 0
        || heads == 0
        || !width.is_multiple_of(heads)
        || factor == 0
        || mlp == 0
        || block_size < 2
    {
        return Err(Error::Geometry("corrector geometry".into()));
    }
    let slots = block_size - 1;
    let nodes = product(&[slots, top_k])?;
    let affine = |binder: &mut Tensors<'_>,
                      stem: &str,
                      outputs: u64,
                      inputs: u64|
     -> Result<Affine, Error> {
        Ok(Affine {
            weight: binder.bind(&format!("{stem}.weight"), &[outputs, inputs])?,
            bias: binder.bind(&format!("{stem}.bias"), &[outputs])?,
        })
    };
    let rms = |weight| RmsNorm { weight, epsilon };
    let token = if binder.contains("lilicorr.token_proj.weight") {
        Some(affine(binder, "lilicorr.token_proj", width, hidden)?)
    } else {
        None
    };
    let pass = affine(binder, "lilicorr.pass_hidden_proj", width, hidden)?;
    let context = affine(binder, "lilicorr.context_proj", width, hidden)?;
    let features = block_count + layers;
    let feature_norm = CenteredNorm {
        weight: binder.bind(
            &format!("blk.{features}.ffn_norm.weight"),
            &[CORRECTOR_FEATURES],
        )?,
        bias: binder.bind(
            &format!("blk.{features}.ffn_norm.bias"),
            &[CORRECTOR_FEATURES],
        )?,
        epsilon: CORRECTOR_FEATURE_EPSILON,
    };
    let feature_up = affine(
        binder,
        &format!("blk.{features}.ffn_up"),
        width,
        CORRECTOR_FEATURES,
    )?;
    let feature_down = affine(binder, &format!("blk.{features}.ffn_down"), width, width)?;
    let slot_embedding = binder.bind("lilicorr.slot_embedding.weight", &[slots, width])?;
    let rank_embedding = binder.bind("lilicorr.rank_embedding.weight", &[top_k, width])?;
    let attention_bias = binder.bind("lilicorr.attention_bias.bias", &[heads, nodes, nodes])?;
    let corrector_layers = (0..layers)
        .map(|index| {
            let block = block_count + index;
            Ok(CorrectorLayer {
                attention_norm: rms(binder.bind(&format!("blk.{block}.attn_norm.weight"), &[width])?),
                attention_input: affine(
                    binder,
                    &format!("lilicorr.layers.{index}.attn.in_proj"),
                    product(&[3, width])?,
                    width,
                )?,
                attention_output: affine(
                    binder,
                    &format!("blk.{block}.attn_output"),
                    width,
                    width,
                )?,
                mlp_norm: rms(binder.bind(&format!("blk.{block}.ffn_norm.weight"), &[width])?),
                up: affine(binder, &format!("blk.{block}.ffn_up"), mlp, width)?,
                down: affine(binder, &format!("blk.{block}.ffn_down"), width, mlp)?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(CandidateCorrector {
        width,
        top_k,
        heads,
        factor,
        token,
        pass,
        context,
        feature_norm,
        feature_up,
        feature_down,
        slot_embedding,
        rank_embedding,
        attention_bias,
        layers: corrector_layers,
        output_norm: rms(binder.bind("lilicorr.output_norm.weight", &[width])?),
        anchor_norm: rms(binder.bind("lilicorr.anchor_norm.weight", &[width])?),
        factor_input: affine(
            binder,
            "lilicorr.factor_input_proj",
            width,
            product(&[3, width])?,
        )?,
        outgoing: affine(binder, "lilicorr.out_head", factor, width)?,
        incoming: affine(binder, "lilicorr.in_head", factor, width)?,
        anchor: affine(binder, "lilicorr.anchor_out_head", factor, width)?,
        vector_epsilon,
        logit_scale,
    })
}

/// DeepSeek V4.1 DSpark blocks (`dflash.hyper_connection.single_pass = true`):
/// 4-stream single-pass Sinkhorn mHC, window-only MLA attention (trailing
/// adjacent rotary with inverse output rotation), and sqrt-softplus MoE with a
/// shared expert and SwiGLU clamps. Shares its operator binders with
/// `deepseek41`.
#[allow(clippy::too_many_arguments)]
fn deepseek41_dspark_blocks(
    m: &Metadata<'_>,
    binder: &mut Tensors<'_>,
    block_count: u64,
    hidden: u64,
    heads: u64,
    kv_heads: u64,
    width: u64,
    epsilon: f64,
) -> Result<(ResidualForm, Vec<Block>, Vec<BlockAttention>), Error> {
    if kv_heads != 1 {
        return Err(Error::Geometry(format!(
            "DeepSeek V4.1 DSpark requires attention.head_count_kv = 1, got {kv_heads}"
        )));
    }
    m.invariant_count(
        "expert_gating_func",
        binders::SQRT_SOFTPLUS,
        "4 (sqrt-softplus)",
    )?;
    let ratios = m.counts("attention.compress_ratios")?;
    if ratios.len() as u64 != block_count || ratios.iter().any(|ratio| *ratio != 0) {
        return Err(Error::Geometry(format!(
            "DeepSeek V4.1 DSpark requires {block_count} zero compress_ratios, got {ratios:?}"
        )));
    }
    let streams = m.integer("hyper_connection.count")?;
    let sinkhorn_iterations = m.integer("hyper_connection.sinkhorn_iterations")?;
    let hc_epsilon = m.positive_number("hyper_connection.epsilon")?;
    let residual = ResidualForm::HyperConnections(HyperConnections {
        streams,
        mixing: HyperConnectionMixing::Sinkhorn {
            iterations: sinkhorn_iterations,
            epsilon: hc_epsilon,
            norm_epsilon: epsilon,
            input: SinkhornInput::PreviousSite,
        },
    });
    let window = m.integer("attention.sliding_window")?;
    let block_attention = match m.value("attention.causal") {
        // DeepSeek V4.1 DSpark's block rows attend the full block bidirectionally
        // (`get_dspark_topk_idxs`), and the converter omits `dflash.attention.causal`.
        None => vec![BlockAttention::Bidirectional; block_count as usize],
        Some(Value::Scalar(Scalar::Bool(causal))) => vec![
            if *causal {
                BlockAttention::Causal
            } else {
                BlockAttention::Bidirectional
            };
            block_count as usize
        ],
        Some(Value::Array(_)) => m
            .flags("attention.causal", block_count)?
            .into_iter()
            .map(|causal| {
                if causal {
                    BlockAttention::Causal
                } else {
                    BlockAttention::Bidirectional
                }
            })
            .collect(),
        Some(_) => {
            return Err(Error::Metadata {
                key: m.key("attention.causal"),
                expected: "a BOOL or ARRAY<BOOL> with one entry per layer",
            })
        }
    };
    let rotated = m.integer("rope.dimension_count")?;
    let base = m.positive_number("rope.freq_base")?;
    let rotary = rotary::laid_out(
        rotated,
        width,
        base,
        binders::DEEPSEEK41_ROTARY_LAYOUT,
        OutputRotary::Inverse,
    )?;
    let attention_form = binders::MlaWindowForm {
        hidden,
        heads,
        width,
        query_rank: m.integer("attention.q_lora_rank")?,
        window,
        groups: m.integer("attention.output_group_count")?,
        output_rank: m.integer("attention.output_lora_rank")?,
        epsilon,
    };
    let experts = m.integer("expert_count")?;
    let selected = m.integer("expert_used_count")?;
    let intermediate = m.integer("expert_feed_forward_length")?;
    let shared = match m.default_count("expert_shared_count", 0)? {
        0 => None,
        count => Some(product(&[count, intermediate])?),
    };
    let normalization = match m.flag("expert_weights_norm")? && selected > 1 {
        true => RouteNormalization::SumPlusEpsilon(binders::ROUTE_EPSILON),
        false => RouteNormalization::None,
    };
    let scale = m.positive_number("expert_weights_scale")?;
    let clamp = |key: &str| -> Result<Vec<Option<GluLimit>>, Error> {
        m.uniform_or_layer_numbers(key, block_count)?
            .into_iter()
            .map(|limit| match limit == 0.0 {
                true => Ok(None),
                false => binders::glu_limit(m, key, limit)
                    .map(Some)
                    .map_err(|error| Error::Geometry(error.to_string())),
            })
            .collect()
    };
    let clamps: Vec<(Option<GluLimit>, Option<GluLimit>)> = clamp("swiglu_clamp_exp")?
        .into_iter()
        .zip(clamp("swiglu_clamp_shexp")?)
        .collect();
    let mut blocks = Vec::with_capacity(block_count as usize);
    for layer in 0..block_count {
        let prefix = format!("blk.{layer}.");
        let mut bind = |name: &str, shape: &[u64]| -> Result<WeightDescriptor, FamilyError> {
            binder
                .bind(&format!("{prefix}{name}"), shape)
                .map_err(FamilyError::from)
        };
        let attention = binders::mla_window_attention(&mut bind, &attention_form, rotary.clone(), None)
            .map_err(|error| Error::Geometry(error.to_string()))?;
        let (routed_limit, shared_limit) = clamps[layer as usize];
        let routed = binders::routed(
            &mut bind,
            binders::RoutedForm {
                experts,
                selected,
                intermediate,
                hidden,
                shared,
                score: ScoreFunction::SqrtSoftplus,
                media_bias: false,
                normalization,
                scale,
                routed_limit,
                shared_limit,
            },
        )
        .map_err(|error| Error::Geometry(error.to_string()))?;
        let attn_sublayer = binders::sinkhorn_sited(
            &mut bind,
            streams,
            hidden,
            epsilon,
            "attn",
            "attn_norm",
            Operator::Attention(Box::new(attention)),
        )
        .map_err(|error| Error::Geometry(error.to_string()))?;
        let ffn_sublayer = binders::sinkhorn_sited(
            &mut bind,
            streams,
            hidden,
            epsilon,
            "ffn",
            "ffn_norm",
            Operator::RoutedFfn(Box::new(routed)),
        )
        .map_err(|error| Error::Geometry(error.to_string()))?;
        blocks.push(Block {
            sublayers: vec![attn_sublayer, ffn_sublayer],
        });
    }
    Ok((residual, blocks, block_attention))
}

/// History windows and block causality are independent checkpoint properties.
/// Causality must be explicit: one BOOL broadcasts, ARRAY<BOOL> names each layer.
/// A sliding layer reads its declared window; a full layer reads at most the
/// draft's trained context (`context_length`), never further back than any
/// sequence it learned from.
fn layer_attention(
    m: &Metadata<'_>,
    block_count: u64,
    context_length: u64,
    default_bidirectional: bool,
) -> Result<Vec<(TokenDomain, BlockAttention)>, Error> {
    let window = m.optional_integer("attention.sliding_window")?;
    let causal = match m.value("attention.causal") {
        None if default_bidirectional => vec![false; block_count as usize],
        None => return Err(Error::MissingCausality),
        Some(Value::Scalar(Scalar::Bool(causal))) => vec![*causal; block_count as usize],
        Some(Value::Array(_)) => m.flags("attention.causal", block_count)?,
        Some(_) => {
            return Err(Error::Metadata {
                key: m.key("attention.causal"),
                expected: "a BOOL or ARRAY<BOOL> with one entry per layer",
            })
        }
    };
    let sliding = m
        .optional_flags("attention.sliding_window_pattern", block_count)?
        .unwrap_or_else(|| vec![window.is_some(); block_count as usize]);
    sliding
        .into_iter()
        .zip(causal)
        .map(|(sliding, causal)| {
            let domain = match (sliding, window) {
                (true, Some(tokens)) => TokenDomain::Window { tokens },
                (true, None) => {
                    return Err(Error::Metadata {
                        key: m.key("attention.sliding_window"),
                        expected: "present for sliding layers",
                    })
                }
                (false, _) => TokenDomain::Window {
                    tokens: context_length,
                },
            };
            let attention = match causal {
                true => BlockAttention::Causal,
                false => BlockAttention::Bidirectional,
            };
            Ok((domain, attention))
        })
        .collect()
}
