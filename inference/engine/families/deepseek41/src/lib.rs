//! DeepSeek V4.1 Flash (`deepseek41`) artifact recognition and numerical
//! interpretation, from the converter's GGUF (`inference/conversion`
//! `deepseek41.py`: llama.cpp v0.9.0 names and metadata, FP8 projections as
//! E4M3 codes beside their I8 grids of UE8M0 scales).
//!
//! The model is defined by DeepSeek's released code for
//! deepseek-ai/DeepSeek-V4.1-Flash-1017. Every layer is a window-attention
//! sublayer and a routed feed-forward, each between a Sinkhorn
//! hyper-connection mix site (`hc_*_fn`, `_base`, `_scale`) and a scatter
//! over the streams:
//!
//! - the mixing is single-pass: a sublayer reads the streams with the
//!   previous site's `pre`, its own norm after; `epsilon` is
//!   `hyper_connection.epsilon`, the one epsilon of `pre` and of Sinkhorn,
//!   `norm_epsilon` the model's RMS epsilon, and the exit mixes by the last
//!   site's `pre` under `output_norm`;
//! - attention: a low-rank query without a head norm, one 512-wide key head
//!   normalized by `attn_kv_a_norm` that is also its value, a 128-row
//!   window, per-head sinks, adjacent rotary pairs over the trailing
//!   `rope.dimension_count` dimensions turned back on the attended row, and
//!   the grouped low-rank output (`o_groups` blocks of `o_lora_rank` rows);
//! - routing: sqrt-softplus scores ranked on a selection bias (a second one,
//!   `exp_probs_b_vl`, for media rows), weights over their sum plus 1e-20
//!   scaled by `expert_weights_scale`, SwiGLU experts and an ungated shared
//!   expert clamped at each layer's `swiglu_clamp_*`.
//!
//! An execution may also round activations: each sublayer's normed input to
//! E4M3, the grouped output's first product, the expansions' outputs per
//! token and the mixed stream row to BF16. These are execution roundings,
//! not the model definition's (the reference, `families/deepseek41.py`,
//! excludes them), so this definition states none.
//!
//! An Engram layer (`ple.layers`) enters the streams between layers: the
//! trailing n-gram sublayer of the block before it, applied on the trunk
//! after the previous sublayer's post and before the layer's attention pre.
//! Its rows hash compressed token ids (`engram.token_map`), a predecessor
//! before the sequence reading the compressed pad and no token cutting; a
//! media row blocks. It adds `σ(signed sqrt(gate)) · value` with no
//! convolution, gated by the stored product `q_weight·k_weight` at the
//! model's RMS epsilon; its key and value are the two row ranges of the one
//! fused projection.
//!
//! A layer that compresses its history (`attention.compress_ratios` above
//! 0) reads a second source beside its window, in one softmax: one pooled
//! row per `ratio` tokens (`attn_compressor_*`, softmax-gated, normed,
//! rotated at the group's first token), of which a query reads the
//! indexer's picks. A kv and index source (Full) pools and indexes its own
//! rows: its indexer's query projects the low-rank query row, its keys the
//! rows' pre-rotary latent, with learned per-head weights. At ratio 1 a row
//! is its token's own (Identity). An index source that is no kv source
//! (Reindex) indexes the latest kv source's rows with that source's keys and
//! its own query and head weights. Every other compressed layer reads the
//! latest kv source's rows through the latest index source's picks (Reuse).
//! The candidate source (`attention.indexer.candidate_source_layer`)
//! produces a pool of index blocks from its scores, to which every later
//! index source restricts. The window, compressed and index rows carry
//! their trained roundings.

mod family;

pub use family::DeepSeek41Family;

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    PackageIdentity, UnsupportedReason,
};
use magnitude_family_common::{
    rotary::{self, Yarn, YarnAmplitude},
    HeaderError, Metadata, Tensors,
};
use magnitude_family_contracts::{
    checked_product, grouped_output_columns, ActivationDType, ActivationFunction, Attention, AttentionGate, Block,
    BlockElement, BlockRounding, BlockScale, Candidates, Decoder, EmbeddingScale, HeadWeights, HistoryReads,
    Indexer, IndexerInput, IndexerKeys, IndexerQuery, Pool, PooledRows, Selection, SublayerIndex, EntryForm, ExitForm, ExitNorm, ExitSite, ExpertSelection,
    FamilyError, FamilyId, FeedForwardUp, GluLimit, HeadNorm, HistoryRows, HistorySource,
    HyperConnectionExit, HyperConnectionInput, HyperConnectionMixing, HyperConnectionWeights,
    HyperConnections, ImportTransform, InputNorm, InputSemantics, MediaRowAttention,
    ProjectedRows, RowProducer, RowRounding, TokenDomain,
    ModelDefinition, NGramGate, NGramHash, NGramInput, NGramMedia, Operator, OutputForm, OutputProjection, OutputRotary, QueryProjection, ResidualForm, RmsNorm,
    Rotary, RotaryLayout, RotaryPairing, RotaryPlacement, RouteNormalization, RoutedFfn, Router,
    RouterInput, RowRange, ScoreFunction, SharedExpert, SharedExpertGate, SinkhornSite, Sublayer,
    ValueNorm,
    ValueSource, WeightDescriptor,
};

pub const FAMILY_ID: &str = "deepseek41";

/// The trained roundings of V4.1's history rows, part of its numerics
/// (`kernel.py`'s `act_quant` and `fp4_act_quant`; the reference's
/// `WINDOW`, `COMPRESSED` and `INDEX`), each with the model's own amax
/// floor.
/// - A window row, after norm and rotary (each held in BF16, as
///   `Attention._window_kv` holds them): E4M3 per 32 values, a power-of-two
///   scale, floor `1e-4`.
const WINDOW_ROUNDING: RowRounding = RowRounding::Block(BlockRounding {
    element: BlockElement::E4M3,
    block: 32,
    scale: BlockScale::Ue8m0,
    amax_floor: 1e-4,
});
/// - A compressed row: E2M1 per 16 values, an E4M3 scale, floor `6·2⁻⁹`.
const COMPRESSED_ROUNDING: RowRounding = RowRounding::Block(BlockRounding {
    element: BlockElement::E2M1,
    block: 16,
    scale: BlockScale::E4M3,
    amax_floor: 6.0 / 512.0,
});
/// - An indexer query or key: E2M1 per 32 values, a power-of-two scale,
///   floor `6·2⁻¹²⁶`.
const INDEX_ROUNDING: RowRounding = RowRounding::Block(BlockRounding {
    element: BlockElement::E2M1,
    block: 32,
    scale: BlockScale::Ue8m0,
    // 2⁻¹²⁶, F32's smallest normal.
    amax_floor: 6.0 * f32::MIN_POSITIVE as f64,
});

/// The keys every V4.1 file states. The YaRN, compressed-theta and indexer
/// keys describe compressed layers, which a file without them still states.
const KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.q_lora_rank",
    "attention.sliding_window",
    "attention.output_group_count",
    "attention.output_lora_rank",
    "attention.compress_ratios",
    "attention.compress_rope_freq_base",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast",
    "rope.scaling.yarn_beta_slow",
    "expert_count",
    "expert_used_count",
    "expert_gating_func",
    "expert_feed_forward_length",
    "expert_shared_count",
    "expert_weights_scale",
    "expert_weights_norm",
    "swiglu_clamp_exp",
    "swiglu_clamp_shexp",
    "hyper_connection.count",
    "hyper_connection.sinkhorn_iterations",
    "hyper_connection.epsilon",
    "hash_layer_count",
];

/// The keys only compressed layers read: their kv and index sources and
/// the candidate blocks. The converter omits each list no layer names.
const COMPRESSED_KEYS: &[&str] = &[
    "attention.kv_source_layers",
    "attention.indexer.source_layers",
    "attention.indexer.candidate_source_layer",
    "attention.indexer.candidate_top_k_blocks",
    "attention.indexer.candidate_block_size",
];

/// The keys only Engram layers read.
const ENGRAM_KEYS: &[&str] = &[
    "ple.layers",
    "ple.ngram_size",
    "ple.heads_per_ngram",
    "ple.layer_multipliers",
    "ple.head_offsets",
    "ple.head_vocab_sizes",
    "embedding_length_per_layer_input",
    "engram.pad_token_id",
    "engram.token_map",
];

/// llama.cpp's `expert_gating_func` for sqrt-softplus scores
/// (`LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`), the one V4.1 routes by.
const SQRT_SOFTPLUS: u64 = 4;

/// The combine weights' normalizer, `sum + 1e-20`: the released router's
/// own constant, as trained, not the model's RMS epsilon.
const ROUTE_EPSILON: f64 = 1e-20;

/// Where V4.1's rotary pairs sit: adjacent pairs `(2p, 2p + 1)` over a
/// head's trailing dimensions, from `head_dim - rope_dim` on.
pub const ROTARY_LAYOUT: RotaryLayout = RotaryLayout {
    pairs: RotaryPairing::Adjacent,
    placement: RotaryPlacement::Trailing,
};

pub fn recognize(directory: &Directory) -> Result<(), FamilyError> {
    match directory
        .value("general.architecture")
        .and_then(Value::string)
    {
        Some(FAMILY_ID) => Ok(()),
        architecture => Err(FamilyError::invalid(format!(
            "GGUF architecture {architecture:?} is not {FAMILY_ID:?}"
        ))),
    }
}

/// The rotary of a layer compressing its history at `ratio` (0: a window
/// layer), over `width`-wide heads in [`ROTARY_LAYOUT`], turned back on the
/// attended row. A window layer turns at `rope.freq_base`; a compressed one
/// at `attention.compress_rope_freq_base`, YaRN-scaled when the header
/// declares it, at unit amplitude: V4.1's YaRN has no mscale.
pub fn rotary(m: &Metadata, ratio: u64, width: u64) -> Result<Rotary, FamilyError> {
    let rotated = m.integer("rope.dimension_count")?;
    let output = OutputRotary::Inverse;
    let (base, scaling) = match ratio {
        0 => (m.positive_number("rope.freq_base")?, None),
        _ => (
            m.positive_number("attention.compress_rope_freq_base")?,
            m.optional_string("rope.scaling.type")?,
        ),
    };
    match scaling {
        None => Ok(rotary::laid_out(
            rotated,
            width,
            base,
            ROTARY_LAYOUT,
            output,
        )?),
        Some("yarn") => Ok(Yarn::read(m, YarnAmplitude::Unit)?.table(
            rotated,
            width,
            base,
            ROTARY_LAYOUT,
            output,
        )?),
        Some(kind) => Err(FamilyError::unsupported(
            vec![UnsupportedReason::RopeScaling { kind: kind.into() }],
            format!("{FAMILY_ID} rope scaling {kind:?}"),
        )),
    }
}

/// How one layer's attention reads compressed history beside its window
/// (`attention.compress_ratios`, the kv and index source lists; each layer
/// follows the latest source at or before it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compression {
    /// Its window alone (ratio 0).
    None,
    /// A kv and index source: it pools its own rows at `ratio` and indexes
    /// them, producing the candidate pool when it is the candidate source.
    Full { ratio: u64, pool: Option<CandidatePool> },
    /// An index source only (Reindex): it reads the rows and index keys of
    /// the latest kv source, `source`, with its own query and head weights,
    /// restricted to the candidate pool of `restrict` when the candidate
    /// source precedes it.
    Reindex { ratio: u64, source: u64, restrict: Option<u64> },
    /// Neither: it reads the rows of the latest kv source, `source`, and
    /// the picks of the latest index source, `indexer` (that source, or a
    /// Reindex layer over its rows).
    Reuse { ratio: u64, source: u64, indexer: u64 },
}

/// The candidate pool's blocks and units a block
/// (`attention.indexer.candidate_top_k_blocks`, `candidate_block_size`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidatePool {
    blocks: u64,
    block_units: u64,
}

/// Every decoder layer's [`Compression`]. Index sources after the candidate
/// source restrict to its pool, which therefore must index the rows they
/// read: the latest kv source.
fn compressions(m: &Metadata, layers: u64) -> Result<Vec<Compression>, FamilyError> {
    let ratios = m.counts("attention.compress_ratios")?;
    // The entries past the decoder's layers are the DSpark drafter's, a
    // separate file, whose layers attend a window.
    let key = m.key("attention.compress_ratios");
    let (main, drafter) = ratios.split_at_checked(layers as usize).ok_or_else(|| {
        FamilyError::metadata(&key, format!("{key} names fewer than {layers} layers"))
    })?;
    if drafter.iter().any(|ratio| *ratio != 0) {
        return Err(FamilyError::metadata(
            &key,
            format!("{key}: the drafter's layers compress"),
        ));
    }
    if main.iter().all(|ratio| *ratio == 0) {
        return Ok(vec![Compression::None; layers as usize]);
    }
    // The converter omits a list no kept layer names.
    let list = |key: &str| match m.value(key) {
        None => Ok(Vec::new()),
        Some(_) => m.counts(key),
    };
    let kv = list("attention.kv_source_layers")?;
    let index = list("attention.indexer.source_layers")?;
    // A source of -1 is the converter's none.
    let candidate = match m.value("attention.indexer.candidate_source_layer") {
        Some(Value::Scalar(Scalar::Signed(-1))) | None => None,
        Some(value) => Some(value.unsigned().ok_or_else(|| {
            let key = m.key("attention.indexer.candidate_source_layer");
            FamilyError::metadata(&key, format!("{key} is neither a layer nor -1"))
        })?),
    };
    // The pool is produced by a compressed layer of the decoder.
    if let Some(candidate) = candidate {
        match main.get(candidate as usize).copied() {
            None => {
                return Err(FamilyError::invalid(format!(
                    "{FAMILY_ID}: candidate source {candidate} is past the decoder's {layers} layers"
                )));
            }
            Some(0) => {
                return Err(FamilyError::invalid(format!(
                    "{FAMILY_ID}: candidate source {candidate} compresses no history"
                )));
            }
            Some(_) => {}
        }
    }
    // The pool is produced by a kv and index source, from its own rows.
    if let Some(candidate) = candidate.filter(|c| !kv.contains(c) || !index.contains(c)) {
        return Err(FamilyError::invalid(format!(
            "{FAMILY_ID}: candidate source {candidate} is not a kv and index source"
        )));
    }
    let pool = candidate
        .map(|_| {
            Ok::<_, FamilyError>(CandidatePool {
                blocks: m.integer("attention.indexer.candidate_top_k_blocks")?,
                block_units: m.integer("attention.indexer.candidate_block_size")?,
            })
        })
        .transpose()?;
    let latest = |sources: &[u64], layer: u64| sources.iter().copied().filter(|s| *s <= layer).max();
    (0..layers)
        .map(|layer| {
            let ratio = main[layer as usize];
            if ratio == 0 {
                return Ok(Compression::None);
            }
            let (Some(source), Some(indexer)) = (latest(&kv, layer), latest(&index, layer)) else {
                return Err(FamilyError::invalid(format!(
                    "{FAMILY_ID}: compressed layer {layer} follows no kv and index source"
                )));
            };
            if main[source as usize] != ratio {
                return Err(FamilyError::invalid(format!(
                    "{FAMILY_ID}: layer {layer} compresses at {ratio}, its source {source} at {}",
                    main[source as usize]
                )));
            }
            // The candidate source restricts the index sources after it.
            let restrict = candidate.filter(|candidate| *candidate < layer);
            match (source == layer, indexer == layer) {
                (true, true) if restrict.is_some() => Err(FamilyError::invalid(format!(
                    "{FAMILY_ID}: kv source {layer} follows the candidate source"
                ))),
                (true, true) => Ok(Compression::Full {
                    ratio,
                    pool: pool.filter(|_| candidate == Some(layer)),
                }),
                (false, true) if restrict.is_some_and(|candidate| candidate != source) => {
                    Err(FamilyError::invalid(format!(
                        "{FAMILY_ID}: layer {layer} reads kv source {source}, not the candidate source"
                    )))
                }
                (false, true) => Ok(Compression::Reindex { ratio, source, restrict }),
                (false, false) => Ok(Compression::Reuse { ratio, source, indexer }),
                (true, false) => Err(FamilyError::invalid(format!(
                    "{FAMILY_ID}: kv source {layer} does not index"
                ))),
            }
        })
        .collect()
}

/// The Engram layers of a file that has them (`ple.layers`), each with its
/// row rule: its constants are its position's in `ple.layers`.
struct Engram {
    layers: Vec<(u64, NGramHash)>,
    /// Row width (`embedding_length_per_layer_input`).
    width: u64,
}

impl Engram {
    fn read(m: &Metadata, block_count: u64) -> Result<Option<Self>, FamilyError> {
        if m.value("ple.layers").is_none() {
            return Ok(None);
        }
        let layers = m.counts("ple.layers")?;
        let ngram = m.integer("ple.ngram_size")?;
        let context = ngram.checked_sub(1).filter(|context| *context > 0).ok_or_else(|| {
            FamilyError::metadata(
                m.key("ple.ngram_size"),
                format!("{FAMILY_ID} n-grams span at least two tokens"),
            )
        })?;
        let heads_per_order = m.integer("ple.heads_per_ngram")?;
        let heads = product(&[context, heads_per_order])?;
        let multipliers = m.counts("ple.layer_multipliers")?;
        let sizes = m.counts("ple.head_vocab_sizes")?;
        let offsets = m.counts("ple.head_offsets")?;
        let count = layers.len() as u64;
        for (key, values, per) in [
            ("ple.layer_multipliers", &multipliers, ngram),
            ("ple.head_vocab_sizes", &sizes, heads),
            ("ple.head_offsets", &offsets, heads),
        ] {
            if values.len() as u64 != product(&[count, per])? {
                return Err(FamilyError::metadata(
                    m.key(key),
                    format!("{} states {per} values per Engram layer", m.key(key)),
                ));
            }
        }
        let token_map = m
            .counts("engram.token_map")?
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| {
                FamilyError::metadata(
                    m.key("engram.token_map"),
                    format!("{} maps a token past u32", m.key("engram.token_map")),
                )
            })?;
        // A predecessor before the sequence reads the compressed pad: the
        // token map's entry for the pad token.
        let pad = m.integer("engram.pad_token_id")?;
        let fill = token_map.get(pad as usize).copied().ok_or_else(|| {
            FamilyError::metadata(
                m.key("engram.pad_token_id"),
                format!("{} is past the token map", m.key("engram.pad_token_id")),
            )
        })?;
        let rules = layers
            .iter()
            .enumerate()
            .map(|(position, layer)| {
                if *layer >= block_count {
                    return Err(FamilyError::metadata(
                        m.key("ple.layers"),
                        format!("Engram layer {layer} exceeds the block count"),
                    ));
                }
                let span = |per: u64| (position as u64 * per) as usize..((position as u64 + 1) * per) as usize;
                Ok((
                    *layer,
                    NGramHash {
                        context,
                        heads_per_order,
                        multipliers: multipliers[span(ngram)].to_vec(),
                        sizes: sizes[span(heads)].to_vec(),
                        offsets: offsets[span(heads)].to_vec(),
                        token_map: Some(token_map.clone()),
                        fill: u64::from(fill),
                        boundary: None,
                        media: NGramMedia::Blocking,
                    },
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            layers: rules,
            width: m.integer("embedding_length_per_layer_input")?,
        }))
    }
}

/// The header's geometry every V4.1 layer reads: its attention (the window,
/// the indexer over compressed sources, the grouped output projection), its
/// routed and shared experts with each layer's clamps, and the
/// hyper-connection streams its sublayers and Engram inputs mix.
struct Geometry {
    hidden: u64,
    heads: u64,
    width: u64,
    query_rank: u64,
    window: u64,
    /// The indexer's heads, their width and the rows it picks.
    index_heads: u64,
    index_width: u64,
    index_top: u64,
    groups: u64,
    output_rank: u64,
    epsilon: f64,
    experts: u64,
    selected: u64,
    intermediate: u64,
    /// The shared expert's width, when the layer has one.
    shared: Option<u64>,
    normalization: RouteNormalization,
    scale: f64,
    streams: u64,
    /// Each layer's routed and shared expansion clamps.
    clamps: Vec<(Option<GluLimit>, Option<GluLimit>)>,
}

impl Geometry {
    fn read(m: &Metadata, layers: u64) -> Result<Self, FamilyError> {
        let heads = m.integer("attention.head_count")?;
        let width = m.integer("attention.key_length")?;
        let equal = |key: &str, expected: u64| -> Result<(), FamilyError> {
            match m.integer(key)? {
                value if value == expected => Ok(()),
                value => Err(FamilyError::metadata(
                    m.key(key),
                    format!("{} is {value}, V4.1 states {expected}", m.key(key)),
                )),
            }
        };
        // One key head, which is also the value head.
        equal("attention.head_count_kv", 1)?;
        equal("attention.value_length", width)?;
        equal("expert_gating_func", SQRT_SOFTPLUS)?;
        if m.count("hash_layer_count")? != 0 {
            let key = m.key("hash_layer_count");
            return Err(FamilyError::metadata(
                &key,
                format!("{key}: V4.1 hashes no layer"),
            ));
        }
        let clamp = |key: &str| -> Result<Vec<Option<GluLimit>>, FamilyError> {
            m.layer_numbers(key, layers)?
                .into_iter()
                // A limit of 0 is no clamp, as the released definition reads it.
                .map(|limit| match limit == 0.0 {
                    true => Ok(None),
                    false => GluLimit::new(limit)
                        .map(Some)
                        .map_err(|error| FamilyError::metadata(m.key(key), error.to_string())),
                })
                .collect()
        };
        let shared = match m.count("expert_shared_count")? {
            0 => None,
            count => Some(product(&[count, m.integer("expert_feed_forward_length")?])?),
        };
        let selected = m.integer("expert_used_count")?;
        Ok(Self {
            hidden: m.integer("embedding_length")?,
            heads,
            width,
            query_rank: m.integer("attention.q_lora_rank")?,
            window: m.integer("attention.sliding_window")?,
            index_heads: m.integer("attention.indexer.head_count")?,
            index_width: m.integer("attention.indexer.key_length")?,
            index_top: m.integer("attention.indexer.top_k")?,
            groups: m.integer("attention.output_group_count")?,
            output_rank: m.integer("attention.output_lora_rank")?,
            epsilon: m.positive_number("attention.layer_norm_rms_epsilon")?,
            experts: m.integer("expert_count")?,
            selected,
            intermediate: m.integer("expert_feed_forward_length")?,
            shared,
            // A lone selected expert keeps its raw score: the released router
            // normalizes only a selection of more than one.
            normalization: match m.flag("expert_weights_norm")? && selected > 1 {
                true => RouteNormalization::SumPlusEpsilon(ROUTE_EPSILON),
                false => RouteNormalization::None,
            },
            scale: m.positive_number("expert_weights_scale")?,
            streams: m.integer("hyper_connection.count")?,
            clamps: clamp("swiglu_clamp_exp")?
                .into_iter()
                .zip(clamp("swiglu_clamp_shexp")?)
                .collect(),
        })
    }

    fn rms(&self, weight: WeightDescriptor) -> RmsNorm {
        RmsNorm {
            weight,
            epsilon: self.epsilon,
        }
    }

    /// One sublayer between its Sinkhorn site `hc_{site}_*` and the
    /// streams, its mixed input normalized by `{norm}.weight`.
    fn sited(
        &self,
        weight: &mut Weight,
        site: &str,
        norm: &str,
        op: Operator,
    ) -> Result<Sublayer, FamilyError> {
        let mixes = product(&[2 + self.streams, self.streams])?;
        let wide = product(&[self.streams, self.hidden])?;
        Ok(Sublayer {
            input: InputNorm::HyperConnectionMix(HyperConnectionInput {
                site: HyperConnectionWeights::Sinkhorn(SinkhornSite {
                    function: weight(&format!("hc_{site}_fn.weight"), &[mixes, wide])?,
                    base: weight(&format!("hc_{site}_base.weight"), &[mixes])?,
                    scale: weight(&format!("hc_{site}_scale.weight"), &[3])?,
                }),
                norm: Some(self.rms(weight(&format!("{norm}.weight"), &[self.hidden])?)),
            }),
            op,
            output: OutputForm::Scatter,
        })
    }

    /// A compressed layer's second history source (part 1): its own pooled
    /// rows, indexed by its own indexer (Full); a kv source's rows and index
    /// keys, indexed by its own query (Reindex); or a kv source's rows read
    /// through an index source's picks (Reuse).
    fn compressed(
        &self,
        weight: &mut Weight,
        compression: Compression,
    ) -> Result<Option<HistorySource>, FamilyError> {
        let (width, hidden, index_width) = (self.width, self.hidden, self.index_width);
        match compression {
            Compression::None => Ok(None),
            Compression::Reuse { source, indexer, .. } => Ok(Some(HistorySource {
                rows: HistoryRows::Shared { source: attention_layer(source)?, part: 1 },
                reads: HistoryReads::Selected(Selection::Reuse {
                    source: attention_layer(indexer)?,
                    part: 1,
                }),
            })),
            Compression::Reindex { source, restrict, .. } => {
                let source = attention_layer(source)?;
                let candidates = match restrict {
                    Some(producer) => Candidates::Restrict { source: attention_layer(producer)?, part: 1 },
                    None => Candidates::All,
                };
                let indexer = self.indexer(weight, |_| Ok(IndexerKeys::Shared { source, part: 1 }), candidates)?;
                Ok(Some(HistorySource {
                    rows: HistoryRows::Shared { source, part: 1 },
                    reads: HistoryReads::Selected(Selection::Own(Box::new(indexer))),
                }))
            }
            Compression::Full { ratio, pool: candidate_pool } => {
                let pool = match ratio {
                    1 => Pool::Identity,
                    _ => Pool::SoftmaxGated {
                        gate: weight("attn_compressor_gate.weight", &[width, hidden])?,
                    },
                };
                let rows = PooledRows {
                    rate: ratio,
                    projection: weight("attn_compressor_kv.weight", &[width, hidden])?,
                    pool,
                    norm: self.rms(weight("attn_compressor_norm.weight", &[width])?),
                    rounding: COMPRESSED_ROUNDING,
                };
                let candidates = match candidate_pool {
                    Some(CandidatePool { blocks, block_units }) => Candidates::Produce { blocks, block_units },
                    None => Candidates::All,
                };
                let indexer = self.indexer(
                    weight,
                    |weight| {
                        Ok(IndexerKeys::Latent {
                            projection: weight("indexer.attn_k.weight", &[index_width, width])?,
                            norm: self.rms(weight("indexer.k_norm.weight", &[index_width])?),
                            rounding: INDEX_ROUNDING,
                        })
                    },
                    candidates,
                )?;
                Ok(Some(HistorySource {
                    rows: HistoryRows::Own(RowProducer::Pooled(rows)),
                    reads: HistoryReads::Selected(Selection::Own(Box::new(indexer))),
                }))
            }
        }
    }

    /// An index source's indexer over the low-rank query row, with its
    /// `keys` (read between its query and head weights) and `candidates`.
    fn indexer(
        &self,
        weight: &mut Weight,
        keys: impl FnOnce(&mut Weight) -> Result<IndexerKeys, FamilyError>,
        candidates: Candidates,
    ) -> Result<Indexer, FamilyError> {
        let (heads, index_width) = (self.index_heads, self.index_width);
        Ok(Indexer {
            heads,
            width: index_width,
            query: IndexerQuery {
                input: IndexerInput::QueryRank,
                projection: weight(
                    "indexer.attn_q_b.weight",
                    &[product(&[heads, index_width])?, self.query_rank],
                )?,
                norm: HeadNorm::None,
                rounding: INDEX_ROUNDING,
            },
            keys: keys(weight)?,
            head_weights: HeadWeights::Projected(weight("indexer.proj.weight", &[heads, self.hidden])?),
            // `weights_proj · x · (index_head_dim · index_n_heads)^-½`.
            scale: ((index_width * heads) as f64).powf(-0.5),
            top: self.index_top,
            candidates,
        })
    }

    fn attention(
        &self,
        weight: &mut Weight,
        rotary: Rotary,
        compression: Compression,
    ) -> Result<Attention, FamilyError> {
        let (heads, width, hidden) = (self.heads, self.width, self.hidden);
        let rows = product(&[heads, width])?;
        let columns = grouped_output_columns(rows, self.groups)
            .map_err(|error| FamilyError::invalid(format!("{FAMILY_ID}: {error}")))?;
        let ranks = product(&[self.groups, self.output_rank])?;
        Ok(Attention {
            heads,
            kv_heads: 1,
            width,
            query: QueryProjection::LowRank {
                down: weight("attn_q_a.weight", &[self.query_rank, hidden])?,
                norm: self.rms(weight("attn_q_a_norm.weight", &[self.query_rank])?),
                up: weight("attn_q_b.weight", &[rows, self.query_rank])?,
            },
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            history: vec![HistorySource::visible(HistoryRows::Own(RowProducer::Projected(
                ProjectedRows {
                    key: weight("attn_kv.weight", &[width, hidden])?,
                    value: ValueSource::StoredKey,
                    key_norm: HeadNorm::Rms(
                        self.rms(weight("attn_kv_a_norm.weight", &[width])?),
                    ),
                    value_norm: ValueNorm::None,
                    domain: TokenDomain::Window {
                        tokens: self.window,
                    },
                    rounding: WINDOW_ROUNDING,
                },
            )))]
            .into_iter()
            .chain(self.compressed(weight, compression)?)
            .collect(),
            rotary,
            // The softmax scale is `head_dim ** -0.5`.
            scale: (width as f64).powf(-0.5),
            media_rows: MediaRowAttention::Causal,
            sinks: Some(weight("attn_sinks.weight", &[heads])?),
            output: OutputProjection::GroupedLowRank {
                groups: self.groups,
                rank: self.output_rank,
                a: weight("attn_output_a.weight", &[ranks, columns])?,
                b: weight("attn_output_b.weight", &[hidden, ranks])?,
            },
        })
    }

    fn routed(&self, weight: &mut Weight, layer: u64) -> Result<RoutedFfn, FamilyError> {
        let (experts, intermediate, hidden) = (self.experts, self.intermediate, self.hidden);
        let (routed_limit, shared_limit) = self.clamps[layer as usize];
        let silu = |limit, gate, up| FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            limit,
            gate,
            up,
        };
        let stacked = [experts, intermediate, hidden];
        Ok(RoutedFfn {
            experts,
            selected: self.selected,
            intermediate,
            router: Router {
                weight: weight("ffn_gate_inp.weight", &[experts, hidden])?,
                input: RouterInput::Operator,
                score: ScoreFunction::SqrtSoftplus,
                selection: ExpertSelection::TopK {
                    bias: Some(weight("exp_probs_b.bias", &[experts])?),
                },
                media_bias: Some(weight("exp_probs_b_vl.bias", &[experts])?),
                normalization: self.normalization,
                scale: self.scale,
            },
            expert_up: silu(
                routed_limit,
                weight("ffn_gate_exps.weight", &stacked)?,
                weight("ffn_up_exps.weight", &stacked)?,
            ),
            expert_down: weight("ffn_down_exps.weight", &[experts, hidden, intermediate])?,
            expert_scale: None,
            latent: None,
            shared: self
                .shared
                .map(|shared| -> Result<SharedExpert, FamilyError> {
                    Ok(SharedExpert {
                        intermediate: shared,
                        up: silu(
                            shared_limit,
                            weight("ffn_gate_shexp.weight", &[shared, hidden])?,
                            weight("ffn_up_shexp.weight", &[shared, hidden])?,
                        ),
                        down: weight("ffn_down_shexp.weight", &[hidden, shared])?,
                        gate: SharedExpertGate::None,
                    })
                })
                .transpose()?,
        })
    }
}

impl Geometry {
    /// Engram layer `layer`'s n-gram input over `table`: its key and value
    /// are the row ranges `[0, S·H)` and `[S·H, (S + 1)·H)` of the fused
    /// projection `engram_kv` (unflattened to `S + 1` rows of `H`), its gate
    /// the stored product of the key and query weights over the streams
    /// (`[S, H]`).
    fn engram(
        &self,
        tensors: &mut Tensors,
        layer: u64,
        hash: NGramHash,
        width: u64,
        table: &WeightDescriptor,
    ) -> Result<Sublayer, FamilyError> {
        let p = format!("blk.{layer}.");
        let wide = product(&[self.streams, self.hidden])?;
        let heads = hash
            .heads()
            .map_err(|error| FamilyError::invalid(error.to_string()))?;
        let fused = format!("{p}engram_kv.weight");
        let stored = [
            product(&[self.streams + 1, self.hidden])?,
            product(&[heads, width])?,
        ];
        let scale = tensors.companion_scale(&fused, 1)?;
        let mut rows = |start, rows| {
            tensors.bind_transformed(
                &fused,
                &stored,
                scale
                    .iter()
                    .cloned()
                    .chain([ImportTransform::Rows(RowRange { start, rows })])
                    .collect(),
            )
        };
        let key = rows(0, wide)?;
        let value = rows(wide, self.hidden)?;
        let gate = tensors.bind_transformed(
            &format!("{p}engram_gate.weight"),
            &[self.streams, self.hidden],
            vec![ImportTransform::Flatten],
        )?;
        Ok(Sublayer {
            input: InputNorm::None,
            op: Operator::NGramInput(Box::new(NGramInput {
                hash,
                width,
                table: table.clone(),
                key,
                value,
                gate: NGramGate::Product(gate),
                epsilon: self.epsilon,
                mixing: None,
            })),
            output: OutputForm::Residual,
        })
    }
}

/// Binds one layer's weight by its name within the layer.
type Weight<'a> = dyn FnMut(&str, &[u64]) -> Result<WeightDescriptor, FamilyError> + 'a;

/// Decoder layer `layer`'s attention sublayer.
fn attention_layer(layer: u64) -> Result<SublayerIndex, FamilyError> {
    Ok(SublayerIndex {
        block: u32::try_from(layer).map_err(|_| FamilyError::invalid(format!("{FAMILY_ID}: layer {layer}")))?,
        sublayer: 0,
    })
}

fn product(dimensions: &[u64]) -> Result<u64, FamilyError> {
    checked_product(dimensions).map_err(|error| FamilyError::invalid(error.to_string()))
}

/// Inspect already-opened package components. This exists for composition
/// roots and header-only fixtures; component ownership remains with
/// `Package`.
pub fn inspect_components(
    directory: &Directory,
    projector: Option<&Directory>,
    identity: PackageIdentity,
) -> Result<ModelDefinition, FamilyError> {
    recognize(directory)?;
    if projector.is_some() {
        return Err(FamilyError::mismatch(format!(
            "{FAMILY_ID}'s vision projector is not interpreted yet"
        )));
    }
    // Every layer kind's keys, to refuse the unbound forms by name.
    let every = Metadata::new(
        directory,
        FAMILY_ID,
        &[KEYS, COMPRESSED_KEYS, ENGRAM_KEYS].concat(),
    )?;
    let layers = every.integer("block_count")?;
    // Each layer requires distinct stored weights. Reject impossible counts
    // before allocating per-layer vectors from untrusted metadata.
    if layers > directory.tensors.len() as u64 {
        return Err(FamilyError::invalid(format!(
            "{FAMILY_ID} layer count exceeds available weight roles"
        )));
    }
    let compressions = compressions(&every, layers)?;
    let engram = Engram::read(&every, layers)?;
    let compressed = compressions.iter().any(|layer| *layer != Compression::None);
    let keys = [
        KEYS,
        if compressed { COMPRESSED_KEYS } else { &[] },
        if engram.is_some() { ENGRAM_KEYS } else { &[] },
    ]
    .concat();
    let m = Metadata::new(directory, FAMILY_ID, &keys)?;
    let geometry = Geometry::read(&m, layers)?;
    let hidden = geometry.hidden;
    let streams = geometry.streams;
    let window_rotary = rotary(&m, 0, geometry.width)?;
    let mut tensors = Tensors::new(directory);
    let vocabulary = match tensors.shape("token_embd.weight")? {
        [vocabulary, _] => *vocabulary,
        shape => {
            return Err(HeaderError::NotMatrix {
                name: "token_embd.weight".into(),
                shape: shape.to_vec(),
            }
            .into());
        }
    };
    // An FP8 projection binds with its UE8M0 grid.
    let mut tensor = |name: &str, shape: &[u64]| -> Result<WeightDescriptor, FamilyError> {
        Ok(tensors.bind_scaled(name, shape)?)
    };
    let embedding = tensor("token_embd.weight", &[vocabulary, hidden])?;
    let output = tensor("output.weight", &[vocabulary, hidden])?;
    let output_norm = geometry.rms(tensor("output_norm.weight", &[hidden])?);
    let mut blocks = (0..layers)
        .map(|layer| {
            let p = format!("blk.{layer}.");
            let weight: &mut Weight = &mut |name, shape| tensor(&format!("{p}{name}"), shape);
            let compression = compressions[layer as usize];
            let attention = match compression {
                Compression::None => {
                    geometry.attention(weight, window_rotary.clone(), compression)?
                }
                Compression::Full { ratio, .. }
                | Compression::Reindex { ratio, .. }
                | Compression::Reuse { ratio, .. } => {
                    geometry.attention(weight, rotary(&m, ratio, geometry.width)?, compression)?
                }
            };
            let routed = geometry.routed(weight, layer)?;
            Ok(Block {
                sublayers: vec![
                    geometry.sited(
                        weight,
                        "attn",
                        "attn_norm",
                        Operator::Attention(Box::new(attention)),
                    )?,
                    geometry.sited(
                        weight,
                        "ffn",
                        "ffn_norm",
                        Operator::RoutedFfn(Box::new(routed)),
                    )?,
                ],
            })
        })
        .collect::<Result<Vec<_>, FamilyError>>()?;
    // Each Engram layer is the trailing sublayer of the block before it,
    // its weights keeping its own layer's names; every one reads the one
    // table, its rows at its own offsets.
    if let Some(engram) = engram {
        let name = "per_layer_token_embd.weight";
        let table = tensors.bind_scaled(name, &[tensors.rows(name)?, engram.width])?;
        for (layer, hash) in engram.layers {
            let sublayer = geometry.engram(&mut tensors, layer, hash, engram.width, &table)?;
            layer
                .checked_sub(1)
                .and_then(|before| blocks.get_mut(before as usize))
                .ok_or_else(|| {
                    FamilyError::invalid(format!(
                        "{FAMILY_ID}: Engram layer {layer} follows no decoder layer"
                    ))
                })?
                .sublayers
                .push(sublayer);
        }
    }
    let definition = ModelDefinition {
        family: FamilyId(FAMILY_ID.into()),
        artifact_identity: identity,
        inputs: InputSemantics { coordinate_axes: 1 },
        decoder: Decoder {
            activation_dtype: ActivationDType::BF16,
            hidden,
            vocabulary,
            context_limit: m.integer("context_length")?,
            residual: ResidualForm::HyperConnections(HyperConnections {
                streams,
                mixing: HyperConnectionMixing::Sinkhorn {
                    iterations: m.integer("hyper_connection.sinkhorn_iterations")?,
                    epsilon: m.positive_number("hyper_connection.epsilon")?,
                    norm_epsilon: geometry.epsilon,
                },
            }),
            entry: EntryForm {
                embedding,
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
            },
            blocks,
            exit: ExitForm {
                norm: ExitNorm::HyperConnectionHead(HyperConnectionExit {
                    site: ExitSite::PreviousSite,
                    norm: Some(output_norm),
                }),
                output,
                softcap: None,
            },
        },
        head: None,
        vision: None,
        draft: None,
    };
    tensors.finish()?;
    definition
        .validate()
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
    Ok(definition)
}
