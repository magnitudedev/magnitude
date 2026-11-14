//! GLM-5.3-Flash (`glm5next`) artifact recognition and numerical
//! interpretation, from the converter's GGUF (`inference/conversion`
//! `glm5next.py`: upstream llama.cpp v0.9.0 `glm5-next` names and metadata,
//! FP8 projections as E4M3 codes beside their F32 128×128 tile grids, the
//! MTP layer last in llama.cpp's nextn layout).
//!
//! The model is defined by transformers 5.17.0's `Glm5NextTextModel` (the
//! reference, `families/glm5next.py`). Every trunk layer is a mixer and a
//! feed-forward, each between a Sinkhorn hyper-connection site (`hc_*_fn`,
//! `_base`, `_scale`) and a scatter over the streams; a sublayer mixes the
//! streams by its *own* site's `pre`, then its norm, and the exit is the
//! streams' mean under `output_norm`. `epsilon` is `hyper_connection.epsilon`,
//! the projection's RMS epsilon the model's.
//!
//! - The mixer of a layer with `attention.head_count_kv` 0 is Kimi Delta
//!   Attention: q, k and v stored apart, each convolved by its own plane; the
//!   per-channel decay `lower·sigmoid(exp(A_log)·(f_b·f_a·x + dt_bias))`
//!   (`ssm_a` holds `−exp(A_log)`); a low-rank sigmoid output gate.
//! - With `head_count_kv` 1 it is multi-head latent attention without
//!   positions: a low-rank query, one 512-wide latent row per token that is
//!   key and value, each query head absorbed by `attn_k_b` at run time and
//!   each head's output by `attn_v_b`, over the rows its own indexer picks.
//! - The indexer projects its query from the low-rank query row; its keys are
//!   each token's `LayerNorm(wk·x)` with a bias, pooled per `kpool` tokens by
//!   a softmax gate with a per-place term (`ape`); a row reads the rows of
//!   its best `top_k / kpool` complete pools and its tail.
//! - The first `leading_dense_block_count` feed-forwards are dense, clamped
//!   at `swiglu_clamp_shexp`; the rest route by sigmoid scores ranked on a
//!   selection bias, weights over their sum plus 1e-20 scaled by
//!   `expert_weights_scale`, SwiGLU experts clamped at `swiglu_clamp_exp` and
//!   a shared expert at `swiglu_clamp_shexp`.
//! - The `nextn_predict_layers` trailing blocks (none where the key is
//!   absent, as upstream writes a file without them) are the MTP head: the next
//!   token's normed embedding beside the normed final hidden row, combined
//!   by `eh_proj`, through one plain-residual layer of the same attention and
//!   routed feed-forward, normed by `shared_head_norm` into the target's
//!   output projection. Its attention reads the rows its own indexer picks,
//!   as every layer's does: the head keeps pooled indexer state of its own.

mod family;
pub mod projector;

pub use family::Glm5NextFamily;

use magnitude_artifacts::{
    gguf::{Directory, Value},
    PackageIdentity,
};
use magnitude_family_common::{Metadata, Tensors, binders};
use magnitude_family_contracts::{
    checked_product, ActivationDType, ActivationFunction, Attention, AttentionGate, Block,
    Candidates, DecayGate, Decoder, DenseFfn, EmbeddingScale, EntryForm, ExitForm, ExitNorm,
    ExitSite, FamilyError, FamilyId, FeedForwardUp, GateFunction,
    GateProjection, GatedDelta, GluLimit, Head, HeadBlock, HeadChain, HeadComponent, HeadHistory,
    HeadInput, HeadNorm, HeadReadout, HeadWeights, HistoryReads, HistoryRows,
    HistorySource, HyperConnectionExit, HyperConnectionMixing,
    HyperConnections, ImportTransform, Indexer, IndexerInput, InputNorm,
    IndexerKeys, IndexerQuery, InputSemantics, LayerNorm, LowRank,
    MediaRowAttention, ModelDefinition, Operator, OutputForm, OutputProjection, Pool, PoolNorm,
    PooledRows, ProjectedRows, QueryKeys, QueryProjection, RecurrentDecay, RecurrentGate,
    RecurrentHeadMapping, ResidualForm, RmsNorm, Rotary, RouteNormalization, RoutedFfn,
    RowProducer, RowRounding, ScoreFunction, Selection, SinkhornInput, Sublayer, TokenDomain, ValueNorm,
    ValueSource, VocabularyProjection, WeightDescriptor,
};

pub const FAMILY_ID: &str = "glm5-next";

/// The keys a GLM-5.3-Flash file may state, each read below; every other `glm5-next.*` key is refused.
const KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "vocab_size",
    "feed_forward_length",
    "leading_dense_block_count",
    "nextn_predict_layers",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.layer_norm_rms_epsilon",
    "attention.layer_norm_epsilon",
    "attention.q_lora_rank",
    "attention.kv_lora_rank",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "attention.indexer.kpool",
    "attention.indexer.kpool_select_tail",
    "attention.indexer.types",
    "rope.dimension_count",
    "ssm.conv_kernel",
    "kda.head_dim",
    "kda.gate_lower_bound",
    "hyper_connection.count",
    "hyper_connection.sinkhorn_iterations",
    "hyper_connection.epsilon",
    "expert_count",
    "expert_used_count",
    "expert_group_count",
    "expert_group_used_count",
    "expert_gating_func",
    "expert_feed_forward_length",
    "expert_shared_count",
    "expert_weights_scale",
    "expert_weights_norm",
    "swiglu_clamp_exp",
    "swiglu_clamp_shexp",
];

/// llama.cpp's `expert_gating_func` for sigmoid scores
/// (`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`), the one GLM routes by.
const SIGMOID: u64 = 2;


/// The epsilon of KDA's q and k L2 norms, `x / sqrt(Σx² + 1e-6)`:
/// transformers' `l2norm`, which the header does not state.
const KDA_KEY_NORM_EPSILON: f64 = 1e-6;

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

/// A block's mixer (`attention.head_count_kv`: 0 KDA, 1 latent attention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mixer {
    Kda,
    Latent,
}

/// The header's geometry every layer reads.
struct Geometry {
    hidden: u64,
    heads: u64,
    epsilon: f64,
    streams: u64,
    /// KDA: head width, convolution taps, the decay's lower bound.
    kda_width: u64,
    taps: u64,
    lower: f64,
    /// Latent attention: query rank, latent width, query and value head
    /// widths.
    query_rank: u64,
    latent: u64,
    key_width: u64,
    value_width: u64,
    /// The indexer: heads, width, rows it picks, tokens a pool and the key
    /// norm's epsilon.
    index_heads: u64,
    index_width: u64,
    index_top: u64,
    kpool: u64,
    index_epsilon: f64,
    /// Feed-forwards: the dense width, then the routed experts.
    dense: u64,
    dense_width: u64,
    experts: u64,
    selected: u64,
    expert_width: u64,
    shared: u64,
    normalization: RouteNormalization,
    route_scale: f64,
    /// Each block's routed and shared (and dense) expansion clamps.
    clamps: Vec<(GluLimit, GluLimit)>,
}

impl Geometry {
    fn read(m: &Metadata, blocks: u64) -> Result<Self, FamilyError> {
        let equal = |key: &str, value: u64, expected: u64| match value == expected {
            true => Ok(()),
            false => Err(FamilyError::metadata(
                m.key(key),
                format!("{} is {value}, {FAMILY_ID} states {expected}", m.key(key)),
            )),
        };
        equal("rope.dimension_count", m.count("rope.dimension_count")?, 0)?;
        equal("expert_gating_func", m.integer("expert_gating_func")?, SIGMOID)?;
        equal("expert_group_count", m.integer("expert_group_count")?, 1)?;
        equal("expert_group_used_count", m.integer("expert_group_used_count")?, 1)?;
        let latent = m.integer("attention.kv_lora_rank")?;
        // Each history row is the latent, read as key and value.
        equal("attention.key_length", m.integer("attention.key_length")?, latent)?;
        equal("attention.value_length", m.integer("attention.value_length")?, latent)?;
        let kpool = m.integer("attention.indexer.kpool")?;
        let top = m.integer("attention.indexer.top_k")?;
        if kpool < 2 || !top.is_multiple_of(kpool) {
            let key = m.key("attention.indexer.top_k");
            return Err(FamilyError::metadata(
                &key,
                format!("{key} {top} is not a whole number of pools of {kpool} tokens"),
            ));
        }
        // A row before its first complete pool reads its tail alone.
        if !m.flag("attention.indexer.kpool_select_tail")? {
            let key = m.key("attention.indexer.kpool_select_tail");
            return Err(FamilyError::metadata(&key, format!("{key}: every row reads its tail")));
        }
        let clamp = |key: &str| -> Result<Vec<GluLimit>, FamilyError> {
            m.layer_numbers(key, blocks)?
                .into_iter()
                .map(|limit| binders::glu_limit(&m, key, limit))
                .collect()
        };
        Ok(Self {
            hidden: m.integer("embedding_length")?,
            heads: m.integer("attention.head_count")?,
            epsilon: m.positive_number("attention.layer_norm_rms_epsilon")?,
            streams: m.integer("hyper_connection.count")?,
            kda_width: m.integer("kda.head_dim")?,
            taps: m.integer("ssm.conv_kernel")?,
            lower: m.number("kda.gate_lower_bound")?,
            query_rank: m.integer("attention.q_lora_rank")?,
            latent,
            key_width: m.integer("attention.key_length_mla")?,
            value_width: m.integer("attention.value_length_mla")?,
            index_heads: m.integer("attention.indexer.head_count")?,
            index_width: m.integer("attention.indexer.key_length")?,
            index_top: top / kpool,
            kpool,
            index_epsilon: m.positive_number("attention.layer_norm_epsilon")?,
            dense: m.count("leading_dense_block_count")?,
            dense_width: m.integer("feed_forward_length")?,
            experts: m.integer("expert_count")?,
            selected: m.integer("expert_used_count")?,
            expert_width: m.integer("expert_feed_forward_length")?,
            shared: m.count("expert_shared_count")?,
            normalization: match m.flag("expert_weights_norm")? {
                true => RouteNormalization::SumPlusEpsilon(binders::ROUTE_EPSILON),
                false => RouteNormalization::None,
            },
            route_scale: m.positive_number("expert_weights_scale")?,
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
    fn sited(&self, layer: &mut Layer, site: &str, norm: &str, op: Operator) -> Result<Sublayer, FamilyError> {
        let mut weight = |name: &str, shape: &[u64]| layer.weight(name, shape);
        binders::sinkhorn_sited(&mut weight, self.streams, self.hidden, self.epsilon, site, norm, op)
    }

    /// One plain-residual sublayer (the MTP head's), its input normalized by
    /// `{norm}.weight`.
    fn plain(&self, layer: &mut Layer, norm: &str, op: Operator) -> Result<Sublayer, FamilyError> {
        Ok(Sublayer {
            input: InputNorm::Rms(self.rms(layer.weight(&format!("{norm}.weight"), &[self.hidden])?)),
            op,
            output: OutputForm::Residual,
        })
    }

    /// Kimi Delta Attention, its q, k and v projections and convolution
    /// planes each stored apart.
    fn kda(&self, layer: &mut Layer) -> Result<GatedDelta, FamilyError> {
        let (heads, width, hidden) = (self.heads, self.kda_width, self.hidden);
        let inner = product(&[heads, width])?;
        let projection = [inner, hidden];
        let plane = [1, inner, 1, self.taps];
        Ok(GatedDelta {
            convolution_width: self.taps,
            key_heads: heads,
            value_heads: heads,
            width,
            head_mapping: RecurrentHeadMapping::Grouped,
            query_key_value: layer.parts(
                ("attn_q.weight", &projection),
                &[("attn_k.weight", &projection), ("attn_v.weight", &projection)],
                false,
            )?,
            convolution: layer.parts(
                ("ssm_conv1d_q.weight", &plane),
                &[("ssm_conv1d_k.weight", &plane), ("ssm_conv1d_v.weight", &plane)],
                true,
            )?,
            beta: layer.weight("ssm_beta.weight", &[heads, hidden])?,
            decay: RecurrentDecay::PerChannel {
                projection: self.low_rank(layer, "ssm_f_a.weight", "ssm_f_b.weight")?,
                bias: layer.weight("ssm_dt.bias", &[inner])?,
                decay: layer.weight("ssm_a", &[heads])?,
                gate: DecayGate::BoundedSigmoid { lower: self.lower },
            },
            key_norm_epsilon: KDA_KEY_NORM_EPSILON,
            norm: self.rms(layer.weight("ssm_norm.weight", &[width])?),
            output_gate: RecurrentGate {
                projection: GateProjection::LowRank(self.low_rank(layer, "ssm_g_a.weight", "ssm_g_b.weight")?),
                function: GateFunction::Sigmoid,
            },
            output: layer.weight("attn_output.weight", &[hidden, inner])?,
        })
    }

    /// A low-rank projection `up·(down·x)` from the hidden row to KDA's
    /// heads, its rank the stored `down`'s rows.
    fn low_rank(&self, layer: &mut Layer, down: &str, up: &str) -> Result<LowRank, FamilyError> {
        let rank = layer.rows(down)?;
        Ok(LowRank {
            down: layer.weight(down, &[rank, self.hidden])?,
            up: layer.weight(up, &[product(&[self.heads, self.kda_width])?, rank])?,
        })
    }

    /// Latent attention without positions over the rows its own indexer
    /// picks.
    fn latent(&self, layer: &mut Layer) -> Result<Attention, FamilyError> {
        let (heads, hidden, latent) = (self.heads, self.hidden, self.latent);
        let query_rows = product(&[heads, self.key_width])?;
        let query = QueryProjection::LowRank {
            down: layer.weight("attn_q_a.weight", &[self.query_rank, hidden])?,
            norm: self.rms(layer.weight("attn_q_a_norm.weight", &[self.query_rank])?),
            up: layer.weight("attn_q_b.weight", &[query_rows, self.query_rank])?,
        };
        // Head h's block of `attn_k_b` [heads, latent, key width] is W_UK_hᵀ.
        let query_keys = QueryKeys::Absorbed {
            weight: layer.merged("attn_k_b.weight", &[heads, latent, self.key_width])?,
            width: self.key_width,
        };
        let rows = ProjectedRows {
            key: layer.weight("attn_kv_a_mqa.weight", &[latent, hidden])?,
            value: ValueSource::StoredKey,
            key_norm: HeadNorm::Rms(self.rms(layer.weight("attn_kv_a_norm.weight", &[latent])?)),
            value_norm: ValueNorm::None,
            domain: TokenDomain::Token,
            rounding: RowRounding::None,
        };
        let reads = HistoryReads::Selected(Selection::Own(Box::new(self.indexer(layer)?)));
        Ok(Attention {
            heads,
            kv_heads: 1,
            width: latent,
            query,
            query_keys,
            gate: AttentionGate::None,
            query_norm: HeadNorm::None,
            history: vec![HistorySource {
                rows: HistoryRows::Own(RowProducer::Projected(rows)),
                reads,
            }],
            rotary: Rotary::None,
            // The softmax scale is `qk_nope_head_dim ** -0.5`.
            scale: (self.key_width as f64).powf(-0.5),
            media_rows: MediaRowAttention::Causal,
            sinks: None,
            // Head h's block of `attn_v_b` [heads, value width, latent] is
            // W_UV_h, then the output projection.
            output: OutputProjection::GroupedLowRank {
                groups: heads,
                rank: self.value_width,
                a: layer.merged("attn_v_b.weight", &[heads, self.value_width, latent])?,
                b: layer.weight("attn_output.weight", &[hidden, product(&[heads, self.value_width])?])?,
            },
        })
    }

    /// The layer's indexer over its own token rows: its query from the
    /// low-rank query row, its keys pooled per `kpool` tokens from each
    /// token's LayerNorm-ed key.
    fn indexer(&self, layer: &mut Layer) -> Result<Indexer, FamilyError> {
        let (heads, width, hidden) = (self.index_heads, self.index_width, self.hidden);
        Ok(Indexer {
            heads,
            width,
            query: IndexerQuery {
                input: IndexerInput::QueryRank,
                projection: layer.weight("indexer.attn_q_b.weight", &[product(&[heads, width])?, self.query_rank])?,
                norm: HeadNorm::None,
                rounding: RowRounding::None,
            },
            keys: IndexerKeys::Pooled(PooledRows {
                rate: self.kpool,
                projection: layer.weight("indexer.attn_k.weight", &[width, hidden])?,
                pool: Pool::SoftmaxGated {
                    gate: layer.weight("indexer_compressor_gate.weight", &[width, hidden])?,
                    position: Some(layer.weight("indexer_compressor_ape.weight", &[self.kpool, width])?),
                },
                norm: PoolNorm::Token {
                    norm: LayerNorm {
                        weight: layer.weight("indexer.k_norm.weight", &[width])?,
                        epsilon: self.index_epsilon,
                    },
                    bias: layer.weight("indexer.k_norm.bias", &[width])?,
                },
                rounding: RowRounding::None,
            }),
            head_weights: HeadWeights::Projected(layer.weight("indexer.proj.weight", &[heads, hidden])?),
            // `relu(q·K / sqrt(width))` weighted by `proj(x) / sqrt(heads)`.
            scale: ((width * heads) as f64).powf(-0.5),
            top: self.index_top,
            candidates: Candidates::All,
        })
    }

    fn gated(limit: GluLimit, gate: WeightDescriptor, up: WeightDescriptor) -> FeedForwardUp {
        FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            limit: Some(limit),
            gate,
            up,
        }
    }

    /// Block `block`'s dense feed-forward, clamped at its shared limit.
    fn dense(&self, layer: &mut Layer, block: u64) -> Result<DenseFfn, FamilyError> {
        let (width, hidden) = (self.dense_width, self.hidden);
        let (_, limit) = self.clamps[block as usize];
        Ok(DenseFfn {
            intermediate: width,
            up: Self::gated(
                limit,
                layer.weight("ffn_gate.weight", &[width, hidden])?,
                layer.weight("ffn_up.weight", &[width, hidden])?,
            ),
            down: layer.weight("ffn_down.weight", &[hidden, width])?,
        })
    }

    /// Block `block`'s routed feed-forward and shared expert.
    fn routed(&self, layer: &mut Layer, block: u64) -> Result<RoutedFfn, FamilyError> {
        let (routed, shared) = self.clamps[block as usize];
        let mut weight = |name: &str, shape: &[u64]| layer.weight(name, shape);
        binders::routed(
            &mut weight,
            binders::RoutedForm {
                experts: self.experts,
                selected: self.selected,
                intermediate: self.expert_width,
                hidden: self.hidden,
                shared: match self.shared {
                    0 => None,
                    count => Some(product(&[count, self.expert_width])?),
                },
                score: ScoreFunction::Sigmoid,
                media_bias: false,
                normalization: self.normalization,
                scale: self.route_scale,
                routed_limit: Some(routed),
                shared_limit: Some(shared),
            },
        )
    }

    /// Block `block`'s feed-forward: dense for the leading blocks, routed
    /// after.
    fn feed_forward(&self, layer: &mut Layer, block: u64) -> Result<Operator, FamilyError> {
        Ok(match block < self.dense {
            true => Operator::DenseFfn(Box::new(self.dense(layer, block)?)),
            false => Operator::RoutedFfn(Box::new(self.routed(layer, block)?)),
        })
    }

    fn mixer(&self, layer: &mut Layer, mixer: Mixer) -> Result<Operator, FamilyError> {
        Ok(match mixer {
            Mixer::Kda => Operator::GatedDelta(Box::new(self.kda(layer)?)),
            Mixer::Latent => Operator::Attention(Box::new(self.latent(layer)?)),
        })
    }
}

/// One block's weights, bound by their names within the block (`blk.N.`):
/// every FP8 projection with its scale grid.
struct Layer<'t, 'a> {
    tensors: &'t mut Tensors<'a>,
    prefix: String,
}

impl Layer<'_, '_> {
    fn name(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    fn weight(&mut self, name: &str, shape: &[u64]) -> Result<WeightDescriptor, FamilyError> {
        let name = self.name(name);
        Ok(self.tensors.bind_scaled(&name, shape)?)
    }

    /// The stored rows of the matrix `name`.
    fn rows(&self, name: &str) -> Result<u64, FamilyError> {
        Ok(self.tensors.rows(&self.name(name))?)
    }

    /// The stack `name`, stored as `stored`, read as one matrix of all its
    /// matrices' rows.
    fn merged(&mut self, name: &str, stored: &[u64]) -> Result<WeightDescriptor, FamilyError> {
        let name = self.name(name);
        let &[matrices, _, _] = stored else { unreachable!("a stack is stored as three axes") };
        let scale = self.tensors.companion_scale(&name, matrices)?;
        Ok(self.tensors.bind_transformed(
            &name,
            stored,
            scale.into_iter().chain([ImportTransform::MergeLeadingAxes]).collect(),
        )?)
    }

    /// One weight stored in parts: `first`'s rows, then each part's
    /// (`Tensors::bind_rows`).
    fn parts(
        &mut self,
        first: (&str, &[u64]),
        appended: &[(&str, &[u64])],
        merge_leading_axes: bool,
    ) -> Result<WeightDescriptor, FamilyError> {
        let first_name = self.name(first.0);
        let names = appended.iter().map(|(name, _)| self.name(name)).collect::<Vec<_>>();
        let appended = names
            .iter()
            .zip(appended)
            .map(|(name, (_, shape))| (name.as_str(), *shape))
            .collect::<Vec<_>>();
        Ok(self.tensors.bind_rows((first_name.as_str(), first.1), &appended, merge_leading_axes)?)
    }
}

/// Each trunk block's mixer, from `attention.head_count_kv`, and the MTP
/// blocks', which attend. Every indexer is the layer's own
/// (`attention.indexer.types` all true): a layer reusing the previous one's
/// selection is refused.
fn mixers(m: &Metadata, blocks: u64, trunk: u64) -> Result<Vec<Mixer>, FamilyError> {
    let key = m.key("attention.head_count_kv");
    let mixers = m
        .layer_counts("attention.head_count_kv", blocks)?
        .into_iter()
        .enumerate()
        .map(|(block, kv)| match kv {
            0 if (block as u64) < trunk => Ok(Mixer::Kda),
            1 => Ok(Mixer::Latent),
            kv => Err(FamilyError::metadata(
                &key,
                format!("{key}: block {block} states {kv} key heads"),
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let types = m.flags("attention.indexer.types", trunk)?;
    if let Some(block) = types.iter().position(|own| !own) {
        let key = m.key("attention.indexer.types");
        return Err(FamilyError::metadata(
            &key,
            format!("{key}: block {block} reuses another layer's selection, which {FAMILY_ID} does not interpret"),
        ));
    }
    Ok(mixers)
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
    let m = Metadata::new(directory, FAMILY_ID, KEYS)?;
    let blocks = m.integer("block_count")?;
    // Each block requires distinct stored weights. Reject impossible counts
    // before allocating per-block vectors from untrusted metadata.
    if blocks > directory.tensors.len() as u64 {
        return Err(FamilyError::invalid(format!(
            "{FAMILY_ID} block count exceeds available weight roles"
        )));
    }
    let predicted = m.optional_count("nextn_predict_layers")?.unwrap_or(0);
    let trunk = blocks.checked_sub(predicted).filter(|trunk| *trunk > 0).ok_or_else(|| {
        let key = m.key("nextn_predict_layers");
        FamilyError::metadata(&key, format!("{key} leaves no trunk block"))
    })?;
    let mixers = mixers(&m, blocks, trunk)?;
    let geometry = Geometry::read(&m, blocks)?;
    let hidden = geometry.hidden;
    let mut tensors = Tensors::new(directory);
    let vocabulary = m.integer("vocab_size")?;
    let embedding = tensors.bind_scaled("token_embd.weight", &[vocabulary, hidden])?;
    let output = tensors.bind_scaled("output.weight", &[vocabulary, hidden])?;
    let output_norm = geometry.rms(tensors.bind_scaled("output_norm.weight", &[hidden])?);
    let trunk_blocks = (0..trunk)
        .map(|block| {
            let layer = &mut Layer {
                tensors: &mut tensors,
                prefix: format!("blk.{block}."),
            };
            let mixer = geometry.mixer(layer, mixers[block as usize])?;
            let feed_forward = geometry.feed_forward(layer, block)?;
            Ok(Block {
                sublayers: vec![
                    geometry.sited(layer, "attn", "attn_norm", mixer)?,
                    geometry.sited(layer, "ffn", "ffn_norm", feed_forward)?,
                ],
            })
        })
        .collect::<Result<Vec<_>, FamilyError>>()?;
    let head = (predicted > 0)
        .then(|| {
            let head_blocks = (trunk..blocks)
                .map(|block| {
                    let layer = &mut Layer {
                        tensors: &mut tensors,
                        prefix: format!("blk.{block}."),
                    };
                    let mixer = geometry.mixer(layer, mixers[block as usize])?;
                    let feed_forward = geometry.feed_forward(layer, block)?;
                    Ok(HeadBlock {
                        residual: ResidualForm::Single,
                        input: HeadInput::Combine {
                            embedding_norm: geometry.rms(layer.weight("nextn.enorm.weight", &[hidden])?),
                            hidden_norm: geometry.rms(layer.weight("nextn.hnorm.weight", &[hidden])?),
                            combine: layer.weight("nextn.eh_proj.weight", &[hidden, product(&[2, hidden])?])?,
                        },
                        blocks: vec![Block {
                            sublayers: vec![
                                geometry.plain(layer, "attn_norm", mixer)?,
                                geometry.plain(layer, "ffn_norm", feed_forward)?,
                            ],
                        }],
                        output_norm: ExitNorm::Rms(
                            geometry.rms(layer.weight("nextn.shared_head_norm.weight", &[hidden])?),
                        ),
                        readout: HeadReadout::full(VocabularyProjection::Target),
                        chain: HeadChain::Output,
                    })
                })
                .collect::<Result<Vec<_>, FamilyError>>()?;
            Ok::<_, FamilyError>(Head {
                blocks: head_blocks,
                history: HeadHistory::Own,
                component: HeadComponent::Target,
            })
        })
        .transpose()?;
    tensors.finish()?;
    let vision = projector
        .map(|projector| projector::describe_projector(projector, hidden))
        .transpose()?;
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
                streams: geometry.streams,
                mixing: HyperConnectionMixing::Sinkhorn {
                    iterations: m.integer("hyper_connection.sinkhorn_iterations")?,
                    epsilon: m.positive_number("hyper_connection.epsilon")?,
                    norm_epsilon: geometry.epsilon,
                    input: SinkhornInput::OwnSite,
                },
            }),
            entry: EntryForm {
                embedding,
                scale: EmbeddingScale::Unit,
                norm: None,
                per_layer: None,
            },
            blocks: trunk_blocks,
            exit: ExitForm {
                norm: ExitNorm::HyperConnectionHead(HyperConnectionExit {
                    site: ExitSite::Mean,
                    norm: Some(output_norm),
                }),
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
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
    Ok(definition)
}
