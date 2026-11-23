//! Sublayers families bind the same way from the same tensor names: a
//! sublayer between its Sinkhorn hyper-connection site and the streams, and a
//! DeepSeek-form routed feed-forward with its shared expert. A family states
//! what varies (its score function, clamps, media bias) and binds each tensor
//! through its own weight lookup.

use crate::Metadata;
use magnitude_family_contracts::{
    ActivationFunction, Attention, AttentionGate, BlockElement, BlockRounding, BlockScale,
    ExpertSelection, FamilyError, FeedForwardUp, GluLimit, HeadNorm, HistoryRows, HistorySource,
    HyperConnectionInput, HyperConnectionWeights, InputNorm, MediaRowAttention, Operator,
    OutputForm, OutputProjection, ProjectedRows, QueryKeys, QueryProjection, RmsNorm, Rotary,
    RotaryLayout, RotaryPairing, RotaryPlacement, RouteNormalization, RoutedFfn, Router,
    RouterInput, RowProducer, RowRounding, ScoreFunction, SharedExpert, SharedExpertGate,
    SinkhornSite, Sublayer, TokenDomain, ValueNorm, ValueSource, WeightDescriptor,
    checked_product, grouped_output_columns,
};

/// llama.cpp's `expert_gating_func` for sqrt-softplus scores
/// (`LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`), used by DeepSeek V4.1 and
/// its DSpark drafter.
pub const SQRT_SOFTPLUS: u64 = 4;

/// Where DeepSeek V4.1's rotary pairs sit: adjacent pairs `(2p, 2p + 1)` over
/// a head's trailing dimensions, from `head_dim - rope_dim` on.
pub const DEEPSEEK41_ROTARY_LAYOUT: RotaryLayout = RotaryLayout {
    pairs: RotaryPairing::Adjacent,
    placement: RotaryPlacement::Trailing,
};

/// DeepSeek V4.1's trained window-KV rounding (`act_quant` per 32 values,
/// power-of-two scale, floor `1e-4`), shared by the target and its DSpark
/// drafter.
pub const DEEPSEEK41_WINDOW_ROUNDING: RowRounding = RowRounding::Block(BlockRounding {
    element: BlockElement::E4M3,
    block: 32,
    scale: BlockScale::Ue8m0,
    amax_floor: 1e-4,
});

/// A family's binding of its stored tensor `name` (the family's own naming of
/// a layer's tensors) of shape `shape`.
pub type BindWeight<'a> = dyn FnMut(&str, &[u64]) -> Result<WeightDescriptor, FamilyError> + 'a;

/// The combine weights' normalizer, `sum + 1e-20`
/// (`RouteNormalization::SumPlusEpsilon`): the released DeepSeek V4.1 router's
/// constant and transformers' `Glm5NextTextTopkRouter`'s, as trained, not a
/// model's RMS epsilon.
pub const ROUTE_EPSILON: f64 = 1e-20;

/// `limit` (the metadata `key`'s entry for a layer) as a GLU clamp, or the
/// metadata error that names the key.
pub fn glu_limit(m: &Metadata, key: &str, limit: f64) -> Result<GluLimit, FamilyError> {
    GluLimit::new(limit).map_err(|error| FamilyError::metadata(m.key(key), error.to_string()))
}

fn product(dimensions: &[u64]) -> Result<u64, FamilyError> {
    checked_product(dimensions).map_err(|error| FamilyError::invalid(error.to_string()))
}

/// One sublayer between its Sinkhorn site `hc_{site}_*` and `streams` streams
/// of `hidden` columns, its mixed input RMS-normed by `{norm}.weight` at
/// `epsilon`.
pub fn sinkhorn_sited(
    weight: &mut BindWeight,
    streams: u64,
    hidden: u64,
    epsilon: f64,
    site: &str,
    norm: &str,
    op: Operator,
) -> Result<Sublayer, FamilyError> {
    let mixes = product(&[2 + streams, streams])?;
    let wide = product(&[streams, hidden])?;
    Ok(Sublayer {
        input: InputNorm::HyperConnectionMix(HyperConnectionInput {
            site: HyperConnectionWeights::Sinkhorn(SinkhornSite {
                function: weight(&format!("hc_{site}_fn.weight"), &[mixes, wide])?,
                base: weight(&format!("hc_{site}_base.weight"), &[mixes])?,
                scale: weight(&format!("hc_{site}_scale.weight"), &[3])?,
            }),
            norm: Some(RmsNorm {
                weight: weight(&format!("{norm}.weight"), &[hidden])?,
                epsilon,
            }),
        }),
        op,
        output: OutputForm::Scatter,
    })
}

/// What a DeepSeek-form routed feed-forward varies by: `experts` stacked
/// SiLU-gated experts of `intermediate` rows over `hidden`, `selected` a row,
/// routed by `score` over the router's biased top-k (with the media bias
/// `exp_probs_b_vl` where `media_bias`), and a shared expert of `shared` rows
/// where one is stored, each expansion clamped at its limit.
pub struct RoutedForm {
    pub experts: u64,
    pub selected: u64,
    pub intermediate: u64,
    pub hidden: u64,
    pub shared: Option<u64>,
    pub score: ScoreFunction,
    pub media_bias: bool,
    pub normalization: RouteNormalization,
    pub scale: f64,
    pub routed_limit: Option<GluLimit>,
    pub shared_limit: Option<GluLimit>,
}

/// The routed feed-forward of `form` from its stored tensors (`ffn_gate_inp`,
/// `exp_probs_b`, `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`).
pub fn routed(weight: &mut BindWeight, form: RoutedForm) -> Result<RoutedFfn, FamilyError> {
    let RoutedForm {
        experts,
        selected,
        intermediate,
        hidden,
        ..
    } = form;
    let silu = |limit, gate, up| FeedForwardUp::Gated {
        activation: ActivationFunction::Silu,
        limit,
        gate,
        up,
    };
    let stacked = [experts, intermediate, hidden];
    let shared = match form.shared {
        None => None,
        Some(shared) => Some(SharedExpert {
            intermediate: shared,
            up: silu(
                form.shared_limit,
                weight("ffn_gate_shexp.weight", &[shared, hidden])?,
                weight("ffn_up_shexp.weight", &[shared, hidden])?,
            ),
            down: weight("ffn_down_shexp.weight", &[hidden, shared])?,
            gate: SharedExpertGate::None,
        }),
    };
    Ok(RoutedFfn {
        experts,
        selected,
        intermediate,
        router: Router {
            weight: weight("ffn_gate_inp.weight", &[experts, hidden])?,
            input: RouterInput::Operator,
            score: form.score,
            selection: ExpertSelection::TopK {
                bias: Some(weight("exp_probs_b.bias", &[experts])?),
            },
            media_bias: if form.media_bias {
                Some(weight("exp_probs_b_vl.bias", &[experts])?)
            } else {
                None
            },
            normalization: form.normalization,
            scale: form.scale,
        },
        expert_up: silu(
            form.routed_limit,
            weight("ffn_gate_exps.weight", &stacked)?,
            weight("ffn_up_exps.weight", &stacked)?,
        ),
        expert_down: weight("ffn_down_exps.weight", &[experts, hidden, intermediate])?,
        expert_scale: None,
        latent: None,
        shared,
    })
}

/// Geometry of a DeepSeek V4.1 window-MLA attention layer (low-rank query,
/// single shared key/value head with `attn_kv_a_norm`, sliding window with
/// trained E4M3 rounding, per-head sinks, and grouped low-rank output).
pub struct MlaWindowForm {
    pub hidden: u64,
    pub heads: u64,
    pub width: u64,
    pub query_rank: u64,
    pub window: u64,
    pub groups: u64,
    pub output_rank: u64,
    pub epsilon: f64,
}

/// Bind a DeepSeek V4.1 window-MLA attention layer from its stored tensors,
/// appending any optional `extra_history` (e.g. a target layer's compressed
/// history source).
pub fn mla_window_attention(
    weight: &mut BindWeight,
    form: &MlaWindowForm,
    rotary: Rotary,
    extra_history: Option<HistorySource>,
) -> Result<Attention, FamilyError> {
    let MlaWindowForm {
        hidden,
        heads,
        width,
        query_rank,
        window,
        groups,
        output_rank,
        epsilon,
    } = *form;
    let rows = product(&[heads, width])?;
    let columns = grouped_output_columns(rows, groups)
        .map_err(|error| FamilyError::invalid(error.to_string()))?;
    let ranks = product(&[groups, output_rank])?;
    let rms = |weight| RmsNorm { weight, epsilon };
    Ok(Attention {
        heads,
        kv_heads: 1,
        width,
        query: QueryProjection::LowRank {
            down: weight("attn_q_a.weight", &[query_rank, hidden])?,
            norm: rms(weight("attn_q_a_norm.weight", &[query_rank])?),
            up: weight("attn_q_b.weight", &[rows, query_rank])?,
        },
        query_keys: QueryKeys::Direct,
        gate: AttentionGate::None,
        query_norm: HeadNorm::None,
        history: vec![HistorySource::visible(HistoryRows::Own(
            RowProducer::Projected(ProjectedRows {
                key: weight("attn_kv.weight", &[width, hidden])?,
                value: ValueSource::StoredKey,
                key_norm: HeadNorm::Rms(rms(weight("attn_kv_a_norm.weight", &[width])?)),
                value_norm: ValueNorm::None,
                domain: TokenDomain::Window { tokens: window },
                rounding: DEEPSEEK41_WINDOW_ROUNDING,
            }),
        ))]
        .into_iter()
        .chain(extra_history)
        .collect(),
        rotary,
        scale: (width as f64).powf(-0.5),
        media_rows: MediaRowAttention::Causal,
        sinks: Some(weight("attn_sinks.weight", &[heads])?),
        output: OutputProjection::GroupedLowRank {
            groups,
            rank: output_rank,
            a: weight("attn_output_a.weight", &[ranks, columns])?,
            b: weight("attn_output_b.weight", &[hidden, ranks])?,
        },
    })
}
