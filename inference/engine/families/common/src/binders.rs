//! Sublayers families bind the same way from the same tensor names: a
//! sublayer between its Sinkhorn hyper-connection site and the streams, and a
//! DeepSeek-form routed feed-forward with its shared expert. A family states
//! what varies (its score function, clamps, media bias) and binds each tensor
//! through its own weight lookup.

use crate::Metadata;
use magnitude_family_contracts::{
    ActivationFunction, ExpertSelection, FamilyError, FeedForwardUp, GluLimit,
    HyperConnectionInput, HyperConnectionWeights, InputNorm, Operator, OutputForm, RmsNorm,
    RouteNormalization, RoutedFfn, Router, RouterInput, ScoreFunction, SharedExpert,
    SharedExpertGate, SinkhornSite, Sublayer, WeightDescriptor, checked_product,
};

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
