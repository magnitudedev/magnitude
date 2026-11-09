//! The general routed feed-forward (`Operator::RoutedFfn` in every form but
//! Qwen's fused one, which `routed_route/expand/output/…` keep): the
//! projection lane's `routed_select` over any score, selection bias,
//! normalization, scale and per-expert scale; gated or up-only experts in
//! the hidden width or a latent width; an ungated shared expert.
//!
//! One sublayer runs, over the block's residual rows:
//!
//! 1. the shared expert, when present, as a dense feed-forward of the same
//!    normalized input (`dense_expand` or `dense_up`, then `dense_output`
//!    onto the residual): the `base` the routed sum adds to;
//! 2. `routed_select`: routes, weights and the normalized expert input;
//! 3. a latent operator projects the normalized input down
//!    (`project_rows` into activations);
//! 4. decode rows: `routed_gate_up` or `routed_up`, then `routed_down`;
//!    larger rows: `routed_group`, `routed_experts` or `routed_experts_up`,
//!    then `routed_scatter`. The sum lands on `base`, or for a latent
//!    operator on zeros in the latent width;
//! 5. a latent operator projects the latent sum up onto `base`
//!    (`dense_output`).

pub(crate) mod fused_graph;
pub(crate) mod graph;
pub(crate) mod grouped;
pub(crate) mod split;
pub(crate) mod streamed;

use crate::error::PlanError;
use crate::{DenseScales, RoutedBinding, ScalableWeight};
use magnitude_family_contracts::{
    ActivationFunction, Decoder, ExpertSelection, FeedForwardUp, GluLimit, Operator,
    RouteNormalization, RoutedFfn, RouterInput, ScoreFunction, SharedExpert, SharedExpertGate,
    WeightKind,
};
use seismic::Element;

use super::dense_ffn::activation_code;

/// An expansion's form, activation code and clamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Expansion {
    /// Gated (`act(gate) · up`) or up-only (`act(up)`).
    pub gated: bool,
    pub activation: i32,
    /// The gated expansion's clamp (`glu`'s, its entries' LB).
    pub limit: Option<GluLimit>,
}

impl Expansion {
    pub(crate) fn of(up: &FeedForwardUp) -> Self {
        Self {
            gated: up.gate().is_some(),
            activation: activation_code(up.activation()),
            limit: up.limit(),
        }
    }
}

/// Every static axis and numerical parameter of a general routed operator
/// over a `hidden`-wide residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeneralRoutedShape {
    pub hidden: u64,
    pub experts: u64,
    pub selected: u64,
    pub intermediate: u64,
    /// The width the experts act in: the latent width, or `hidden`.
    pub expert_hidden: u64,
    pub latent: bool,
    pub experts_expansion: Expansion,
    /// The shared expert's width and expansion.
    pub shared: Option<(u64, Expansion)>,
    /// `routed_select`'s `score` code: 0 softmax, 1 sigmoid, 2 sqrt(softplus).
    pub score: i32,
    /// `routed_select`'s `normalization` code: 0 none, 1 sum, 2 sum + ε,
    /// 3 max(sum, ε).
    pub normalization: i32,
    /// ε of the normalization, as F32 bits (0 when unused).
    pub normalization_epsilon: u32,
    /// The combine weights' multiplier, as F32 bits.
    pub scale: u32,
    /// Whether the router ranks on a stored selection bias.
    pub bias: bool,
    /// Whether per-expert output scales multiply the combine weights.
    pub expert_scale: bool,
    /// Whether the router reads its own normalization of the residual.
    pub router_norm: bool,
}

impl GeneralRoutedShape {
    /// The shape of an admitted general routed operator.
    pub(crate) fn of(hidden: u64, routed: &RoutedFfn) -> Result<Self, PlanError> {
        admit(routed)?;
        let router = &routed.router;
        let (normalization, epsilon) = match router.normalization {
            RouteNormalization::None => (0, 0.0),
            RouteNormalization::Sum => (1, 0.0),
            RouteNormalization::SumPlusEpsilon(epsilon) => (2, epsilon),
            RouteNormalization::ClampedSum(epsilon) => (3, epsilon),
        };
        Ok(Self {
            hidden,
            experts: routed.experts,
            selected: routed.selected,
            intermediate: routed.intermediate,
            expert_hidden: routed.expert_hidden(hidden),
            latent: routed.latent.is_some(),
            experts_expansion: Expansion::of(&routed.expert_up),
            shared: routed
                .shared
                .as_ref()
                .map(|shared| (shared.intermediate, Expansion::of(&shared.up))),
            score: match router.score {
                ScoreFunction::Softmax => 0,
                ScoreFunction::Sigmoid => 1,
                ScoreFunction::SqrtSoftplus => 2,
            },
            normalization,
            normalization_epsilon: (epsilon as f32).to_bits(),
            scale: (router.scale as f32).to_bits(),
            bias: matches!(router.selection, ExpertSelection::TopK { bias: Some(_) }),
            expert_scale: routed.expert_scale.is_some(),
            router_norm: matches!(router.input, RouterInput::Residual(_)),
        })
    }

    /// The extent of the combine's per-expert scale port (`routed_select`'s
    /// ES) given the down weight's per-expert scale extent `down`: the stored
    /// output scales' (E) where the operator has them, else `down` (plan
    /// admission refuses both at once).
    pub fn expert_scale_extent(&self, down: u64) -> u64 {
        match self.expert_scale {
            true => self.experts,
            false => down,
        }
    }

    /// `routed_select`'s dimensions.
    pub fn select_dimensions(&self, rows: u64) -> [(&'static str, u64); 4] {
        [
            ("M", rows),
            ("H", self.hidden),
            ("E", self.experts),
            ("K", self.selected),
        ]
    }

    /// The decode expert entries' dimensions, in the experts' width.
    pub fn expert_dimensions(&self, rows: u64) -> [(&'static str, u64); 5] {
        [
            ("M", rows),
            ("H", self.expert_hidden),
            ("E", self.experts),
            ("K", self.selected),
            ("F", self.intermediate),
        ]
    }
}

/// The resident elements of a general routed operator's weights that are
/// not fixed F32 (the selection bias and expert scales are).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeneralRoutedBinding {
    pub shape: GeneralRoutedShape,
    /// The sublayer's input RMS weight.
    pub norm: Element,
    /// The router's own input norm, else `norm`.
    pub router_norm: Element,
    pub router: Element,
    /// Absent for up-only experts.
    pub expert_gate: Option<Element>,
    pub expert_up: Element,
    pub expert_down: Element,
    /// `(down, up)` of a latent operator.
    pub latent: Option<(Element, Element)>,
    /// `(gate, up, down)` of the shared expert; `gate` absent for up-only.
    pub shared: Option<(Option<Element>, Element, Element)>,
    pub activation: Element,
    pub scales: GeneralRoutedScales,
}

/// The extents of the routed entries' per-expert scale ports (`[S] f32`,
/// `routed.seismic`): each stacked expert weight's resident second-level
/// scales, E where it has them, else 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExpertScales {
    /// The gate and up weights': the expansions scale each expert's
    /// accumulators (GS, US).
    pub gate: u64,
    pub up: u64,
    /// The down weight's: the route folds it into each choice's combine
    /// weight (ES), as it does stored per-expert output scales.
    pub down: u64,
}

/// The extents of a general routed operator's accumulator-scale ports: its
/// projection weights' resident second-level scales, 0 where a weight has
/// none. The router binds no scale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct GeneralRoutedScales {
    pub experts: ExpertScales,
    /// The shared expert's dense entries.
    pub shared: DenseScales,
    /// A latent operator's `(down, up)` projections, per tensor.
    pub latent: (u64, u64),
}

/// The routed feed-forward of a feed-forward operator: its own, or a
/// parallel sublayer's routed branch's.
pub(crate) fn routed_ffn(op: &Operator) -> Option<&RoutedFfn> {
    match op {
        Operator::RoutedFfn(routed) => Some(routed),
        Operator::Parallel(branches) => branches.iter().find_map(|branch| routed_ffn(&branch.op)),
        _ => None,
    }
}

/// The expansion function of the fused form's experts (`is_fused`).
pub(crate) const FUSED_ACTIVATION: i32 = magnitude_kernels::cpu_library::cpu::lib::core::functions::SILU;

/// The routed form Qwen's fused entries implement: a softmax top-k router on
/// the normalized input without bias, the plain sum normalization, SiLU-gated
/// experts in the hidden width and one sigmoid-gated SiLU shared expert.
pub(crate) fn is_fused(routed: &RoutedFfn) -> bool {
    let router = &routed.router;
    let silu_gated = |up: &FeedForwardUp| {
        matches!(
            up,
            FeedForwardUp::Gated {
                activation: ActivationFunction::Silu,
                limit: None,
                ..
            }
        )
    };
    router.input == RouterInput::Operator
        && router.score == ScoreFunction::Softmax
        && router.selection == (ExpertSelection::TopK { bias: None })
        && matches!(
            router.normalization,
            RouteNormalization::None | RouteNormalization::Sum
        )
        && router.scale == 1.0
        && silu_gated(&routed.expert_up)
        && routed.expert_scale.is_none()
        && routed.latent.is_none()
        && routed.shared.as_ref().is_some_and(|shared| {
            silu_gated(&shared.up) && matches!(shared.gate, SharedExpertGate::Sigmoid(_))
        })
}

/// The weights a streamed layer's decode route reads of the layer it
/// predicts in place of its own, in its hint ports' order: the fused
/// route's input norm and router; the general selection's router norm
/// (its own where the router reads the residual, else the input norm),
/// router and selection bias.
pub(crate) fn hint_kinds(routed: &RoutedFfn) -> Vec<WeightKind> {
    match is_fused(routed) {
        true => vec![WeightKind::InputNorm, WeightKind::Router],
        false => vec![
            match routed.router.input {
                RouterInput::Residual(_) => WeightKind::RouterInputNorm,
                _ => WeightKind::InputNorm,
            },
            WeightKind::Router,
            WeightKind::RouterSelectionBias,
        ],
    }
}

/// The forms the general routed entries implement: every score function
/// with top-k selection, any normalization and finite scale, a router input
/// norm of the sublayer's epsilon, gated or up-only experts, and an ungated
/// shared expert.
pub(crate) fn admit(routed: &RoutedFfn) -> Result<(), PlanError> {
    let router = &routed.router;
    if !router.scale.is_finite() {
        return Err(PlanError::Unsupported("expert router scale"));
    }
    if routed
        .shared
        .as_ref()
        .is_some_and(|shared| shared.gate != SharedExpertGate::None)
    {
        return Err(PlanError::Unsupported(
            "gated shared expert with general routing",
        ));
    }
    Ok(())
}

/// Media rows of a decoder whose routers rank them on their own selection
/// bias (`Router::media_bias`): no selection reads that bias yet, so a
/// decoder with a vision component and one is refused, rather than ranking
/// its media rows on the text bias. Without a vision component there are no
/// media rows, and text rows never read it.
pub(crate) fn admit_media(decoder: &Decoder, vision: bool) -> Result<(), PlanError> {
    let media_biased = decoder.sublayers().any(|(_, sublayer)| {
        routed_ffn(&sublayer.op).is_some_and(|routed| routed.router.media_bias.is_some())
    });
    match vision && media_biased {
        true => Err(PlanError::Unsupported(
            "media rows ranked on their own expert selection bias",
        )),
        false => Ok(()),
    }
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(routed: &'a RoutedFfn, push: &mut super::WeightPush<'_, 'a>) {
    let router = &routed.router;
    push(WeightKind::Router, &router.weight);
    if let RouterInput::Residual(norm) = &router.input {
        push(WeightKind::RouterInputNorm, &norm.weight);
    }
    if let ExpertSelection::TopK { bias: Some(bias) } = &router.selection {
        push(WeightKind::RouterSelectionBias, bias);
    }
    if let Some(SharedExpert {
        gate: SharedExpertGate::Sigmoid(gate),
        ..
    }) = &routed.shared
    {
        push(WeightKind::SharedRouter, gate);
    }
    if let Some(gate) = routed.expert_up.gate() {
        push(WeightKind::ExpertGate, gate);
    }
    push(WeightKind::ExpertUp, routed.expert_up.up());
    push(WeightKind::ExpertDown, &routed.expert_down);
    if let Some(scale) = &routed.expert_scale {
        push(WeightKind::ExpertScale, scale);
    }
    if let Some(latent) = &routed.latent {
        push(WeightKind::LatentDown, &latent.down);
        push(WeightKind::LatentUp, &latent.up);
    }
    if let Some(shared) = &routed.shared {
        if let Some(gate) = shared.up.gate() {
            push(WeightKind::SharedGate, gate);
        }
        push(WeightKind::SharedUp, shared.up.up());
        push(WeightKind::SharedDown, &shared.down);
    }
}

/// Every numerical parameter of Qwen's fused form a sealed graph holds apart
/// from its weights.
pub(super) fn fused_shape_key(routed: &RoutedFfn) -> String {
    format!(
        "routed {} {} {} {:?} {:?}",
        routed.experts,
        routed.selected,
        routed.intermediate,
        routed.shared.as_ref().map(|shared| shared.intermediate),
        routed.router.normalization
    )
}

/// The program binding of Qwen's fused form (`is_fused`); `lookup` resolves
/// the planned element of a role in its scope that its entry binds without
/// an accumulator-scale port (the shared expert's), `scalable` one it binds
/// with one (the experts').
pub(super) fn fused_binding(
    routed: &RoutedFfn,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<ScalableWeight, PlanError>,
    activation: Element,
) -> Result<RoutedBinding, PlanError> {
    let (gate, up, down) = (
        scalable(WeightKind::ExpertGate)?,
        scalable(WeightKind::ExpertUp)?,
        scalable(WeightKind::ExpertDown)?,
    );
    Ok(RoutedBinding {
        hidden,
        experts: routed.experts,
        selected: routed.selected,
        features: routed.intermediate,
        shared: routed
            .shared
            .as_ref()
            .ok_or(PlanError::Unsupported(
                "routed feed-forward without shared expert",
            ))?
            .intermediate,
        normalize_selected: routed.router.normalization == RouteNormalization::Sum,
        norm: lookup(WeightKind::InputNorm)?,
        router: lookup(WeightKind::Router)?,
        expert_gate: gate.element,
        expert_up: up.element,
        expert_down: down.element,
        shared_gate: lookup(WeightKind::SharedGate)?,
        shared_up: lookup(WeightKind::SharedUp)?,
        shared_down: lookup(WeightKind::SharedDown)?,
        activation,
        scales: ExpertScales {
            gate: gate.scale,
            up: up.scale,
            down: down.scale,
        },
    })
}

/// The program binding of a general routed operator; `lookup` resolves the
/// planned element of a role in its scope that its entry binds without an
/// accumulator-scale port, `scalable` one it binds with one.
pub(super) fn binding(
    routed: &RoutedFfn,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<ScalableWeight, PlanError>,
    activation: Element,
) -> Result<GeneralRoutedBinding, PlanError> {
    let shape = GeneralRoutedShape::of(hidden, routed)?;
    let norm = lookup(WeightKind::InputNorm)?;
    let mut scales = GeneralRoutedScales::default();
    let expert_gate = shape
        .experts_expansion
        .gated
        .then(|| scalable(WeightKind::ExpertGate))
        .transpose()?;
    let expert_up = scalable(WeightKind::ExpertUp)?;
    let expert_down = scalable(WeightKind::ExpertDown)?;
    // The route's one per-expert scale port holds either.
    if expert_down.scale != 0 && shape.expert_scale {
        return Err(PlanError::Unsupported(
            "a scaled expert down weight beside stored per-expert output scales",
        ));
    }
    scales.experts = ExpertScales {
        gate: expert_gate.map_or(0, |gate| gate.scale),
        up: expert_up.scale,
        down: expert_down.scale,
    };
    let (expert_gate, expert_up) = (expert_gate.map(|gate| gate.element), expert_up.element);
    let latent = if shape.latent {
        let (down, up) = (
            scalable(WeightKind::LatentDown)?,
            scalable(WeightKind::LatentUp)?,
        );
        scales.latent = (down.scale, up.scale);
        Some((down.element, up.element))
    } else {
        None
    };
    let shared = match shape.shared {
        Some((_, expansion)) => {
            let gate = expansion
                .gated
                .then(|| scalable(WeightKind::SharedGate))
                .transpose()?;
            let (up, down) = (
                scalable(WeightKind::SharedUp)?,
                scalable(WeightKind::SharedDown)?,
            );
            scales.shared = DenseScales {
                gate: gate.map_or(0, |gate| gate.scale),
                up: up.scale,
                down: down.scale,
            };
            Some((gate.map(|gate| gate.element), up.element, down.element))
        }
        None => None,
    };
    Ok(GeneralRoutedBinding {
        shape,
        norm,
        router_norm: if shape.router_norm {
            lookup(WeightKind::RouterInputNorm)?
        } else {
            norm
        },
        router: lookup(WeightKind::Router)?,
        expert_gate,
        expert_up,
        expert_down: expert_down.element,
        latent,
        shared,
        activation,
        scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planning::fixtures::descriptor;
    use magnitude_family_contracts::Router;

    /// DeepSeek V4.1 Flash's routed feed-forward (engine map §7): sqrt-softplus
    /// scores ranked with a selection bias, weights over sum + 1e-20 scaled by
    /// 1.5, 384 SiLU-gated experts clamped at 10 of which 6 are selected, and
    /// one ungated shared expert clamped the same.
    fn deepseek41() -> RoutedFfn {
        const HIDDEN: u64 = 5120;
        const EXPERTS: u64 = 384;
        const FEATURES: u64 = 2304;
        let silu = |gate, up| FeedForwardUp::Gated {
            activation: ActivationFunction::Silu,
            limit: Some(GluLimit::new(10.0).unwrap()),
            gate,
            up,
        };
        RoutedFfn {
            experts: EXPERTS,
            selected: 6,
            intermediate: FEATURES,
            router: Router {
                weight: descriptor("ffn_gate_inp", &[EXPERTS, HIDDEN]),
                input: RouterInput::Operator,
                score: ScoreFunction::SqrtSoftplus,
                selection: ExpertSelection::TopK {
                    bias: Some(descriptor("exp_probs_b", &[EXPERTS])),
                },
                media_bias: None,
                normalization: RouteNormalization::SumPlusEpsilon(1e-20),
                scale: 1.5,
            },
            expert_up: silu(
                descriptor("ffn_gate_exps", &[EXPERTS, FEATURES, HIDDEN]),
                descriptor("ffn_up_exps", &[EXPERTS, FEATURES, HIDDEN]),
            ),
            expert_down: descriptor("ffn_down_exps", &[EXPERTS, HIDDEN, FEATURES]),
            expert_scale: None,
            latent: None,
            shared: Some(SharedExpert {
                intermediate: FEATURES,
                up: silu(
                    descriptor("ffn_gate_shexp", &[FEATURES, HIDDEN]),
                    descriptor("ffn_up_shexp", &[FEATURES, HIDDEN]),
                ),
                down: descriptor("ffn_down_shexp", &[HIDDEN, FEATURES]),
                gate: SharedExpertGate::None,
            }),
        }
    }

    /// Until a selection reads it, a router's media-row bias admits a
    /// text-only decoder (its text rows never read it) and refuses one with
    /// a vision component by name: no media row is ranked on the text bias.
    #[test]
    fn a_media_row_bias_is_refused_beside_vision_and_ignored_without_it() {
        let mut definition = crate::planning::fixtures::sinkhorn_fixture_definition(1);
        let decoder = &mut definition.decoder;
        assert_eq!(admit_media(decoder, true), Ok(()));
        let Operator::RoutedFfn(routed) = &mut decoder.blocks[0].sublayers[1].op else {
            panic!("the fixture's feed-forward is routed");
        };
        routed.router.media_bias = Some(descriptor("exp_probs_b_vl", &[routed.experts]));
        assert_eq!(admit_media(decoder, false), Ok(()));
        assert_eq!(
            admit_media(decoder, true),
            Err(PlanError::Unsupported(
                "media rows ranked on their own expert selection bias"
            ))
        );
    }

    #[test]
    fn sqrt_softplus_routing_takes_the_general_path() {
        let routed = deepseek41();
        assert!(!is_fused(&routed));
        let silu_gated = Expansion {
            gated: true,
            activation: 0,
            limit: Some(GluLimit::new(10.0).unwrap()),
        };
        assert_eq!(
            GeneralRoutedShape::of(5120, &routed).unwrap(),
            GeneralRoutedShape {
                hidden: 5120,
                experts: 384,
                selected: 6,
                intermediate: 2304,
                expert_hidden: 5120,
                latent: false,
                experts_expansion: silu_gated,
                shared: Some((2304, silu_gated)),
                score: 2,
                normalization: 2,
                normalization_epsilon: 1e-20_f32.to_bits(),
                scale: 1.5_f32.to_bits(),
                bias: true,
                expert_scale: false,
                router_norm: false,
            }
        );
    }
}
