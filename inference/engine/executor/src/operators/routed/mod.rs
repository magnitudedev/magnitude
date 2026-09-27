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

use crate::error::PlanError;
use crate::RoutedBinding;
use magnitude_family_contracts::{
    ActivationFunction, ExpertSelection, FeedForwardUp, RouteNormalization, RoutedFfn,
    RouterInput, ScoreFunction, SharedExpert, SharedExpertGate, WeightKind,
};
use seismic::Element;

use super::dense_ffn::activation_code;

/// An expansion's form and activation code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Expansion {
    /// Gated (`act(gate) · up`) or up-only (`act(up)`).
    pub gated: bool,
    pub activation: i32,
}

impl Expansion {
    fn of(up: &FeedForwardUp) -> Self {
        Self {
            gated: up.gate().is_some(),
            activation: activation_code(up.activation()),
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
    /// `routed_select`'s `score` code: 0 softmax, 1 sigmoid.
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
                ScoreFunction::SqrtSoftplus => {
                    return Err(PlanError::Unsupported("sqrt-softplus expert scores"))
                }
            },
            normalization,
            normalization_epsilon: (epsilon as f32).to_bits(),
            scale: (router.scale as f32).to_bits(),
            bias: matches!(router.selection, ExpertSelection::TopK { bias: Some(_) }),
            expert_scale: routed.expert_scale.is_some(),
            router_norm: matches!(router.input, RouterInput::Residual(_)),
        })
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
}

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

/// The forms the general routed entries implement: softmax or sigmoid
/// scores with top-k selection, any normalization and finite scale, a router
/// input norm of the sublayer's epsilon, gated or up-only experts, and an
/// ungated shared expert.
pub(crate) fn admit(routed: &RoutedFfn) -> Result<(), PlanError> {
    let router = &routed.router;
    if !matches!(router.selection, ExpertSelection::TopK { .. }) {
        return Err(PlanError::Unsupported("expert selection"));
    }
    if !matches!(router.score, ScoreFunction::Softmax | ScoreFunction::Sigmoid) {
        return Err(PlanError::Unsupported("expert score function"));
    }
    if !router.scale.is_finite() {
        return Err(PlanError::Unsupported("expert router scale"));
    }
    if routed
        .shared
        .as_ref()
        .is_some_and(|shared| shared.gate != SharedExpertGate::None)
    {
        return Err(PlanError::Unsupported("gated shared expert with general routing"));
    }
    Ok(())
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
/// the planned element of a role in its scope.
pub(super) fn fused_binding(
    routed: &RoutedFfn,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<RoutedBinding, PlanError> {
    Ok(RoutedBinding {
        hidden,
        experts: routed.experts,
        selected: routed.selected,
        features: routed.intermediate,
        shared: routed
            .shared
            .as_ref()
            .ok_or(PlanError::Unsupported("routed feed-forward without shared expert"))?
            .intermediate,
        normalize_selected: routed.router.normalization == RouteNormalization::Sum,
        norm: lookup(WeightKind::InputNorm)?,
        router: lookup(WeightKind::Router)?,
        expert_gate: lookup(WeightKind::ExpertGate)?,
        expert_up: lookup(WeightKind::ExpertUp)?,
        expert_down: lookup(WeightKind::ExpertDown)?,
        shared_gate: lookup(WeightKind::SharedGate)?,
        shared_up: lookup(WeightKind::SharedUp)?,
        shared_down: lookup(WeightKind::SharedDown)?,
        activation,
    })
}

/// The program binding of a general routed operator; `lookup` resolves the
/// planned element of a role in its scope.
pub(super) fn binding(
    routed: &RoutedFfn,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<GeneralRoutedBinding, PlanError> {
    let shape = GeneralRoutedShape::of(hidden, routed)?;
    let norm = lookup(WeightKind::InputNorm)?;
    let expert_gate = if shape.experts_expansion.gated {
        Some(lookup(WeightKind::ExpertGate)?)
    } else {
        None
    };
    let latent = if shape.latent {
        Some((lookup(WeightKind::LatentDown)?, lookup(WeightKind::LatentUp)?))
    } else {
        None
    };
    let shared = match shape.shared {
        Some((_, expansion)) => Some((
            if expansion.gated {
                Some(lookup(WeightKind::SharedGate)?)
            } else {
                None
            },
            lookup(WeightKind::SharedUp)?,
            lookup(WeightKind::SharedDown)?,
        )),
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
        expert_up: lookup(WeightKind::ExpertUp)?,
        expert_down: lookup(WeightKind::ExpertDown)?,
        latent,
        shared,
        activation,
    })
}
