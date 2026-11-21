//! Gated delta-rule recurrence (`Operator::GatedDelta`: Qwen 3.5's Gated
//! DeltaNet over the `gated_delta_*` entries, GLM's Kimi Delta Attention over
//! the `kimi_delta_*` entries): the forms the recurrent entries implement,
//! their weight roles and program binding.

pub(crate) mod graph;

use super::{PlanError, WeightPush};
use crate::{RecurrentBinding, RecurrentForm};
use magnitude_family_contracts::{
    DecayGate, GateFunction, GateProjection, GatedDelta, RecurrentDecay, WeightKind,
};
use seismic::Element;

/// The output gate functions the recurrent entries implement
/// (`output_gate` in `functions.seismic`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdmittedGateFunction {
    Silu,
    Sigmoid,
}

impl AdmittedGateFunction {
    /// Admit a gated delta output gate function supported by the recurrent
    /// entries.
    pub(crate) fn admit(function: GateFunction) -> Result<Self, PlanError> {
        match function {
            GateFunction::Silu => Ok(Self::Silu),
            GateFunction::Sigmoid => Ok(Self::Sigmoid),
            GateFunction::Softplus => {
                Err(PlanError::Unsupported("softplus gated delta output gate"))
            }
        }
    }

    /// The recurrent entries' `gate_function` code: 0 SiLU, 1 sigmoid
    /// (`output_gate` in `functions.seismic`).
    pub(crate) fn code(self) -> i32 {
        match self {
            Self::Silu => 0,
            Self::Sigmoid => 1,
        }
    }
}

/// An admitted gated delta-rule operator in one of the two forms the executor
/// implements: per-head decay with a full gate projection (GDN,
/// `gated_delta_*`) or per-channel decay with low-rank gate and decay
/// projections (KDA, `kimi_delta_*`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum AdmittedGatedDelta<'a> {
    PerHead {
        operator: &'a GatedDelta,
        gate_function: AdmittedGateFunction,
    },
    PerChannel {
        operator: &'a GatedDelta,
        gate_function: AdmittedGateFunction,
        decay_lower: f32,
        ranks: [u64; 2],
    },
}

impl<'a> AdmittedGatedDelta<'a> {
    pub(crate) fn operator(&self) -> &'a GatedDelta {
        match self {
            Self::PerHead { operator, .. } | Self::PerChannel { operator, .. } => operator,
        }
    }

    pub(crate) fn gate_function(&self) -> AdmittedGateFunction {
        match self {
            Self::PerHead { gate_function, .. } | Self::PerChannel { gate_function, .. } => {
                *gate_function
            }
        }
    }

    /// The per-channel decay's lower bound (the `kimi_delta_*` entries'
    /// `decay_lower`): none for the per-head decay.
    pub(crate) fn decay_lower(&self) -> Option<f32> {
        match self {
            Self::PerHead { .. } => None,
            Self::PerChannel { decay_lower, .. } => Some(*decay_lower),
        }
    }

    /// A per-channel layer's low-rank gate and decay ranks: the rows of their
    /// down projections, `attention_project`'s gate and key segments (`GR`,
    /// `K`); none for the per-head form.
    pub(crate) fn ranks(&self) -> Option<[u64; 2]> {
        match self {
            Self::PerHead { .. } => None,
            Self::PerChannel { ranks, .. } => Some(*ranks),
        }
    }
}

/// The recurrent entries take one norm epsilon: the recurrent norm's must be
/// the input norm's. Their output gate is SiLU or sigmoid. They take either
/// the per-head decay with a full gate projection (GDN, `gated_delta_*`) or
/// the per-channel decay with a low-rank one (KDA, `kimi_delta_*`), whose up
/// projections run as `project_rows`.
pub(crate) fn admit<'a>(
    delta: &'a GatedDelta,
    input_epsilon: f64,
) -> Result<AdmittedGatedDelta<'a>, PlanError> {
    if delta.norm.epsilon != input_epsilon {
        return Err(PlanError::Unsupported(
            "gated delta norm epsilon differing from its input norm",
        ));
    }
    let gate_function = AdmittedGateFunction::admit(delta.output_gate.function)?;
    match (&delta.decay, &delta.output_gate.projection) {
        (RecurrentDecay::PerHead { .. }, GateProjection::Full(_)) => {
            Ok(AdmittedGatedDelta::PerHead {
                operator: delta,
                gate_function,
            })
        }
        (
            RecurrentDecay::PerChannel {
                projection,
                gate: DecayGate::BoundedSigmoid { lower },
                ..
            },
            GateProjection::LowRank(gate),
        ) => Ok(AdmittedGatedDelta::PerChannel {
            operator: delta,
            gate_function,
            decay_lower: *lower as f32,
            ranks: [gate.down.shape[0], projection.down.shape[0]],
        }),
        (RecurrentDecay::PerHead { .. }, GateProjection::LowRank(_)) => Err(
            PlanError::Unsupported("gated delta per-head decay with a low-rank gate"),
        ),
        (RecurrentDecay::PerChannel { .. }, GateProjection::Full(_)) => Err(
            PlanError::Unsupported("gated delta per-channel decay with a full gate"),
        ),
    }
}

/// The weights the operator binds, in the order its entries consume them:
/// a low-rank projection's `down` takes the projection segment's role and
/// its `up` the role's `Up`.
pub(super) fn weights<'a>(delta: &'a GatedDelta, push: &mut WeightPush<'_, 'a>) {
    push(WeightKind::RecurrentQueryKeyValue, &delta.query_key_value);
    match &delta.output_gate.projection {
        GateProjection::Full(gate) => push(WeightKind::RecurrentGate, gate),
        GateProjection::LowRank(gate) => {
            push(WeightKind::RecurrentGate, &gate.down);
            push(WeightKind::RecurrentGateUp, &gate.up);
        }
    }
    let (time_bias, decay) = match &delta.decay {
        RecurrentDecay::PerHead {
            alpha,
            time_bias,
            decay,
        } => {
            push(WeightKind::RecurrentAlpha, alpha);
            (time_bias, decay)
        }
        RecurrentDecay::PerChannel {
            projection,
            bias,
            decay,
            gate: _,
        } => {
            push(WeightKind::RecurrentAlpha, &projection.down);
            push(WeightKind::RecurrentAlphaUp, &projection.up);
            (bias, decay)
        }
    };
    push(WeightKind::RecurrentBeta, &delta.beta);
    push(WeightKind::RecurrentConvolution, &delta.convolution);
    push(WeightKind::RecurrentDecay, decay);
    push(WeightKind::RecurrentTimeBias, time_bias);
    push(WeightKind::RecurrentNorm, &delta.norm.weight);
    push(WeightKind::RecurrentOutput, &delta.output);
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(delta: AdmittedGatedDelta<'_>) -> String {
    let operator = delta.operator();
    format!(
        "gated delta {} {} {} {} {:?} {:?} {} {:?}",
        operator.convolution_width,
        operator.key_heads,
        operator.value_heads,
        operator.width,
        operator.head_mapping,
        operator.output_gate.function,
        operator.key_norm_epsilon,
        delta.decay_lower(),
    )
}

/// The program binding of a gated delta sublayer; `lookup` resolves the
/// planned element of a role in its scope, `scalable` that of a projection
/// bound with an accumulator-scale port: the output's (`attention_output`)
/// and KDA's input projections' (`attention_project`, `project_rows`).
/// `gated_delta_project` has none, so GDN's input projections take `lookup`.
pub(super) fn binding(
    delta: AdmittedGatedDelta<'_>,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<crate::ScalableWeight, PlanError>,
    activation: Element,
) -> Result<RecurrentBinding, PlanError> {
    let operator = delta.operator();
    let form = match delta {
        AdmittedGatedDelta::PerHead { .. } => RecurrentForm::PerHead,
        AdmittedGatedDelta::PerChannel {
            decay_lower,
            ranks: [gate_rank, alpha_rank],
            ..
        } => RecurrentForm::PerChannel {
            decay_lower: decay_lower.to_bits(),
            gate_up: scalable(WeightKind::RecurrentGateUp)?.element,
            alpha_up: scalable(WeightKind::RecurrentAlphaUp)?.element,
            gate_rank,
            alpha_rank,
        },
    };
    let projection = |kind| match form {
        RecurrentForm::PerHead => lookup(kind),
        RecurrentForm::PerChannel { .. } => Ok(scalable(kind)?.element),
    };
    Ok(RecurrentBinding {
        key_heads: operator.key_heads,
        value_heads: operator.value_heads,
        width: operator.width,
        convolution_width: operator.convolution_width,
        decay_width: operator.decay_width(),
        key_norm_epsilon: (operator.key_norm_epsilon as f32).to_bits(),
        gate_function: delta.gate_function().code(),
        norm: lookup(WeightKind::InputNorm)?,
        qkv: projection(WeightKind::RecurrentQueryKeyValue)?,
        gate: projection(WeightKind::RecurrentGate)?,
        alpha: projection(WeightKind::RecurrentAlpha)?,
        beta: projection(WeightKind::RecurrentBeta)?,
        recurrent_norm: lookup(WeightKind::RecurrentNorm)?,
        output: scalable(WeightKind::RecurrentOutput)?.element,
        activation,
        form,
    })
}
