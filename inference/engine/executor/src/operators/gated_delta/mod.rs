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

/// The recurrent entries take one norm epsilon: the recurrent norm's must be
/// the input norm's. Their output gate is SiLU or sigmoid. They take either
/// the per-head decay with a full gate projection (GDN, `gated_delta_*`) or
/// the per-channel decay with a low-rank one (KDA, `kimi_delta_*`), whose up
/// projections run as `project_rows`.
pub(super) fn admit(delta: &GatedDelta, input_epsilon: f64) -> Result<(), PlanError> {
    if delta.norm.epsilon != input_epsilon {
        return Err(PlanError::Unsupported(
            "gated delta norm epsilon differing from its input norm",
        ));
    }
    if delta.output_gate.function == GateFunction::Softplus {
        return Err(PlanError::Unsupported("softplus gated delta output gate"));
    }
    match (&delta.decay, &delta.output_gate.projection) {
        (RecurrentDecay::PerHead { .. }, GateProjection::Full(_))
        | (RecurrentDecay::PerChannel { .. }, GateProjection::LowRank(_)) => Ok(()),
        (RecurrentDecay::PerHead { .. }, GateProjection::LowRank(_)) => Err(
            PlanError::Unsupported("gated delta per-head decay with a low-rank gate"),
        ),
        (RecurrentDecay::PerChannel { .. }, GateProjection::Full(_)) => Err(
            PlanError::Unsupported("gated delta per-channel decay with a full gate"),
        ),
    }
}

/// The recurrent entries' `gate_function` code: 0 SiLU, 1 sigmoid
/// (`output_gate` in functions.seismic). [`admit`] rejects every other gate.
pub(crate) fn gate_function(delta: &GatedDelta) -> i32 {
    match delta.output_gate.function {
        GateFunction::Silu => 0,
        GateFunction::Sigmoid => 1,
        GateFunction::Softplus => unreachable!("admission rejects softplus gated delta gates"),
    }
}

/// The per-channel decay's lower bound (the `kimi_delta_*` entries'
/// `decay_lower`): none for the per-head decay.
pub(crate) fn decay_lower(delta: &GatedDelta) -> Option<f32> {
    match &delta.decay {
        RecurrentDecay::PerHead { .. } => None,
        RecurrentDecay::PerChannel {
            gate: DecayGate::BoundedSigmoid { lower },
            ..
        } => Some(*lower as f32),
    }
}

/// A per-channel layer's low-rank gate and decay ranks: the rows of their
/// down projections, `attention_project`'s gate and key segments (`GR`,
/// `K`); none for the per-head form.
pub(crate) fn ranks(delta: &GatedDelta) -> Option<[u64; 2]> {
    match (&delta.decay, &delta.output_gate.projection) {
        (RecurrentDecay::PerHead { .. }, _) => None,
        (RecurrentDecay::PerChannel { projection, .. }, GateProjection::LowRank(gate)) => {
            Some([gate.down.shape[0], projection.down.shape[0]])
        }
        (RecurrentDecay::PerChannel { .. }, GateProjection::Full(_)) => {
            unreachable!("admission rejects a per-channel decay with a full gate")
        }
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
pub(super) fn shape_key(delta: &GatedDelta) -> String {
    format!(
        "gated delta {} {} {} {} {:?} {:?} {} {:?}",
        delta.convolution_width,
        delta.key_heads,
        delta.value_heads,
        delta.width,
        delta.head_mapping,
        delta.output_gate.function,
        delta.key_norm_epsilon,
        decay_lower(delta),
    )
}

/// The program binding of a gated delta sublayer; `lookup` resolves the
/// planned element of a role in its scope, `scalable` that of a projection
/// bound with an accumulator-scale port: the output's (`attention_output`)
/// and KDA's input projections' (`attention_project`, `project_rows`).
/// `gated_delta_project` has none, so GDN's input projections take `lookup`.
pub(super) fn binding(
    delta: &GatedDelta,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<crate::ScalableWeight, PlanError>,
    activation: Element,
) -> Result<RecurrentBinding, PlanError> {
    let form = match (ranks(delta), decay_lower(delta)) {
        (None, None) => RecurrentForm::PerHead,
        (Some([gate_rank, alpha_rank]), Some(decay_lower)) => RecurrentForm::PerChannel {
            decay_lower: decay_lower.to_bits(),
            gate_up: scalable(WeightKind::RecurrentGateUp)?.element,
            alpha_up: scalable(WeightKind::RecurrentAlphaUp)?.element,
            gate_rank,
            alpha_rank,
        },
        _ => unreachable!("a per-channel decay has low-rank projections and a lower bound, a per-head decay neither"),
    };
    let projection = |kind| match form {
        RecurrentForm::PerHead => lookup(kind),
        RecurrentForm::PerChannel { .. } => Ok(scalable(kind)?.element),
    };
    Ok(RecurrentBinding {
        key_heads: delta.key_heads,
        value_heads: delta.value_heads,
        width: delta.width,
        convolution_width: delta.convolution_width,
        decay_width: delta.decay_width(),
        key_norm_epsilon: (delta.key_norm_epsilon as f32).to_bits(),
        gate_function: gate_function(delta),
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
