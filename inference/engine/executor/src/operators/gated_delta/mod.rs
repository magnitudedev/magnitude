//! Gated delta-rule recurrence (`Operator::GatedDelta`, Qwen 3.5): the
//! form the recurrent entries implement, its weight roles and program
//! binding.

pub(crate) mod graph;

use super::{PlanError, WeightPush};
use crate::RecurrentBinding;
use magnitude_family_contracts::{GateFunction, GatedDelta, WeightKind};
use seismic::Element;

/// The recurrent entries take one norm epsilon: the recurrent norm's must be
/// the input norm's. Their output gate is SiLU or sigmoid.
pub(super) fn admit(delta: &GatedDelta, input_epsilon: f64) -> Result<(), PlanError> {
    if delta.norm.epsilon != input_epsilon {
        return Err(PlanError::Unsupported(
            "gated delta norm epsilon differing from its input norm",
        ));
    }
    if delta.output_gate == GateFunction::Softplus {
        return Err(PlanError::Unsupported("softplus gated delta output gate"));
    }
    Ok(())
}

/// The recurrent entries' `gate_function` code: 0 SiLU, 1 sigmoid
/// (`output_gate` in functions.seismic). [`admit`] rejects every other gate.
pub(crate) fn gate_function(delta: &GatedDelta) -> i32 {
    match delta.output_gate {
        GateFunction::Silu => 0,
        GateFunction::Sigmoid => 1,
        GateFunction::Softplus => unreachable!("admission rejects softplus gated delta gates"),
    }
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(delta: &'a GatedDelta, push: &mut WeightPush<'_, 'a>) {
    push(WeightKind::RecurrentQueryKeyValue, &delta.query_key_value);
    push(WeightKind::RecurrentGate, &delta.gate);
    push(WeightKind::RecurrentAlpha, &delta.alpha);
    push(WeightKind::RecurrentBeta, &delta.beta);
    push(WeightKind::RecurrentConvolution, &delta.convolution);
    push(WeightKind::RecurrentDecay, &delta.decay);
    push(WeightKind::RecurrentTimeBias, &delta.time_bias);
    push(WeightKind::RecurrentNorm, &delta.norm.weight);
    push(WeightKind::RecurrentOutput, &delta.output);
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(delta: &GatedDelta) -> String {
    format!(
        "gated delta {} {} {} {} {:?} {:?}",
        delta.convolution_width,
        delta.key_heads,
        delta.value_heads,
        delta.width,
        delta.head_mapping,
        delta.output_gate
    )
}

/// The program binding of a gated delta sublayer; `lookup` resolves the
/// planned element of a role in its scope, `scalable` that of a projection
/// bound with an accumulator-scale port (the output's, `attention_output`;
/// `gated_delta_project` has none, so its weights take `lookup`).
pub(super) fn binding(
    delta: &GatedDelta,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<crate::ScalableWeight, PlanError>,
    activation: Element,
) -> Result<RecurrentBinding, PlanError> {
    Ok(RecurrentBinding {
        key_heads: delta.key_heads,
        value_heads: delta.value_heads,
        width: delta.width,
        convolution_width: delta.convolution_width,
        norm: lookup(WeightKind::InputNorm)?,
        qkv: lookup(WeightKind::RecurrentQueryKeyValue)?,
        gate: lookup(WeightKind::RecurrentGate)?,
        alpha: lookup(WeightKind::RecurrentAlpha)?,
        beta: lookup(WeightKind::RecurrentBeta)?,
        recurrent_norm: lookup(WeightKind::RecurrentNorm)?,
        output: scalable(WeightKind::RecurrentOutput)?.element,
        activation,
    })
}
