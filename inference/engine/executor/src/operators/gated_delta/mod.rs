//! Gated delta-rule recurrence (`Operator::GatedDelta`, Qwen 3.5): the
//! form the recurrent entries implement, its weight roles and program
//! binding.

pub(crate) mod graph;

use super::{PlanError, WeightPush};
use crate::RecurrentBinding;
use magnitude_family_contracts::{GatedDelta, WeightKind};
use seismic::Element;

/// The recurrent entries take one norm epsilon: the recurrent norm's must be
/// the input norm's.
pub(super) fn admit(delta: &GatedDelta, input_epsilon: f64) -> Result<(), PlanError> {
    if delta.norm.epsilon != input_epsilon {
        return Err(PlanError::Unsupported(
            "gated delta norm epsilon differing from its input norm",
        ));
    }
    Ok(())
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
        "gated delta {} {} {} {} {:?}",
        delta.convolution_width, delta.key_heads, delta.value_heads, delta.width, delta.head_mapping
    )
}

/// The program binding of a gated delta sublayer; `lookup` resolves the
/// planned element of a role in its scope.
pub(super) fn binding(
    delta: &GatedDelta,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
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
        output: lookup(WeightKind::RecurrentOutput)?,
        activation,
    })
}
