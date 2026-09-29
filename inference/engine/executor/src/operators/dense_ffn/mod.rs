//! Dense feed-forward (`Operator::DenseFfn`): the gated forms
//! `dense_expand` computes, their activation codes, weight roles and program
//! binding.

pub(crate) mod graph;

use super::{tail, PlanError, WeightPush};
use crate::{DenseBinding, DenseScales, ScalableWeight};
use magnitude_family_contracts::{
    ActivationFunction, DenseFfn, FeedForwardUp, OutputForm, WeightKind,
};
use seismic::Element;

/// The gated activations `dense_expand` computes.
pub(super) fn admit(dense: &DenseFfn) -> Result<(), PlanError> {
    match dense.up {
        FeedForwardUp::Gated {
            activation: ActivationFunction::Silu | ActivationFunction::GeluTanh,
            ..
        } => Ok(()),
        _ => Err(PlanError::Unsupported("feed-forward activation")),
    }
}

/// The feed-forward entries' `activation` code: 0 SiLU, 1 GELU-tanh,
/// 2 ReLU².
pub(crate) fn activation_code(function: ActivationFunction) -> i32 {
    match function {
        ActivationFunction::Silu => 0,
        ActivationFunction::GeluTanh => 1,
        ActivationFunction::ReluSquared => 2,
    }
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(dense: &'a DenseFfn, push: &mut WeightPush<'_, 'a>) {
    if let Some(gate) = dense.up.gate() {
        push(WeightKind::DenseGate, gate);
    }
    push(WeightKind::DenseUp, dense.up.up());
    push(WeightKind::DenseDown, &dense.down);
}

/// Every numerical parameter of the operator a sealed graph holds apart from
/// its weights.
pub(super) fn shape_key(dense: &DenseFfn) -> String {
    format!("dense {} {:?}", dense.intermediate, dense.up.activation())
}

/// The program binding of a dense sublayer with `output`; `lookup` resolves
/// the planned element of a role in its scope, `scalable` that of a
/// projection its entries bind with an accumulator-scale port.
pub(super) fn binding(
    dense: &DenseFfn,
    output: &OutputForm,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    scalable: impl Fn(WeightKind) -> Result<ScalableWeight, PlanError>,
    activation: Element,
) -> Result<DenseBinding, PlanError> {
    let (gate, up, down) = (
        scalable(WeightKind::DenseGate)?,
        scalable(WeightKind::DenseUp)?,
        scalable(WeightKind::DenseDown)?,
    );
    Ok(DenseBinding {
        features: dense.intermediate,
        norm: lookup(WeightKind::InputNorm)?,
        gate: gate.element,
        up: up.element,
        down: down.element,
        activation,
        tail: tail(output, &lookup)?,
        scales: DenseScales {
            gate: gate.scale,
            up: up.scale,
            down: down.scale,
        },
    })
}
