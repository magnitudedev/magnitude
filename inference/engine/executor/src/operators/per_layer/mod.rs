//! Per-layer input sublayers (`Operator::PerLayerInput`, Gemma PLE,
//! model-family plan §3.7): the form the per-layer entries implement, its
//! weight roles and program binding. The per-layer rows they read come from
//! the decoder's per-layer entry ([`graph::per_layer_entry`]).

pub(crate) mod graph;

use super::{post_norm_epsilon, tail, PlanError, WeightPush};
use crate::PerLayerBinding;
use magnitude_family_contracts::{
    ActivationFunction, InputNorm, OutputForm, PerLayerEntry, PerLayerInput, Sublayer, WeightKind,
};
use seismic::Element;

/// The per-layer input sublayer of a [`super::PairedBlock`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct PerLayerSublayer<'a> {
    pub op: &'a PerLayerInput,
    pub output: &'a OutputForm,
}

/// The form `per_layer_gate` implements: an unnormalized input, a GELU-tanh
/// gate and a post-norm tail.
pub(super) fn admit(sublayer: &Sublayer) -> Result<PerLayerSublayer<'_>, PlanError> {
    let magnitude_family_contracts::Operator::PerLayerInput(op) = &sublayer.op else {
        return Err(super::unsupported_operator(&sublayer.op, "per-layer input"));
    };
    if sublayer.input != InputNorm::None
        || !matches!(
            sublayer.output,
            OutputForm::PostNorm(_) | OutputForm::ScaledPostNorm { .. }
        )
        || op.activation != ActivationFunction::GeluTanh
    {
        return Err(PlanError::Unsupported("per-layer input sublayer form"));
    }
    Ok(PerLayerSublayer {
        op,
        output: &sublayer.output,
    })
}

/// The weights the operator binds, in the order its entries consume them.
pub(super) fn weights<'a>(input: &'a PerLayerInput, push: &mut WeightPush<'_, 'a>) {
    push(WeightKind::PerLayerGate, &input.gate);
    push(WeightKind::PerLayerProjection, &input.projection);
}

/// Every numerical parameter of the sublayer a sealed graph holds apart
/// from its weights: it reads its layer's slice of the per-layer rows by
/// index.
pub(super) fn shape_key(sublayer: PerLayerSublayer<'_>) -> String {
    format!(
        "per-layer {} {} {:?}",
        sublayer.op.layer,
        sublayer.op.width,
        post_norm_epsilon(sublayer.output)
    )
}

/// The program binding of a per-layer input sublayer over the decoder's
/// per-layer `entry`; `lookup` resolves the planned element of a role in its
/// scope.
pub(crate) fn binding(
    sublayer: PerLayerSublayer<'_>,
    entry: &PerLayerEntry,
    hidden: u64,
    lookup: impl Fn(WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<PerLayerBinding, PlanError> {
    Ok(PerLayerBinding {
        hidden,
        layers: entry.layers,
        width: entry.width,
        gate: lookup(WeightKind::PerLayerGate)?,
        projection: lookup(WeightKind::PerLayerProjection)?,
        activation,
        tail: tail(sublayer.output, &lookup)?,
    })
}
