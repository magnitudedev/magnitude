//! Parallel branches (`Operator::Parallel`) in the one form the executor
//! runs: a gated dense branch beside a general routed branch over the same
//! residual (Gemma 4 26B), joined by `moe_tail` through the sublayer's
//! post-norm tail.

pub(crate) mod graph;

use super::{dense_ffn, routed, tail, PlanError};
use crate::{DenseBranchBinding, ParallelBinding, SublayerTail};
use magnitude_family_contracts::{
    Branch, BranchOutput, DenseFfn, InputNorm, Operator, OutputForm, RmsNorm, RoutedFfn,
    RouterInput, SublayerIndex, WeightKind, WeightScope,
};
use seismic::Element;

/// A dense branch beside a general routed branch over the same residual,
/// each normalized on entry and on exit; `moe_tail` normalizes their sum
/// through the sublayer's post-norm tail. Branch 0 is the dense branch,
/// branch 1 the routed one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseBesideRouted<'a> {
    pub dense: &'a DenseFfn,
    pub dense_input: &'a RmsNorm,
    pub dense_output: &'a RmsNorm,
    pub routed: &'a RoutedFfn,
    pub routed_input: &'a RmsNorm,
    pub routed_output: &'a RmsNorm,
}

impl DenseBesideRouted<'_> {
    /// The branches' weight scopes (dense, routed) in target sublayer
    /// `sublayer`.
    pub fn scopes(sublayer: SublayerIndex) -> [WeightScope; 2] {
        [0, 1].map(|branch| WeightScope::TargetBranch { sublayer, branch })
    }

    /// The norms the sublayer's kernels share one epsilon with.
    pub(super) fn norms(&self) -> [&RmsNorm; 4] {
        [
            self.dense_input,
            self.dense_output,
            self.routed_input,
            self.routed_output,
        ]
    }
}

/// The parallel form the executor runs: a gated dense branch then a general
/// routed branch, each with an RMS input norm and an RMS output norm, in a
/// sublayer without an input norm and with a post-norm tail.
pub(super) fn admit<'a>(
    branches: &'a [Branch],
    input: &InputNorm,
    output: &OutputForm,
) -> Result<DenseBesideRouted<'a>, PlanError> {
    let form = || PlanError::Unsupported("parallel branches other than dense beside routed");
    if *input != InputNorm::None || *output == OutputForm::Residual {
        return Err(PlanError::Unsupported("parallel branches sublayer form"));
    }
    let [dense, routed] = branches else {
        return Err(form());
    };
    let (
        Branch {
            input: InputNorm::Rms(dense_input),
            op: Operator::DenseFfn(dense_op),
            output: BranchOutput::Norm(dense_output),
        },
        Branch {
            input: InputNorm::Rms(routed_input),
            op: Operator::RoutedFfn(routed_op),
            output: BranchOutput::Norm(routed_output),
        },
    ) = (dense, routed)
    else {
        return Err(form());
    };
    dense_ffn::admit(dense_op)?;
    if routed::is_fused(routed_op) {
        return Err(form());
    }
    routed::admit(routed_op)?;
    if let RouterInput::Residual(norm) = &routed_op.router.input {
        if norm.epsilon != routed_input.epsilon {
            return Err(PlanError::Unsupported(
                "expert router norm epsilon differing from its input norm",
            ));
        }
    }
    Ok(DenseBesideRouted {
        dense: dense_op,
        dense_input,
        dense_output,
        routed: routed_op,
        routed_input,
        routed_output,
    })
}

/// Every numerical parameter of the sublayer a sealed graph holds apart
/// from its weights.
pub(super) fn shape_key(branches: DenseBesideRouted<'_>) -> String {
    format!(
        "{} beside routed {:?}",
        dense_ffn::shape_key(branches.dense),
        routed::GeneralRoutedShape::of(0, branches.routed)
    )
}

/// The program binding of a parallel sublayer `sublayer` with `output`;
/// `lookup` resolves the planned element of a role in a scope. `moe_tail`
/// reads the branch and tail norms as one element.
pub(super) fn binding(
    branches: DenseBesideRouted<'_>,
    sublayer: SublayerIndex,
    output: &OutputForm,
    hidden: u64,
    lookup: impl Fn(WeightScope, WeightKind) -> Result<Element, PlanError>,
    activation: Element,
) -> Result<ParallelBinding, PlanError> {
    let [dense, routed] = DenseBesideRouted::scopes(sublayer);
    let SublayerTail::PostNorm { norm, scaled } = tail(output, &|kind| {
        lookup(WeightScope::TargetSublayer(sublayer), kind)
    })?
    else {
        return Err(PlanError::Unsupported("parallel branches without a post-norm tail"));
    };
    if [dense, routed]
        .into_iter()
        .map(|branch| lookup(branch, WeightKind::PostNorm))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .any(|element| *element != norm)
    {
        return Err(PlanError::Unsupported(
            "parallel branch norms resident unlike the tail norm",
        ));
    }
    Ok(ParallelBinding {
        hidden,
        dense: DenseBranchBinding {
            features: branches.dense.intermediate,
            norm: lookup(dense, WeightKind::InputNorm)?,
            gate: lookup(dense, WeightKind::DenseGate)?,
            up: lookup(dense, WeightKind::DenseUp)?,
            down: lookup(dense, WeightKind::DenseDown)?,
            activation,
        },
        // The parallel graph binds no scale ports: `lookup` refuses scaled
        // weights.
        routed: routed::binding(
            branches.routed,
            hidden,
            |kind| lookup(routed, kind),
            |kind| {
                lookup(routed, kind).map(|element| crate::ScalableWeight { element, scale: 0 })
            },
            activation,
        )?,
        norm,
        scaled,
    })
}
